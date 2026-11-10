"""Day 3 features: clues that still work when the ball is missing.

    python possession/features.py

Reads data/derived/frames.parquet and atd.npz, writes data/derived/features.parquet.

What it adds, and why (see the health report):
  * our own ball speed     the provided ball speed column is not a ball speed (median 1.3)
  * player movement        player speeds are in m/s; everyone walking is a stoppage clue
  * camera aim             the operator follows the ball; homography is valid on every frame,
                           so the pitch point at the image centre tracks the ball even when
                           the tracker has lost it (missing stretches last 10 to 30 s)
  * ball proxy             real ball when visible, otherwise camera aim corrected by a fit
                           learned on frames where both are known (no labels involved)
  * team shape             who is pushed up, who is compact, which way teams are moving
  * tag votes              tag names say whether the tagged team has the ball or not
  * dead ball clues        ball outside the lines, ball standing still, players walking
No feature reads a label.
"""

import os
import warnings

import numpy as np
import pandas as pd

import sources as D
from build_table import interval_mask

warnings.filterwarnings("ignore", category=RuntimeWarning)
L, W = D.PITCH_LENGTH, D.PITCH_WIDTH

# Tag code -> does the tagged team have the ball?
HAS_BALL_CODES = ["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                  "ATTACKING TRANSITION", "LONG BALL", "BALL IN THE BOX", "PLAYERS IN THE BOX"]
NO_BALL_CODES = ["DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                 "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"]


def same_half_shift(values, half, k):
    """values shifted by k frames, NaN where the shift would cross a half boundary."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    n = len(values)
    if k > 0:
        out[:n - k] = values[k:]
        bad = np.r_[half[:n - k] != half[k:], np.ones(k, bool)]
    else:
        k = -k
        out[k:] = values[:n - k]
        bad = np.r_[np.ones(k, bool), half[k:] != half[:n - k]]
    out[bad] = np.nan
    return out


def rolling(series, window, how="mean"):
    roll = pd.Series(series).rolling(window, center=True, min_periods=max(1, window // 5))
    return getattr(roll, how)().values.astype(np.float32)


def image_box(H, players, frames):
    """Where players appear in pixels on sample frames -> (u_min, u_max, v_min, v_max).

    The tracker only follows players the camera can see, so the spread of their pixel
    positions outlines the image. Percentiles guard against a few predicted off screen tracks.
    """
    Hs = H[frames].reshape(-1, 3, 3)
    x, y = players[frames, :, 0], players[frames, :, 1]
    P = np.einsum("mij,mkj->mki", Hs, np.stack([x, y, np.ones_like(x)], axis=-1))
    u, v = P[..., 0] / P[..., 2], P[..., 1] / P[..., 2]
    return (np.nanpercentile(u, 0.5), np.nanpercentile(u, 99.5),
            np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5))


def main():
    frame = pd.read_parquet(os.path.join(D.DERIVED_DIR, "frames.parquet"))
    atd = np.load(os.path.join(D.DERIVED_DIR, "atd.npz"))
    players, teams = atd["players"], atd["teams"]
    tags = pd.read_csv(os.path.join(D.DERIVED_DIR, "tags.csv"), keep_default_na=False)
    H, _, valid = D.load_homography(D.require(["homography"])[0])
    half = frame["half"].values
    scored = half > 0
    home, away = teams == D.HOME, teams == D.AWAY
    f = {}

    print("\n1. Ball speed from positions")
    bx, by = frame["ball_x"].values.astype(np.float64), frame["ball_y"].values.astype(np.float64)
    k = 2
    vx = (same_half_shift(bx, half, k) - same_half_shift(bx, half, -k)) / (2 * k / D.FPS)
    vy = (same_half_shift(by, half, k) - same_half_shift(by, half, -k)) / (2 * k / D.FPS)
    speed = np.hypot(vx, vy)
    speed[speed > 45] = np.nan                      # > 160 km/h is a tracking jump, not a kick
    f["ball_speed_own"] = speed.astype(np.float32)
    f["ball_speed_1s"] = rolling(speed, 25, "median")
    f["ball_still_1s"] = (f["ball_speed_1s"] < 0.7).astype(np.int8)
    f["ball_fast"] = (speed > 8).astype(np.int8)    # a pass or a shot, ball not under control
    vis = np.isfinite(speed) & scored
    print("   own ball speed m/s  median {:.1f}  p90 {:.1f}  p99 {:.1f}".format(
        *np.percentile(speed[vis], [50, 90, 99])))

    print("\n2. Player movement (provided speeds, m/s)")
    ps = players[:, :, 2].astype(np.float64)
    present = np.isfinite(players[:, :, 0])
    ps[~present] = np.nan
    f["speed_mean_all"] = np.nanmean(ps, axis=1).astype(np.float32)
    f["speed_mean_home"] = np.nanmean(ps[:, home], axis=1).astype(np.float32)
    f["speed_mean_away"] = np.nanmean(ps[:, away], axis=1).astype(np.float32)
    f["share_walking"] = (np.nansum(ps < 2.0, axis=1) / np.maximum(present.sum(axis=1), 1)
                          ).astype(np.float32)
    f["speed_mean_all_2s"] = rolling(f["speed_mean_all"], 50)
    # Stoppages show up as the whole pitch slowing to a walk, over seconds rather than frames.
    for secs in (3, 5, 10):
        f["speed_mean_all_{}s".format(secs)] = rolling(f["speed_mean_all"], secs * D.FPS)
    f["speed_drop_3s"] = (f["speed_mean_all_3s"]
                          - same_half_shift(f["speed_mean_all_3s"], half, -3 * D.FPS)).astype(np.float32)
    f["speed_rise_3s"] = (same_half_shift(f["speed_mean_all_3s"], half, 3 * D.FPS)
                          - f["speed_mean_all_3s"]).astype(np.float32)
    print("   mean player speed, scored frames: median {:.2f} m/s".format(
        np.nanmedian(f["speed_mean_all"][scored])))
    print("   share of scored frames with 3 s mean speed under 1.2 m/s: {:.1f}%".format(
        100 * np.mean(f["speed_mean_all_3s"][scored] < 1.2)))

    # How far the ball travels over 3 s, from 1 s medians at each end (robust to jitter).
    mx, my = rolling(bx, 25, "median"), rolling(by, 25, "median")
    f["ball_disp_3s"] = np.hypot(same_half_shift(mx, half, 37) - same_half_shift(mx, half, -37),
                                 same_half_shift(my, half, 37) - same_half_shift(my, half, -37)
                                 ).astype(np.float32)

    print("\n3. Camera aim from homography")
    sample = np.flatnonzero(scored & valid)[::50]
    u0, u1, v0, v1 = image_box(H, players, sample)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    print("   players appear in pixels u {:.0f}..{:.0f}, v {:.0f}..{:.0f}; image centre ({:.0f}, {:.0f})"
          .format(u0, u1, v0, v1, cu, cv))
    Hinv = np.full((D.N_FRAMES, 3, 3), np.nan)
    Hinv[valid] = np.linalg.inv(H[valid].reshape(-1, 3, 3))
    q = Hinv @ np.array([cu, cv, 1.0])
    aim_x, aim_y = q[:, 0] / q[:, 2] * L, q[:, 1] / q[:, 2] * W
    f["aim_x"], f["aim_y"] = aim_x.astype(np.float32), aim_y.astype(np.float32)
    f["camera_pan_1s"] = np.abs(same_half_shift(aim_x, half, 12) - same_half_shift(aim_x, half, -12)
                                ).astype(np.float32)

    print("\n4. Ball proxy: fit ball position from camera aim on frames where both are known")
    fit = scored & np.isfinite(bx) & np.isfinite(aim_x)
    A = np.column_stack([np.ones(fit.sum()), aim_x[fit], aim_y[fit]])
    cx, *_ = np.linalg.lstsq(A, bx[fit], rcond=None)
    cy, *_ = np.linalg.lstsq(A, by[fit], rcond=None)
    full = np.column_stack([np.ones(D.N_FRAMES), aim_x, aim_y])
    est_x, est_y = full @ cx, full @ cy
    err = np.hypot(est_x[fit] - bx[fit], est_y[fit] - by[fit])
    raw = np.hypot(aim_x[fit] - bx[fit], aim_y[fit] - by[fit])
    print("   distance camera aim -> real ball   median {:.1f} m (raw)  {:.1f} m (fitted)  p90 {:.1f} m"
          .format(np.median(raw), np.median(err), np.percentile(err, 90)))
    visible = np.isfinite(bx)
    prox_x = np.where(visible, bx, est_x)
    prox_y = np.where(visible, by, est_y)
    f["proxy_x"], f["proxy_y"] = prox_x.astype(np.float32), prox_y.astype(np.float32)

    px = players[:, :, 0].astype(np.float64) * L
    py = players[:, :, 1].astype(np.float64) * W

    def around(point_x, point_y, tag):
        """Team counts and nearest distances around a point. `tag` names the point."""
        d = np.hypot(px - point_x[:, None], py - point_y[:, None])
        hs, as_ = np.sort(d[:, home], axis=1), np.sort(d[:, away], axis=1)
        f["near_home_" + tag] = hs[:, 0].astype(np.float32)
        f["near_away_" + tag] = as_[:, 0].astype(np.float32)
        f["gap_" + tag] = (as_[:, 0] - hs[:, 0]).astype(np.float32)
        for r in (5, 10, 20):
            f["home_within_{}m_{}".format(r, tag)] = (d[:, home] < r).sum(axis=1).astype(np.int8)
            f["away_within_{}m_{}".format(r, tag)] = (d[:, away] < r).sum(axis=1).astype(np.int8)
        f["gap_{}_1s".format(tag)] = rolling(f["gap_" + tag], 25)
        f["gap_{}_5s".format(tag)] = rolling(f["gap_" + tag], 125)

    # proxy: real ball when visible (used by the model that may see the ball)
    # cam:   camera estimate on every frame (used by the ball blind model, so that what it
    #        learns on visible frames looks the same as what it meets on missing ones)
    around(prox_x, prox_y, "proxy")
    around(est_x, est_y, "cam")
    f["cam_x"], f["cam_y"] = est_x.astype(np.float32), est_y.astype(np.float32)

    print("\n5. Team shape, normalised so +x is the direction Barcelona attack")
    direction = np.zeros(D.N_FRAMES)
    for h in (1, 2):
        m = half == h
        sign = 1.0 if np.nanmean(frame.loc[m, "home_centroid_x"]) < np.nanmean(
            frame.loc[m, "away_centroid_x"]) else -1.0
        direction[m] = sign
        print("   half {}: Barcelona attack {}".format(h, "+x (left to right)" if sign > 0
                                                       else "-x (right to left)"))
    hx = np.where(present[:, home], px[:, home], np.nan)
    ax_ = np.where(present[:, away], px[:, away], np.nan)
    hy = np.where(present[:, home], py[:, home], np.nan)
    ay = np.where(present[:, away], py[:, away], np.nan)
    hc, ac = np.nanmean(hx, axis=1), np.nanmean(ax_, axis=1)
    f["proxy_x_attack"] = ((prox_x - L / 2) * direction).astype(np.float32)
    f["cam_x_attack"] = ((est_x - L / 2) * direction).astype(np.float32)
    f["cam_to_touchline"] = np.minimum(est_y, W - est_y).astype(np.float32)
    f["cam_to_goal_line"] = np.minimum(est_x, L - est_x).astype(np.float32)
    f["home_push"] = ((hc - ac) * direction).astype(np.float32)
    f["home_spread_x"] = np.nanstd(hx, axis=1).astype(np.float32)
    f["away_spread_x"] = np.nanstd(ax_, axis=1).astype(np.float32)
    f["home_spread_y"] = np.nanstd(hy, axis=1).astype(np.float32)
    f["away_spread_y"] = np.nanstd(ay, axis=1).astype(np.float32)
    f["home_forward_1s"] = ((same_half_shift(hc, half, 12) - same_half_shift(hc, half, -12))
                            * direction).astype(np.float32)
    f["away_forward_1s"] = ((same_half_shift(ac, half, 12) - same_half_shift(ac, half, -12))
                            * -direction).astype(np.float32)

    print("\n6. Tag votes")
    vote_home = np.zeros(D.N_FRAMES, np.int16)
    vote_away = np.zeros(D.N_FRAMES, np.int16)
    for codes, ours in ((HAS_BALL_CODES, True), (NO_BALL_CODES, False)):
        for team, is_home in (("Barcelona", True), ("Atletic Club", False)):
            rows = tags[tags["code"].isin(codes) & (tags["team"] == team)]
            mask = interval_mask(rows["start"], rows["end"])
            if is_home == ours:
                vote_home += mask
            else:
                vote_away += mask
    f["tag_vote_home"], f["tag_vote_away"] = vote_home, vote_away
    f["tag_vote"] = (vote_home - vote_away).astype(np.int16)

    # Label free consistency check: on frames where proximity alone is decisive,
    # do the tag votes point the same way? If the code mapping were backwards this is < 50%.
    gap = frame["gap_away_minus_home"].values
    clear = scored & (np.fmin(frame["near_home"], frame["near_away"]).values < 1.2) & (np.abs(gap) > 3)
    voted = clear & (f["tag_vote"] != 0)
    agree = np.mean(np.sign(f["tag_vote"][voted]) == np.sign(gap[voted]))
    print("   tag vote agrees with clear proximity on {:.1f}% of {} frames".format(
        100 * agree, voted.sum()))

    print("\n7. Dead ball clues")
    out = visible & ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1))
    f["ball_out"] = out.astype(np.int8)
    f["ball_out_1s"] = rolling(out.astype(float), 25)
    f["ball_visible_5s"] = rolling(visible.astype(float), 125)
    print("   ball clearly outside the lines on {:.1f}% of scored frames".format(
        100 * out[scored].mean()))

    print("\n8. Timing clues: since the ball went out, and around set piece tags")
    cap = 60 * D.FPS

    def since_event(event):
        """Frames since the last event frame, within the same half; capped at 60 s."""
        result = np.full(D.N_FRAMES, cap, np.int32)
        edges = np.flatnonzero(np.diff(half)) + 1
        for first, last in zip(np.r_[0, edges], np.r_[edges, D.N_FRAMES]):
            idx = np.arange(last - first)
            seen = np.maximum.accumulate(np.where(event[first:last], idx, -1))
            result[first:last] = np.where(seen >= 0, np.minimum(idx - seen, cap), cap)
        return result

    def until_event(event):
        reversed_since = since_event_reversed(event[::-1])
        return reversed_since[::-1]

    def since_event_reversed(event):
        # same as since_event on a reversed timeline, with the half boundaries reversed too
        result = np.full(D.N_FRAMES, cap, np.int32)
        rhalf = half[::-1]
        edges = np.flatnonzero(np.diff(rhalf)) + 1
        for first, last in zip(np.r_[0, edges], np.r_[edges, D.N_FRAMES]):
            idx = np.arange(last - first)
            seen = np.maximum.accumulate(np.where(event[first:last], idx, -1))
            result[first:last] = np.where(seen >= 0, np.minimum(idx - seen, cap), cap)
        return result

    f["since_ball_out"] = since_event(out)
    starts = np.zeros(D.N_FRAMES, bool)
    set_pieces = tags[tags["code"] == "SET PIECES"]
    starts[np.clip(set_pieces["start"].values, 0, D.N_FRAMES - 1)] = True
    f["since_setpiece_start"] = since_event(starts)
    f["until_setpiece_start"] = until_event(starts)
    f["since_any_tag_start"] = since_event(np.isin(np.arange(D.N_FRAMES),
                                                   np.clip(tags["start"].values, 0, D.N_FRAMES - 1)))

    print("\n9. Control: who last had the ball under control, and who has it next")
    # CHALLENGE.md: possession changes only when the other team gains CONTROLLED possession;
    # while the ball is loose or contested it stays with the last team. A controlled touch here
    # is one team's player staying within 1 m of the ball, closer than any opponent by 0.5 m,
    # for at least 5 frames (0.2 s) in a row. Single touches and deflections do not qualify.
    near_h, near_a = frame["near_home"].values, frame["near_away"].values
    touch_h = visible & (near_h < 1.0) & (near_h + 0.5 < np.nan_to_num(near_a, nan=99.0))
    touch_a = visible & (near_a < 1.0) & (near_a + 0.5 < np.nan_to_num(near_h, nan=99.0))

    def runs_at_least(mask, k):
        out_mask = np.zeros_like(mask)
        change = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
        for start, end in zip(change[::2], change[1::2]):
            if end - start >= k:
                out_mask[start:end] = True
        return out_mask

    ctrl_h = runs_at_least(touch_h, 5)
    ctrl_a = runs_at_least(touch_a, 5)
    ctrl = np.where(ctrl_h, 1, np.where(ctrl_a, -1, 0)).astype(np.int8)

    def carry(values, backwards=False):
        """Last non zero value (or next one, backwards), never across a half; and its distance."""
        h = half[::-1] if backwards else half
        v = values[::-1] if backwards else values
        edges = np.flatnonzero(np.diff(h)) + 1
        t_out = np.zeros(D.N_FRAMES, np.int8)
        d_out = np.full(D.N_FRAMES, cap, np.int32)
        for first, last in zip(np.r_[0, edges], np.r_[edges, D.N_FRAMES]):
            idx = np.arange(last - first)
            seen = np.maximum.accumulate(np.where(v[first:last] != 0, idx, -1))
            ok_ = seen >= 0
            t_out[first:last] = np.where(ok_, v[first:last][np.maximum(seen, 0)], 0)
            d_out[first:last] = np.where(ok_, np.minimum(idx - seen, cap), cap)
        if backwards:
            t_out, d_out = t_out[::-1], d_out[::-1]
        return t_out, d_out

    f["ctrl_now"] = ctrl
    f["ctrl_last_team"], f["ctrl_since"] = carry(ctrl)
    f["ctrl_next_team"], f["ctrl_until"] = carry(ctrl, backwards=True)
    f["ctrl_same_both_sides"] = ((f["ctrl_last_team"] == f["ctrl_next_team"])
                                 & (f["ctrl_last_team"] != 0)).astype(np.int8)
    # share of the last 3 s / next 3 s that each team spent in control
    for secs in (3,):
        w = secs * D.FPS
        ch = pd.Series(ctrl_h.astype(float))
        ca = pd.Series(ctrl_a.astype(float))
        f["ctrl_home_past_{}s".format(secs)] = ch.rolling(w, min_periods=1).mean().values.astype(np.float32)
        f["ctrl_away_past_{}s".format(secs)] = ca.rolling(w, min_periods=1).mean().values.astype(np.float32)
        f["ctrl_home_next_{}s".format(secs)] = ch[::-1].rolling(w, min_periods=1).mean().values[::-1].astype(np.float32)
        f["ctrl_away_next_{}s".format(secs)] = ca[::-1].rolling(w, min_periods=1).mean().values[::-1].astype(np.float32)
    n_runs = int(np.sum(np.diff(np.r_[0, (ctrl != 0).astype(np.int8)]) == 1))
    print("   controlled touch frames: Barcelona {:.1f}%  Athletic {:.1f}% of scored; {} control spells"
          .format(100 * ctrl_h[scored].mean(), 100 * ctrl_a[scored].mean(), n_runs))
    flips = np.flatnonzero(np.diff(f["ctrl_last_team"]) != 0)
    print("   last controlling team changes {} times across the match".format(
        int(np.sum(scored[flips]))))

    feats = pd.concat([frame, pd.DataFrame(f)], axis=1)
    path = os.path.join(D.DERIVED_DIR, "features.parquet")
    feats.to_parquet(path, index=False)
    print("\n   wrote {} columns to {}\n".format(feats.shape[1], os.path.relpath(path, D.ROOT)))


if __name__ == "__main__":
    main()

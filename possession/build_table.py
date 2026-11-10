"""Build the master frame table: one row per t = 0..199318, every source attached.

    python possession/build_table.py

Writes (inside data/, which git ignores):
    data/derived/atd.npz          raw tracking arrays on the master index
    data/derived/frames.parquet   one row per frame: half, ball, team distances, tags, quality
    data/derived/tags.csv         the smart tags with master frame intervals

Nothing here looks at labels. The public ground truth is attached only so later reports can
check the pipeline; no feature is computed from it.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import sources as D
from sources import AWAY, HOME, N_FRAMES, PITCH_LENGTH, PITCH_WIDTH

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all NaN slices are expected

TAG_CODES = [
    "PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PROGRESSION",
    "DEFENDING IN MIDDLE THIRD", "SET PIECES", "CREATING CHANCES",
    "DEFENDING IN DEFENSIVE THIRD", "LONG BALL", "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION", "BUILD UP", "DEFENDING IN ATTACKING THIRD", "GOALS",
]
SET_PIECE_TYPES = ["throw-in", "corner kick", "goal kick", "free kick",
                   "kick off (after goal)", "kick off (start)"]


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def interval_mask(starts, ends):
    """Count of intervals [start, end) covering each t, via a difference array."""
    diff = np.zeros(N_FRAMES + 1, dtype=np.int32)
    for start, end in zip(starts, ends):
        start, end = max(0, int(start)), min(N_FRAMES, int(end))
        if end > start:
            diff[start] += 1
            diff[end] -= 1
    return np.cumsum(diff[:-1]).astype(np.int16)


def gap_features(visible, half):
    """Frames since the ball was last seen and until it is next seen, never across a half.

    Inside each block of constant `half` value the count restarts, so a ball seen during
    halftime never explains a frame of the second half.
    """
    big = np.int32(10 ** 6)
    since = np.full(N_FRAMES, big, dtype=np.int32)
    until = np.full(N_FRAMES, big, dtype=np.int32)
    run_len = np.zeros(N_FRAMES, dtype=np.int32)
    edges = np.flatnonzero(np.diff(half)) + 1
    for first, last in zip(np.r_[0, edges], np.r_[edges, N_FRAMES]):
        seen = visible[first:last]
        idx = np.arange(last - first)
        last_seen = np.maximum.accumulate(np.where(seen, idx, -1))
        since[first:last] = np.where(last_seen >= 0, idx - last_seen, big)
        rev = seen[::-1]
        next_seen = np.maximum.accumulate(np.where(rev, idx, -1))
        until[first:last] = np.where(next_seen >= 0, idx - next_seen, big)[::-1]
        # length of the missing run each frame sits in (0 when visible)
        missing = ~seen
        if missing.any():
            starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
            ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
            block = np.zeros(last - first, dtype=np.int32)
            for s, e in zip(starts, ends):
                block[s:e] = e - s
            run_len[first:last] = block
    return since, until, run_len


def main():
    began = time.time()
    meta, raw, homography, smart = D.require(["atd_meta", "atd_raw", "homography",
                                              "smart_tagging"])
    os.makedirs(D.DERIVED_DIR, exist_ok=True)

    print("\n1. Reading sources and checking the timeline contract")
    players, ball, teams = D.load_atd(meta, raw)
    H, reproj, valid = D.load_homography(homography)
    tags = D.load_tags(smart)
    gt = D.load_public_gt()
    half = D.half_of_frames()
    ok_counts = {h: int(np.sum(half == h)) for h in (1, 2)}
    if ok_counts != {1: 79467, 2: 79390}:
        D.fail("scored half sizes are {}, expected 79467 and 79390".format(ok_counts))
    D.ok("scored windows 11185..90651 and 110808..190197 ({} graded frames)".format(
        sum(ok_counts.values())))
    if np.any(gt[half == 0] != "X"):
        D.fail("public ground truth has labels outside the scored halves")
    D.ok("public ground truth is X everywhere outside the scored halves")

    np.savez(os.path.join(D.DERIVED_DIR, "atd.npz"), players=players, ball=ball, teams=teams)
    tags.to_csv(os.path.join(D.DERIVED_DIR, "tags.csv"), index=False)

    print("\n2. Computing per frame features")
    bx = ball[:, 0] * PITCH_LENGTH
    by = ball[:, 1] * PITCH_WIDTH
    ball_visible = np.isfinite(bx) & np.isfinite(by)

    px = players[:, :, 0] * PITCH_LENGTH
    py = players[:, :, 1] * PITCH_WIDTH
    present = np.isfinite(px) & np.isfinite(py)
    is_home = teams == HOME
    is_away = teams == AWAY

    dist = np.hypot(px - bx[:, None], py - by[:, None]).astype(np.float32)  # NaN if either missing
    home_sorted = np.sort(dist[:, is_home], axis=1)   # NaN sorts last
    away_sorted = np.sort(dist[:, is_away], axis=1)

    frame = pd.DataFrame({
        "t": np.arange(N_FRAMES, dtype=np.int32),
        "half": half,
        "scored": half > 0,
        "gt_public": gt,
        # ball
        "ball_visible": ball_visible,
        "ball_x": bx.astype(np.float32),
        "ball_y": by.astype(np.float32),
        "ball_speed": ball[:, 2],
        # players
        "n_home": present[:, is_home].sum(axis=1).astype(np.int8),
        "n_away": present[:, is_away].sum(axis=1).astype(np.int8),
        "home_centroid_x": np.nanmean(np.where(present[:, is_home], px[:, is_home], np.nan),
                                      axis=1).astype(np.float32),
        "away_centroid_x": np.nanmean(np.where(present[:, is_away], px[:, is_away], np.nan),
                                      axis=1).astype(np.float32),
        # ball relative, team level (slot ids are never used as identities)
        "near_home": home_sorted[:, 0],
        "near_home_2": home_sorted[:, 1],
        "near_away": away_sorted[:, 0],
        "near_away_2": away_sorted[:, 1],
        "home_within_2m": (dist[:, is_home] < 2).sum(axis=1).astype(np.int8),
        "away_within_2m": (dist[:, is_away] < 2).sum(axis=1).astype(np.int8),
        "home_within_5m": (dist[:, is_home] < 5).sum(axis=1).astype(np.int8),
        "away_within_5m": (dist[:, is_away] < 5).sum(axis=1).astype(np.int8),
        # camera / calibration quality
        "homography_valid": valid,
        "reprojection_error": reproj,
    })
    frame["gap_away_minus_home"] = (frame["near_away"] - frame["near_home"]).astype(np.float32)
    frame["dist_to_touchline"] = np.minimum(frame["ball_y"], PITCH_WIDTH - frame["ball_y"])
    frame["dist_to_goal_line"] = np.minimum(frame["ball_x"], PITCH_LENGTH - frame["ball_x"])

    since, until, run_len = gap_features(ball_visible, half)
    frame["frames_since_ball"] = since
    frame["frames_until_ball"] = until
    frame["ball_missing_run"] = run_len

    for code in TAG_CODES:
        rows = tags[tags["code"] == code]
        frame["tag_" + slug(code)] = interval_mask(rows["start"], rows["end"])
    for kind in SET_PIECE_TYPES:
        rows = tags[tags["type"].str.lower() == kind]
        frame["setpiece_" + slug(kind)] = interval_mask(rows["start"], rows["end"])
    for team, name in (("Barcelona", "barca"), ("Atletic Club", "athletic")):
        rows = tags[tags["team"] == team]
        frame["tag_team_" + name] = interval_mask(rows["start"], rows["end"])
    frame["tag_any"] = interval_mask(tags["start"], tags["end"])

    path = os.path.join(D.DERIVED_DIR, "frames.parquet")
    frame.to_parquet(path, index=False)
    print("  wrote {} rows x {} columns to {}".format(len(frame), frame.shape[1],
                                                       os.path.relpath(path, D.ROOT)))
    print("\nDone in {:.0f} s. Next: python possession/health_report.py\n".format(
        time.time() - began))


if __name__ == "__main__":
    main()

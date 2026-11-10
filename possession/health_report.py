"""Health report on the master frame table. Describes the data; never uses labels to decide anything.

    python possession/health_report.py

Prints the report and saves it to data/derived/health_report.txt (aggregate numbers only).
"""

import os
import warnings

import numpy as np
import pandas as pd

import sources as D

warnings.filterwarnings("ignore", category=RuntimeWarning)

LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def seconds(frames):
    return frames / D.FPS


def main():
    frame = pd.read_parquet(os.path.join(D.DERIVED_DIR, "frames.parquet"))
    tags = pd.read_csv(os.path.join(D.DERIVED_DIR, "tags.csv"), keep_default_na=False)
    atd = np.load(os.path.join(D.DERIVED_DIR, "atd.npz"))
    players, teams = atd["players"], atd["teams"]

    scored = frame["scored"].values
    s = frame[scored]

    say("=" * 72)
    say("HEALTH REPORT  tactical data on the master index t = 0..{}".format(D.N_FRAMES - 1))
    say("=" * 72)

    # 1. Ball visibility --------------------------------------------------------------------
    say("\n1. BALL VISIBILITY")
    say("   all frames      {:5.1f}%".format(pct(frame["ball_visible"].sum(), len(frame))))
    say("   scored frames   {:5.1f}%".format(pct(s["ball_visible"].sum(), len(s))))
    for h in (1, 2):
        part = frame[frame["half"] == h]
        say("   half {}          {:5.1f}%".format(h, pct(part["ball_visible"].sum(), len(part))))

    # 2. Missing ball runs ------------------------------------------------------------------
    say("\n2. MISSING BALL STRETCHES (scored frames only, never across halves)")
    lengths = []
    for h in (1, 2):
        vis = frame.loc[frame["half"] == h, "ball_visible"].values
        miss = ~vis
        starts = np.flatnonzero(miss & ~np.r_[False, miss[:-1]])
        ends = np.flatnonzero(miss & ~np.r_[miss[1:], False]) + 1
        lengths.extend((ends - starts).tolist())
    lengths = np.array(lengths)
    say("   number of stretches   {}".format(len(lengths)))
    if len(lengths):
        say("   median length         {:.2f} s".format(seconds(np.median(lengths))))
        say("   90th percentile       {:.2f} s".format(seconds(np.percentile(lengths, 90))))
        say("   longest               {:.1f} s".format(seconds(lengths.max())))
        total = lengths.sum()
        for limit in (1, 2, 5, 10):
            share = lengths[lengths > limit * D.FPS].sum()
            say("   missing frames in stretches > {:>2} s : {:5.1f}%".format(limit,
                                                                           pct(share, total)))

    # 3. Players ----------------------------------------------------------------------------
    say("\n3. TRACKED PLAYERS (scored frames)")
    n_obj = s["n_home"] + s["n_away"]
    say("   mean objects       {:.1f}   (home {:.1f}, away {:.1f})".format(
        n_obj.mean(), s["n_home"].mean(), s["n_away"].mean()))
    say("   min / max          {} / {}".format(int(n_obj.min()), int(n_obj.max())))
    say("   frames with > 22   {:5.1f}%   (duplicated tracks)".format(pct((n_obj > 22).sum(),
                                                                        len(s))))
    say("   frames with < 18   {:5.1f}%   (players out of camera view)".format(
        pct((n_obj < 18).sum(), len(s))))
    say("   frames with 0 home or 0 away   {:5.1f}%".format(
        pct(((s["n_home"] == 0) | (s["n_away"] == 0)).sum(), len(s))))

    # 4. Coordinates and pitch orientation --------------------------------------------------
    say("\n4. COORDINATES AND ORIENTATION")
    vb = s[s["ball_visible"]]
    say("   ball x range (m)   {:.1f} .. {:.1f}   (pitch length {})".format(
        vb["ball_x"].min(), vb["ball_x"].max(), D.PITCH_LENGTH))
    say("   ball y range (m)   {:.1f} .. {:.1f}   (pitch width {})".format(
        vb["ball_y"].min(), vb["ball_y"].max(), D.PITCH_WIDTH))
    outside = ((vb["ball_x"] < 0) | (vb["ball_x"] > D.PITCH_LENGTH)
               | (vb["ball_y"] < 0) | (vb["ball_y"] > D.PITCH_WIDTH))
    say("   ball outside the pitch lines   {:5.1f}% of visible scored frames".format(
        pct(outside.sum(), len(vb))))
    for h in (1, 2):
        part = frame["half"].values == h
        say("   half {}: team mean x  home {:5.1f} m  away {:5.1f} m".format(
            h, np.nanmean(frame.loc[part, "home_centroid_x"]),
            np.nanmean(frame.loc[part, "away_centroid_x"])))
        for slot, name in ((23, "home GK slot 24"), (45, "away GK slot 46")):
            xs = players[part, slot, 0] * D.PITCH_LENGTH
            seen = np.isfinite(xs)
            say("           {:<16} seen {:5.1f}%  mean x {:5.1f} m".format(
                name, pct(seen.sum(), part.sum()), np.nanmean(xs) if seen.any() else float("nan")))
    say("   (Barcelona attack left to right in half 1 per the metadata: expect home mean x")
    say("    below 52.6 m in half 1 if x = 0 is the left goal line.)")

    # 5. Ball speed -------------------------------------------------------------------------
    say("\n5. BALL SPEED (visible scored frames, km/h)")
    speed = vb["ball_speed"].dropna()
    if len(speed):
        for q in (50, 90, 99):
            say("   {}th percentile   {:6.1f}".format(q, np.percentile(speed, q)))
        say("   above 130 km/h     {:5.2f}%   (faster than almost any real shot: tracking noise)"
            .format(pct((speed > 130).sum(), len(speed))))
    else:
        say("   no ball speed values present")

    # 6. Distances ---------------------------------------------------------------------------
    say("\n6. NEAREST PLAYER TO THE BALL (visible scored frames)")
    nearest = np.fmin(vb["near_home"], vb["near_away"])
    say("   median nearest distance        {:.1f} m".format(np.nanmedian(nearest)))
    for limit in (1, 2, 5, 10):
        say("   nearest within {:>2} m            {:5.1f}%".format(
            limit, pct((nearest < limit).sum(), len(vb))))
    gap = vb["gap_away_minus_home"]
    say("   |gap| under 1 m (contested)    {:5.1f}%".format(pct((gap.abs() < 1).sum(),
                                                               gap.notna().sum())))

    # 7. Homography --------------------------------------------------------------------------
    say("\n7. HOMOGRAPHY")
    say("   valid, all frames      {:5.1f}%".format(pct(frame["homography_valid"].sum(),
                                                       len(frame))))
    say("   valid, scored frames   {:5.1f}%".format(pct(s["homography_valid"].sum(), len(s))))
    say("   median reprojection error  scored {:.2f}   unscored {:.2f}".format(
        s["reprojection_error"].median(), frame.loc[~scored, "reprojection_error"].median()))

    # 8. Smart tags --------------------------------------------------------------------------
    say("\n8. SMART TAGS")
    say("   instances {}   median length {:.1f} s".format(
        len(tags), (tags["end_s"] - tags["start_s"]).median()))
    for code, count in tags["code"].value_counts().items():
        rows = tags[tags["code"] == code]
        say("   {:<30} {:>4}   median {:5.1f} s".format(
            code, count, (rows["end_s"] - rows["start_s"]).median()))
    say("   set piece types: " + ", ".join(
        "{} {}".format(k, v) for k, v in tags.loc[tags["type"] != "", "type"].value_counts()
        .items()))
    say("   team tag values: " + ", ".join(
        "{} {}".format(k if k else "(none)", v) for k, v in tags["team"].value_counts().items()))
    say("   scored frames covered by any tag      {:5.1f}%".format(
        pct((s["tag_any"] > 0).sum(), len(s))))
    say("   scored frames covered by a set piece  {:5.1f}%".format(
        pct((s["tag_set_pieces"] > 0).sum(), len(s))))
    both = (s["tag_team_barca"] > 0) & (s["tag_team_athletic"] > 0)
    say("   scored frames tagged with BOTH teams  {:5.1f}%   (tags overlap, so they are clues only)"
        .format(pct(both.sum(), len(s))))

    # 9. Second half before kickoff ---------------------------------------------------------
    first2 = D.half_video_blocks(D.GAME)[1][1]
    pre = frame[(frame["t"] >= first2) & (frame["t"] < D.KICKOFF_FRAMES[2])]
    say("\n9. SECOND HALF, SCORED BUT BEFORE KICKOFF")
    say("   frames {}..{}  = {:.0f} s scored before the second half kickoff".format(
        first2, D.KICKOFF_FRAMES[2] - 1, seconds(len(pre))))
    say("   ball visible {:5.1f}%   mean objects {:.1f}".format(
        pct(pre["ball_visible"].sum(), len(pre)), (pre["n_home"] + pre["n_away"]).mean()))

    # 10. Public ground truth (pipeline check only) -----------------------------------------
    say("\n10. PUBLIC GROUND TRUTH (pipeline check only: never used to tune)")
    g = frame["gt_public"]
    say("   H {}  A {}  D {}   (expected 3454 / 2577 / 1469)".format(
        int((g == "H").sum()), int((g == "A").sum()), int((g == "D").sum())))

    say("\n" + "=" * 72)
    path = os.path.join(D.DERIVED_DIR, "health_report.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(LINES) + "\n")
    print("saved to " + os.path.relpath(path, D.ROOT))


if __name__ == "__main__":
    main()

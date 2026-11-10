"""Look at the 21 stretches of the public five minutes, to understand the labelling convention.

    python possession/inspect_public.py            # compares against evaluation/results/v1.csv
    python possession/inspect_public.py v2         # or another submission

CHALLENGE.md asks participants to use the public five minutes to see how cases are labelled.
This prints, for each ground truth stretch, what the data looked like and what we predicted.
It is for understanding only: no number from it is fitted, and nothing here trains a model.
"""

import os
import sys

import numpy as np
import pandas as pd

import sources as D


def share(mask):
    return 100.0 * np.mean(mask) if len(mask) else float("nan")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v2"
    F = pd.read_parquet(os.path.join(D.DERIVED_DIR, "features.parquet"))
    pred = pd.read_csv(os.path.join(D.ROOT, "evaluation", "results", name + ".csv"))["label"].values
    gt = F["gt_public"].values
    rows = np.flatnonzero(gt != "X")
    first, last = rows[0], rows[-1] + 1

    lab = gt[first:last]
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts, ends = first + np.r_[0, change], first + np.r_[change, len(lab)]

    print("\nPublic five minutes, one line per ground truth stretch ({} stretches)".format(len(starts)))
    print("times are seconds since kickoff; 'pred' is what {} said on those frames\n".format(name))
    head = ("  #  gt   from     to    len | ball  ball  ball  plyr  setpc  vote | ball first/last seen   "
            "| pred H  A  D")
    print(head)
    print("                               | seen  out   still m/s   tag    H-A  | x,y (m)               |")
    print("-" * len(head))
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        seg = F.iloc[s:e]
        vis = seg["ball_visible"].values
        xs, ys = seg["ball_x"].values[vis], seg["ball_y"].values[vis]
        where = "{:>5.1f},{:>4.1f} > {:>5.1f},{:>4.1f}".format(xs[0], ys[0], xs[-1], ys[-1]) \
            if vis.any() else "   never seen          "
        kinds = []
        for col in [c for c in F.columns if c.startswith("setpiece_")]:
            short = {"throw": "thr", "corner": "cor", "goal_kick": "gk", "free": "fk",
                     "kick_off": "ko"}
            if (seg[col] > 0).mean() > 0.3:
                kinds.append(next((v for k, v in short.items() if k in col), col[9:13]))
        p = pred[s:e]
        print("{:>3}  {}  {:>6.1f} {:>6.1f} {:>5.1f} | {:>3.0f}%  {:>3.0f}%  {:>3.0f}%  {:>4.1f}  {:<5}  {:>+4.1f} | {} | {:>4.0f} {:>3.0f} {:>3.0f}"
              .format(i, lab[s - first], (s - first) / D.FPS, (e - first) / D.FPS, (e - s) / D.FPS,
                      share(vis), share(seg["ball_out"].values[vis] > 0) if vis.any() else 0,
                      share(seg["ball_still_1s"].values[vis] > 0) if vis.any() else 0,
                      np.nanmean(seg["speed_mean_all"]),
                      "/".join(sorted(set(kinds))) or "-",
                      np.mean(seg["tag_vote"]), where,
                      share(p == "H"), share(p == "A"), share(p == "D")))
    print("\nball seen/out/still: % of the stretch; out and still are % of the frames where it is seen")
    print("plyr m/s: mean player speed; setpc: set piece tag active over 30% of the stretch")
    print("vote: mean tag vote, + Barcelona has the ball, - Athletic\n")


if __name__ == "__main__":
    main()

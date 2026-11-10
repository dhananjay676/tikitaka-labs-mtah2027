"""Pseudo labels -> per frame H/A/D probabilities -> sticky decoding per half -> submission.

    python possession/model.py                 # writes evaluation/results/v1.csv
    python possession/model.py --name v1b      # another name

No human labels anywhere. The public ground truth is never read here.

Step 1  Pseudo labels. A fixed, generic rule marks the easy frames:
          H  ball visible, a Barcelona player within 1.2 m, no Athletic player within 3 m
          A  the same, reversed
          D  ball clearly outside the lines for most of a second, or the whole pitch
             walking (mean < 1.2 m/s over 3 s) while the ball moves under 3 m in 3 s
             (H and A also require players running, mean > 1.5 m/s over 3 s)
        Nobody decided any individual frame; the rule is the same for the whole match.

Step 2  Two learners trained on those easy frames (gradient boosted trees):
          seeing   all features, used where the ball is visible
          blind    no ball features at all, only camera aim, team shape, player speed and
                   tags, used where the ball is missing (often for 10 to 30 s)
        Cross fitted by half: trained on half 1 to predict half 2 and the other way round,
        so no frame is scored by a model that trained on it.

Step 3  Sticky decoding (Viterbi), each half on its own. Switching team costs as much as
        a typical possession length says it should. The one free number, how much a single
        frame of evidence counts, is chosen so possessions average about 12 s, a general
        football figure. It is not chosen with any ground truth.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import sources as D

warnings.filterwarnings("ignore")
STATES = np.array(["H", "A", "D"])
H_, A_, D_ = 0, 1, 2

MEAN_POSSESSION_S = 12.0     # general football prior for an unbroken team possession
MEAN_DEAD_S = 10.0           # general prior for a stoppage
TURNOVER_SHARE = 0.5         # of possession endings: half go straight to the other team
EVIDENCE_WEIGHTS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]

NOT_FEATURES = {"t", "half", "scored", "gt_public"}

# Tried and dropped (v3): the "control" clues (who last had the ball under control) are built
# from the same ingredients as the H/A pseudo label rule, so the learners copy the rule instead
# of learning from independent evidence. Label free symptoms: the ball blind model's agreement
# with the easy labels jumped from 94.6% to 98.5% although it cannot see the ball, and the
# predicted dead share rose to 52%, above the 40 to 45% typical of a top flight match.
# The features are still computed in features.py so the experiment can be repeated.
EXCLUDED_PREFIXES = ("ctrl_",)
BALL_WORDS = ("ball", "near_", "gap", "within_2m", "within_5m", "proxy", "dist_to")


def pseudo_labels(F):
    scored = F["scored"].values
    vis = F["ball_visible"].values & scored
    nh, na = F["near_home"].values, F["near_away"].values
    calm = ~(F["ball_speed_own"].values > 8)          # NaN counts as calm
    inside = F["ball_out"].values == 0
    pace = F["speed_mean_all_3s"].values
    moving = pace > 1.5                               # play is live: players are running
    walking = pace < 1.2                              # a stoppage: the whole pitch walks
    ball_quiet = ~(F["ball_disp_3s"].values > 3.0)    # moved under 3 m in 3 s, or not seen
    easy_h = vis & inside & calm & moving & (nh < 1.2) & (na > 3.0)
    easy_a = vis & inside & calm & moving & (na < 1.2) & (nh > 3.0)
    easy_d = scored & ((F["ball_out_1s"].values > 0.6) | (walking & ball_quiet))
    y = np.full(len(F), -1, np.int8)
    y[easy_h], y[easy_a], y[easy_d] = H_, A_, D_
    y[(easy_h.astype(int) + easy_a + easy_d) > 1] = -1   # conflicting rules: leave unlabelled
    return y


def feature_sets(F):
    """(seeing, blind) column lists.

    Blind keeps only what exists whether or not the ball is visible: tags, camera based
    estimates, team shape, player speeds, calibration quality. Anything computed from the
    tracked ball is dropped, so the blind model meets the same kind of input on missing
    frames as it trained on.
    """
    seeing = [c for c in F.columns if c not in NOT_FEATURES
              and not c.startswith(EXCLUDED_PREFIXES)
              and (np.issubdtype(F[c].dtype, np.number) or F[c].dtype == bool)]
    blind = [c for c in seeing
             if c.startswith(("tag_", "setpiece_")) or "cam" in c
             or not any(word in c for word in BALL_WORDS)]
    return seeing, blind


def cross_fit(F, y, columns):
    """P(H, A, D) for every scored frame, each half predicted by a model trained on the other."""
    X = F[columns].astype(np.float32).values
    half = F["half"].values
    probs = np.full((len(F), 3), np.nan, np.float32)
    for test in (1, 2):
        train = (half == 3 - test) & (y >= 0)
        if len(np.unique(y[train])) < 2:
            raise SystemExit("half {} has fewer than two pseudo label classes; check the rules"
                             .format(3 - test))
        # drop columns that carry no information on the training rows (all NaN or constant)
        Xt = X[train]
        useful = np.array([np.unique(col[np.isfinite(col)]).size > 1 for col in Xt.T])
        X_use = X[:, useful]
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                               max_leaf_nodes=31, min_samples_leaf=200,
                                               class_weight="balanced", random_state=0)
        model.fit(X_use[train], y[train])
        rows = half == test
        p = np.zeros((rows.sum(), 3), np.float32)
        p[:, model.classes_] = model.predict_proba(X_use[rows])
        probs[rows] = p
    return probs


def log_transitions():
    leave_p = 1.0 / (MEAN_POSSESSION_S * D.FPS)
    leave_d = 1.0 / (MEAN_DEAD_S * D.FPS)
    T = np.array([
        [1 - leave_p, leave_p * TURNOVER_SHARE, leave_p * (1 - TURNOVER_SHARE)],
        [leave_p * TURNOVER_SHARE, 1 - leave_p, leave_p * (1 - TURNOVER_SHARE)],
        [leave_d / 2, leave_d / 2, 1 - leave_d],
    ])
    return np.log(T)


def viterbi(log_emit, log_T):
    n = len(log_emit)
    back = np.zeros((n, 3), np.int8)
    score = np.log(np.full(3, 1 / 3)) + log_emit[0]
    for t in range(1, n):
        cand = score[:, None] + log_T
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + log_emit[t]
    path = np.zeros(n, np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(probs, half, weight):
    log_T = log_transitions()
    labels = np.full(len(half), "X", dtype="<U1")
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        p = np.clip(probs[rows], 0.01, 0.98)
        labels[rows] = STATES[viterbi(weight * np.log(p), log_T)]
    return labels


def spell_stats(labels, half):
    """(possessions, mean possession seconds, dead share) over scored frames."""
    lengths = []
    for h in (1, 2):
        lab = labels[half == h]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(lab)]
        lengths += [e - s for s, e in zip(starts, ends) if lab[s] in ("H", "A")]
    scored = half > 0
    return len(lengths), np.mean(lengths) / D.FPS if lengths else 0.0, np.mean(labels[scored] == "D")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v2")
    args = parser.parse_args()
    began = time.time()

    F = pd.read_parquet(os.path.join(D.DERIVED_DIR, "features.parquet"))
    half = F["half"].values
    scored = half > 0

    print("\n1. Pseudo labels (algorithmic rule, no human labels)")
    y = pseudo_labels(F)
    for code, name in enumerate("HAD"):
        print("   easy {}  {:>7} frames  ({:4.1f}% of scored)".format(
            name, int(np.sum(y == code)), 100 * np.mean(y[scored] == code)))
    print("   unlabelled {:>7} frames  ({:4.1f}% of scored)".format(
        int(np.sum((y < 0) & scored)), 100 * np.mean(y[scored] < 0)))

    print("\n2. Learners, cross fitted by half")
    seeing, blind = feature_sets(F)
    print("   seeing model {} features, blind model {} features".format(len(seeing), len(blind)))
    p_seeing = cross_fit(F, y, seeing)
    p_blind = cross_fit(F, y, blind)
    visible = F["ball_visible"].values
    probs = np.where(visible[:, None], p_seeing, p_blind)
    for name, p in (("seeing", p_seeing), ("blind", p_blind)):
        known = scored & (y >= 0)
        agree = np.mean(np.argmax(p[known], axis=1) == y[known])
        print("   {:<6} model reproduces the easy labels of the other half on {:.1f}%".format(
            name, 100 * agree))
    missing = scored & ~visible
    print("   on ball missing frames the blind model says H {:.0f}%  A {:.0f}%  D {:.0f}%".format(
        *(100 * np.mean(np.argmax(probs[missing], axis=1)[:, None] == np.arange(3), axis=0))))
    pd.DataFrame({"t": F["t"], "pH": probs[:, 0], "pA": probs[:, 1], "pD": probs[:, 2]}
                 ).to_parquet(os.path.join(D.DERIVED_DIR, "probs_{}.parquet".format(args.name)),
                              index=False)

    print("\n3. Sticky decoding per half; choosing evidence weight by a football prior")
    print("   target: possessions averaging about {:.0f} s".format(MEAN_POSSESSION_S))
    print("   {:>7}  {:>11}  {:>13}  {:>9}".format("weight", "possessions", "mean length", "dead"))
    best = None
    for w in EVIDENCE_WEIGHTS:
        labels = decode(probs, half, w)
        n, mean_s, dead = spell_stats(labels, half)
        print("   {:>7.2f}  {:>11}  {:>11.1f} s  {:>8.1f}%".format(w, n, mean_s, 100 * dead))
        gap = abs(np.log(max(mean_s, 0.1) / MEAN_POSSESSION_S))
        if best is None or gap < best[0]:
            best = (gap, w, labels)
    _, weight, labels = best
    print("   chosen weight {:.2f}".format(weight))

    labels[~scored] = "X"
    path = os.path.join(D.ROOT, "evaluation", "results", "{}.csv".format(args.name))
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(t, lab) for t, lab in enumerate(labels))
    print("\n   wrote {}  ({:.0f} s)".format(os.path.relpath(path, D.ROOT), time.time() - began))
    print("   next:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, D.ROOT)))
    print("          python evaluation/run_evaluation.py\n")


if __name__ == "__main__":
    main()

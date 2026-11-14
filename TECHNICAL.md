# TikiTaka Labs: possession from incomplete tracking

More Than a Hack 2027, Phase 1. FC Barcelona vs Athletic Club.\
Repository: https://github.com/dhananjay676/tikitaka-labs-mtah2027 (code in `possession/`)

## 1. Summary

We label every frame H, A or D from the provided tactical tracking, Smart Tagging and homography.
No human labels are used anywhere. A fixed rule labels the frames where the answer is obvious,
two gradient boosted tree models learn from those frames, and a hidden Markov model decoded with
Viterbi turns the per frame probabilities into stable possessions, one half at a time.

The method is built around two facts we measured in the data:

1. **The ball is missing in long blocks, not short blinks.** 41% of scored frames have no ball,
   in only 150 stretches with a median length of 10.7 s (longest 221 s). Any method that needs
   the ball on every frame fails there, so we built clues that work without it.
2. **Dead ball time is the biggest lever on spellF1.** A missed stoppage glues two possessions
   into one; a false one splits a possession in two. The clearest stoppage signal turned out to
   be the whole pitch slowing to a walk.

| | spellF1 | macroF1 | Possessions (true 17) |
|---|---|---|---|
| Official baseline smooth50 | 0.326 | 0.621 | 26 |
| **Our submission (v5)** | **0.469** | **0.820** | **18** |

Scores on the public five minutes, which we used only as a pipeline check (section 5).
The full match ground truth was never available to us.

## 2. Data used, and how

All sources are mapped onto one index `t = 0 … 199,318` before anything else runs. The loader
stops with an error if any mapping fails, so an alignment bug cannot reach the output.

| Asset | Used for | Mapping, checked in code |
|---|---|---|
| Tactical ATD (`.txt`, `.xml`) | Positions and team of every tracked object; the ball | `t = frame_id − 1`; exactly 199,319 consecutive ids |
| Smart Tagging | Which team the tags say has the ball; set piece timing | `t = round(seconds × 25)`, interval `[start, end)`; both kickoff tags contain the published kickoff frames |
| Tactical homography | Camera direction on every frame | `t = frame`; exactly 0 … 199,318 |
| Public ground truth | Pipeline checks and understanding the labelling convention only | `t = frame` |

What we learned from the data, and what we did about it:

| Finding | Action |
|---|---|
| Track slots are not people (46 slots for 22 players, slots switch identity) | Only team level quantities; no slot is ever followed as a player |
| The provided ball speed is not a ball speed (median 1.3) | Computed our own from positions (median 5.1 m/s, p99 23.8 m/s) |
| The provided player speed is in m/s, not km/h (median 1.4, p90 3.9) | Used as player pace |
| x = 0 is the left goal line; teams swap ends at halftime | Attack direction derived per half from team centroids |
| 42% of frames carry tags for both teams at once | The tag **name** decides the side: "Build Up" for team X means X has the ball, "Defending in Middle Third" for X means X does not |

**Dropped or not used:** the tactical video (only watched to understand the data, never labelled),
the whole broadcast recording (not synchronised, out of scope for the time available), and the
provided ball speed column.

## 3. Method

### 3.1 Clues (about 110 per frame)

Positions are converted to metres (105.3 × 68.0).

* **Ball proximity:** nearest and second nearest player of each team, the gap between the two
  nearest (`d_Athletic − d_Barcelona`), players within 2 m and 5 m.
* **Ball motion:** our own speed from a central difference over 4 frames; displacement over 1 s
  and over 3 s from 1 s median positions, which is robust to tracker jitter.
* **Player pace:** mean speed of all tracked players over 3, 5 and 10 s, and its change.
* **Camera aim:** the pitch point at the image centre, `H⁻¹ · [u_c, v_c, 1]ᵀ`. It is coarse
  (median 23.6 m from the ball) because the tactical camera shows a wide area, but it exists on
  every frame and says which part of the pitch play is in.
* **Team shape:** centroids, spreads and relative push up, normalised to Barcelona's attacking
  direction.
* **Tag votes:** count of active tags saying each team has the ball. On frames where proximity is
  unambiguous the vote agrees 77% of the time, which confirms the mapping.
* **Stoppage and restart timing:** ball outside the lines, time since it last went out, time
  around set piece tags, time since a restart kick (3.4).

### 3.2 Pseudo labels

A fixed rule, the same for the whole match, labels only obvious frames:

| Label | Rule | Share of scored frames |
|---|---|---|
| H | Ball visible and inside the pitch; a Barcelona player within 1.2 m; no Athletic player within 3 m; ball under 8 m/s; players running (3 s pace over 1.5 m/s) | 6.5% |
| A | Same, mirrored | 3.4% |
| D | Ball outside the lines for most of a second, or 3 s pace under 1.2 m/s while the ball moves under 3 m in 3 s, unless a restart kick happened in the last 8 s | 28.7% |

The other 61% of frames are left unlabelled and inferred.

### 3.3 Two learners

Gradient boosted trees (`HistGradientBoostingClassifier`, 200 iterations, balanced classes,
fixed seed):

* a **seeing** model with all clues, used where the ball is visible;
* a **blind** model with no clue computed from the tracked ball, used where it is missing.

Training one model on visible frames and applying it to missing ones would feed it inputs it has
never seen. The blind model trains and predicts on the same kind of input. Both are **cross fitted
by half**: trained on one half, applied to the other, so no frame is predicted by a model that saw
its own pseudo label.

### 3.4 Sticky decoding

Possession is a hidden state (H, A or D) that changes rarely. Transition probabilities come from
football knowledge, not from labels:

* a possession lasts about 12 s, so the chance of leaving is 1/300 per frame;
* 65% of possessions end directly with the other team (roughly 100 to 130 stoppages against
  250 to 350 possessions per match), 35% at a stoppage;
* a stoppage lasts about 10 s, so 1/250 per frame, split evenly between the teams.

Each frame adds `w · log P(state)` as evidence. Viterbi finds the best path through each half
separately; nothing is carried across halftime. A team switch costs about 6.1 in log terms, and a
confident frame contributes about 0.23, so roughly one second of consistent evidence is needed to
switch team. A single frame never can. The one free number `w` is set so possessions average
about 12 s (a football figure); it is 0.08.

**Restart kick.** A restart sends the ball travelling while most players still walk, so the pitch
looks like a stoppage for a few more seconds. A kick is the ball moving more than 10 m in 1 s,
seen on most frames of that second; it is a restart kick if the pitch walked in the 5 s before.
We find 139 across the match. After one, no frame is pseudo labelled D for 8 s.

**Short stoppages.** A predicted stoppage shorter than 3 s with the **same team** before and after
is replaced by that team (H D H becomes H H H). Fetching the ball and restarting takes longer than
that. H D A is left alone. This removes 24 of 156 predicted stoppages, leaving 132 with a median of
12.4 s.

## 4. Possession rules we implemented

| Situation | Our rule |
|---|---|
| Loose or contested ball | Possession stays with the last team. The decoder needs about a second of consistent evidence before switching |
| A touch, deflection or aerial duel without control | Does not change possession. Pseudo labels need a player within 1.2 m, no opponent within 3 m and a slow ball; the decoder ignores short blips |
| Change of possession | The other team shows clear, sustained proximity to a slow ball |
| Ball out of play, fouls, waiting for a restart | D: ball outside the lines, or the whole pitch walking while the ball barely moves |
| Restart | Play is live from the restart kick, even if players are still walking |
| Very short stoppage between the same team | Treated as noise, kept as that team |
| Who "owns" a dead ball | Not predicted. A stoppage is D, as the challenge asks |
| `X` | Only outside the two scored halves. Never used inside them |

## 5. Training and validation data

**No human labelled data was used, from this match or any other.** All training labels are the
pseudo labels of 3.2, produced by a fixed rule; no person decided any individual frame.

The public five minutes were used to check alignment and file format, and, as CHALLENGE.md
invites, to see how cases are labelled (we listed what the data looked like in each of its
21 stretches). That inspection is where we learned that stoppages show up as players walking.

We did not search any parameter on it. Every change was kept or dropped on criteria written down
before running it: label free checks on the full match (mean possession length, share of dead
time, number of stoppages, whether the blind model's agreement jumped), and a football reason we
could state. The public score was reported alongside as an alarm. With 17 possessions it cannot
rank close versions; the organisers show the best baseline on it is not the best on the full match.

| Version | Change | spellF1 | macroF1 | Kept |
|---|---|---|---|---|
| v1 | Clues, pseudo labels, decoder | 0.264 | 0.538 | Replaced: the D rule found only 2% of frames |
| v2 | Pace based D rule, timing clues | 0.297 | 0.797 | Yes |
| v3 | "Who last had control" clues | 0.276 | 0.744 | **No** (below) |
| v4 | Restart kick, 65% direct turnovers | 0.444 | 0.811 | Yes |
| v5 | Merge short same team stoppages | **0.469** | **0.820** | **Submitted** |

**Why v3 was dropped.** Its clues were built from the same ingredients as the H/A pseudo label
rule, so the learners could copy the rule. Label free symptoms: the blind model's agreement with
the pseudo labels jumped from 94.6% to 98.5% although it cannot see the ball, and predicted dead
time rose to 52%. Lesson: no clue may be built from the pseudo label rule itself.

**Testing v4's two changes separately.** The restart kick alone moved spellF1 from 0.297 to 0.424;
the turnover share alone left it at 0.297. We kept both, because each has a football reason and
neither hurts any check, but the turnover share's own effect is small.

## 6. Limitations

* **Dead time is probably over predicted.** 47% of scored time is D, against roughly 40 to 45%
  in a typical top flight match. Slow live build up (for example after a goal kick) is still
  sometimes called dead.
* **Switches are late.** About 0.45 s on the public window, and possessions under about 3 s are
  often missed. We did not shift switches earlier, because that delay is only measurable with labels.
* **Long missing ball stretches** rely on pace, shape, tags and a coarse camera direction. Who has
  the ball inside a 30 s gap is the weakest part of the system.
* **Thresholds come from football reasoning,** not from labelled data. They were never fitted, but
  they were never validated on a full labelled match either.
* **Single match.** Pseudo labels and models are trained on the test match itself, without labels.
  Behaviour on another match is untested.

**With more time:** find the ball in the tactical video where tracking loses it, and project it back
with the homography; separate air balls from ground balls using the path the ball traces under the
camera's view; synchronise the broadcast feed; model a minimum stoppage length inside the decoder
rather than as a post step.

## 7. Reproduction

Requirements: Python 3.11, the pinned `requirements-possession.txt`, and the four tactical files
from the organisers' bundle in `data/tactical/` (the video and broadcast files are not needed).

```bash
git clone https://github.com/dhananjay676/tikitaka-labs-mtah2027.git
cd tikitaka-labs-mtah2027
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-possession.txt
python possession/run_pipeline.py          # about 30 s on a laptop
```

This writes `evaluation/results/TikiTaka_Labs.csv` and runs the official format check. The
pipeline is deterministic: after deleting all intermediate files and rerunning from the raw data,
the output was byte for byte identical (`cmp`). Intermediate files go to `data/derived/`, which
git ignores, because they are frame level derivatives of the licensed data.

| File | Role |
|---|---|
| `possession/sources.py` | Reads every source onto `t`, with hard alignment checks |
| `possession/build_table.py` | Master table, one row per frame |
| `possession/features.py` | All clues |
| `possession/model.py` | Pseudo labels, learners, decoder, submission |
| `possession/run_pipeline.py` | Runs everything and validates |
| `possession/health_report.py`, `inspect_public.py` | Diagnostics, not part of the submission path |

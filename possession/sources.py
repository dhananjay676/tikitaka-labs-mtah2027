"""Read every tactical source and put it on one master frame index.

The master index is the submission frame:

    t = 0 .. 199318

and every source is attached to it with exactly one rule:

    source               native index              rule
    ATD tracking         frame_id 1 .. 199319      t = frame_id - 1
    homography           frame 0 .. 199318         t = frame
    smart tagging        seconds                   t = round(seconds * 25), interval [start, end)
    public ground truth  frame 0 .. 199318         t = frame

The broadcast recording is never read here. It is on another timeline.

Every loader checks its own contract and stops with a readable message if the file breaks it,
so an alignment bug becomes an error instead of a score that is quietly wrong everywhere.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]          # 199319
FPS = GAMES[GAME]["fps"]                          # 25
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]  # 105.3 m, 68.0 m
N_SLOTS = 46
FIELDS_PER_LINE = 1 + N_SLOTS * 3 + 3             # frame id, 46 x (x, y, speed), ball (x, y, speed)

HOME, AWAY = 0, 1

# Second half kickoff, published by the organisers in starter/load_smart_tagging.py and FAQ.md.
# The first half kicks off on the first scored frame.
KICKOFF_FRAMES = {1: 11185, 2: 113742}

# Derived files live inside data/, which .gitignore keeps out of the public repository.
# They are frame level tracking derivatives, so they must never be published.
DERIVED_DIR = os.path.join(ROOT, "data", "derived")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def ok(message):
    print("  CHECK ok   " + message)


def fail(message):
    raise SystemExit("\n  CHECK FAILED   " + message + "\n")


# ---------------------------------------------------------------------------------------------
# Halves
# ---------------------------------------------------------------------------------------------

def half_of_frames():
    """int8 array over t: 1 or 2 inside a scored half, 0 everywhere else."""
    half = np.zeros(N_FRAMES, dtype=np.int8)
    for number, first, last in half_video_blocks(GAME):
        half[first:last + 1] = number
    return half


# ---------------------------------------------------------------------------------------------
# ATD tracking
# ---------------------------------------------------------------------------------------------

def slot_teams(meta_path):
    """int8 array over the 46 slots: 0 home (Barcelona), 1 away (Athletic Club).

    The first team in the metadata is the home side, the same convention the official baseline
    uses. Slots are tracklets, not people: only the team is trustworthy.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = [team.findtext("Name") for team in root.iter("Team")]
    if len(team_ids) != 2:
        fail("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    teams = np.array([HOME if player.get("teamId") == team_ids[0] else AWAY
                      for player in root.iter("Player")], dtype=np.int8)
    if len(teams) != N_SLOTS:
        fail("expected {} track slots, found {}".format(N_SLOTS, len(teams)))
    ok("{} slots: {} home ({}), {} away ({})".format(
        N_SLOTS, int(np.sum(teams == HOME)), names[0], int(np.sum(teams == AWAY)), names[1]))
    return teams


def _parse_atd_fast(text):
    """Whole file in one numpy call. Returns (n_lines, 142) float64 or None if the shape is off."""
    text = text.replace("\r", "").strip()
    n_lines = text.count("\n") + 1
    flat = np.fromstring(text.replace(":", ",").replace(";", ",").replace("\n", ","), sep=",")
    if flat.size != n_lines * FIELDS_PER_LINE:
        return None
    return flat.reshape(n_lines, FIELDS_PER_LINE)


def _parse_atd_slow(text):
    """Line by line. Only used if a line does not have exactly 142 fields."""
    rows = []
    for number, line in enumerate(text.replace("\r", "").strip().split("\n"), start=1):
        fields = line.replace(":", ",").replace(";", ",").split(",")
        if len(fields) != FIELDS_PER_LINE:
            fail("ATD line {} has {} fields, expected {}: {}...".format(
                number, len(fields), FIELDS_PER_LINE, line[:80]))
        rows.append([float(value) for value in fields])
    return np.array(rows)


def load_atd(meta_path, raw_path):
    """(players, ball, teams) on the master index.

    players  float32 (199319, 46, 3)   x, y normalised 0..1, speed km/h, NaN when missing
    ball     float32 (199319, 3)       same layout
    teams    int8    (46,)             0 home, 1 away
    """
    teams = slot_teams(meta_path)
    with open(raw_path) as handle:
        text = handle.read()
    table = _parse_atd_fast(text)
    if table is None:
        print("  note: fast parser could not read every line, using the slow one (about 2 min)")
        table = _parse_atd_slow(text)
    del text

    frame_id = table[:, 0].astype(np.int64)
    if len(frame_id) != N_FRAMES:
        fail("ATD has {} lines, expected {}".format(len(frame_id), N_FRAMES))
    if not np.array_equal(frame_id, np.arange(1, N_FRAMES + 1)):
        fail("ATD frame ids are not exactly 1..{} in order".format(N_FRAMES))
    ok("ATD has {} lines with frame_id 1..{}; mapped by t = frame_id - 1".format(
        N_FRAMES, N_FRAMES))

    players = table[:, 1:1 + N_SLOTS * 3].reshape(N_FRAMES, N_SLOTS, 3).astype(np.float32)
    ball = table[:, 1 + N_SLOTS * 3:].astype(np.float32)
    return players, ball, teams


# ---------------------------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------------------------

def load_homography(path):
    """(matrices (n, 9), reprojection_error (n,), valid (n,) bool) on the master index."""
    table = pd.read_csv(path)
    columns = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]
    missing = [name for name in ["frame", "reprojection_error", "valid"] + columns
               if name not in table.columns]
    if missing:
        fail("homography is missing columns {}".format(missing))
    if not np.array_equal(table["frame"].values, np.arange(N_FRAMES)):
        fail("homography frames are not exactly 0..{} in order".format(N_FRAMES - 1))
    ok("homography has {} rows with frame 0..{}; t = frame".format(N_FRAMES, N_FRAMES - 1))
    return (table[columns].values.astype(np.float64),
            table["reprojection_error"].values.astype(np.float32),
            table["valid"].values.astype(bool))


# ---------------------------------------------------------------------------------------------
# Smart tagging
# ---------------------------------------------------------------------------------------------

def load_tags(path):
    """One row per tagged instance, with start/end converted to master frames.

    Interval convention, used everywhere in this project: a tag covers t in [start, end),
    start = round(start_seconds * 25), end = round(end_seconds * 25).
    """
    rows = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {(label.findtext("group") or "").strip(): (label.findtext("text") or "").strip()
                for label in instance.findall("label")}
        rows.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_s": start_s,
            "end_s": end_s,
            "start": int(round(start_s * FPS)),
            "end": int(round(end_s * FPS)),
            "team": tags.get("Team", ""),
            "type": tags.get("Type", ""),
            "half_tag": tags.get("Half", ""),
        })
    tags = pd.DataFrame(rows)
    if tags.empty:
        fail("smart tagging file has no instances")

    # Time base check: each half's kickoff tag must contain the published kickoff frame.
    slack = 2 * FPS
    for half, kickoff in KICKOFF_FRAMES.items():
        kickoffs = tags[(tags["code"] == "SET PIECES")
                        & (tags["type"].str.lower() == "kick off (start)")]
        hit = kickoffs[(kickoffs["start"] - slack <= kickoff) & (kickoff < kickoffs["end"] + slack)]
        if hit.empty:
            fail("no 'Kick Off (Start)' tag contains kickoff frame {} (half {}); "
                 "the seconds * 25 rule may be wrong".format(kickoff, half))
        ok("kickoff tag for half {} spans t {}..{}, contains published kickoff {}".format(
            half, int(hit["start"].iloc[0]), int(hit["end"].iloc[0]), kickoff))
    ok("{} smart tags read; t = round(seconds * 25), interval [start, end)".format(len(tags)))
    return tags


# ---------------------------------------------------------------------------------------------
# Public ground truth (checking only: never used to train or tune)
# ---------------------------------------------------------------------------------------------

def load_public_gt():
    table = pd.read_csv(PUBLIC_GT)
    if not np.array_equal(table["frame"].values, np.arange(N_FRAMES)):
        fail("public ground truth frames are not 0..{}".format(N_FRAMES - 1))
    labels = table["label"].astype(str).str.strip().str.upper().values
    ok("public ground truth has {} rows; graded frames {}".format(
        N_FRAMES, int(np.sum(labels != "X"))))
    return labels

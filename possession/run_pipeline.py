"""One command from the raw match data to the submission file.

    python possession/run_pipeline.py                     # writes evaluation/results/v2.csv
    python possession/run_pipeline.py --name my_team      # writes evaluation/results/my_team.csv

Steps, in order (each is also runnable on its own):
    1. build_table.py    read every tactical source onto t = 0..199318, with alignment checks
    2. features.py       per frame clues
    3. model.py          pseudo labels, learners, sticky decoding, submission CSV
    4. validate          the organisers' own format check on the written file

Everything is deterministic: the same data gives the same file, byte for byte.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(step, *args):
    print("\n" + "=" * 72 + "\n" + step + "\n" + "=" * 72, flush=True)
    result = subprocess.run([sys.executable] + list(args), cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("\nStopped: '{}' failed (exit code {}).".format(step, result.returncode))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v2", help="submission file name, without .csv")
    args = parser.parse_args()
    began = time.time()

    run("1/4  Building the master frame table", os.path.join(HERE, "build_table.py"))
    run("2/4  Computing features", os.path.join(HERE, "features.py"))
    run("3/4  Model and decoding", os.path.join(HERE, "model.py"), "--name", args.name)
    path = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    run("4/4  Validating the submission", os.path.join(ROOT, "evaluation", "validate_submission.py"),
        path)
    print("\nAll done in {:.0f} s. Submission: {}\n".format(time.time() - began,
                                                          os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()

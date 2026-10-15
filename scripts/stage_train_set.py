"""Copy the games of a frozen training-id list into a folder the job can mount as train.

scripts/build_train500.py writes the id list (one trial-dir name per line); this script
copies each listed trial directory, under its task folder, from ALFWorld's train split into
a new folder laid out like train500_v2 (<task folder>/<trial dir>/game.tw-pddl ...). Copies,
not symlinks: ALFWorld finds games with os.walk, which does not descend into symlinked
directories.

    python scripts/stage_train_set.py --ids data/train1500_gamefiles.txt \
        --src $ALFWORLD_SRC/json_2.1.1/train --dest /path/to/train1500_v1

Queue a run on it with TRAIN_TASKS=/path/to/train1500_v1 TRAIN_EPS=1500 (slurm/arm.slurm).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="ALFWorld json_2.1.1/train")
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()

    ids = read_ids(args.ids)
    if len(set(ids)) != len(ids):
        sys.exit(f"ERROR: {args.ids} lists {len(ids) - len(set(ids))} ids twice")
    where = {}
    for game in args.src.glob("*/*/game.tw-pddl"):
        where.setdefault(game.parent.name, []).append(game.parent)
    missing = [i for i in ids if i not in where]
    ambiguous = [i for i in ids if len(where.get(i, [])) > 1]
    if missing or ambiguous:
        sys.exit(f"ERROR: {len(missing)} ids not found, {len(ambiguous)} found twice "
                 f"(e.g. {(missing or ambiguous)[:3]})")
    if args.dest.exists() and any(args.dest.iterdir()):
        sys.exit(f"ERROR: {args.dest} is not empty; refusing to mix training sets")

    for i in ids:
        trial = where[i][0]
        shutil.copytree(trial, args.dest / trial.parent.name / trial.name)
    n = sum(1 for _ in args.dest.glob("*/*/game.tw-pddl"))
    tasks = sum(1 for p in args.dest.iterdir() if p.is_dir())
    print(f"Copied {n} games in {tasks} task folders to {args.dest}")
    if n != len(ids):
        sys.exit(f"ERROR: expected {len(ids)} games, found {n}")


if __name__ == "__main__":
    main()

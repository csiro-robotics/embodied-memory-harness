"""Build the frozen train-500 game list for the unified protocol (SDD F5, §4.2).

Enumerates ALFWorld's train split, classifies each game into one of the 6 families by its
trial-directory name prefix (no need to load the game), stratified-samples 500 preserving
family proportions with Random(42), and writes release-stable ids (the trial-dir name) to
data/train500_gamefiles.txt. Commit that file so every arm trains on byte-identical games.

Run on a machine with ALFWORLD_DATA set and the pinned alfworld package installed:

    python scripts/build_train500.py --out data/train500_gamefiles.txt --n 500 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alfworld_env import game_id  # release-stable id = trial-dir name

# ALFWorld task-type prefix (trial-dir name) -> our 6-family label.
_PREFIX_TO_FAMILY = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "look_at_obj_in_light": "examine_in_light",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
}


def family_from_path(game_file: str) -> str:
    """Classify a game by its task-folder name prefix; '' if unrecognised.

    ALFWorld lays games out as <task folder>/<trial dir>/game.tw-pddl, and only the task
    folder ("pick_heat_then_place_in_recep-Egg-None-Fridge-13") names the task type; the
    trial dir is "trial_T...". Reading the trial dir classified nothing (H20).
    """
    task = Path(game_file).parent.parent.name
    for prefix, fam in _PREFIX_TO_FAMILY.items():
        if task.startswith(prefix):
            return fam
    return ""


def list_train_games() -> list[str]:
    """Return all ALFWorld train game files (before any init_env truncation)."""
    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic

    data = os.environ.get("ALFWORLD_DATA", "")
    base_config = os.path.join(data, "base_config.yaml")
    if not os.path.exists(base_config):
        raise FileNotFoundError(f"base_config.yaml not found at {base_config}")

    saved = sys.argv[:]
    sys.argv = [sys.argv[0], base_config]
    config = generic.load_config()
    sys.argv = saved
    config["env"]["type"] = "AlfredTWEnv"
    config.setdefault("dataset", {}).setdefault("num_train_games", 0)  # 0 = all

    env = environment.get_environment("AlfredTWEnv")(config, train_eval="train")
    return list(env.game_files)


def stratified_sample(games: list[str], n: int, seed: int) -> list[str]:
    """Sample n games preserving family proportions, deterministically."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for g in games:
        fam = family_from_path(g)
        if fam:
            by_family[fam].append(g)

    total = sum(len(v) for v in by_family.values())
    if total == 0:
        raise RuntimeError("No games classified — check trial-dir naming / data path.")

    rng = random.Random(seed)
    # Proportional allocation with largest-remainder rounding to hit exactly n.
    raw = {fam: n * len(v) / total for fam, v in by_family.items()}
    alloc = {fam: int(x) for fam, x in raw.items()}
    remainder = n - sum(alloc.values())
    for fam in sorted(by_family, key=lambda f: raw[f] - alloc[f], reverse=True)[:remainder]:
        alloc[fam] += 1

    chosen: list[str] = []
    for fam, k in alloc.items():
        pool = sorted(by_family[fam], key=game_id)  # stable, id-sorted
        chosen += rng.sample(pool, min(k, len(pool)))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/train500_gamefiles.txt"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    games = list_train_games()
    chosen = stratified_sample(games, args.n, args.seed)

    from collections import Counter
    dist = Counter(family_from_path(g) for g in chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# train{args.n} game ids (seed {args.seed}) — release-stable trial-dir names"]
    lines += [f"# family distribution: {dict(sorted(dist.items()))}"]
    lines += sorted(game_id(g) for g in chosen)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(chosen)} ids to {args.out}")
    print(f"Family distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()

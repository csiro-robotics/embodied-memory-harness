"""Thin wrapper around ALFWorld text-mode environment."""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic game selection & ordering  (SDD F4/F5/N1, §4.2)
#
# ALFWorld bakes game order into textworld.gym.register_games() INSIDE init_env,
# building the list via os.walk (filesystem order, machine-dependent). To make the
# task stream identical across every memory arm we must control the game_files list
# BEFORE init_env, using a single shared seed. These helpers are pure (no ALFWorld
# import) so they are unit-testable on any machine.
# ---------------------------------------------------------------------------

def game_id(game_file: str) -> str:
    """Release-stable identifier for an ALFWorld game file.

    The TW game lives at .../<trial_dir>/game.tw-pddl; the trial directory name encodes
    the task and is stable across ALFWorld data releases, unlike the filename stem
    (often just 'game') or the absolute path (release/layout dependent).
    """
    return Path(game_file).parent.name


def select_and_order_games(
    all_files: list[str],
    split: str,
    seed: int,
    num_episodes: int,
    train500_ids: list[str] | None = None,
) -> list[str]:
    """Return the exact, deterministic game-file sequence to run.

    train split: intersect with the frozen train500 id list, HARD-FAIL on shortfall,
    then shuffle by seed and take the first num_episodes. eval split: shuffle by seed
    and take the first num_episodes. The same (split, seed) yields the same order for
    every arm, which is the whole point of the unified protocol.
    """
    files = list(all_files)

    if split == "train" and train500_ids:
        wanted = set(train500_ids)
        files = [f for f in files if game_id(f) in wanted]
        if len(files) != len(wanted):
            raise RuntimeError(
                f"train500 intersection produced {len(files)} games but the frozen list "
                f"has {len(wanted)}. The ids in train500_gamefiles.txt do not all exist "
                f"under the current ALFWorld release. Rebuild data/train500_gamefiles.txt "
                f"against the pinned release (scripts/build_train500.py)."
            )

    # Canonicalise the input order (by stable game id) BEFORE shuffling. shuffle permutes
    # the list as-given, so without this the result would depend on ALFWorld's
    # machine-dependent os.walk order. Sorting first makes the order a pure function of
    # (set of games, seed) — identical across arms AND reproducible across machines/reruns.
    files = sorted(files, key=game_id)
    random.Random(seed).shuffle(files)
    if num_episodes and num_episodes > 0:
        files = files[:num_episodes]
    return files

_FAMILY_PATTERNS = [
    (r"put\s+(a|an|some|two)\s+.+\s+in/on\s+", "pick_and_place"),
    (r"clean\s+(a|an|some)\s+", "pick_clean_then_place"),
    (r"heat\s+(a|an|some)\s+", "pick_heat_then_place"),
    (r"cool\s+(a|an|some)\s+", "pick_cool_then_place"),
    (r"examine\s+.+\s+with\s+the\s+", "examine_in_light"),
    (r"put\s+two\s+", "pick_two_obj_and_place"),
    (r"find\s+two\s+", "pick_two_obj_and_place"),
]

_OBJECT_MODIFIERS = frozenset({
    "heated", "hot", "cold", "cool", "clean", "dirty", "fresh",
    "used", "sliced", "cooked", "boiled", "chilled",
})

def extract_target_objects(task_desc: str) -> list[str]:
    """Extract the primary target object(s) from an ALFWorld task description.

    ALFWorld tasks consistently follow: <verb> (a|an|some|two) [modifier] <object> ...
    Examples:
      "heat up an apple and put it in the fridge"   → ["apple"]
      "clean some candles under running water"       → ["candle"]
      "put two peppershakers on the counter"         → ["peppershaker"]
      "examine some book under the bright light"     → ["book"]

    Returns a list (usually 1 item; pick_two_obj_and_place both objects are the
    same type so still 1 item).  Returns [] if extraction fails.
    """
    desc = task_desc.lower().strip()
    tokens = [t.rstrip("s.,;") for t in desc.split()]  # strip plural/punctuation
    # W17: the examine family is phrased "examine the bowl with the desklamp" or "look at
    # bowl under the desklamp", with no a/an/some/two, and "cd" is two letters. All 18
    # examine tasks and the cd tasks returned [], so the wiki gave them no locations. "the"
    # only opens the object after "examine", and "at" only after "look".
    articles = {"a", "an", "some", "two"}
    openers = {("examine", "the"), ("look", "at")}

    for i, tok in enumerate(tokens):
        if tok not in articles and (tokens[i - 1] if i else "", tok) not in openers:
            continue
        j = i + 1
        if j >= len(tokens):
            break
        candidate = tokens[j]
        # Skip known modifiers (e.g. "heated") to reach the actual object noun.
        if candidate in _OBJECT_MODIFIERS and j + 1 < len(tokens):
            j += 1
            candidate = tokens[j]
        # Accept if it looks like a noun (alpha, length > 1, not a stop word).
        stop = {"the", "a", "an", "some", "two", "and", "or", "in", "on", "at", "to"}
        if len(candidate) > 1 and candidate.isalpha() and candidate not in stop:
            return [candidate]

    return []


def classify_task_family(task_desc: str) -> str:
    """Map a task description to one of ALFWorld's 6 task families."""
    desc = task_desc.lower().strip()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, desc):
            return family
    if "clean" in desc:
        return "pick_clean_then_place"
    if "heat" in desc or "hot" in desc:
        return "pick_heat_then_place"
    if "cool" in desc or "cold" in desc:
        return "pick_cool_then_place"
    if "light" in desc or "lamp" in desc or "examine" in desc:
        return "examine_in_light"
    if "two" in desc:
        return "pick_two_obj_and_place"
    return "pick_and_place"

def extract_task_description(initial_obs: str) -> str:
    """Extract the task instruction from ALFWorld's initial observation."""
    match = re.search(r"[Yy]our task is to:\s*(.+?)(?:\n|$)", initial_obs)
    if match:
        return match.group(1).strip().rstrip(".")
    lines = [l.strip() for l in initial_obs.strip().split("\n") if l.strip()]
    return lines[-1] if lines else initial_obs

class ALFWorldEnv:
    """Wrapper for ALFWorld text-mode that provides a clean per-episode interface."""

    def __init__(
        self,
        train_eval: str = "eval_out_of_distribution",
        max_steps: int = 50,
        seed: int = 42,
        num_episodes: int = -1,
        train500_path: str | Path | None = None,
        domain_randomization: bool = False,
    ):
        import sys
        import os
        import alfworld as _alf
        import alfworld.agents.environment as environment
        import alfworld.agents.modules.generic as generic

        self.max_steps = max_steps
        self._seed = seed
        # alfworld's load_config() parses sys.argv; give it base_config.yaml from ALFWORLD_DATA
        _alfworld_data = os.environ.get("ALFWORLD_DATA", "")
        _base_config = os.path.join(_alfworld_data, "base_config.yaml")
        if not os.path.exists(_base_config):
            raise FileNotFoundError(
                f"ALFWorld base_config.yaml not found at {_base_config}. "
                f"Create it or set ALFWORLD_DATA correctly."
            )
        _saved_argv = sys.argv[:]
        sys.argv = [sys.argv[0], _base_config]
        self._config = generic.load_config()
        sys.argv = _saved_argv
        self._config["env"]["type"] = "AlfredTWEnv"
        # The protocol's frozen key, applied here rather than left to whatever
        # base_config.yaml says (H15). ALFWorld shuffles object names when it is on.
        self._config["env"]["domain_randomization"] = bool(domain_randomization)
        # ALFWorld checks dataset.num_train_games when loading the train split.
        # base_config.yaml may not include this key; default 0 = use all games.
        self._config.setdefault("dataset", {})
        self._config["dataset"].setdefault("num_train_games", 0)

        env_type = self._config["env"]["type"]
        raw_env = environment.get_environment(env_type)(self._config, train_eval=train_eval)

        # Control game ordering deterministically BEFORE init_env registers the games.
        # train_eval is "train" / "eval_in_distribution" / "eval_out_of_distribution";
        # the selection helper only intersects the train500 list on the train split.
        if hasattr(raw_env, "game_files") and raw_env.game_files:
            split = "train" if train_eval == "train" else train_eval
            train500_ids = None
            if train500_path and split == "train":
                ids = [
                    line.strip()
                    for line in Path(train500_path).read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                train500_ids = ids
            ordered = select_and_order_games(
                list(raw_env.game_files),
                split=split,
                seed=seed,
                num_episodes=num_episodes,
                train500_ids=train500_ids,
            )
            raw_env.game_files = ordered
            log.info(
                "[env] %s: ordered %d games (seed=%d%s)",
                train_eval, len(ordered), seed,
                ", train500-intersected" if train500_ids else "",
            )

        self._env = raw_env.init_env(batch_size=1)
        self._current_task_desc: str = ""
        self._current_task_family: str = ""
        self._episode_count: int = 0
        self._game_file: str = ""

    def reset(self) -> tuple[str, list[str]]:
        obs, info = self._env.reset()
        observation = obs[0]
        admissible = list(info["admissible_commands"][0])
        self._current_task_desc = extract_task_description(observation)
        self._current_task_family = classify_task_family(self._current_task_desc)
        self._episode_count += 1
        # ALFWorld requests the `gamefile` extra, so reset() reports which game file this
        # episode plays. Logging only (H12); no arm sees it.
        gf = info.get("extra.gamefile") if isinstance(info, dict) else None
        self._game_file = str(gf[0]) if gf else ""
        if not self._game_file and self._episode_count == 1:
            log.warning("[env] reset() reported no extra.gamefile; game files will not be logged")
        return observation, admissible

    def step(self, action: str) -> tuple[str, float, bool, list[str]]:
        obs, scores, dones, infos = self._env.step([action])
        observation = obs[0]
        score = scores[0]
        done = dones[0]
        admissible = list(infos["admissible_commands"][0])
        return observation, score, done, admissible

    @property
    def task_description(self) -> str:
        return self._current_task_desc

    @property
    def task_family(self) -> str:
        return self._current_task_family

    @property
    def task_id(self) -> str:
        """A per-phase episode counter, `episode_<n>`.

        This used to try to read the game file and fall back to the counter. The lookup
        always failed, so every run so far (Run 10 included) recorded `episode_<n>`, and
        the arms use task_id as a unique per-episode label. It stays the counter so that no
        arm changes; the game file is logged separately (see `game_file`).
        """
        return f"episode_{self._episode_count}"

    @property
    def game_file(self) -> str:
        """Path of the ALFWorld game file the current episode plays ("" if unknown)."""
        return self._game_file

    @property
    def num_episodes(self) -> int:
        try:
            return len(self._env.gamefiles)
        except Exception:
            return -1

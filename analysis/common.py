"""Shared paths and helpers for the analysis scripts.

The scripts read the run logs in results/. Analyses that check where a task's object really starts also need
the ALFWorld game files: set ALFWORLD_DATA to a standard ALFWorld download (the folder holding json_2.1.1/).
"""
import gzip
import json
import os
import re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FINAL = os.path.join(ROOT, "results", "final")
ABLATION = os.path.join(ROOT, "results", "ablation")

# The 500-game runs, three replicates per arm. "pm" is the procedural memory arm (MACLA).
RUNS = {
    "wiki": ["f500-r1-wiki_v3-60561790", "f500-r2-wiki_v3-60561791", "f500-r3-wiki_v3-60561792"],
    "kg": ["f500-r1-kg_hybrid-60561796", "f500-r2-kg_hybrid-60561797", "f500-r3-kg_hybrid-60561798"],
    "pm": ["f500-r1-macla-p28-60692948", "f500-r2-macla-p28-60692949", "f500-r3-macla-p28-60692950"],
    "control": ["r1-no_memory-60496024", "r2-no_memory-60496030", "r3-no_memory-60496034"],
}
RUNS_1500 = {
    "wiki": "f1500-wiki_v3-60561800",
    "kg": "f1500-kg_hybrid-60561804",
    "pm": "f1500-macla-p28-60692947",
    "control": "f1500-no_memory-60561799",
}
FAMS = ["examine_in_light", "pick_and_place", "pick_clean_then_place", "pick_cool_then_place",
        "pick_heat_then_place", "pick_two_obj_and_place"]


def load_episodes(run_dir, phase="eval"):
    """Every episode record of one phase (eval or eval_seen) of a run, or None if the log is missing."""
    p = os.path.join(run_dir, phase, "episodes.jsonl.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def final_episodes(run, phase="eval"):
    return load_episodes(os.path.join(FINAL, run), phase)


def key(game_file):
    """The game's path below valid_unseen/ or valid_seen/, the same on every machine."""
    return re.split(r"valid_(?:un)?seen/", game_file, 1)[1]


def target(desc):
    """The object type the task is about, e.g. 'egg' in 'put a clean egg in microwave'."""
    return re.search(r"(?:put|place|examine|look at|clean|heat|cool|find)\s+(?:a |an |the |two |some )?"
                     r"(?:clean |hot |cool |cold )?(\w+)", desc).group(1)


def saw(r, obj):
    """True if an object of the task's type appeared in any reply the agent received."""
    obs = [t["observation"] for t in r["trajectory"][1:]] + [str(r.get("final_observation", ""))]
    return any(re.search(r"\b" + obj + r" \d", o.lower()) for o in obs)


def game_dir(split="valid_unseen"):
    data = os.environ.get("ALFWORLD_DATA")
    if not data:
        raise SystemExit("Set ALFWORLD_DATA to an ALFWorld download (the folder that holds json_2.1.1/).")
    return os.path.join(data, "json_2.1.1", split)


_start = {}


def start_places(k, obj, split="valid_unseen"):
    """Receptacle types that hold an object of the task's type at the start (from the game's PDDL),
    and all receptacle types in the room."""
    if k not in _start:
        with open(os.path.join(game_dir(split), k)) as f:
            p = json.load(f)["pddl_problem"]
        pairs = re.findall(r"\(inReceptacle (\w+?)_bar_\S* (\w+?)_bar_\S*\)", p)
        recs = set(r.lower() for r in re.findall(r"(\w+?)_bar_\S* - receptacle", p))
        _start[k] = (pairs, recs)
    pairs, recs = _start[k]
    alias = {"sink": "sinkbasin", "bathtub": "bathtubbasin"}
    norm = lambda r: alias.get(r.lower(), r.lower())
    return {norm(r) for o, r in pairs if o.lower() == obj}, {norm(r) for r in recs} | {norm(r) for _, r in pairs}


def hint_places(arm, text, obj, recs):
    """Receptacle types the memory text tells the agent to search for the object, in order of mention."""
    text = (text or "").lower()
    if arm == "kg":
        m = re.search(r"(?:search for|find|locate|look for|get) (?:the |a |an |two |both )?" + obj + r"s?[^\n]*", text)
        seg = m.group(0) if m else ""
    elif arm == "pm":
        m = re.search(r"likely locations of the " + obj + r"[^\n]*", text)
        seg = m.group(0) if m else ""
    else:
        parts = re.findall(r"(?:search|check|look (?:in|at|on)|take [^.]*? from|find [^.]*? (?:on|in|at))([^.]*)", text)
        seg = " ".join(re.split(r"\b(?:move|put|clean|cool|heat|use|examine)\b", p)[0] for p in parts)
    hits = []
    for m in re.finditer(r"[a-z]+", seg):
        w = m.group(0)
        if w in recs and w not in hits:
            hits.append(w)
    return hits

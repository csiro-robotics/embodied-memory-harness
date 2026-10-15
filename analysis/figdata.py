"""Data for the cost figure and the learning-curve figure, read from the run logs only.
Usage: python analysis/figdata.py > analysis/figdata.json"""
import json
import os

from common import FINAL, RUNS, RUNS_1500

SIZE = {"wiki": "total_words", "kg": "graph_edges", "pm": "n_procedures", "pm1500": "n_procedures"}
ARMS = {"wiki": RUNS["wiki"], "kg": RUNS["kg"], "pm": RUNS["pm"], "ctrl": RUNS["control"], "pm1500": [RUNS_1500["pm"]]}


def outcomes(path):
    """Success of every episode, in order, from a phase's diagnostics.jsonl (one line per episode)."""
    with open(path) as f:
        return [bool(json.loads(line)["success"]) for line in f if line.strip()]


res = {}
for arm, runs in ARMS.items():
    res[arm] = []
    for r in runs:
        d = os.path.join(FINAL, r)
        tr = outcomes(os.path.join(d, "train", "diagnostics.jsonl"))
        ev = outcomes(os.path.join(d, "eval", "diagnostics.jsonl"))
        with open(os.path.join(d, "train", "summary.json")) as f:
            s = json.load(f)
        row = {"run": r, "n_train": len(tr), "train_sr": round(100 * sum(tr) / len(tr), 1),
               "n_eval": len(ev), "eval_sr": round(100 * sum(ev) / len(ev), 1),
               "train_mem_calls": s["token_usage"]["memory"]["n_calls"],
               "blocks50": [round(100 * sum(tr[i:i + 50]) / len(tr[i:i + 50]), 1) for i in range(0, len(tr), 50)]}
        g = os.path.join(d, "train", "memory_growth.json")
        if arm in SIZE and os.path.exists(g):
            with open(g) as f:
                row["growth"] = [(p["ep"], p[SIZE[arm]]) for p in json.load(f)]
        res[arm].append(row)
print(json.dumps(res, indent=1))

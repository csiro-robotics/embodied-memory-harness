"""Section 6 of the paper: per-family paired tests, how often the agent never saw the task's object,
and the PM split by whether a procedure reached the agent at the start of the episode.
Usage: python analysis/sec6_analysis.py"""
import random
import re
from collections import Counter, defaultdict

from common import FAMS, RUNS, final_episodes, key, saw, target


def sign_flip_p(diffs, resamples=100000, seed=42):
    rng = random.Random(seed)
    nz = [d for d in diffs if d]
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    hits = sum(1 for _ in range(resamples) if abs(sum(d if rng.random() < 0.5 else -d for d in nz)) >= obs - 1e-12)
    return (hits + 1) / (resamples + 1)


data = {a: [final_episodes(r) for r in runs] for a, runs in RUNS.items()}
fam_of = {key(r["game_file"]): r["task_family"] for r in data["control"][0]}
succ = {a: defaultdict(float) for a in RUNS}
for a, reps in data.items():
    for rep in reps:
        assert len(rep) == 134
        for r in rep:
            succ[a][key(r["game_file"])] += r["success"] / 3

print("=== per-family pooled sign-flip p ===")
for fam in FAMS:
    ts = [t for t in fam_of if fam_of[t] == fam]
    rates = {a: 100 * sum(succ[a][t] for t in ts) / len(ts) for a in RUNS}
    line = fam + " n=%d " % len(ts) + " ".join("%s=%.0f" % (a, v) for a, v in rates.items())
    for x, y in [("wiki", "kg"), ("wiki", "pm"), ("kg", "pm"), ("wiki", "control"), ("kg", "control"), ("pm", "control")]:
        line += " | %s-%s p=%.3f" % (x, y, sign_flip_p([succ[x][t] - succ[y][t] for t in ts]))
    print(line)

print("\n=== episodes in which the task's object never appeared in a reply ===")
for a, reps in data.items():
    n = never = 0
    fails = Counter()
    fails_never = Counter()
    for rep in reps:
        for r in rep:
            n += 1
            seen = saw(r, target(r["task_description"]))
            never += not seen
            if not r["success"]:
                fails[r["task_family"]] += 1
                fails_never[r["task_family"]] += not seen
    print("%-8s never saw the object in %d of %d episodes (%.1f%%)" % (a, never, n, 100 * never / n))
    print("         failures by family (never saw / failed):",
          {f[5:15]: "%d/%d" % (fails_never[f], fails[f]) for f in FAMS})

print("\n=== PM: procedure in the memory block at step 0 ===")
by = defaultdict(lambda: [0, 0])
paths = defaultdict(Counter)
for rep in data["pm"]:
    for r in rep:
        has = "MACLA PROCEDURE" in (r.get("memory_context") or "")
        by[("all", has)][0] += r["success"]
        by[("all", has)][1] += 1
        by[(r["task_family"], has)][0] += r["success"]
        by[(r["task_family"], has)][1] += 1
        paths[r["task_family"]][(r.get("memory_metadata") or {}).get("path")] += 1
for k in sorted(by):
    s, t = by[k]
    print(k, "%d/%d = %.1f%%" % (s, t, 100 * s / t))
for f, c in paths.items():
    print("path at step 0", f, dict(c))

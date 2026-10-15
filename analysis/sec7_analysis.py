"""Section 7 of the paper:
A. Did the memory name the place where the task's object really starts (from the game's PDDL)?
B. Does the PM pass a procedure at the start of more episodes after 1,500 training games?
C. How far do the Wiki's and the KG's solved tasks overlap?
Analysis A needs ALFWORLD_DATA (see common.py). Usage: python analysis/sec7_analysis.py"""
from collections import defaultdict

from common import FAMS, RUNS, RUNS_1500, final_episodes, hint_places, key, saw, start_places, target

data = {a: [final_episodes(r) for r in runs] for a, runs in RUNS.items()}
ctrl = defaultdict(float)
for rep in data["control"]:
    for r in rep:
        ctrl[key(r["game_file"])] += r["success"] / 3

print("=== A. did the memory's search hint name a place where the object starts? ===")
for arm in ("wiki", "kg", "pm"):
    g = defaultdict(lambda: [0, 0, 0.0, 0])  # success, n, control success on same tasks, never saw
    for rep in data[arm]:
        for r in rep:
            k = key(r["game_file"])
            obj = target(r["task_description"])
            true, recs = start_places(k, obj)
            hint = hint_places(arm, r.get("memory_context"), obj, recs)
            if not hint:
                grp = "no hint"
            elif hint[0] in true:
                grp = "first place right"
            elif true & set(hint):
                grp = "right place named later"
            else:
                grp = "right place not named"
            x = g[grp]
            x[0] += r["success"]; x[1] += 1; x[2] += ctrl[k]; x[3] += not saw(r, obj)
    tot = sum(v[1] for v in g.values())
    for grp, (s, n, c, nv) in sorted(g.items()):
        print("%-5s %-25s %3d eps (%4.1f%%)  success %5.1f%%  never saw %5.1f%%  control on same tasks %5.1f%%"
              % (arm, grp, n, 100 * n / tot, 100 * s / n, 100 * nv / n, 100 * c / n))

print("\n=== B. PM procedure at step 0, 500 vs 1,500 training games ===")
for name, runs in [("500 (3 reps)", RUNS["pm"]), ("1500", [RUNS_1500["pm"]])]:
    by = defaultdict(lambda: [0, 0])
    for run in runs:
        for r in final_episodes(run):
            has = "MACLA PROCEDURE" in (r.get("memory_context") or "")
            for f in ("all", r["task_family"]):
                by[(f, has)][0] += r["success"]
                by[(f, has)][1] += 1
    for f in ["all"] + FAMS:
        w, wo = by[(f, True)], by[(f, False)]
        n = w[1] + wo[1]
        print("%-14s %-24s procedure at start %3d/%3d (%4.1f%%)  success with %s  without %s" % (
            name, f, w[1], n, 100 * w[1] / n, "%d/%d" % tuple(w), "%d/%d" % tuple(wo)))

print("\n=== C. overlap of solved tasks, Wiki and KG ===")
for i in range(3):
    w = {key(r["game_file"]) for r in data["wiki"][i] if r["success"]}
    k = {key(r["game_file"]) for r in data["kg"][i] if r["success"]}
    print("replicate %d: both %d, Wiki only %d, KG only %d, union %d" % (i + 1, len(w & k), len(w - k), len(k - w), len(w | k)))
W = {key(r["game_file"]) for rep in data["wiki"] for r in rep if r["success"]}
K = {key(r["game_file"]) for rep in data["kg"] for r in rep if r["success"]}
P = {key(r["game_file"]) for rep in data["pm"] for r in rep if r["success"]}
print("solved at least once: Wiki %d, KG %d, PM %d, Wiki or KG %d, all three %d, PM only %d" % (
    len(W), len(K), len(P), len(W | K), len(W | K | P), len(P - W - K)))

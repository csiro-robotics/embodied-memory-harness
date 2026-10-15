"""The two Wiki interventions in the paper, both evaluated on copies of the trained Wiki stores:
1. Search-order ablation: no ranked search order in the briefing, paired task by task against the original
   evaluation of each replicate.
2. Replicate 3's examine page edited by hand. fix-r3 deletes the faulty step, fix2-r3 rewrites the steps.
   Both run on valid_unseen and valid_seen, against the unedited store.
The location check needs ALFWORLD_DATA (see common.py). Usage: python analysis/ablation_analysis.py"""
import os
import random
from collections import defaultdict
from math import comb

from common import ABLATION, FAMS, RUNS, final_episodes, hint_places, key, load_episodes, saw, start_places, target


def abl(name, phase="eval"):
    return load_episodes(os.path.join(ABLATION, name), phase)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(new, old):
    """new, old: lists over replicates of {task: success}. Pooled per-task mean difference, sign-flip p, bootstrap CI."""
    tasks = sorted(set.intersection(*[set(x) for x in new + old]))
    d = [sum(n[t] - o[t] for n, o in zip(new, old)) / len(new) for t in tasks]
    obs = sum(d) / len(d)
    rng = random.Random(0)
    flips = sum(abs(sum(x if rng.random() < .5 else -x for x in d) / len(d)) >= abs(obs) - 1e-12 for _ in range(20000))
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
    return 100 * obs, (100 * boots[124], 100 * boots[4874]), (flips + 1) / 20001, len(tasks)


def by_task(eps):
    return {key(r["game_file"]): int(r["success"]) for r in eps}


def sr(eps):
    return 100 * sum(r["success"] for r in eps) / len(eps)


def fam(eps, f):
    x = [r["success"] for r in eps if r["task_family"] == f]
    return "%d/%d" % (sum(x), len(x))


orig = [final_episodes(r) for r in RUNS["wiki"]]
ctrl = defaultdict(float)
for r in RUNS["control"]:
    for e in final_episodes(r):
        ctrl[key(e["game_file"])] += e["success"] / 3

print("=== 1. search-order ablation (valid_unseen) ===")
ab = [abl("nosearch-r%d" % n) for n in (1, 2, 3)]
done = [i for i in range(3) if ab[i] and len(ab[i]) == 134]
for i in range(3):
    if not ab[i]:
        continue
    o, a = by_task(orig[i]), by_task(ab[i])
    b = sum(1 for t in a if a[t] and not o[t])
    c = sum(1 for t in a if o[t] and not a[t])
    so = sum("SEARCH ORDER" in (r.get("memory_context") or "") for r in ab[i])
    print("r%d  n=%d  ablated %.1f%%  original %.1f%%  gained %d lost %d  McNemar p=%.4f  briefings with search order %d"
          % (i + 1, len(ab[i]), sr(ab[i]), 100 * sum(o[t] for t in a) / len(a), b, c, mcnemar(b, c), so))
if len(done) == 3:
    diff, ci, p, nt = paired([by_task(x) for x in ab], [by_task(x) for x in orig])
    print("pooled over 3 replicates: %.1f pp [%.1f, %.1f], sign-flip p=%.4f, %d tasks; ablated mean %.1f%%"
          % (diff, ci[0], ci[1], p, nt, sum(sr(x) for x in ab) / 3))
    for f in FAMS:
        print("  %-24s ablated %s  original %s" % (f, " ".join(fam(x, f) for x in ab), " ".join(fam(x, f) for x in orig)))

print("\n--- location hint and finding the object (Section 7 analysis A), Wiki original vs ablated ---")
for name, sets in (("original", [orig[i] for i in done]), ("ablated", [ab[i] for i in done])):
    g = defaultdict(lambda: [0, 0, 0.0, 0])
    for rep in sets:
        for r in rep:
            k = key(r["game_file"])
            obj = target(r["task_description"])
            true, recs = start_places(k, obj)
            hint = hint_places("wiki", r.get("memory_context"), obj, recs)
            grp = ("no hint" if not hint else "first place right" if hint[0] in true
                   else "right place named later" if true & set(hint) else "right place not named")
            x = g[grp]
            x[0] += r["success"]; x[1] += 1; x[2] += ctrl[k]; x[3] += not saw(r, obj)
    tot = sum(v[1] for v in g.values())
    nv = sum(v[3] for v in g.values())
    print("%s: never saw the object in %.1f%% of %d episodes" % (name, 100 * nv / tot, tot))
    for grp, (s, n, c, v) in sorted(g.items()):
        print("  %-25s %3d eps (%4.1f%%)  success %5.1f%%  never saw %5.1f%%  control same tasks %5.1f%%"
              % (grp, n, 100 * n / tot, 100 * s / n, 100 * v / n, 100 * c / n))


def compare(name, new, old, ref=()):
    o, f = by_task(old), by_task(new)
    b = sum(1 for t in f if f[t] and not o.get(t, 0))
    c = sum(1 for t in f if o.get(t, 0) and not f[t])
    print("%s n=%d  edited %.1f%%  unedited %.1f%%  gained %d lost %d  McNemar p=%.4f"
          % (name, len(new), sr(new), sr(old), b, c, mcnemar(b, c)))
    for fm in FAMS:
        print("  %-24s edited %s  unedited %s%s" % (fm, fam(new, fm), fam(old, fm),
                                                   "".join("  %s %s" % (n, fam(e, fm)) for n, e in ref)))


def examine(name, eps):
    ex = [r for r in eps if r["task_family"] == "examine_in_light"]
    mv = sum("move" in (r.get("memory_context") or "").lower() for r in ex)
    putback = sum(sum(1 for t in r["trajectory"] if str(t.get("action", "")).startswith("move ")) for r in ex)
    lamp = sum(any(str(t.get("action", "")).startswith("use desklamp") for t in r["trajectory"]) for r in ex)
    print("  examine %-14s briefings mentioning move %d/%d, move actions %d, tasks with the lamp switched on %d"
          % (name, mv, len(ex), putback, lamp))


seen_base = abl("r3-seen")
for title, d in (("step deleted", "fix-r3"), ("page corrected", "fix2-r3")):
    print("\n=== 2. replicate 3, examine page %s ===" % title)
    u = abl(d)
    if u and len(u) == 134:
        compare("valid_unseen", u, orig[2], (("r1", orig[0]), ("r2", orig[1])))
        examine(title, u)
        examine("unedited r3", orig[2])
    se = abl(d, "eval_seen")
    if se and seen_base and len(se) == len(seen_base) == 140:
        compare("valid_seen", se, seen_base)
        examine(title + " seen", se)
        examine("unedited seen", seen_base)
    elif se:
        print("valid_seen incomplete: %d edited, %d unedited" % (len(se), len(seen_base or [])))

"""The examine-in-light tasks of the three Wiki replicates (Section 6): per replicate, where the lamp was
switched on, and how often the agent put the object back where it took it from.
Usage: python analysis/examine_analysis.py"""
import re
from collections import Counter

from common import RUNS, final_episodes

for rep, run in zip(("r1", "r2", "r3"), RUNS["wiki"]):
    eps = [r for r in final_episodes(run) if r["task_family"] == "examine_in_light"]
    lamp_at = Counter()
    went_lamp = putback_ep = putback_n = lamp_on = fails = fails_putback = 0
    for e in eps:
        obj = re.search(r"(?:examine the|look at) (\w+)", e["task_description"]).group(1)
        actions = [t["action"] for t in e["trajectory"]]
        replies = [t["observation"] for t in e["trajectory"][1:]] + [str(e.get("final_observation", ""))]
        loc = took_from = None
        pb = 0
        visited = set()
        used = False
        for a, r in zip(actions, replies):
            m = re.match(r"go to (.+)$", a)
            if m:
                loc = m.group(1); visited.add(loc)
            m = re.match(r"use (desklamp|floorlamp) \d+", a)
            if m and r.lower().startswith("you turn on"):
                used = True; lamp_at[loc] += 1
            m = re.match(r"take (" + obj + r" \d+) from (.+)$", a)
            if m and r.lower().startswith("you pick up"):
                took_from = m.group(2)
            m = re.match(r"move (" + obj + r" \d+) to (.+)$", a)
            if m and r.lower().startswith("you move") and m.group(2) == took_from:
                pb += 1
        lamp_on += used
        went_lamp += "desk 1" in visited
        putback_ep += pb > 0
        putback_n += pb
        if not e["success"]:
            fails += 1
            fails_putback += pb > 0
    print(rep, "solved", sum(x["success"] for x in eps), "of", len(eps), "| lamp turned on in", lamp_on,
          "| lamp locations", dict(lamp_at), "| visited desk 1 in", went_lamp,
          "| episodes with a put-back", putback_ep, "(total put-backs", putback_n, ")",
          "| failures", fails, "of which with put-back", fails_putback)

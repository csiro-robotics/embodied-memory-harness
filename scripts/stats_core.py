"""The statistics behind the 500-game comparison in the paper. scripts/final_stats.py runs it.

Reads the 12 runs in results/final/ (4 arms x 3 replicates, valid_unseen, 134 tasks
each) and prints:

  * evaluation success per replicate and as mean [min, max]
  * success by task family, as mean [min, max] over replicates
  * paired comparisons: an exact McNemar test in each replicate, and a pooled test over
    all three (a task-level sign-flip permutation test on each task's mean success over
    the replicates, with a bootstrap 95% interval over tasks)
  * training success, steps, memory calls, memory text size and time per episode
  * the size of each arm's trained store

It also writes every number to results/final/paper_numbers.json.

Tasks are paired by their game file, not by position: replicate N uses seed 42 + N - 1,
which changes the order the 134 tasks are played in. Within one replicate the
four arms play the same order.

Why the pooled test works per task: the three replicates replay the same 134 tasks, so
their outcomes are not independent. Adding the three McNemar tables would count each task
three times. Averaging each task over the replicates first keeps the task as the unit.

Run it through scripts/final_stats.py.
"""

import json
import math
import os
import random
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "final"))

# The run list; scripts/final_stats.py sets it to the paper's runs.
ARMS = OrderedDict()
MEMORY_ARMS = ["kg_hybrid", "wiki_v3", "macla"]
PAIRS = [("kg_hybrid", "no_memory"), ("wiki_v3", "no_memory"), ("macla", "no_memory"),
         ("kg_hybrid", "wiki_v3"), ("kg_hybrid", "macla"), ("wiki_v3", "macla")]
FAMILIES = ["examine_in_light", "pick_and_place", "pick_clean_then_place",
            "pick_cool_then_place", "pick_heat_then_place", "pick_two_obj_and_place"]
N_EVAL = 134
N_TRAIN = 500
RESAMPLES = 100000
SEED = 42


def game_key(path):
    """The game's own folder, without the per-job copy prefix."""
    for split in ("valid_unseen/", "valid_seen/", "train/"):
        if split in path:
            return path.split(split, 1)[1]
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(run):
    base = os.path.join(RESULTS, run)
    with open(os.path.join(base, "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    return {
        "report": report,
        "eval": read_jsonl(os.path.join(base, "eval", "diagnostics.jsonl")),
        "train": read_jsonl(os.path.join(base, "train", "diagnostics.jsonl")),
    }


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired 0/1 lists. Returns (a only, b only, p)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2 ** n)


def sign_flip_p(diffs, resamples=RESAMPLES, seed=SEED):
    """Two-sided permutation p for mean(diffs) == 0, flipping each task's sign."""
    rng = random.Random(seed)
    nz = [d for d in diffs if d]
    observed = abs(sum(nz))
    if not nz:
        return 1.0
    hits = 0
    for _ in range(resamples):
        if abs(sum(d if rng.random() < 0.5 else -d for d in nz)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def bootstrap_ci(diffs, resamples=20000, seed=SEED):
    rng = random.Random(seed)
    n = len(diffs)
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def spread(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values),
            "per_replicate": values}


def pct(s):
    return "%.1f [%.1f, %.1f]" % (100 * s["mean"], 100 * s["min"], 100 * s["max"])


def num(s, fmt="%.1f"):
    return (fmt + " [" + fmt + ", " + fmt + "]") % (s["mean"], s["min"], s["max"])


def main():
    data = {arm: [load(r) for r in runs] for arm, runs in ARMS.items()}
    out = {"runs": ARMS, "checks": {}}

    # ---- checks: same tasks everywhere; same order within a replicate; 500 distinct games
    tasks = None
    for arm, reps in data.items():
        for i, rep in enumerate(reps):
            keys = [game_key(r["game_file"]) for r in sorted(rep["eval"], key=lambda r: r["ep_idx"])]
            if len(keys) != N_EVAL or len(set(keys)) != N_EVAL:
                raise SystemExit("%s replicate %d: %d eval episodes, %d distinct"
                                 % (arm, i + 1, len(keys), len(set(keys))))
            tasks = tasks or sorted(keys)
            if sorted(keys) != tasks:
                raise SystemExit("%s replicate %d plays a different task set" % (arm, i + 1))
            ref = [game_key(r["game_file"]) for r in
                   sorted(data["no_memory"][i]["eval"], key=lambda r: r["ep_idx"])]
            if keys != ref:
                raise SystemExit("%s replicate %d: order differs from no_memory" % (arm, i + 1))
            train_games = {game_key(r["game_file"]) for r in rep["train"]}
            if len(rep["train"]) != N_TRAIN or len(train_games) != N_TRAIN:
                raise SystemExit("%s replicate %d: %d training episodes, %d distinct games"
                                 % (arm, i + 1, len(rep["train"]), len(train_games)))
    out["checks"] = {"eval_tasks": N_EVAL, "train_distinct_games": N_TRAIN,
                     "same_order_within_replicate": True}
    print("checks: 12 runs, the same %d tasks, the same order within each replicate, "
          "%d distinct training games per run\n" % (N_EVAL, N_TRAIN))

    family_of = {game_key(r["game_file"]): r["task_family"] for r in data["no_memory"][0]["eval"]}
    # success[arm][replicate][task] = 0/1
    success = {arm: [{game_key(r["game_file"]): int(bool(r["success"])) for r in rep["eval"]}
                     for rep in reps] for arm, reps in data.items()}

    # ---- evaluation success ------------------------------------------------------------
    print("=== evaluation success, % (mean [min, max] over 3 replicates) ===")
    out["eval_success"] = {}
    for arm in ARMS:
        rates = [sum(s.values()) / N_EVAL for s in success[arm]]
        solved = [sum(s.values()) for s in success[arm]]
        out["eval_success"][arm] = {**spread(rates), "solved": solved}
        print("%-10s %s   solved %s of %d" % (arm, pct(spread(rates)),
                                               "/".join(map(str, solved)), N_EVAL))

    print("\n=== evaluation success by task family, % ===")
    out["family_success"] = {}
    n_fam = {f: sum(1 for t in tasks if family_of[t] == f) for f in FAMILIES}
    print("%-28s %s" % ("family (n)", "  ".join("%-20s" % a for a in ARMS)))
    for fam in FAMILIES:
        cells, row = [], {}
        for arm in ARMS:
            rates = [sum(v for t, v in s.items() if family_of[t] == fam) / n_fam[fam]
                     for s in success[arm]]
            row[arm] = spread(rates)
            cells.append("%-20s" % pct(row[arm]))
        out["family_success"][fam] = {"n": n_fam[fam], **row}
        print("%-28s %s" % ("%s (%d)" % (fam, n_fam[fam]), "  ".join(cells)))

    # ---- paired tests ------------------------------------------------------------------
    print("\n=== paired comparisons ===")
    print("per replicate: A-only / B-only tasks and exact McNemar p;"
          " pooled: mean difference [95% CI], sign-flip p")
    out["paired"] = {}
    for a, b in PAIRS:
        reps = []
        for i in range(3):
            x = [success[a][i][t] for t in tasks]
            y = [success[b][i][t] for t in tasks]
            reps.append(mcnemar_exact(x, y))
        diffs = [sum(success[a][i][t] - success[b][i][t] for i in range(3)) / 3 for t in tasks]
        mean = sum(diffs) / len(diffs)
        lo, hi = bootstrap_ci(diffs)
        p = sign_flip_p(diffs)
        out["paired"]["%s_vs_%s" % (a, b)] = {
            "per_replicate": [{"a_only": r[0], "b_only": r[1], "p": r[2]} for r in reps],
            "pooled": {"mean_diff": mean, "ci95": [lo, hi], "p": p}}
        floor = 1.5 / (RESAMPLES + 1)   # no resample reached the observed difference
        print("%-9s vs %-9s | %s | %+5.1f pp [%+.1f, %+.1f]  p%s"
              % (a, b, "  ".join("%d/%d p=%.2g" % r for r in reps),
                 100 * mean, 100 * lo, 100 * hi,
                 "<1e-5" if p < floor else "=%.2g" % p))

    # ---- which tasks the arms solve ----------------------------------------------------
    print("\n=== tasks solved in at least one replicate ===")
    out["coverage"] = {}
    ever = {arm: {t for t in tasks if any(s[t] for s in success[arm])} for arm in ARMS}
    always = {arm: {t for t in tasks if all(s[t] for s in success[arm])} for arm in ARMS}
    for arm in ARMS:
        out["coverage"][arm] = {"ever": len(ever[arm]), "all_three": len(always[arm])}
        print("%-10s solved in any replicate %3d, in all three %3d" %
              (arm, len(ever[arm]), len(always[arm])))
    union = set().union(*(ever[a] for a in MEMORY_ARMS))
    nobody = [t for t in tasks if not any(t in ever[a] for a in ARMS)]
    out["coverage"]["any_memory_arm"] = len(union)
    out["coverage"]["no_arm"] = len(nobody)
    print("any memory arm: %d of %d; no arm at all: %d" % (len(union), N_EVAL, len(nobody)))

    # ---- training, steps, cost ---------------------------------------------------------
    print("\n=== training and cost, mean [min, max] over replicates ===")
    out["cost"] = {}
    rows = [
        ("training success, %", lambda r: 100 * r["train"]["sr"]),
        ("eval steps, solved tasks", lambda r: r["eval"]["steps"]["avg_ok"]),
        ("eval steps, all tasks", lambda r: r["eval"]["steps"]["avg"]),
        ("memory LLM calls / train ep", lambda r: r["train"]["token_usage"]["calls_per_episode"]["memory"]),
        ("memory LLM calls / eval ep", lambda r: r["eval"]["token_usage"]["calls_per_episode"]["memory"]),
        ("action LLM calls / eval ep", lambda r: r["eval"]["token_usage"]["calls_per_episode"]["action"]),
        ("memory tokens / eval step", lambda r: r["eval"]["memory_injection"]["injected_tokens_per_step"]),
        ("memory tokens / eval ep", lambda r: r["eval"]["memory_injection"]["injected_tokens_per_episode"]),
        ("peak memory tokens (eval)", lambda r: r["eval"]["memory_injection"]["peak_offered_tokens"]),
        ("steps cut to the budget, %", lambda r: 100 * r["eval"]["memory_injection"]["truncated_step_rate"]),
        ("train time, h", lambda r: r["train"]["elapsed_s"] / 3600),
        ("eval time, h", lambda r: r["eval"]["elapsed_s"] / 3600),
        ("seconds / eval ep", lambda r: r["eval"]["avg_ep_s"]),
    ]
    print("%-30s %s" % ("", "  ".join("%-22s" % a for a in ARMS)))
    for label, fn in rows:
        cells, row = [], {}
        for arm in ARMS:
            vals = []
            for rep in data[arm]:
                try:
                    vals.append(float(fn(rep["report"])))
                except (KeyError, TypeError):
                    pass
            row[arm] = spread(vals) if vals else None
            cells.append("%-22s" % (num(row[arm]) if vals else "n/a"))
        out["cost"][label] = row
        print("%-30s %s" % (label, "  ".join(cells)))

    # ---- store size --------------------------------------------------------------------
    print("\n=== trained store, per replicate ===")
    out["store"] = {}
    fields = {"kg_hybrid": ["graph_edges", "recipe_families", "episodic_steps"],
              "wiki_v3": ["n_pages", "total_words", "priors_objects", "evidence_forms"],
              "macla": ["n_procedures", "n_meta_procedures", "avg_posterior"]}
    for arm, keys in fields.items():
        vals = {k: [rep["report"]["train"]["memory_stats"].get(k) for rep in data[arm]]
                for k in keys}
        out["store"][arm] = vals
        print("%-10s %s" % (arm, "  ".join("%s %s" % (k, "/".join(map(str, v)))
                                            for k, v in vals.items())))

    path = os.path.join(RESULTS, "paper_numbers.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("\nwrote %s" % os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()

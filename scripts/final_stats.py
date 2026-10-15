"""Every run-derived number for the paper, from the final runs (results/final/).

The 500-game comparison is computed by stats_core.main on the final run list. The
control's runs are older than the memory arms' runs, because the control's code path did
not change between the two (the only shared change is called by the Wiki and MACLA alone).
It then adds the one-replicate 1,500-game runs, on valid_unseen and valid_seen.

Run from anywhere:  python scripts/final_stats.py
"""

import json
import os
from collections import OrderedDict

import stats_core as s6

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "final"))

ARMS_500 = OrderedDict([
    ("kg_hybrid", ["f500-r1-kg_hybrid-60561796", "f500-r2-kg_hybrid-60561797",
                   "f500-r3-kg_hybrid-60561798"]),
    ("wiki_v3",   ["f500-r1-wiki_v3-60561790", "f500-r2-wiki_v3-60561791",
                   "f500-r3-wiki_v3-60561792"]),
    ("macla",     ["f500-r1-macla-p28-60692948", "f500-r2-macla-p28-60692949",
                   "f500-r3-macla-p28-60692950"]),
    ("no_memory", ["r1-no_memory-60496024", "r2-no_memory-60496030", "r3-no_memory-60496034"]),
])
ARMS_1500 = OrderedDict([
    ("kg_hybrid", "f1500-kg_hybrid-60561804"),
    ("wiki_v3",   "f1500-wiki_v3-60561800"),
    ("macla",     "f1500-macla-p28-60692947"),
    ("no_memory", "f1500-no_memory-60561799"),
])
BLOCK = 250


def runs_1500():
    print("\n\n######## 1,500 training games, one run per arm ########")
    out = {}
    data = {}
    for arm, run in ARMS_1500.items():
        base = os.path.join(RESULTS, run)
        with open(os.path.join(base, "report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        train = s6.read_jsonl(os.path.join(base, "train", "diagnostics.jsonl"))
        ev = s6.read_jsonl(os.path.join(base, "eval", "diagnostics.jsonl"))
        seen = s6.read_jsonl(os.path.join(base, "eval_seen", "diagnostics.jsonl"))
        if len(train) != 1500 or len({s6.game_key(r["game_file"]) for r in train}) != 1500:
            raise SystemExit("%s: training is not 1,500 distinct games" % run)
        if len(ev) != 134 or len(seen) != 140:
            raise SystemExit("%s: %d test and %d valid_seen episodes" % (run, len(ev), len(seen)))
        data[arm] = {"report": rep, "train": train, "eval": ev, "seen": seen}

    order = [s6.game_key(r["game_file"]) for r in sorted(data["no_memory"]["train"],
                                                          key=lambda r: r["ep_idx"])]
    for arm, d in data.items():
        if [s6.game_key(r["game_file"]) for r in sorted(d["train"], key=lambda r: r["ep_idx"])] != order:
            raise SystemExit("%s: training order differs from no_memory" % arm)
    print("checks: 4 runs, the same 1,500 training games in the same order, 134 + 140 eval tasks")

    print("\n%-10s %-12s %-12s %-12s %s" % ("", "train %", "test %", "valid_seen %", "train % per 250 games"))
    for arm, d in data.items():
        tr = [int(bool(r["success"])) for r in sorted(d["train"], key=lambda r: r["ep_idx"])]
        blocks = [100 * sum(tr[i:i + BLOCK]) / BLOCK for i in range(0, 1500, BLOCK)]
        te = sum(int(bool(r["success"])) for r in d["eval"])
        se = sum(int(bool(r["success"])) for r in d["seen"])
        out[arm] = {"run": ARMS_1500[arm], "train": sum(tr) / 1500, "test_solved": te,
                    "test": te / 134, "seen_solved": se, "seen": se / 140,
                    "train_blocks": blocks}
        print("%-10s %-12.1f %-12s %-12s %s" % (arm, 100 * sum(tr) / 1500,
              "%.1f (%d)" % (100 * te / 134, te), "%.1f (%d)" % (100 * se / 140, se),
              " ".join("%.1f" % b for b in blocks)))

    tasks = sorted(s6.game_key(r["game_file"]) for r in data["no_memory"]["eval"])
    succ = {a: {s6.game_key(r["game_file"]): int(bool(r["success"])) for r in d["eval"]}
            for a, d in data.items()}
    print("\ntest split, exact McNemar against no_memory and between memory arms:")
    out["mcnemar"] = {}
    for a, b in s6.PAIRS:
        x = [succ[a][t] for t in tasks]
        y = [succ[b][t] for t in tasks]
        r = s6.mcnemar_exact(x, y)
        out["mcnemar"]["%s_vs_%s" % (a, b)] = {"a_only": r[0], "b_only": r[1], "p": r[2]}
        print("  %-9s vs %-9s %d/%d p=%.2g" % (a, b, *r))

    keys = [("memory LLM calls / train ep", lambda r: r["train"]["token_usage"]["calls_per_episode"]["memory"]),
            ("train time, h", lambda r: r["train"]["elapsed_s"] / 3600),
            ("memory tokens / eval step", lambda r: r["eval"]["memory_injection"]["injected_tokens_per_step"])]
    print()
    for label, fn in keys:
        cells = []
        for arm, d in data.items():
            try:
                cells.append("%s %.1f" % (arm, fn(d["report"])))
            except (KeyError, TypeError):
                cells.append("%s n/a" % arm)
        print("%-30s %s" % (label, "  ".join(cells)))
    stats = {a: d["report"]["train"].get("memory_stats", {}) for a, d in data.items()}
    print("store:", json.dumps({a: {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
                               for a, s in stats.items()})[:1500])
    return out


def main():
    s6.RESULTS = RESULTS
    s6.ARMS = ARMS_500
    print("######## 500 training games, three replicates per arm ########")
    s6.main()
    path = os.path.join(RESULTS, "paper_numbers.json")
    with open(path, encoding="utf-8") as f:
        numbers = json.load(f)
    numbers["runs_1500"] = runs_1500()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(numbers, f, indent=1)
        f.write("\n")
    print("\nwrote %s" % os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()

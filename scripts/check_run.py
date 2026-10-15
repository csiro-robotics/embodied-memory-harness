"""Validate a completed arm run: not just "it didn't crash" but "it behaved correctly".

A smoke job exiting 0 proves very little. This checks the things that would silently
invalidate a comparison if they were wrong, and that a human would not notice by reading
a log: that the arm actually built its memory, that evaluation was genuinely frozen, that
only the arms with a model in the memory path spent inference there, and that every arm
was scored over the same number of episodes.

Usage:
    python scripts/check_run.py <run_dir> [<run_dir> ...]
    python scripts/check_run.py $RUNS/smoke-*-585947*/

Exit code is non-zero if any run fails a check, so it can gate the real submissions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Where each arm persists its trained memory, relative to a phase directory.
ARM_ARTIFACTS = {
    "kg_hybrid": ["kg_hybrid/graph.json", "kg_hybrid/episodic.jsonl"],
    "wiki_v3": ["wiki_v3/journal.jsonl", "wiki_v3/priors.json", "wiki_v3/evidence.json"],
    "macla": ["macla/state.pkl"],
    "no_memory": [],
}

# Only arms with a language model in the memory path may spend inference there.
# kg_hybrid runs its AriGraph triplet extractor on the memory LLM during training;
# at eval its memory path is mechanical (frozen graph + embedding lookups), so the
# eval-phase memory-calls check still expects ~0 for it (see check()).
# kg_hybrid joined with DEC-10: its knowledge-guided planner makes one call per episode.
ARMS_WITH_MEMORY_LLM = {"wiki_v3", "kg_hybrid"}
# MACLA calls its model during training only, through the harness client since H11.
ARMS_WITH_TRAIN_MEMORY_LLM = {"wiki_v3", "kg_hybrid", "macla"}

# MACLA calls its model once per training episode (segmentation, DEC-11) unless the
# arm has no model client; kept as a warning so a smoke run is judged on wiring only.
MAY_HAVE_NO_TRAIN_CALLS_AT_SMOKE_SIZE = {"macla"}

# MACLA keeps from a failed episode only substeps the game confirmed (DEC-18 item 1), so
# a smoke run whose few training episodes all fail may leave nothing to show at eval.
LEARNS_FROM_SUCCESSES_ONLY = {"macla"}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def check(run_dir: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True

    def bad(m: str) -> None:
        nonlocal ok
        ok = False
        notes.append(f"FAIL  {m}")

    def good(m: str) -> None:
        notes.append(f"ok    {m}")

    train, ev = run_dir / "train", run_dir / "eval"
    ts, es = _load(train / "summary.json"), _load(ev / "summary.json")
    if not ts:
        return False, [f"FAIL  no train/summary.json in {run_dir}"]
    if not es:
        return False, [f"FAIL  no eval/summary.json in {run_dir}"]

    arm = ts.get("memory_type", "?")
    good(f"arm = {arm}")

    # 1. Episode counts. train and eval SHOULD differ in a real run (500 vs 134); the
    # invariant that matters is that eval n is identical ACROSS arms, checked in main().
    good(f"episodes: train n={ts['n']}, eval n={es['n']}")
    for phase, s in (("train", ts), ("eval", es)):
        n_err = s.get("errored_episodes", 0)
        if n_err:
            bad(f"{phase}: {n_err} errored episode(s) — these count as failures but "
                f"indicate a runtime problem")
    if not any(s.get("errored_episodes", 0) for s in (ts, es)):
        good("no errored episodes")

    # 2. The arm actually built its memory during training.
    expected = ARM_ARTIFACTS.get(arm, [])
    for rel in expected:
        if (train / rel).exists():
            good(f"built {rel}")
        else:
            bad(f"missing trained artifact train/{rel} — the arm learned nothing")
    if arm == "no_memory" and not expected:
        good("no artifact expected (baseline)")

    # 3. Evaluation was frozen: no memory artifact may appear under eval/.
    leaked = [rel for rel in expected if (ev / rel).exists()]
    if leaked:
        bad(f"eval wrote memory ({', '.join(leaked)}) — freeze() is not holding")
    elif expected:
        good("eval wrote no memory (freeze holding)")

    # 4. Inference accounting: only arms with a memory LLM may spend calls there.
    cpe = (es.get("token_usage") or {}).get("calls_per_episode") or {}
    mem_calls = cpe.get("memory", 0)
    act_calls = cpe.get("action", 0)
    if act_calls <= 0:
        bad("eval made no action LLM calls per episode — agent did not act")
    else:
        good(f"eval action calls/ep = {act_calls}")
    if arm in ARMS_WITH_MEMORY_LLM:
        if mem_calls <= 0:
            bad(f"{arm} reports {mem_calls} memory calls/ep but should use a memory LLM")
        else:
            good(f"eval memory calls/ep = {mem_calls} (declared for this arm)")
    elif mem_calls:
        bad(f"{arm} spent {mem_calls} memory LLM calls/ep but has no memory LLM — "
            f"unfair inference asymmetry")
    else:
        good("no memory LLM inference (as expected)")

    # 4b. Train-phase inference accounting: kg_hybrid (extractor) and wiki learn with a
    # model in the memory path; every other arm's training must be mechanical.
    t_mem_calls = ((ts.get("token_usage") or {}).get("calls_per_episode") or {}).get("memory", 0)
    if arm in ARMS_WITH_TRAIN_MEMORY_LLM:
        smoke = any(d.get("key") == "SMOKE_RUN" for d in (ts.get("deviations") or []))
        if t_mem_calls <= 0 and smoke and arm in MAY_HAVE_NO_TRAIN_CALLS_AT_SMOKE_SIZE:
            notes.append(f"warn  {arm} made 0 train memory calls (expected at smoke size)")
        elif t_mem_calls <= 0:
            bad(f"{arm} reports {t_mem_calls} train memory calls/ep but should learn with a memory LLM")
        else:
            good(f"train memory calls/ep = {t_mem_calls} (declared for this arm)")
    elif t_mem_calls:
        bad(f"{arm} spent {t_mem_calls} memory LLM calls/ep during training but has no memory LLM")

    # 4b2. Store updates that raised (H10): the harness records the error and carries on,
    # so the episode keeps its result, but the store may be missing that episode.
    for phase, s in (("train", ts), ("eval", es)):
        n_upd_err = (s.get("memory_update_totals") or {}).get("update_errors", 0)
        if n_upd_err:
            bad(f"{phase}: memory.update raised in {n_upd_err} episode(s); see update_error "
                f"in episodes.jsonl")

    # 4b3. Retrieval calls that raised (H13). The arm served less than it holds, and the
    # episode still counts, so the run is not a fair measure of the arm.
    for phase, s in (("train", ts), ("eval", es)):
        ms = s.get("memory_stats") or {}
        n_ret_err = ms.get("retrieve_errors", 0) + ms.get("read_errors", 0)
        if n_ret_err:
            bad(f"{phase}: {n_ret_err} memory retrieval call(s) raised; see pipeline.log")

    # 4b4. The trained store is byte-identical before and after evaluation (H17).
    sha = es.get("trained_store_sha256")
    if sha is None:
        notes.append("warn  no trained_store_sha256 in eval summary (run predates H17)")
    elif sha.get("before_eval") is None and arm != "no_memory":
        bad("no trained store was found to hash before evaluation")
    elif not sha.get("unchanged"):
        bad("the trained store changed during evaluation — freeze() did not hold")
    elif arm != "no_memory":
        good("trained store unchanged by evaluation (SHA-256)")

    # 4c. Memory injection: did the arm actually deliver anything to the agent, and how
    # much of it survived the budget? A memory arm that retrieves nothing scores like
    # no_memory and looks like a legitimate negative result, so silence here is failed
    # plumbing until proven otherwise. Heavy truncation is reported but NOT failed — it is
    # a real property of the arm under the protocol budget, not a defect.
    mi = es.get("memory_injection") or {}
    if not mi:
        notes.append("warn  no memory_injection block — run predates injection telemetry")
    else:
        rate = mi.get("retrieval_rate", 0.0)
        if arm == "no_memory":
            if rate:
                bad(f"no_memory retrieved on {rate:.0%} of steps — baseline is contaminated")
            else:
                good("no_memory injected nothing (baseline clean)")
        elif (rate <= 0.0 and arm in LEARNS_FROM_SUCCESSES_ONLY and not ts.get("sr")
              and any(d.get("key") == "SMOKE_RUN" for d in (ts.get("deviations") or []))):
            notes.append(f"warn  {arm} showed no memory: every smoke training episode failed, "
                         f"so its store is empty (expected at smoke size)")
        elif rate <= 0.0:
            bad(f"{arm} retrieved on 0% of eval steps — memory never reached the agent")
        else:
            good(f"eval retrieval_rate = {rate:.0%} "
                 f"({mi.get('injected_tokens_per_step', 0):.0f} tok/step injected)")
        disc = mi.get("discarded_fraction", 0.0)
        if disc >= 0.25:
            notes.append(
                f"warn  {arm} discards {disc:.0%} of retrieved content to the "
                f"{mi.get('peak_offered_tokens', 0):.0f}-token peak vs the protocol budget "
                f"— report this alongside its SR"
            )

    # 4d. The server's context window (H1). arm.slurm writes ollama_context_check.txt after
    # the run; a prompt the server cut to fit means the model saw less than the arm sent.
    ctx_file = run_dir / "ollama_context_check.txt"
    if ctx_file.exists():
        text = ctx_file.read_text(encoding="utf-8")
        cut = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                    if l.startswith("truncated_prompts:")), "")
        if cut.isdigit() and int(cut) > 0:
            bad(f"the Ollama server truncated {cut} prompt(s) to fit its context window")
        elif cut.isdigit():
            good("no prompt truncated by the Ollama server")
        else:
            notes.append("warn  ollama_context_check.txt has no truncated_prompts count")
    else:
        notes.append("warn  no ollama_context_check.txt (run predates H1 or ran outside SLURM)")

    # 5. Declared deviations are recorded (SDD F9).
    devs = es.get("deviations")
    if devs is None:
        bad("no `deviations` field in summary — SDD F9 not emitted")
    else:
        good(f"deviations recorded = {len(devs)}")

    # 6. Sanity, not a result: SR must be a real fraction.
    for phase, s in (("train", ts), ("eval", es)):
        sr = s.get("sr")
        if sr is None or not (0.0 <= sr <= 1.0):
            bad(f"{phase}: implausible sr={sr}")
    good(f"sr train={ts.get('sr'):.0%} eval={es.get('sr'):.0%} (not meaningful at smoke size)")

    return ok, notes


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        sys.exit(2)
    all_ok = True
    eval_n: dict[str, int] = {}
    eval_split: dict[str, str] = {}
    train_n: dict[str, int] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        ok, notes = check(d)
        all_ok &= ok
        print(f"\n{'=' * 64}\n{d.name}   {'PASS' if ok else 'FAIL'}\n{'=' * 64}")
        for n in notes:
            print("  " + n)
        s = _load(d / "eval" / "summary.json")
        if s:
            key = d.name  # run dir, not memory_type: replicates share a memory_type
            eval_n[key] = s["n"]
            eval_split[key] = s.get("eval_split", "eval_out_of_distribution")
        t = _load(d / "train" / "summary.json")
        if t:
            train_n[d.name] = t["n"]

    # Cross-arm invariants. Every arm in a round must be scored over the identical eval
    # set, on the identical split, having trained on the identical number of episodes.
    # If any of those differ the success RATES are not commensurable and the round is void.
    if len(eval_n) > 1:
        print(f"\n{'=' * 64}\ncross-arm\n{'=' * 64}")
        if len(set(eval_n.values())) == 1:
            print(f"  ok    all {len(eval_n)} arms evaluated on n={next(iter(eval_n.values()))}")
        else:
            all_ok = False
            print(f"  FAIL  eval episode counts differ across arms: {eval_n}")
        if len(set(eval_split.values())) == 1:
            split = next(iter(eval_split.values()))
            label = "DEV (valid_seen)" if split == "eval_in_distribution" else "TEST (valid_unseen)"
            print(f"  ok    all arms on the same split: {split}  -> {label}")
        else:
            all_ok = False
            print(f"  FAIL  arms evaluated on DIFFERENT splits — scores not comparable: "
                  f"{eval_split}")
        if len(set(train_n.values())) == 1:
            print(f"  ok    all arms trained on n={next(iter(train_n.values()))}")
        else:
            all_ok = False
            print(f"  FAIL  training episode counts differ across arms: {train_n}")

    print(f"\n{'ALL RUNS PASS' if all_ok else 'SOME RUNS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

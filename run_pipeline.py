"""Two-phase pipeline: build memory on ALFWorld training games, then evaluate it frozen.

Phase 1 — Training:
  Runs episodes from the train split (the fixed 500-game list in train500_path).
  After each episode the arm's update() adds to its store.

Phase 2 — Evaluation:
  Runs episodes from the eval split (valid_unseen, 134 games, by default). The arm
  loads the store that training built and is frozen: retrieve() works, update() and
  finalize() do nothing.

Output layout:
  <out>/
    train/
      <memory_type>/       <- the trained store (wiki pages, graph, procedures, ...)
      episodes.jsonl       <- one JSON line per training episode
      diagnostics.jsonl    <- a short per-episode record for quick analysis
      memory_growth.json   <- store-size snapshots every 50 episodes
      summary.json
    eval/
      episodes.jsonl
      diagnostics.jsonl
      summary.json
    report.json            <- combined train + eval results
    pipeline_config.yaml   <- the merged config actually used

Usage:
  python run_pipeline.py --config configs/arm_wiki_v3.yaml --out results/runs/my_run

Flags:
  --skip-train   Skip training and evaluate the store already in <out>/train/<memory_type>/
  --skip-eval    Skip the eval phase (train only)
  --smoke N      Cap both phases at N episodes (a shakedown, never a result)
  -v / --verbose Enable DEBUG logging
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alfworld_env import ALFWorldEnv
from memory import create_memory
from protocol_config import load_config
from runner import build_llm, compute_summary, print_family_table, print_summary, run_phase, sep

_W = 72


def store_digest(path: Path) -> str | None:
    """SHA-256 over every file under `path` (relative names and contents), or None.

    Taken before and after evaluation (H17): equal digests show the trained store was not
    written during evaluation, which freeze() is meant to guarantee.
    """
    import hashlib

    if not path.exists():
        return None
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def check_protocol_keys(config: dict) -> list[str]:
    """Problems with frozen keys that nothing else reads (H15); empty if all hold."""
    import importlib.metadata as md

    problems = []
    want = str(config.get("alfworld_version", ""))
    try:
        have = md.version("alfworld")
    except md.PackageNotFoundError:
        have = None
    if want and have is not None and have != want:
        problems.append(f"alfworld_version is {want} in the config but {have} is installed")
    fewshot = str(config.get("fewshot", "none")).lower()
    if fewshot != "none":
        problems.append(f"fewshot is {fewshot!r}, but no few-shot path exists; only 'none' is valid")
    return problems


def _resolve_llm_cfgs(phase_cfg: dict) -> tuple[dict, dict]:
    """Return (action_llm_cfg, memory_llm_cfg) for a phase config.

    action_llm: (else llm:) is used for action selection.
    memory_llm: is used for the arm's own model calls (retrieval, store updates).
    An arm without a memory_llm block falls back to the action config, so its reported
    memory model is the model it would actually call, not a library default.
    """
    base = phase_cfg.get("llm", {})
    action_cfg = phase_cfg.get("action_llm", base)
    memory_cfg = phase_cfg.get("memory_llm", action_cfg)
    return action_cfg, memory_cfg


def _print_comparison(train_sum: dict, eval_sum: dict) -> None:
    """Print a side-by-side train vs eval comparison."""
    print(f"\n{'=' * _W}")
    print("  PIPELINE COMPARISON  (train → eval)")
    print("=" * _W)
    print(f"  Training SR:  {train_sum['n_success']}/{train_sum['n']} ({train_sum['sr']:.1%})")
    print(f"  Eval SR:      {eval_sum['n_success']}/{eval_sum['n']}  ({eval_sum['sr']:.1%})")
    delta = eval_sum["sr"] - train_sum["sr"]
    print(f"  Delta:        {delta:+.1%}")

    # Per-family side-by-side
    all_fams = sorted(
        set(train_sum.get("per_family", {}).keys()) | set(eval_sum.get("per_family", {}).keys())
    )
    if all_fams:
        print(f"\n  {'Family':<30}  {'Train SR':>8}  {'Eval SR':>8}  {'Delta':>7}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*7}")
        for fam in all_fams:
            t = train_sum.get("per_family", {}).get(fam)
            e = eval_sum.get("per_family", {}).get(fam)
            t_str = f"{t['rate']:.0%} ({t['n']})" if t else "    -"
            e_str = f"{e['rate']:.0%} ({e['n']})" if e else "    -"
            if t and e:
                d = e["rate"] - t["rate"]
                d_str = f"{d:+.0%}"
            else:
                d_str = "    -"
            print(f"  {fam:<30}  {t_str:>8}  {e_str:>8}  {d_str:>7}")

    # Memory injection, train vs eval. Reported before the store description because an
    # arm can hold a large store and still inject almost none of it; these two blocks
    # answer "what did it accumulate" and "what did it actually deliver" respectively.
    for _phase, _s in (("train", train_sum), ("eval", eval_sum)):
        if mi := _s.get("memory_injection"):
            print(
                f"\n  MEMORY INJECTION at {_phase}  "
                f"retrieval_rate={mi['retrieval_rate']:.0%}  "
                f"offered={mi['offered_tokens_per_step']:,.0f} tok/step  "
                f"injected={mi['injected_tokens_per_step']:,.0f} tok/step  "
                f"discarded={mi['discarded_fraction']:.0%}"
            )

    # Memory-store description, from EVAL — what the agent actually had access to.
    # Arm-neutral: every arm names its own store contents (pages, procedures, graph edges),
    # and no two share a vocabulary, so this prints the keys it was handed.
    if ws := eval_sum.get("memory_stats"):
        print(f"\n  MEMORY STORE at eval ({eval_sum.get('memory_type', '?')})")
        for k, v in ws.items():
            print(f"    {k + ':':<28} {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train + eval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline YAML config")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training — evaluate the store already in <out>/train/<memory_type>/",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval phase")
    parser.add_argument(
        "--smoke",
        type=int,
        metavar="N",
        help="Shakedown run: cap both phases at N episodes. num_episodes is a frozen "
             "protocol key so a smoke CONFIG would be (correctly) rejected by the guard; "
             "this flag overrides it explicitly and stamps the output as non-protocol so "
             "a shakedown can never be mistaken for a result.",
    )
    # ---- Iteration-round overrides ---------------------------------------------
    # num_episodes and train_eval are frozen protocol keys, so a config that changed
    # them would be (correctly) rejected by the guard. These flags are the sanctioned
    # override, exactly like --smoke: explicit at the call site, and recorded as a
    # deviation so a dev round can never be mistaken for a protocol test result.
    parser.add_argument(
        "--train-episodes",
        type=int,
        metavar="N",
        help="Override the training episode count (protocol default 500). Every arm in "
             "a round must use the same value.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("eval_out_of_distribution", "eval_in_distribution"),
        help="Which split to evaluate on. eval_out_of_distribution (valid_unseen) is the "
             "held-out TEST set; eval_in_distribution (valid_seen) is the DEV set used "
             "for iteration. Defaults to the config value (test).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        metavar="N",
        help="Override the eval episode count (protocol default 134).",
    )
    parser.add_argument(
        "--eval-seen",
        type=int,
        metavar="N",
        help="Also evaluate the frozen store on N valid_seen tasks (140 is the whole split), "
             "after the test split, into <out>/eval_seen/. Off by default.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        metavar="N",
        default=1,
        help="Replicate number (DEC-15). The seed becomes the protocol seed + N - 1, which "
             "changes only the ORDER of the same 500 training games and 134 evaluation "
             "games. Replicate 1 keeps the protocol seed.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Suppress noisy HTTP wire-level logging from third-party libraries.
    # These flood --verbose output with per-request headers and body dumps.
    for _lib in ("openai", "httpcore", "httpx", "urllib3"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    # load_config applies _base.yaml inheritance + the frozen-key guard when the config
    # declares `extends:` (unified protocol). Legacy self-contained configs load as-is.
    config = load_config(args.config)

    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

    # The pinned ALFWorld release must be resolved before any number is trusted; game
    # identifiers are release-sensitive. Fail loudly rather than silently running against
    # whatever happens to be installed.
    _alf_ver = config.get("alfworld_version")
    if _alf_ver == "CONFIRM_ON_CLUSTER":
        print(
            "ERROR: alfworld_version is still the CONFIRM_ON_CLUSTER placeholder.\n"
            "  Run `pip show alfworld`, set the exact version in configs/_base.yaml,\n"
            "  and pin it in env.yml so every arm resolves the same game files."
        )
        sys.exit(1)

    _problems = check_protocol_keys(config)
    if _problems:
        for _p in _problems:
            print(f"ERROR: {_p}")
        sys.exit(1)

    # H1: the context window is set on the Ollama server, not per request. When the job
    # script started the server (it exports OLLAMA_CONTEXT_LENGTH), the config must record
    # the same value, or the reported setting would not be the one used.
    _server_ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if _server_ctx:
        for _phase_name, _phase_cfg in (("train", train_cfg), ("eval", eval_cfg)):
            for _llm_cfg in _resolve_llm_cfgs(_phase_cfg):
                _cfg_ctx = _llm_cfg.get("num_ctx")
                if _cfg_ctx is not None and int(_cfg_ctx) != int(_server_ctx):
                    print(f"ERROR: {_phase_name} num_ctx={_cfg_ctx} but the Ollama server "
                          f"runs OLLAMA_CONTEXT_LENGTH={_server_ctx}. Set one value.")
                    sys.exit(1)

    # Unified-protocol top-level knobs (absent in legacy configs -> safe defaults).
    seed = config.get("seed", 42)
    # DEC-15 amendment: with a separate memory server (H19) a run is deterministic at
    # temperature 0, so replicates with one seed are copies. Each replicate therefore
    # trains in its own game order; everything else, including the game sets, is fixed.
    if args.replicate < 1:
        print("ERROR: --replicate must be 1 or more")
        sys.exit(1)
    seed = seed + args.replicate - 1
    config["seed"] = seed
    config["replicate"] = args.replicate
    declared_deviations = config.get("deviations", [])

    # Optional second evaluation on valid_seen: the eval phase's settings with the split
    # and episode count changed. It adds a result; the test evaluation is unchanged.
    if args.eval_seen:
        import copy
        config["eval_seen"] = copy.deepcopy(config.get("eval", {}))
        _es_env = config["eval_seen"].setdefault("env", {})
        _es_env["train_eval"] = "eval_in_distribution"
        _es_env["num_episodes"] = args.eval_seen
        declared_deviations = list(declared_deviations) + [{
            "key": "eval_seen",
            "reason": f"Additional evaluation of the frozen store on {args.eval_seen} "
                      f"valid_seen tasks, written to eval_seen/. The test evaluation is "
                      f"unchanged.",
        }]

    if args.smoke:
        for _phase in ("train", "eval", "eval_seen"):
            if _phase in config and "env" in (config[_phase] or {}):
                config[_phase]["env"]["num_episodes"] = args.smoke
        declared_deviations = list(declared_deviations) + [{
            "key": "SMOKE_RUN",
            "reason": f"Shakedown only: both phases capped at {args.smoke} episodes. "
                      f"NOT a protocol run; results are not comparable across arms.",
        }]
        print("=" * _W)
        print(f"  SMOKE RUN — {args.smoke} episodes per phase. NOT a valid result.")
        print("=" * _W)

    # Iteration-round overrides. Each one is recorded so the summary states plainly what
    # protocol value was departed from and why.
    if args.train_episodes:
        config.setdefault("train", {}).setdefault("env", {})["num_episodes"] = args.train_episodes
        declared_deviations = list(declared_deviations) + [{
            "key": "train.env.num_episodes",
            "reason": f"Iteration round: training capped at {args.train_episodes} episodes "
                      f"(protocol 500). All arms in the round share this value, so the "
                      f"comparison is internally valid; absolute SR is not comparable to "
                      f"a 500-episode protocol run.",
        }]
    if args.eval_episodes:
        config.setdefault("eval", {}).setdefault("env", {})["num_episodes"] = args.eval_episodes
        declared_deviations = list(declared_deviations) + [{
            "key": "eval.env.num_episodes",
            "reason": f"Iteration round: eval capped at {args.eval_episodes} episodes.",
        }]
    if args.eval_split:
        config.setdefault("eval", {}).setdefault("env", {})["train_eval"] = args.eval_split
        if args.eval_split != "eval_out_of_distribution":
            declared_deviations = list(declared_deviations) + [{
                "key": "eval.env.train_eval",
                "reason": f"DEV ROUND: evaluated on {args.eval_split} (valid_seen), NOT the "
                          f"held-out test split. Used for iteration so that repeated "
                          f"selection does not overfit valid_unseen. This number is a dev "
                          f"score and must never be reported as the headline result.",
            }]
            print("=" * _W)
            print(f"  DEV ROUND — eval split is {args.eval_split} (valid_seen).")
            print("  This is a DEV score for iteration, NOT a test result.")
            print("=" * _W)

    mem_block_max_tokens = config.get("memory_block_max_tokens")
    _t500 = config.get("train500_path")
    if _t500 and not Path(_t500).is_absolute():
        _t500 = str((Path(__file__).resolve().parent / _t500))
    train500_path = _t500

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pipeline_config.yaml").write_text(yaml.dump(config))
    # Keep the Python log with the results (H12), not only in the job's stdout.
    _fh = logging.FileHandler(args.out / "pipeline.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    train_dir = args.out / "train"
    eval_dir = args.out / "eval"
    # Every arm persists its trained store under train/<memory_type>/. The factory loads it
    # from there at eval, so the pipeline only needs to check that it exists.
    mem_type = (
        (train_cfg.get("memory") or {}).get("type")
        or (eval_cfg.get("memory") or {}).get("type")
    )
    if not mem_type:
        print("ERROR: the config names no memory type (train.memory.type).")
        sys.exit(1)
    store_dir = train_dir / mem_type

    report: dict = {}

    # =========================================================================
    # Phase 1: Training
    # =========================================================================
    if not args.skip_train:
        print(f"\n{'=' * _W}")
        print("  PHASE 1 — TRAINING")
        print("=" * _W)

        train_env_cfg = train_cfg.get("env", {})
        train_mem_cfg = train_cfg.get("memory", {})
        train_agent_cfg = train_cfg.get("agent", {})

        train_action_cfg, train_memory_cfg = _resolve_llm_cfgs(train_cfg)
        train_action_llm = build_llm(train_action_cfg)
        train_memory_llm = build_llm(train_memory_cfg, role="memory")

        num_train = train_env_cfg.get("num_episodes", 500)
        max_steps = train_env_cfg.get("max_steps", 50)

        print(f"  Episodes:   {num_train}  |  Split: train  |  Max steps: {max_steps}")
        print(f"  Output:     {train_dir}")
        if mem_type != "no_memory":
            print(f"  Store:      {store_dir}")
        print(f"  Action LLM: {train_action_llm.model}")
        if train_action_llm.model != train_memory_llm.model:
            print(f"  Memory LLM: {train_memory_llm.model}")

        train_env = ALFWorldEnv(
            train_eval=train_env_cfg.get("train_eval", "train"),
            max_steps=max_steps,
            seed=seed,
            num_episodes=num_train,
            train500_path=train500_path,
            domain_randomization=train_env_cfg.get("domain_randomization", False),
        )
        train_memory = create_memory(
            mem_type,
            run_dir=train_dir,
            llm=train_memory_llm,
            config=train_mem_cfg,
        )

        train_results, train_elapsed = run_phase(
            env=train_env,
            memory=train_memory,
            action_llm=train_action_llm,
            num_episodes=num_train,
            max_steps=max_steps,
            history_window=train_agent_cfg.get("history_window", 20),
            out_dir=train_dir,
            phase="TRAIN",
            store_checkpoints=True,
            verbose=args.verbose,
            memory_block_max_tokens=mem_block_max_tokens,
        )

        train_summary = compute_summary(
            results=train_results,
            memory=train_memory,
            action_llm=train_action_llm,
            memory_llm=train_memory_llm,
            elapsed=train_elapsed,
        )
        train_summary["deviations"] = declared_deviations  # SDD F9
        train_summary["replicate"] = args.replicate
        train_summary["seed"] = seed
        train_summary["errored_episodes"] = sum(1 for r in train_results if r.get("errored"))
        (train_dir / "summary.json").write_text(json.dumps(train_summary, indent=2))
        print_summary(train_summary, "TRAINING COMPLETE")
        report["train"] = train_summary

    else:
        print(f"\n[pipeline] Skipping training — using the store at {store_dir}")
        summary_path = train_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open() as f:
                report["train"] = json.load(f)
            print(f"  Loaded training summary: SR={report['train']['sr']:.1%}  ({report['train']['n']} eps)")

    # =========================================================================
    # Phase 2: Evaluation (memory frozen for every arm)
    # =========================================================================
    # With --eval-seen, a second frozen evaluation on valid_seen runs after the held-out
    # test, from the same trained store, into its own directory, so every tool that reads
    # eval/ keeps reading the test split. MACLA reports both splits (140 + 134 = 274).
    eval_phases = [("eval", eval_cfg, eval_dir, "PHASE 2 — EVALUATION (memory frozen)")]
    if config.get("eval_seen"):
        eval_phases.append(("eval_seen", config["eval_seen"], args.out / "eval_seen",
                            "PHASE 3 — EVALUATION ON valid_seen (memory frozen)"))
    for eval_key, eval_cfg, eval_dir, eval_title in ([] if args.skip_eval else eval_phases):
        print(f"\n{'=' * _W}")
        print(f"  {eval_title}")
        print("=" * _W)

        eval_env_cfg = eval_cfg.get("env", {})
        eval_mem_cfg = dict(eval_cfg.get("memory", {}))
        eval_mem_type = eval_mem_cfg.get("type", mem_type)
        if eval_mem_type != mem_type:
            print(f"ERROR: train memory type {mem_type!r} != eval memory type {eval_mem_type!r}.")
            sys.exit(1)

        # Every arm except the control must find the store that training built. Without
        # this check an arm with a missing store would load an empty one and score as if
        # it had never trained.
        if eval_mem_type != "no_memory" and not store_dir.exists():
            print(f"ERROR: trained store not found at {store_dir}.")
            print("  Run training first (remove --skip-train if set).")
            sys.exit(1)
        eval_agent_cfg = eval_cfg.get("agent", {})

        eval_action_cfg, eval_memory_cfg = _resolve_llm_cfgs(eval_cfg)
        eval_action_llm = build_llm(eval_action_cfg)
        eval_memory_llm = build_llm(eval_memory_cfg, role="memory")

        num_eval = eval_env_cfg.get("num_episodes", 134)
        max_steps = eval_env_cfg.get("max_steps", 50)

        eval_split = eval_env_cfg.get("train_eval", "eval_out_of_distribution")
        _split_label = ("valid_unseen (TEST)" if eval_split == "eval_out_of_distribution"
                        else "valid_seen" + (" (DEV)" if eval_key == "eval" else ""))
        print(f"  Episodes:   {num_eval}  |  Split: {_split_label}  |  Max steps: {max_steps}")
        if eval_mem_type != "no_memory":
            print(f"  Store:      {store_dir}  (frozen, no updates)")
        print(f"  Output:     {eval_dir}")
        print(f"  Action LLM: {eval_action_llm.model}")
        if eval_action_llm.model != eval_memory_llm.model:
            print(f"  Memory LLM: {eval_memory_llm.model}")

        eval_env = ALFWorldEnv(
            train_eval=eval_env_cfg.get("train_eval", "eval_out_of_distribution"),
            max_steps=max_steps,
            seed=seed,
            num_episodes=num_eval,
            train500_path=train500_path,  # ignored for non-train splits
            domain_randomization=eval_env_cfg.get("domain_randomization", False),
        )

        # Force read_only regardless of what the config says (legacy per-arm flag).
        eval_mem_cfg["read_only"] = True
        store_before = store_digest(store_dir)

        eval_memory = create_memory(
            eval_mem_type,
            run_dir=eval_dir,
            llm=eval_memory_llm,
            config=eval_mem_cfg,
            trained_dir=train_dir,  # every arm loads its store from train/<memory_type>/
        )

        # Uniform eval-freeze for EVERY arm (SDD §3.1): freeze() flips
        # MemoryModule._read_only so update(), finalize() and any arm-specific write path
        # are suppressed.
        eval_memory.freeze()

        eval_results, eval_elapsed = run_phase(
            env=eval_env,
            memory=eval_memory,
            action_llm=eval_action_llm,
            num_episodes=num_eval,
            max_steps=max_steps,
            history_window=eval_agent_cfg.get("history_window", 20),
            out_dir=eval_dir,
            phase="EVAL",
            store_checkpoints=False,
            verbose=args.verbose,
            memory_block_max_tokens=mem_block_max_tokens,
        )

        eval_summary = compute_summary(
            results=eval_results,
            memory=eval_memory,
            action_llm=eval_action_llm,
            memory_llm=eval_memory_llm,
            elapsed=eval_elapsed,
        )
        eval_summary["deviations"] = declared_deviations  # SDD F9
        eval_summary["errored_episodes"] = sum(1 for r in eval_results if r.get("errored"))
        store_after = store_digest(store_dir)
        eval_summary["trained_store_sha256"] = {
            "before_eval": store_before,
            "after_eval": store_after,
            "unchanged": store_before == store_after,
        }
        # Which split produced this number. Analysis must never mix dev and test scores,
        # and reading it off the summary is safer than inferring it from the run name.
        eval_summary["eval_split"] = eval_split
        eval_summary["replicate"] = args.replicate
        eval_summary["seed"] = seed
        # valid_seen is a DEV round only when it replaces the test split (--eval-split); as
        # the second evaluation phase it is a reported split.
        eval_summary["is_dev"] = (eval_split != "eval_out_of_distribution"
                                  and eval_key == "eval")
        (eval_dir / "summary.json").write_text(json.dumps(eval_summary, indent=2))
        print_summary(eval_summary, "EVAL COMPLETE")
        report[eval_key] = eval_summary

    # =========================================================================
    # Combined report
    # =========================================================================
    if "train" in report and "eval" in report:
        _print_comparison(report["train"], report["eval"])

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Full report: {args.out / 'report.json'}")
    print("=" * _W)
    logging.getLogger().removeHandler(_fh)
    _fh.close()


if __name__ == "__main__":
    main()

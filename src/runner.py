"""Shared phase-running logic used by run_pipeline.py.

Provides build_llm(), run_phase(), compute_summary(), print_summary(), and helpers.
"""
from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alfworld_env import ALFWorldEnv
    from llm import LLMClient
    from memory.base import MemoryModule

log = logging.getLogger(__name__)

_W = 72  # display width


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(cfg: dict, role: str = "action") -> LLMClient:
    """Instantiate an LLMClient from a config dict.

    OLLAMA_BASE_URL overrides the configured base_url when set. Arms run in parallel land
    on the same compute node, where a hardcoded port 11434 collides: the second job fails
    to bind and then silently talks to the FIRST job's ollama server. arm.slurm therefore
    allocates a free port per job and exports it here. This is a deployment detail, not a
    protocol change: same model, same settings, only the server address differs, so it is
    deliberately not a frozen key.
    """
    import os

    from llm import LLMClient
    # role="memory" goes to the job's second server when there is one (H19), so memory-side
    # calls never touch the cached state of the server that picks actions.
    base_url = (
        (os.environ.get("OLLAMA_MEMORY_BASE_URL") if role == "memory" else None)
        or os.environ.get("OLLAMA_BASE_URL")
        or cfg.get("base_url", "http://localhost:11434/v1")
    )
    return LLMClient(
        model=cfg.get("model", "llama3.1:8b"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 256),
        base_url=base_url,
        api_key=cfg.get("api_key", "ollama"),
        num_ctx=cfg.get("num_ctx", 32768),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _rolling_sr(results: list[dict], window: int = 10) -> float | None:
    if len(results) < 2:
        return None
    chunk = results[-min(window, len(results)):]
    return sum(1 for r in chunk if r["success"]) / len(chunk)


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def sep(label: str = "") -> str:
    if label:
        return f"-- {label} " + "-" * max(0, _W - len(label) - 4)
    return "-" * _W


# ---------------------------------------------------------------------------
# Family breakdown table
# ---------------------------------------------------------------------------

def print_family_table(results: list[dict], label: str) -> None:
    families: dict[str, list[dict]] = {}
    for r in results:
        families.setdefault(r["task_family"], []).append(r)

    print(f"\n{sep(label)}")
    print(f"  {'Family':<30}  {'OK':>3}  {'N':>3}  {'SR':>6}  Bar")
    print(f"  {'-'*30}  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*10}")
    for fam in sorted(families):
        eps = families[fam]
        n_ok = sum(1 for e in eps if e["success"])
        sr = n_ok / len(eps)
        print(f"  {fam:<30}  {n_ok:>3}  {len(eps):>3}  {sr:>5.0%}  {_bar(sr)}")
    print()


# ---------------------------------------------------------------------------
# Memory-store snapshot (logged at training checkpoints)
# ---------------------------------------------------------------------------

def _memory_snapshot(memory: Any) -> dict | None:
    """Return a size/telemetry snapshot of the arm's store, or None if it has none.

    Arm-neutral: every arm that can describe its own store gets a growth curve, and the
    caller must not assume any particular key is present.

    An arm describes its own store through a `stats()` method; an arm without one
    (no_memory) has nothing to report.
    """
    for attr in ("stats", "get_stats"):
        fn = getattr(memory, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                log.debug("memory.%s() failed", attr, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Core phase runner
# ---------------------------------------------------------------------------

def run_phase(
    *,
    env: ALFWorldEnv,
    memory: MemoryModule,
    action_llm: LLMClient,
    num_episodes: int,
    max_steps: int,
    history_window: int,
    out_dir: Path,
    phase: str = "phase",
    store_checkpoints: bool = True,
    verbose: bool = False,
    memory_block_max_tokens: int | None = None,
) -> tuple[list[dict], float]:
    """Run one training or eval phase.

    Returns (results, elapsed_seconds).
    action_llm: used for per-step action selection.
    memory:     holds the memory-side LLM internally, if the arm uses one.
    store_checkpoints: log the store's size every 50 episodes. The pipeline turns
                this on for training and off for eval, where the store is frozen.
    """
    from agent import run_episode

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t_start = time.time()
    memory_growth: list[dict] = []  # snapshots at checkpoints

    diag_path = out_dir / "diagnostics.jsonl"
    with (out_dir / "episodes.jsonl").open("w") as ep_file, diag_path.open("w") as diag_file:
        for ep_idx in range(num_episodes):
            ep_start = time.time()

            elapsed_so_far = ep_start - t_start
            if ep_idx > 0:
                avg_ep = elapsed_so_far / ep_idx
                eta_str = f"  avg={avg_ep:.0f}s/ep  ETA {_fmt_eta(avg_ep * (num_episodes - ep_idx))}"
            else:
                eta_str = ""

            header = f"[{phase}] Ep {ep_idx + 1:3d}/{num_episodes}{eta_str}"
            print(f"\n{sep(header)}")

            try:
                result = run_episode(
                    env=env,
                    memory=memory,
                    llm=action_llm,
                    max_steps=max_steps,
                    history_window=history_window,
                    verbose=verbose,
                    memory_block_max_tokens=memory_block_max_tokens,
                )
                ep_time = time.time() - ep_start

                d = asdict(result)
                d["episode_index"] = ep_idx
                d["ep_time_s"] = round(ep_time, 1)
                results.append(d)
                # default=str here and for the diagnostics record: an arm's metadata may
                # hold values JSON cannot encode (a set, say). Without it the write raises
                # after `results.append`, and the except branch below then adds a second,
                # errored record for the same episode.
                ep_file.write(json.dumps(d, default=str) + "\n")
                ep_file.flush()

                # Diagnostics record — lean version for quick grep/analysis.
                # The full memory text is already in episodes.jsonl (via EpisodeResult).
                diag: dict = {
                    "ep_idx": ep_idx,
                    "task_family": result.task_family,
                    "task_description": result.task_description,
                    "game_file": result.game_file,
                    "success": result.success,
                    "steps": result.steps_taken,
                    # Optional arm hook: a report of the store update this episode
                    # triggered (for example the wiki's consolidation passes).
                    "update_report": result.memory_update or {},
                }
                diag_file.write(json.dumps(diag, default=str) + "\n")
                diag_file.flush()

                n_ok = sum(1 for r in results if r["success"])
                sr = n_ok / len(results)
                rolling = _rolling_sr(results)

                status = "OK  " if result.success else "FAIL"
                rolling_str = f"  R10={rolling:.0%}" if rolling is not None else ""

                # Running average time (updated after each completed episode)
                avg_t = (time.time() - t_start) / len(results)
                avg_t_str = f"  avg={avg_t:.0f}s" if len(results) > 1 else ""

                log.info(
                    "[%s] Ep %d/%d %s | steps=%d | %.1fs (avg %.0fs) | SR=%.1f%% (%d/%d)%s",
                    phase, ep_idx + 1, num_episodes, status.strip(),
                    result.steps_taken, ep_time, avg_t, sr * 100, n_ok, len(results), rolling_str,
                )
                task_short = result.task_description
                if len(task_short) > 120:
                    task_short = task_short[:117] + "..."
                print(
                    f"   {status}  steps={result.steps_taken:2d}  {ep_time:5.1f}s{avg_t_str}"
                    f"  |  SR={sr:.1%} ({n_ok}/{len(results)}){rolling_str}"
                )
                print(f"   {result.task_family}  |  {task_short}")

            except Exception:
                log.exception("[%s] Episode %d error", phase, ep_idx)
                print(f"   ERROR after {time.time() - ep_start:.1f}s — see log")
                # Record the episode as a failure rather than dropping it. Silently
                # skipping made `n` vary between arms, so a crashy arm quietly evaluated
                # on fewer tasks and its success RATE was computed over a smaller
                # denominator. The unified protocol requires every arm to be scored over
                # the identical task set, so an errored episode counts as a failure.
                d = {
                    "task_id": getattr(env, "task_id", f"episode_{ep_idx}"),
                    "task_family": getattr(env, "task_family", "unknown"),
                    "task_description": getattr(env, "task_description", ""),
                    "trajectory": [],
                    "success": False,
                    "steps_taken": 0,
                    "episode_index": ep_idx,
                    "ep_time_s": round(time.time() - ep_start, 1),
                    "errored": True,
                }
                results.append(d)
                ep_file.write(json.dumps(d) + "\n")
                ep_file.flush()

            ep_num = ep_idx + 1

            # Family breakdown every 25 episodes
            if ep_num % 25 == 0 and results:
                print_family_table(results, f"[{phase}] ep {ep_num} breakdown")

            # Store-size snapshot every 50 episodes (training only)
            if store_checkpoints and ep_num % 50 == 0:
                snap = _memory_snapshot(memory)
                if snap:
                    snap["ep"] = ep_num
                    snap["sr_so_far"] = sum(1 for r in results if r["success"]) / len(results) if results else 0.0
                    memory_growth.append(snap)
                    # Every key here is arm-specific (wiki_v3 reports `n_pages`, MACLA
                    # reports `n_procedures`), so the snapshot is logged as whatever it
                    # contains rather than by fixed keys.
                    log.info(
                        "[memory] growth at ep %d: %s",
                        ep_num,
                        ", ".join(
                            f"{k}={v}" for k, v in snap.items()
                            if k not in ("ep", "sr_so_far") and isinstance(v, (int, float))
                        ),
                    )

    elapsed = time.time() - t_start

    # Final end-of-phase processing (training only — finalize is a no-op during eval
    # because every arm's finalize() checks _read_only before doing anything).
    if hasattr(memory, "finalize"):
        memory.finalize()

    # Save the store-growth log alongside episodes.jsonl.
    if memory_growth:
        (out_dir / "memory_growth.json").write_text(json.dumps(memory_growth, indent=2))

    # A compressed copy of the full episode log (H12). The raw file is too large for git,
    # and Run 10 lost its memory text because the raw file was never kept; the .gz copy is
    # small enough to commit with the rest of the run folder.
    _gzip_copy(out_dir / "episodes.jsonl")

    return results, elapsed


def _sum_reports(reports: list[dict]) -> dict:
    """Sum per-episode update reports key by key.

    Numbers are summed, booleans count how often they were true, strings count each
    value ("meta_procedure": {"created": 3, ...}), and nested dicts of numbers are summed
    one level down. Keys ending in "_after" are a store size, not a delta, so the last
    value is kept instead.
    """
    out: dict = {}
    for rep_ in reports:
        for k, v in rep_.items():
            if k.endswith("_after"):
                out[k] = v
            elif isinstance(v, bool):
                out[k] = out.get(k, 0) + int(v)
            elif isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
            elif isinstance(v, str):
                d = out.setdefault(k, {})
                d[v] = d.get(v, 0) + 1
            elif isinstance(v, dict):
                d = out.setdefault(k, {})
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        d[kk] = d.get(kk, 0) + vv
    return out


def _gzip_copy(path: Path) -> None:
    """Write <path>.gz next to path. Failures are logged, never fatal."""
    try:
        with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    except Exception:
        log.warning("could not write %s.gz", path, exc_info=True)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(
    *,
    results: list[dict],
    memory: MemoryModule,
    action_llm: LLMClient,
    memory_llm: LLMClient | None = None,
    elapsed: float,
) -> dict:
    """Compute the full summary dict for one phase."""
    n_ok = sum(1 for r in results if r["success"])
    n = len(results)
    sr = n_ok / n if n else 0.0

    families: dict[str, list[dict]] = {}
    for r in results:
        families.setdefault(r["task_family"], []).append(r)

    per_family: dict[str, dict] = {}
    for fam, eps in sorted(families.items()):
        ok_eps = [e for e in eps if e["success"]]
        fail_eps = [e for e in eps if not e["success"]]
        per_family[fam] = {
            "n": len(eps),
            "success": len(ok_eps),
            "rate": len(ok_eps) / len(eps) if eps else 0.0,
            "avg_steps_ok": sum(e["steps_taken"] for e in ok_eps) / len(ok_eps) if ok_eps else 0.0,
            "avg_steps_fail": sum(e["steps_taken"] for e in fail_eps) / len(fail_eps) if fail_eps else 0.0,
        }

    all_steps = [r["steps_taken"] for r in results]
    ok_steps = [r["steps_taken"] for r in results if r["success"]]
    fail_steps = [r["steps_taken"] for r in results if not r["success"]]

    # Rolling SR windows (size 10)
    sr_windows: list[dict] = []
    for i in range(0, n, 10):
        chunk = [r["success"] for r in results[i: i + 10]]
        sr_windows.append({
            "start": i + 1,
            "end": i + len(chunk),
            "n": len(chunk),
            "success": sum(chunk),
            "rate": sum(chunk) / len(chunk),
        })

    # Memory-store stats (final snapshot). Every arm that can describe its own store gets
    # its description into summary.json, whatever the store is made of.
    memory_stats = _memory_snapshot(memory)

    # Token usage: report action calls and memory-side calls separately.
    action_tok = action_llm.token_usage
    # Only count the memory LLM separately when it is a distinct client. Arms without a
    # memory model fall back to the action config, so the same object can appear
    # twice; double-counting it would invent inference the arm never performed.
    mem_tok = memory_llm.token_usage if (memory_llm is not None and memory_llm is not action_llm) else None
    if mem_tok:
        combined = {k: action_tok.get(k, 0) + mem_tok.get(k, 0) for k in action_tok}
        token_usage = {"action": action_tok, "memory": mem_tok, "total": combined}
    else:
        token_usage = {"action": action_tok, "memory": None, "total": action_tok}

    # Memory-injection telemetry, aggregated from agent.py's per-step measurements. This
    # answers two questions the token_usage block cannot: how OFTEN each arm actually
    # retrieved something, and how much of what it retrieved survived the protocol budget.
    # An arm can retrieve on every step and still deliver only a fraction of it; without
    # the offered/injected split that loss is invisible, since only the truncated block
    # reaches the prompt.
    inj_eps = [r.get("memory_injection") or {} for r in results]
    inj_eps = [e for e in inj_eps if e]
    memory_injection: dict | None = None
    if inj_eps:
        def _sum(key: str) -> float:
            return sum(e.get(key, 0) for e in inj_eps)

        tot_steps = _sum("steps")
        offered = _sum("offered_tokens")
        injected = _sum("injected_tokens")
        n_eps = len(inj_eps)
        memory_injection = {
            "episodes_measured": n_eps,
            "steps_total": int(tot_steps),
            # Share of steps where memory returned anything at all. 0.0 for no_memory by
            # construction; anything unexpectedly near 0.0 elsewhere means the arm is
            # silently failing to retrieve, which no success-rate number would reveal.
            "retrieval_rate": round(_sum("retrieval_steps") / tot_steps, 3) if tot_steps else 0.0,
            "retrieval_steps_per_episode": round(_sum("retrieval_steps") / n_eps, 2),
            "offered_tokens_per_episode": round(offered / n_eps, 1),
            "injected_tokens_per_episode": round(injected / n_eps, 1),
            "offered_tokens_per_step": round(offered / tot_steps, 1) if tot_steps else 0.0,
            "injected_tokens_per_step": round(injected / tot_steps, 1) if tot_steps else 0.0,
            # The headline: fraction of retrieved content the budget threw away.
            "discarded_fraction": round((offered - injected) / offered, 3) if offered else 0.0,
            "truncated_step_rate": round(_sum("truncated_steps") / tot_steps, 3) if tot_steps else 0.0,
            "peak_offered_tokens": round(max(e.get("peak_offered_tokens", 0) for e in inj_eps), 1),
        }

    # Per-episode inference cost (SDD N4): action calls and memory-side calls.
    _n = max(n, 1)
    # How each action was chosen (H14): "digit" is the normal path; everything else is a
    # retry or a fallback. Reported per arm so their frequency is a number, not a guess.
    token_usage["action_pick_paths"] = dict(getattr(action_llm, "pick_paths", {}) or {})
    # Requests resent after a server or connection failure (H11). Identical requests, so
    # they cost time but cannot change a choice at temperature 0.
    token_usage["transport_retries"] = {
        "action": getattr(action_llm, "transport_retries", 0),
        "memory": getattr(memory_llm, "transport_retries", 0)
        if memory_llm is not None and memory_llm is not action_llm else 0,
    }
    # Replies cut off at max_tokens: an incomplete page or JSON object on the memory side.
    token_usage["length_stops"] = {
        "action": getattr(action_llm, "length_stops", 0),
        "memory": getattr(memory_llm, "length_stops", 0)
        if memory_llm is not None and memory_llm is not action_llm else 0,
    }
    token_usage["calls_per_episode"] = {
        "action": round(action_tok.get("n_calls", 0) / _n, 2),
        "memory": round(mem_tok.get("n_calls", 0) / _n, 2) if mem_tok else 0.0,
    }

    # What the arm's update() wrote, summed over the phase (each arm's own keys), and
    # the mean wall time per episode in retrieve / action / update. Logging only.
    memory_update_totals = _sum_reports([r.get("memory_update") or {} for r in results])
    timings = [r.get("memory_timing") or {} for r in results if r.get("memory_timing")]
    memory_timing = (
        {k: round(sum(t.get(k, 0.0) for t in timings) / len(timings), 3) for k in timings[0]}
        if timings else None
    )

    return {
        "memory_type": memory.name,
        "memory_injection": memory_injection,
        "memory_update_totals": memory_update_totals or None,
        "memory_timing_per_episode_s": memory_timing,
        "action_model": action_llm.model,
        "memory_model": memory_llm.model if memory_llm else action_llm.model,
        "n": n,
        "n_success": n_ok,
        "sr": sr,
        "elapsed_s": round(elapsed, 1),
        "avg_ep_s": round(elapsed / n, 1) if n else 0.0,
        "steps": {
            "avg": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0.0,
            "avg_ok": round(sum(ok_steps) / len(ok_steps), 1) if ok_steps else 0.0,
            "avg_fail": round(sum(fail_steps) / len(fail_steps), 1) if fail_steps else 0.0,
        },
        "token_usage": token_usage,
        "per_family": per_family,
        "sr_windows": sr_windows,
        **({"memory_stats": memory_stats} if memory_stats else {}),
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict, label: str = "SUMMARY") -> None:
    print(f"\n{'=' * _W}")
    print(f"  {label}")
    print("=" * _W)
    print(f"  Memory:      {summary['memory_type']}")
    action_model = summary.get("action_model", summary.get("llm_model", "?"))
    memory_model = summary.get("memory_model", action_model)
    if action_model == memory_model:
        print(f"  Model:       {action_model}")
    else:
        print(f"  Action LLM:  {action_model}")
        print(f"  Memory LLM:  {memory_model}")
    print(f"  Success:  {summary['n_success']}/{summary['n']} ({summary['sr']:.1%})")
    if summary["n"]:
        print(f"  Time:     {summary['elapsed_s']:.0f}s  ({summary['avg_ep_s']:.1f}s/ep)")

    s = summary["steps"]
    if s["avg"]:
        print(f"\n  STEPS  avg={s['avg']:.1f}  ok={s['avg_ok']:.1f}  fail={s['avg_fail']:.1f}")

    print(f"\n  SR OVER TIME (window=10)")
    for w in summary.get("sr_windows", []):
        bar = _bar(w["rate"], width=w["n"])
        print(f"    Ep {w['start']:3d}-{w['end']:3d}:  {w['success']}/{w['n']}  ({w['rate']:4.0%})  {bar}")

    print(f"\n  PER-FAMILY")
    print(f"  {'Family':<30}  {'OK':>3}  {'N':>3}  {'SR':>6}  {'Steps-OK':>8}  {'Steps-FAIL':>10}")
    print(f"  {'-'*30}  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*10}")
    for fam, st in summary.get("per_family", {}).items():
        s_ok = f"{st['avg_steps_ok']:.1f}" if st["success"] > 0 else "-"
        s_fail = f"{st['avg_steps_fail']:.1f}" if st["success"] < st["n"] else "-"
        print(
            f"  {fam:<30}  {st['success']:>3}  {st['n']:>3}  {st['rate']:>5.0%}"
            f"  {s_ok:>8}  {s_fail:>10}"
        )

    tok = summary.get("token_usage", {})
    if tok:
        total = tok.get("total", tok)
        total_n = total.get("total_tokens", 0)
        per_ep = total_n // max(summary["n"], 1)
        print(f"\n  TOKENS  total={total_n:,}  per_ep=~{per_ep:,}")
        if tok.get("action") and tok.get("memory"):
            a, w = tok["action"], tok["memory"]
            print(f"    action={a.get('total_tokens',0):,}  memory={w.get('total_tokens',0):,}")
        if cpe := tok.get("calls_per_episode"):
            print(
                f"  LLM CALLS/EP  action={cpe.get('action',0)}  memory={cpe.get('memory',0)}"
                f"   <- memory>0 only for arms with a model in the memory path"
            )
        if paths := tok.get("action_pick_paths"):
            n_pick = sum(paths.values()) or 1
            other = {k: v for k, v in paths.items() if k != "digit"}
            print(
                f"  ACTION PICKS  digit={paths.get('digit', 0) / n_pick:.1%}  "
                + ("  ".join(f"{k}={v}" for k, v in sorted(other.items()))
                   or "no retries or fallbacks")
            )

    if mi := summary.get("memory_injection"):
        print(
            f"\n  MEMORY INJECTION  retrieval_rate={mi['retrieval_rate']:.0%}"
            f"  ({mi['retrieval_steps_per_episode']} steps/ep)"
        )
        print(
            f"    tokens/step   offered={mi['offered_tokens_per_step']:,.0f}"
            f"  injected={mi['injected_tokens_per_step']:,.0f}"
            f"  discarded={mi['discarded_fraction']:.0%}"
        )
        print(
            f"    truncated_steps={mi['truncated_step_rate']:.0%}"
            f"  peak_offered={mi['peak_offered_tokens']:,.0f} tok"
        )

    if ws := summary.get("memory_stats"):
        # Every arm now describes its own store and no two arms share key names, so
        # this prints whatever it was handed. Indexing fixed keys here turned a
        # finished 25-hour run into a KeyError at the moment it tried to report itself.
        print("\n  MEMORY STORE")
        for _k, _v in ws.items():
            print(f"    {_k + ':':<26} {_v}")

"""Base agent loop - observe, retrieve, act, update."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from memory.base import EpisodeResult
from prompts.act import build_action_prompt
from tokens import count_tokens, truncate_to_tokens

if TYPE_CHECKING:
    from alfworld_env import ALFWorldEnv
    from llm import LLMClient
    from memory.base import MemoryModule

log = logging.getLogger(__name__)

def _json_key(d: dict) -> str:
    """A stable string for comparing metadata dicts (values may be unhashable)."""
    import json

    try:
        return json.dumps(d, sort_keys=True, default=str)
    except Exception:
        return repr(d)


def run_episode(
    env: ALFWorldEnv,
    memory: MemoryModule,
    llm: LLMClient,
    max_steps: int = 50,
    history_window: int = 5,
    verbose: bool = False,
    memory_block_max_tokens: int | None = None,
) -> EpisodeResult:
    """Run a single ALFWorld episode with the given memory module.

    verbose=True prints the memory text at episode start and each step's
    action + observation to stdout.  Use -v / --verbose on the CLI to enable.
    """
    observation, admissible = env.reset()
    task = env.task_description
    task_family = env.task_family
    task_id = env.task_id
    game_file = getattr(env, "game_file", "") or ""

    log.debug("Episode start - task=%s family=%s", task, task_family)

    trajectory: list[dict] = []
    score = 0.0
    # Diagnostic: captured once at step 0.
    _initial_memory_context: str = ""
    _initial_memory_metadata: dict = {}
    # Logging only (H12): the served memory text whenever it changes, and per-step sizes.
    memory_log: list[dict] = []
    step_log: list[dict] = []
    _last_logged: tuple[str, str] | None = None
    _timing = {"retrieve_s": 0.0, "action_s": 0.0, "update_s": 0.0}
    _system_prompt = ""
    # Memory-injection telemetry, accumulated per step (EpisodeResult.memory_injection).
    _inj: dict = {
        "steps": 0,
        "retrieval_steps": 0,
        "offered_tokens": 0.0,
        "injected_tokens": 0.0,
        "peak_offered_tokens": 0.0,
        "truncated_steps": 0,
    }

    for step in range(max_steps):
        _t0 = time.perf_counter()
        ctx = memory.retrieve(task, observation, admissible, step)
        _timing["retrieve_s"] += time.perf_counter() - _t0

        # SDD §3.2: cap the injected memory block to a fixed budget so arms with long
        # blocks and short blocks truncate identically, rather than a long block silently
        # eating the action context.
        #
        # Injection telemetry: measure what the arm OFFERED before the cap, and what
        # actually reached the prompt after it. The gap is the number that matters — an
        # arm can retrieve richly and still deliver almost nothing, which is invisible if
        # you only record the post-truncation block. Measured here because this is the one
        # place every arm passes through, so no arm can define the metric differently.
        # Real Qwen3 token counts (H6), for the cap and the meter alike. The kept count
        # includes the truncation marker, and the cap leaves room for it, so a truncated
        # block is at most the budget.
        _offered = count_tokens(ctx.system_context)

        ctx.system_context = truncate_to_tokens(ctx.system_context, memory_block_max_tokens)

        _kept = count_tokens(ctx.system_context)

        _inj["steps"] += 1
        if _offered > 0:
            _inj["retrieval_steps"] += 1
        _inj["offered_tokens"] += _offered
        _inj["injected_tokens"] += _kept
        _inj["peak_offered_tokens"] = max(_inj["peak_offered_tokens"], _offered)
        if _kept < _offered:
            _inj["truncated_steps"] += 1

        if step == 0:
            _initial_memory_context = ctx.system_context or ""
            _initial_memory_metadata = dict(ctx.metadata or {})

        _served = ctx.system_context or ""
        _meta = dict(ctx.metadata or {})
        _key = (_served, _json_key(_meta))
        if _key != _last_logged:
            memory_log.append({
                "step": step,
                "text": _served,
                "metadata": _meta,
                "offered_tokens": round(_offered, 1),
                "injected_tokens": round(_kept, 1),
            })
            _last_logged = _key

        # Verbose: print the memory text once at episode start (step 0)
        if verbose and step == 0:
            ctx_text = ctx.system_context or "(no memory text)"
            print(f"\n{'─'*60}")
            print(f"  MEMORY CONTEXT  [{task_family}]")
            print(f"{'─'*60}")
            print(ctx_text)
            print(f"{'─'*60}\n")

        system_msg, user_msg = build_action_prompt(
            task=task,
            observation=observation,
            admissible_commands=admissible,
            memory_context=ctx.system_context,
            recent_history=trajectory,
            history_window=history_window,
        )
        _t0 = time.perf_counter()
        action = llm.pick_action(system_msg, user_msg, admissible)
        _timing["action_s"] += time.perf_counter() - _t0
        if step == 0:
            _system_prompt = system_msg
        _call = dict(getattr(llm, "last_pick", None) or {})
        step_log.append({
            "step": step,
            "offered_tokens": round(_offered, 1),
            "injected_tokens": round(_kept, 1),
            "prompt_tokens": _call.get("prompt_tokens"),
            "completion_tokens": _call.get("completion_tokens"),
            "action_calls": _call.get("calls"),
            "pick_path": _call.get("path"),
            "raw_output": _call.get("raw"),
            # The exact user prompt the action model received (the system prompt is
            # stored once per episode). Lets any claim about what an arm's agent saw be
            # checked against the log rather than rebuilt.
            "user_prompt": user_msg,
        })

        next_obs, score, done, next_admissible = env.step(action)

        # Verbose: one-line step summary
        if verbose:
            obs_short = observation.replace("\n", " ")[:120]
            done_tag = " ✓" if done else ""
            print(f"  [{step:02d}] {action}{done_tag}")
            print(f"       obs: {obs_short}")

        trajectory.append({
            "step": step,
            "action": action,
            "observation": observation,
            "admissible_commands": admissible,
            "score": score,
        })

        log.debug("  step %02d: %s%s", step, action, " [DONE]" if done else "")

        observation = next_obs
        admissible = next_admissible

        if done:
            break

    success = score > 0
    result = EpisodeResult(
        task_id=task_id,
        task_family=task_family,
        task_description=task,
        trajectory=trajectory,
        success=success,
        steps_taken=len(trajectory),
        scene_id=None,
        memory_context=_initial_memory_context,
        memory_metadata=_initial_memory_metadata,
        game_file=game_file,
        final_observation=observation,
        memory_log=memory_log,
        step_log=step_log,
        system_prompt=_system_prompt,
        memory_injection={
            **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in _inj.items()},
            "discarded_tokens": round(
                _inj["offered_tokens"] - _inj["injected_tokens"], 1
            ),
        },
    )

    # H10: a frozen store is not offered the episode at all, so "update is never called at
    # evaluation" is true of the harness, not only of each arm's own early return. When it
    # is called, a failure is recorded in the episode and reported by check_run.py, instead
    # of escaping and turning a finished episode into an errored one.
    update_report: dict = {}
    if not memory.read_only:
        _t0 = time.perf_counter()
        try:
            memory.update(result)
        except Exception as exc:  # noqa: BLE001  (any arm failure; the run must see it)
            log.exception("memory.update failed for %s", task_id)
            update_report = {"update_errors": 1,
                             "update_error": f"{type(exc).__name__}: {exc}"[:300]}
        else:
            # What this episode wrote to the store (each arm's optional hook).
            update_report = dict(getattr(memory, "last_update_report", None) or {})
        _timing["update_s"] += time.perf_counter() - _t0
    # Logging only, set after update() returns, so the arm never sees them.
    result.memory_timing = {k: round(v, 3) for k, v in _timing.items()}
    result.memory_update = update_report

    log.debug(
        "Episode end - success=%s steps=%d family=%s",
        success,
        result.steps_taken,
        task_family,
    )
    if not success:
        # Concise failure diagnosis — shows whether the robot was looping, lost, or ran out of steps.
        last_actions = " → ".join(t["action"] for t in trajectory[-3:])
        log.warning(
            "FAIL | %s | steps=%d/%d | last: %s",
            task_family, result.steps_taken, max_steps, last_actions,
        )
    return result

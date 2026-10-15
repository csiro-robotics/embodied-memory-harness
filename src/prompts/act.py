"""Prompt builder for action selection.

The shared prompt holds only what an agent needs to use the interface, plus facts the
simulator has stated (DEC-16 in docs/rerun_change_log.md). Guidance about how to solve a
task is the memory module's job, because the memory modules are what is being compared:
any rule written here reaches every arm and hides the difference between them.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You control a household robot. Output ONLY the number of the action to take from \
the numbered ADMISSIBLE ACTIONS list. A single integer, nothing else.

Example:
TASK: put a mug on shelf 1
OBSERVATION: You are in the kitchen. On countertop 1 you see a mug 1.
ADMISSIBLE ACTIONS:
  1. go to countertop 1
  2. go to shelf 1
  3. take mug 1 from countertop 1
Your choice (a single integer from 1 to 3):
3

Rules:
- Pick the action that makes the most progress toward completing the task.
- Only take an object whose name matches the object named in the TASK exactly.
- TASK KNOWLEDGE, when present, is recalled from past episodes in this environment. \
It may be wrong. Treat it as a hint.
"""

# The simulator's own replies to a pickup and to a placement (ALFWorld 0.4.2 says
# "move ... to"; older releases said "put ... in/on"). Reading these replies is how the
# summary knows what the robot holds; the action text alone does not say whether it worked.
_PICKUP_RE = re.compile(r"^you pick up the (.+?) from the ", re.IGNORECASE)
_PLACE_RE = re.compile(r"^you (?:move|put) the (.+?) (?:to|in/on|in|on) the ", re.IGNORECASE)
_ARRIVE_RE = re.compile(r"^you arrive at ", re.IGNORECASE)

# The loop warning fires when the last _LOOP_WINDOW actions use at most _LOOP_DISTINCT
# different commands (for example "go to desk 1", "go to shelf 1" alternating).
_LOOP_WINDOW = 4
_LOOP_DISTINCT = 2


def _with_results(trajectory: list[dict], observation: str) -> list[tuple[str, str]]:
    """Pair each action with the simulator's reply to it (H2).

    Each trajectory entry stores the observation the agent saw BEFORE acting, which is what
    the memory modules learn from, so that record is left as it is. The reply to action i is
    the observation stored with action i+1, and the reply to the last action is the current
    observation.
    """
    replies = [t["observation"] for t in trajectory[1:]] + [observation]
    return [(t["action"], r) for t, r in zip(trajectory, replies)]


def _build_progress(trajectory: list[dict], observation: str) -> str:
    """Facts the simulator has stated so far: what is held, where the robot has been, and
    whether the last few actions repeat.

    Nothing here is inferred from the action text: a pickup counts only if the simulator
    replied that it happened. There is no task-specific state (no heated or cleaned flags)
    and no claim about where an object is not; those are for the memory module to supply.
    """
    if not trajectory:
        return ""

    holding: list[str] = []
    visited: list[str] = []
    for action, reply in _with_results(trajectory, observation):
        reply = reply.strip()
        m = _PICKUP_RE.match(reply)
        if m:
            holding.append(m.group(1).strip())
            continue
        m = _PLACE_RE.match(reply)
        if m:
            obj = m.group(1).strip()
            if obj in holding:
                holding.remove(obj)
            continue
        if action.lower().startswith("go to ") and _ARRIVE_RE.match(reply):
            place = action[6:].strip()
            if place not in visited:
                visited.append(place)

    lines = [
        "PROGRESS:",
        f"  Holding: {', '.join(holding) if holding else 'nothing'}",
        f"  Visited: {', '.join(visited) if visited else 'none yet'}",
    ]

    # Sorted so the text is the same in every process (H5): set order depends on the
    # Python hash seed.
    if len(trajectory) >= _LOOP_WINDOW:
        last = [t["action"] for t in trajectory[-_LOOP_WINDOW:]]
        distinct = sorted(set(last))
        if len(distinct) <= _LOOP_DISTINCT:
            quoted = ", ".join(f"'{a}'" for a in distinct)
            lines.append(
                f"  LOOP WARNING: your last {_LOOP_WINDOW} actions repeated only {quoted}."
            )

    return "\n".join(lines)


def build_action_prompt(
    task: str,
    observation: str,
    admissible_commands: list[str],
    memory_context: str = "",
    recent_history: list[dict] | None = None,
    history_window: int = 5,
) -> tuple[str, str]:
    parts = []

    # Memory context from whichever module is active, under one neutral header. How to read
    # it is up to the arm's own text; the harness adds no legend for any arm's format (H7).
    if memory_context:
        parts.append(
            "TASK KNOWLEDGE (recalled from past episodes in this environment):\n"
            f"{memory_context}\n"
        )

    parts.append(f"TASK: {task}\n")

    if recent_history:
        progress = _build_progress(recent_history, observation)
        if progress:
            parts.append(progress + "\n")

        # Each action followed by the simulator's reply to it (H2), shown in full (H16).
        # The last reply is the current observation, which is also shown below.
        shown = _with_results(recent_history, observation)[-history_window:]
        parts.append("RECENT ACTIONS:")
        for action, reply in shown:
            parts.append(f"  > {action}")
            parts.append(f"    {reply}")
        parts.append("")

    parts.append(f"OBSERVATION: {observation}\n")
    parts.append("ADMISSIBLE ACTIONS:")
    for i, cmd in enumerate(admissible_commands, 1):
        parts.append(f"  {i}. {cmd}")
    n = len(admissible_commands)
    parts.append(f"\nYour choice (a single integer from 1 to {n}):")

    user = "\n".join(parts)
    return SYSTEM_PROMPT, user

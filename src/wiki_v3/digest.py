"""The deterministic digest (D55) — what makes single-model consolidation viable at all.

A 14B model cannot read twenty-five raw ALFWorld transcripts; a 25-episode batch of full
trajectories is tens of thousands of tokens of mostly-repeated room description. Each
record is therefore compacted BY CODE to ~6 lines before the consolidator sees anything,
so a batch lands at ~150 lines and fits alongside the page descriptions.

Two reasons this is a design decision and not an optimisation:

  * D53 puts `qwen3:14b` in the consolidator seat. filesystemmem measured 75.0% -> 89.3%
    from curator strength alone, and the difference was DISTILLATION QUALITY. A
    deterministic compaction step has no distillation to get wrong, so it is a structural
    hedge against exactly the failure mode that single-model consolidation invites.
  * it is the L0 -> L1 compression step from compspectrum, performed without an LLM.

The full trajectory stays in the journal for audit. The consolidator never sees it.
"""
from __future__ import annotations

import re
from typing import Any

from memory.observation_priors import extract_object_locations
from memory.skeleton import spine
from wiki_v3.journal import DOCUMENT, EPISODE

# "You pick up the handtowel 1 from the towelholder 1."
_PICKUP = re.compile(r"you pick up the (\w[\w ]*?)\s*\d+", re.IGNORECASE)
_GOTO = re.compile(r"^go to (\w[\w ]*?)\s*\d+", re.IGNORECASE)


def target_object(task_desc: str) -> str | None:
    """The task's target object type, via the shared extractor used by every arm."""
    from alfworld_env import extract_target_objects

    objs = extract_target_objects(task_desc or "")
    return objs[0] if objs else None


def _receptacles_visited(trajectory: list[dict]) -> list[str]:
    out: list[str] = []
    for step in trajectory or ():
        m = _GOTO.match(str(step.get("action", "") or "").strip())
        if m:
            name = m.group(0)[len("go to ") :].strip().lower()
            if name not in out:
                out.append(name)
    return out


def _found_at(trajectory: list[dict], target: str | None) -> tuple[str | None, int | None]:
    """Where the target was picked up, and how many receptacles had been visited by then.

    'Second receptacle checked' is the quantity that decides ALFWorld episodes (D17), so
    it goes in the digest rather than being left for the LLM to infer from a transcript it
    will not see.
    """
    if not target:
        return None, None
    visited: list[str] = []
    for step in trajectory or ():
        action = str(step.get("action", "") or "").strip()
        m = _GOTO.match(action)
        if m:
            name = m.group(0)[len("go to ") :].strip().lower()
            if name not in visited:
                visited.append(name)
        if action.lower().startswith(f"take {target}"):
            parts = action.lower().split(" from ")
            where = parts[1].strip() if len(parts) > 1 else None
            return where, len(visited)
    return None, None


def _wrong_object_pickups(trajectory: list[dict], target: str | None) -> list[str]:
    """Objects picked up that were not the target.

    Run 3's failure analysis: 13 of 23 failures were wrong-object pickups (cloth ->
    handtowel, apple -> tomato, bowl -> cup). It is the largest single named failure class
    in the project, so the digest names it explicitly instead of hoping the consolidator
    spots it.
    """
    if not target:
        return []
    out: list[str] = []
    for step in trajectory or ():
        m = _PICKUP.search(str(step.get("observation", "") or ""))
        if not m:
            continue
        got = m.group(1).strip().lower()
        if got != target and got not in out:
            out.append(got)
    return out


def _briefing_verdict(record: dict, found_where: str | None) -> str:
    """Did the served briefing help, mislead, or say nothing?

    D23's rationale — the curator must be able to attribute a failure to its own guidance
    — survives into v3 as a deterministic field rather than something the consolidator has
    to work out from prose.
    """
    briefing = (record.get("briefing_served") or "").strip()
    if not briefing:
        return "none served"
    if record.get("outcome") == "success":
        if found_where and found_where.lower() in briefing.lower():
            return "correct — named the location that worked"
        return "served; episode succeeded"
    if found_where and found_where.lower() not in briefing.lower():
        return "MISLEADING — did not name where the object actually was"
    return "served; episode failed"


def digest_episode(record: dict) -> str:
    """~6 lines. Everything the consolidator needs, nothing it does not."""
    traj = record.get("trajectory") or []
    task = record.get("task", "")
    shown = record.get("request_shown") or task
    target = target_object(task)
    found_where, nth = _found_at(traj, target)
    wrong = _wrong_object_pickups(traj, target)
    visited = _receptacles_visited(traj)
    actions = [str(s.get("action", "")) for s in traj]

    lines = [
        f"ep {record.get('index')} · {record.get('family')} · "
        f"{str(record.get('outcome', '')).upper()} · {record.get('steps')} steps",
        f"  request: {shown}",
    ]
    if shown != task:
        lines.append(f"  actual goal: {task}")

    if target:
        if found_where:
            lines.append(
                f"  target: {target}  found at: {found_where}"
                + (f" ({nth} receptacle{'s' if (nth or 0) != 1 else ''} checked)" if nth else "")
            )
        else:
            lines.append(
                f"  target: {target}  NEVER FOUND; checked {len(visited)} receptacles"
            )
    if wrong:
        lines.append(f"  wrong-object pickup: {', '.join(wrong)} (target was {target})")

    sp = spine(actions)
    if sp:
        lines.append(f"  spine: {' -> '.join(sp)}")

    lines.append(f"  briefing was: {_briefing_verdict(record, found_where)}")

    if record.get("outcome") != "success":
        lines.append(f"  failure class: {failure_class(record, target, found_where)}")

    return "\n".join(lines)


def _replies(trajectory: list[dict]) -> list[tuple[str, str]]:
    """(action, the simulator's reply to it). The reply to step i is step i+1's observation."""
    out = []
    for i, step in enumerate(trajectory or ()):
        nxt = trajectory[i + 1].get("observation", "") if i + 1 < len(trajectory) else ""
        out.append((str(step.get("action", "") or "").strip().lower(), str(nxt or "").strip().lower()))
    return out


def _transform_verb(family: str) -> str | None:
    """The verb a `pick_<verb>_then_place` family names, read from the family label."""
    m = re.match(r"^pick_([a-z]+)_then_place$", family or "")
    return m.group(1) if m else None


def _no_transform(record: dict, target: str | None, found_where: str | None) -> bool:
    """W6: the target was picked up, but the family's verb was never accepted.

    In Run 10's late training the agent held the object and never transformed it in 16/20
    heat, 24/29 clean and 21/29 cool failures, and no label could say so.
    """
    verb = _transform_verb(str(record.get("family", "")))
    if not verb or not target or not found_where:
        return False
    return not any(a.startswith(f"{verb} ") and r.startswith(f"you {verb} ")
                   for a, r in _replies(record.get("trajectory") or []))


def _retook_from_destination(record: dict, target: str | None) -> bool:
    """W6/W7: an object of the target type was placed somewhere, then taken back out of it.

    27 of 34 late-training pick-two failures took the first object back out of the
    destination; the page then advised "return to the original location", because the
    digest could not name what happened.
    """
    if not target:
        return False
    placed: set[str] = set()
    for action, reply in _replies(record.get("trajectory") or []):
        m = re.match(rf"^you move the {re.escape(target)} \d+ to the ([a-z]+ \d+)", reply)
        if m:
            placed.add(m.group(1))
            continue
        m = re.match(rf"^take {re.escape(target)} \d+ from ([a-z]+ \d+)$", action)
        if m and m.group(1) in placed and reply.startswith("you pick up"):
            return True
    return False


def failure_class(record: dict, target: str | None, found_where: str | None) -> str:
    """A coarse, deterministic label. 91-95% of failures in this project are search."""
    traj = record.get("trajectory") or []
    labels = []
    if target and not found_where:
        labels.append("search-exhaustion")
    if _wrong_object_pickups(traj, target):
        labels.append("object-confusion")
    if _no_transform(record, target, found_where):
        labels.append("no-transform-issued")
    if _retook_from_destination(record, target):
        labels.append("retook-from-destination")
    if int(record.get("steps", 0)) >= 50:
        labels.append("step-cap")
    return " + ".join(labels) if labels else "unclassified"


def digest_document(record: dict) -> str:
    """A document digest is the document. They are short by design (Part XV D42)."""
    sup = record.get("supersedes") or []
    head = f"doc {record.get('doc_id')} · {record.get('title', '')}"
    if sup:
        head += f"  [SUPERSEDES: {', '.join(sup)}]"
    return f"{head}\n  {str(record.get('text', '')).strip()}"


def digest(record: dict) -> str:
    kind = record.get("kind")
    if kind == EPISODE:
        return digest_episode(record)
    if kind == DOCUMENT:
        return digest_document(record)
    raise ValueError(f"unknown source kind: {kind!r}")


def digest_batch(records: list[dict]) -> str:
    """Documents first — authority before experience, so the consolidator reads the
    handbook before the episodes that may or may not confirm it."""
    docs = [r for r in records if r.get("kind") == DOCUMENT]
    eps = [r for r in records if r.get("kind") == EPISODE]
    parts: list[str] = []
    if docs:
        parts.append("=== DOCUMENTS ===")
        parts.extend(digest(r) for r in docs)
    if eps:
        parts.append("=== EPISODES ===")
        parts.extend(digest(r) for r in eps)
    return "\n".join(parts)


def batch_stats(records: list[dict]) -> dict[str, Any]:
    """Counts the consolidator prompt states explicitly, so it cannot miscount them."""
    eps = [r for r in records if r.get("kind") == EPISODE]
    n_ok = sum(1 for r in eps if r.get("outcome") == "success")
    families: dict[str, dict[str, int]] = {}
    for r in eps:
        fam = str(r.get("family", "unknown"))
        b = families.setdefault(fam, {"n": 0, "ok": 0})
        b["n"] += 1
        if r.get("outcome") == "success":
            b["ok"] += 1
    return {
        "n_episodes": len(eps),
        "n_documents": sum(1 for r in records if r.get("kind") == DOCUMENT),
        "n_success": n_ok,
        "families": families,
    }

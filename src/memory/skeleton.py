"""Shared trajectory abstraction: turn concrete episodes into transferable procedures.

Used by BOTH the procedural and RAG arms (SDD cross-application rule: a generic mechanism
improvement is offered to every arm it applies to, not just the favoured one).

The problem this solves, measured in run 58595498/58595499:

  * Procedural injected the SHORTEST successful trajectory per family, on the theory that
    shortest = most efficient. Shortest actually means LUCKIEST — the run where the object
    happened to be in the first place looked. For examine_in_light the shortest success was
    3 steps against a median of 17; for pick_clean_then_place, 6 against 23. The injected
    "procedure" therefore demonstrated zero search behaviour, and procedural scored BELOW
    the no-memory baseline on pick_and_place (17% vs 29%).

  * Those literal trajectories also carry scene-specific navigation (`go to desk 1`) that
    cannot transfer to a different scene, and sometimes plain noise — the cool procedure
    contained `move winebottle 1 to diningtable 1 -> take winebottle 1 from diningtable 1`,
    a put-down/pick-up detour taught as if it were part of the recipe.

  * RAG stored every episode with no success filter, so 76% of its index was failures, and
    dumped them raw at up to the 50-step cap.

The fix is to stop replaying trajectories and start inducing them. An episode's SPINE is
its sequence of effectful, non-navigation actions with instance numbers abstracted away.
Spines are short, stable within a family, and scene-independent, so they transfer. Search
effort is then reported as a statistic rather than baked in as one lucky path.

Everything here is derived from observed training episodes. No hand-authored task
knowledge: the tool for each transform, the shape of each family, and the search cost are
all counted from data.
"""
from __future__ import annotations

import re
from collections import Counter

# Actions that change world state. Navigation (`go to`), sensing (`look`, `inventory`,
# `examine`) and door handling (`open`/`close`) are excluded from the spine: they are
# either scene-specific or incidental, and including them is what made the old literal
# procedures untransferable.
_EFFECTFUL = ("take", "put", "move", "heat", "cool", "clean", "slice", "use", "toggle")

# "take mug 1 from cabinet 3" -> ("take", "mug", "cabinet")
_INSTANCE = re.compile(r"\b([a-z]+)\s+\d+\b", re.IGNORECASE)


def normalize_action(action: str) -> str:
    """Drop instance numbers so an action names types, not scene-specific objects.

    >>> normalize_action("take mug 1 from cabinet 3")
    'take mug from cabinet'
    """
    return _INSTANCE.sub(r"\1", str(action or "").strip()).strip()


def is_effectful(action: str) -> bool:
    a = str(action or "").strip().lower()
    return bool(a) and a.split()[0] in _EFFECTFUL


def strip_detours(actions: list[str]) -> list[str]:
    """Remove put-down/pick-up round trips that leave the world unchanged.

    `move winebottle 1 to diningtable 1` followed by `take winebottle 1 from diningtable 1`
    returns the agent to the state it started in. Real successful runs contain these, and
    replaying one as a procedure teaches the detour.
    """
    out: list[str] = []
    for a in actions:
        a = str(a or "").strip()
        if not a:
            continue
        if out:
            prev = out[-1]
            pm = re.match(r"^(?:put|move)\s+(\w+\s+\d+)\s+(?:in/on|to)\s+(\w+\s+\d+)", prev, re.I)
            tm = re.match(r"^take\s+(\w+\s+\d+)\s+from\s+(\w+\s+\d+)", a, re.I)
            if pm and tm and pm.group(1).lower() == tm.group(1).lower() \
                    and pm.group(2).lower() == tm.group(2).lower():
                out.pop()          # drop the put; the take restores the prior state
                continue
        out.append(a)
    return out


def spine(actions: list[str]) -> list[str]:
    """The transferable core of a trajectory: effectful actions, abstracted, de-duplicated.

    Consecutive duplicates collapse — repeatedly toggling a lamp is one step in the recipe,
    not five (the wiki arm averaged 5.5 lamp uses per episode against KG's 0.9).

    Object and receptacle TYPES are retained here (only instance numbers go), because the
    RAG arm displays one spine per retrieved episode and the concrete nouns are what make a
    retrieved neighbour informative. For induction across episodes use role_spine().
    """
    out: list[str] = []
    for a in strip_detours(actions):
        if not is_effectful(a):
            continue
        n = normalize_action(a)
        if out and out[-1] == n:
            continue
        out.append(n)
    return out


# "cool cup 1 with fridge 1" -> verb=cool, tool=fridge
_TRANSFORM = re.compile(
    r"^(heat|cool|clean|slice)\s+\w+(?:\s+\d+)?\s+with\s+(?:the\s+)?(\w+)", re.IGNORECASE
)
_TAKE = re.compile(r"^take\s+\w+", re.IGNORECASE)
_PUT = re.compile(r"^(?:put|move)\s+\w+", re.IGNORECASE)
_USE = re.compile(r"^(?:use|toggle)\s+(?:the\s+)?(\w+)", re.IGNORECASE)


def role_action(action: str) -> str | None:
    """Abstract an action to its ROLE in the task, keeping only tool names concrete.

    Induction needs a level of abstraction at which episodes actually agree. Keeping object
    types ("take cd from desk") does not reach it: across 18 successful examine_in_light
    episodes the most common such spine occurred twice, so the "modal procedure" was a rare
    path presented as the recipe, and two families produced no mode at all.

    Roles agree. The object and the source/destination receptacle are scene-specific and
    become slots; the TOOL stays concrete because which appliance performs which state
    change is exactly the stable, learned regularity worth transferring — the same fact the
    KG arm mines into its relations and wiki/attest.py records as evidence. It is read off
    the observed successful action, never hand-written.
    """
    a = str(action or "").strip()
    if m := _TRANSFORM.match(a):
        return f"{m.group(1).lower()} the object with the {m.group(2).lower()}"
    if _TAKE.match(a):
        return "take the target object"
    if _PUT.match(a):
        return "put the object at the goal location"
    if m := _USE.match(a):
        return f"use the {m.group(1).lower()}"
    return None


def _strip_false_starts(steps: list[str]) -> list[str]:
    """Drop acquisitions the episode abandoned.

    `take -> put -> take` means the agent picked something up, put it back down, and went
    for something else: the first pair is a false start, not part of the recipe. Real
    successes contain these, and with only one success to induce from there is no majority
    vote to wash them out — the single stored pick_clean_then_place success began with
    exactly this pattern, so the recipe opened by telling the agent to deliver an object
    before it had cleaned anything.

    Only applied where a later `take` proves the earlier one was discarded, so a genuine
    multi-object delivery (take -> put -> take -> put) is untouched.
    """
    out = list(steps)
    i = 0
    while i + 2 < len(out):
        if (out[i].startswith("take") and out[i + 1].startswith("put")
                and out[i + 2].startswith("take")
                and not (i + 3 < len(out) and out[i + 3].startswith("put"))):
            del out[i:i + 2]
            continue
        i += 1
    return out


def role_spine(actions: list[str]) -> list[str]:
    """Role-abstracted spine: the shape of the solution, independent of scene and object."""
    out: list[str] = []
    for a in strip_detours(actions):
        r = role_action(a)
        if r is None:
            continue
        if out and out[-1] == r:
            continue
        out.append(r)
    return _strip_false_starts(out)


def collapse_cycles(steps: list[str]) -> list[str]:
    """Collapse a repeated block into one pass marked as repeating.

    A modal spine for pick_two_obj_and_place came out as take->put->take->put->take->put:
    THREE delivery cycles, induced from episodes where the agent shuttled an extra object.
    The task needs two, so the enumerated recipe taught over-delivery. The count is a
    property of the episode, not of the family, so the fix is to stop asserting a count
    rather than to hard-code the right one — which would be exactly the hand-authored task
    knowledge the fairness rules forbid.
    """
    n = len(steps)
    for size in range(1, n // 2 + 1):
        if n % size:
            continue
        block = steps[:size]
        if all(steps[i:i + size] == block for i in range(0, n, size)) and n // size > 1:
            return block + [f"(repeat the {size} steps above for each target object)"]
    return steps


def induce_procedure(
    episodes: list[list[str]], min_support: int = 1
) -> tuple[list[tuple[str, int]], int] | None:
    """Induce a family's modal procedure from its successful trajectories.

    Returns (steps, n_episodes) where steps is [(abstracted_action, support), ...], or None
    when there is no evidence at all. Support counts are kept so the injected text can
    state how well attested each step is instead of asserting it.

    min_support is 1, not 2, and that is deliberate. Round 1 ran it at 2 and measured the
    cost: at a 200-episode budget the arm had stored exactly ONE success for each of
    pick_clean/cool/heat_then_place, so induction returned None and the agent was injected
    an EMPTY context on precisely the three families where procedural memory had been
    earning its keep. The arm fell from +11.9pp over the no-memory control to -3.6pp.
    A recipe from one success is weak evidence, but it bootstraps: the arm needs successes
    to build a procedure and the procedure to get successes, and a threshold of 2 severs
    that loop on exactly the hard families. Thin support is now labelled, not withheld.
    """
    spines = [tuple(role_spine(a)) for a in episodes]
    spines = [s for s in spines if s]
    if len(spines) < min_support or not spines:
        return None
    modal, count = Counter(spines).most_common(1)[0]
    if count < 2 and len(spines) >= 3:
        # Enough episodes to expect agreement, but none found: fall back to the steps that
        # appear in a majority, in their most common order. Better a partial recipe than a
        # rare one.
        step_counts = Counter(s for sp in spines for s in set(sp))
        threshold = len(spines) / 2
        common = [s for s, c in step_counts.items() if c > threshold]
        if common:
            order: dict[str, list[int]] = {}
            for sp in spines:
                for i, s in enumerate(sp):
                    order.setdefault(s, []).append(i)
            common.sort(key=lambda s: sum(order[s]) / len(order[s]))
            return [(s, step_counts[s]) for s in common], len(spines)
    return [(s, count) for s in collapse_cycles(list(modal))], len(spines)


def search_stats(episodes: list[list[str]]) -> tuple[float, float]:
    """(median distinct locations visited, median total steps) across successful episodes.

    This is what the old "shortest trajectory" selection threw away. Telling the agent that
    a typical success visited 4 locations in 18 steps is honest and useful; showing it a
    3-step run where the object was found immediately is neither.
    """
    locs, steps = [], []
    for acts in episodes:
        visited = {
            m.group(1).lower()
            for a in acts
            if (m := re.match(r"^go to\s+([a-z]+)", str(a).strip(), re.I))
        }
        locs.append(len(visited))
        steps.append(len(acts))
    if not locs:
        return 0.0, 0.0
    return _median(locs), _median(steps)


def _median(xs: list[int]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def render_procedure(
    family: str, steps: list[tuple[str, int]], n_eps: int, med_locs: float, med_steps: float
) -> str:
    """Render an induced procedure for injection. Compact, and honest about support."""
    label = family.replace("_", " ")
    if n_eps == 1:
        lines = [f"PROCEDURE for {label} — from a SINGLE success, so treat it as a lead "
                 f"rather than a recipe; the current observation outranks it:"]
    else:
        lines = [f"PROCEDURE for {label} — induced from {n_eps} successful episodes:"]
    for i, (step, sup) in enumerate(steps, 1):
        if step.startswith("(repeat"):
            lines.append(f"  {step}")
            continue
        lines.append(f"  {i}. {step}  [seen in {sup}/{n_eps}]")
    if med_locs:
        lines.append(
            f"Typical success searched {med_locs:.0f} locations over {med_steps:.0f} steps — "
            f"expect to search; the recipe above is the goal, not the whole episode."
        )
    lines.append("Object and receptacle instance numbers are omitted; bind them from the "
                 "current observation.")
    return "\n".join(lines)

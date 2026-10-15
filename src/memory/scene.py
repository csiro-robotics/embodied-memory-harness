"""Scene-conditioned search ordering — shared by every arm that holds location priors.

This lives in memory/ rather than wiki/ deliberately. It is a ranking function over
learned priors plus the current observation; nothing about it requires prose, Markdown,
LLM synthesis or agentic retrieval. Under the cross-application rule a generic mechanism
must be offered to every arm it applies to, so the knowledge-graph arm gets it on exactly
the same terms as the wiki. Giving it only to the wiki would make any wiki-vs-KG result an
artifact of unequal mechanism rather than a fact about memory representation.

The problem it solves, measured on Round 2's dev runs (n=140): 91-95% of every arm's
failures are the agent never getting the object in hand, 100% hit the 50-step cap, and
roughly HALF of all navigation in a failing episode goes to a single receptacle type. Every
arm states priors at TYPE level ("1. cabinet, 40%") while search cost is paid at INSTANCE
level, and a room can advertise 27 cabinets. The agent takes the top-ranked type, grinds
cabinet 1 through 27, and the episode ends.
"""
from __future__ import annotations

import re
from collections import Counter

_INSTANCE = re.compile(r"\b([a-z]+)\s+\d+")
_STEPS_PER_CHECK = 2.0


def scene_counts(observation: str) -> Counter:
    """How many instances of each receptacle type this room actually advertises."""
    if not observation:
        return Counter()
    return Counter(_INSTANCE.findall(observation.split("Your task", 1)[0]))


def expected_steps(prob: float, n_instances: int) -> float:
    """Expected steps to find the object if we commit to searching this type.

    On average half the instances are checked before a hit. A type with probability 0.40
    spread over 27 cabinets costs ~68 steps; one with 0.18 in a single coffeemachine costs
    ~6. Ranking by this instead of by probability alone is the entire idea.
    """
    if prob <= 0:
        return float("inf")
    return (max(n_instances, 1) / 2.0) * _STEPS_PER_CHECK / prob


def rank_by_expected_cost(
    counts: dict[str, int], observation: str, top_k: int = 4
) -> list[tuple[str, float, int, float]]:
    """(receptacle_type, prob, n_in_scene, expected_steps), cheapest first.

    `counts` is {receptacle_type: times_observed} from whichever arm's learned priors.
    Types the room does not contain are dropped — they cannot be searched at all.
    """
    clean = {re.sub(r"\s+\d+$", "", str(k).lower()): int(v) for k, v in (counts or {}).items() if v}
    if not clean:
        return []
    total = sum(clean.values())
    present = scene_counts(observation)
    out = []
    for loc, c in clean.items():
        if present and loc not in present:
            continue
        n = present.get(loc, 1)
        p = c / total
        out.append((loc, p, n, expected_steps(p, n)))
    out.sort(key=lambda x: x[3])
    return out[:top_k]


def budget_trap(observation: str, chosen: list[str], threshold: int = 8) -> str | None:
    """Name the receptacle type that will eat the step budget, if it is not already first."""
    present = scene_counts(observation)
    if not present:
        return None
    big, n = present.most_common(1)[0]
    if n >= threshold and (not chosen or chosen[0] != big):
        return (f"This room has {n} {big}s — do not work through them one by one; "
                f"try the places above first.")
    return None


def _instances(observation: str, rtype: str) -> list[str]:
    """The concrete instance names of one receptacle type, in the order the room lists them."""
    head = (observation or "").split("Your task", 1)[0]
    seen, out = set(), []
    for m in re.finditer(r"\b(" + re.escape(rtype) + r")\s+(\d+)\b", head):
        name = f"{m.group(1)} {m.group(2)}"
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def shortlist(counts: dict[str, int], observation: str, k: int = 3) -> list[str]:
    """Name the k concrete receptacle INSTANCES to check, in order.

    Memory in every arm of this project names TYPES. The agent has to pick an instance,
    and told "cabinet" in a room with 27 of them it goes to `cabinet 1`. Measured on the
    Round 2 dev runs: successful episodes found the object within a median of TWO
    receptacle instances (49% in the first one, 100% within six), while failures wandered
    a median of six across all 50 steps and never found it, in rooms advertising a median
    of 22. The 50-step budget permits roughly six checks, so exhaustive search is
    arithmetically impossible and success is decided by the first two or three guesses.

    A ranked list of types does not make those guesses. A list of instances does.
    Low-instance types come first because they are both cheap and unambiguous — there is
    only one `coffeemachine 1` to try.
    """
    out: list[str] = []
    for rtype, _p, _n, _cost in rank_by_expected_cost(counts, observation, top_k=6):
        for name in _instances(observation, rtype):
            out.append(name)
            if len(out) >= k:
                return out
    return out

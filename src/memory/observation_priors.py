"""Object->receptacle extraction from raw observations — shared by every arm.

This lives in memory/ for the same reason memory/scene.py does: it is a regex over the
observation text the environment already returned, and nothing about it requires prose,
Markdown, LLM synthesis or agentic retrieval. Under the cross-application rule it must be
offered to every arm that holds location priors.

WHY IT MATTERS — measured on the FINAL held-out runs (identical 200 training episodes):

    arm            objects   obj->recep edges   observations recorded
    wiki_final        58           361                 8,505
    kg_instances      38            96                   312

A 27x gap in evidence, from the same experience. It exists because the two arms harvest
the trajectory under different rules:

  * the wiki reads EVERY observation of EVERY episode. "On the countertop 1, you see a
    mug 2" records mug@countertop whether or not the agent ever picked the mug up and
    whether or not the episode succeeded.
  * the KG read only ``take <obj> from <recep>`` actions, and only in SUCCESSFUL episodes.
    An object seen and not taken was discarded; a whole failed episode was discarded.

Since Round 6 that asymmetry decides the headline. Once wiki K reduced the injected
payload to a ranked list of receptacle instances, the payload became a pure function of
the priors — so "wiki beats KG" was largely a statement about intake, not representation.
On the 134 held-out episodes the wiki knew the target object 134/134 times against the
KG's 107, and where only the wiki could name instances it scored 50.0% to the KG's 30.6%.

Sharing the extractor removes intake as an explanation. Whatever gap survives is
attributable to how each arm stores and serves what it recorded, which is the actual
research question.

Note this is also the more faithful port: Optimus-1's HDKG builds its location edges by
observing the world, not by conditioning on task success.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# "On the countertop 1, you see a mug 2, a bowl 3."
_OBS_ON = re.compile(r"[Oo]n the (\w[\w ]*?)\s*\d+,\s*you see (.+?)(?:\.|$)")
# "You open the fridge 1. In it, you see a apple 2."
_OBS_IN = re.compile(r"[Yy]ou open the (\w[\w ]*?)\s*\d+\.\s*[Ii]n it,\s*you see (.+?)(?:\.|$)")
# W16: what ALFWorld 0.4.2 actually says, on opening a receptacle or arriving at an open one:
# "You open the fridge 1. The fridge 1 is open. In it, you see a apple 2." and
# "You arrive at cabinet 1. The cabinet 1 is open. In it, you see a cup 1." `_OBS_IN` matched
# neither, so nothing inside a fridge, microwave, safe or closed drawer was ever recorded
# (replicate 1 of the final runs recorded 15,930 sightings, none of them in a fridge).
_OBS_OPEN = re.compile(r"[Tt]he (\w[\w ]*?)\s*\d+ is open\.\s*[Ii]n it,\s*you see (.+?)(?:\.|$)")
_ITEM = re.compile(r"(?:a|an|the)\s+(\w[\w ]*?)\s*\d+")
# "You move the mug 1 to the shelf 1." (ALFWorld 0.4.2) or "You put the mug 1 in/on the shelf 1."
_MOVED = re.compile(r"^[Yy]ou (?:move|put) the (\w[\w ]*?\s*\d+) (?:to|in/on|in|on) the (\w[\w ]*?\s*\d+)")


def extract_object_locations(trajectory: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every (object_type, receptacle_type) pair the agent observed, in order.

    Reads observation text only — no dependence on which action was taken, and no
    dependence on whether the episode succeeded. Duplicates are preserved: a receptacle
    seen holding the object ten times is ten counts of evidence, which is what makes the
    probability estimate meaningful.
    """
    # W14: an object the agent itself put somewhere is not evidence of where that object
    # type is found. Once the simulator reports "You move the mug 1 to the shelf 1", later
    # sightings of mug 1 on shelf 1 in the same episode are skipped. Other instances there
    # still count.
    placed: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for step in trajectory or ():
        obs = str(step.get("observation", "") or "")
        moved = _MOVED.match(obs.strip())
        if moved:
            placed.add((moved.group(1).strip().lower(), moved.group(2).strip().lower()))
        for pattern in (_OBS_ON, _OBS_IN, _OBS_OPEN):
            m = pattern.search(obs)
            if not m:
                continue
            loc = m.group(1).strip().lower()
            loc_inst = m.group(0).split(",")[0].split(".")[0]
            loc_inst = re.sub(r"^(?:[Oo]n the|[Yy]ou open the|[Tt]he)\s+", "", loc_inst)
            loc_inst = re.sub(r"\s+is open$", "", loc_inst).strip().lower()
            for item in _ITEM.finditer(m.group(2)):
                obj = item.group(1).strip().lower()
                inst = item.group(0).split(" ", 1)[1].strip().lower()
                if (inst, loc_inst) in placed:
                    continue
                if obj and loc:
                    out.append((obj, loc))
    return out

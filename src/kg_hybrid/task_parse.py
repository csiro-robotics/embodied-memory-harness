"""Parse an ALFWorld task instruction into (family, object, target, tool, count).

This only *reads* the task string to decide which sub-graph to retrieve from the HDKG.
It does not select actions or drive the agent (that stays with the shared LLM loop), so
the KG remains a pure knowledge source rather than a hand-coded controller.

Family classification is keyword-based and robust to BOTH ALFWorld phrasings:
  "put a clean apple in/on the diningtable"   (state-adjective form)
  "clean some apple and put it in/on the desk" (verb form)
The reference alfworld_env.classify_task_family() matches "put ... in/on" first and would
mislabel the adjective form as pick_and_place, so the KG uses this classifier instead.

Phrasings handled:
  pick_and_place           "put a mug in/on desk"
  pick_clean_then_place    "put a clean apple in/on diningtable" / "clean some X and put it in/on Y"
  pick_heat_then_place     "put a hot egg in/on countertop"      / "heat some X and put it in/on Y"
  pick_cool_then_place     "put a cool tomato in/on sinkbasin"   / "cool some X and put it in/on Y"
  examine_in_light         "examine the book with the desklamp"  / "look at the bowl under the desklamp"
  pick_two_obj_and_place   "put two soapbar in/on garbagecan"    / "find two cd and put them in/on safe"
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ART = r"(?:(?:a|an|some|the|two)\s+)?"  # optional leading article (with trailing space)
_OBJ = r"(?P<obj>[a-z][a-z ]*?)"
_TGT = r"(?P<target>[a-z][a-z ]*?)"
_TOOL = r"(?P<tool>[a-z][a-z ]*?)"
_PREP = r"(?:in/on|in|on|to)"
_END = r"\s*$"

# State adjectives that are NOT part of the object noun.
_STATE_WORDS = {"clean", "hot", "warm", "cool", "cold", "dirty"}


@dataclass
class ParsedTask:
    family: str
    object: str | None
    target: str | None
    tool: str | None
    count: int


def classify_family(task: str) -> str:
    t = " ".join((task or "").strip().lower().split())
    if "examine" in t or "look at" in t or "with the desklamp" in t or "under the desklamp" in t:
        return "examine_in_light"
    if " two " in f" {t} " or t.startswith("find two") or "two " in t:
        return "pick_two_obj_and_place"
    if "clean" in t:
        return "pick_clean_then_place"
    if "hot" in t or "heat" in t or "warm" in t:
        return "pick_heat_then_place"
    if "cool" in t or "cold" in t or "chill" in t:
        return "pick_cool_then_place"
    return "pick_and_place"


def _clean_obj(raw: str) -> str:
    words = [w for w in raw.strip().split() if w and w not in _STATE_WORDS]
    return " ".join(words).strip()


# Ordered (most specific first). Used only to extract object/target/tool spans.
_PATTERNS = [
    (re.compile(rf"^(?:examine|look at)\s+{_ART}{_OBJ}\s+(?:with|under|using)\s+{_ART}{_TOOL}{_END}"),
     "examine"),
    (re.compile(rf"^(?:clean|heat|cool)\s+{_ART}{_OBJ}\s+and\s+put\s+(?:it|them)\s+{_PREP}\s+{_ART}{_TGT}{_END}"),
     "verb"),
    (re.compile(rf"^find\s+two\s+{_OBJ}\s+and\s+put\s+(?:it|them)\s+{_PREP}\s+{_ART}{_TGT}{_END}"),
     "two"),
    (re.compile(rf"^put\s+two\s+{_OBJ}\s+{_PREP}\s+{_ART}{_TGT}{_END}"),
     "two"),
    (re.compile(rf"^put\s+{_ART}{_OBJ}\s+{_PREP}\s+{_ART}{_TGT}{_END}"),
     "put"),
]


def parse_task(task: str) -> ParsedTask:
    text = " ".join((task or "").strip().lower().rstrip(".").split())
    family = classify_family(text)

    for pattern, _kind in _PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        groups = m.groupdict()
        obj = _clean_obj(groups.get("obj", "") or "")
        target = (groups.get("target") or "").strip() or None
        tool = (groups.get("tool") or "").strip() or None
        count = 2 if family == "pick_two_obj_and_place" else 1
        return ParsedTask(family=family, object=obj or None, target=target, tool=tool, count=count)

    # Fallback: grab the first noun-ish token after a leading verb.
    m = re.match(rf"^[a-z]+\s+{_ART}{_OBJ}\b", text)
    obj = _clean_obj(m.group("obj")) if m else None
    count = 2 if family == "pick_two_obj_and_place" else 1
    return ParsedTask(family=family, object=obj or None, target=None, tool=None, count=count)

"""Closed-vocabulary triplets for the AriGraph-hybrid graph (spec §3.1/§3.2)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from kg_hybrid.hdkg import _norm

ADDITIVE_RELATIONS = {"found-at", "affords", "requires", "precedes"}
EXCLUSIVE_RELATIONS = {"is-at"}  # instance-level current state; AriGraph deletion rule
RELATIONS = ADDITIVE_RELATIONS | EXCLUSIVE_RELATIONS

# --- extraction-precision hardening (audit 2026-08-06, gate iteration 1) -----------------
#
# Container/object *state* words the LLM sometimes emits as if they were a location
# ("(drawer 1, is-at, open)", "(drawer, found-at, nothing)"). Never valid as an
# is-at/found-at object (or subject).
STATE_WORDS = {
    "open", "opened", "close", "closed", "empty", "full",
    "on", "off", "nothing", "none", "locked", "unlocked",
    "dirty", "clean", "hot", "cold", "warm", "toggled", "lit", "unlit",
}

# The agent itself is never a graph node.
AGENT_WORDS = {"you", "agent"}

# ALFWorld's receptacle types, exactly `alfworld.gen.constants.RECEPTACLES` (0.4.2),
# lower-cased as the game spells instance names. Used to require that is-at/found-at
# objects are receptacles. Disclosed as hand-written structure (K3): it is the game's
# own type list, not task knowledge. K7: "desklamp", "floorlamp" and "sink" were listed
# too. The lamps are objects in ALFWorld, and as "receptacles" they could never be the
# subject of a location fact, so the graph could not learn where a lamp is.
RECEPTACLE_WORDS = {
    "armchair", "bathtubbasin", "bed", "bowl", "box", "cabinet", "cart", "coffeemachine",
    "coffeetable", "countertop", "cup", "desk", "diningtable", "drawer", "dresser",
    "fridge", "garbagecan", "handtowelholder", "laundryhamper", "microwave", "mug",
    "ottoman", "paintinghanger", "pan", "plate", "pot", "safe", "shelf", "sidetable",
    "sinkbasin", "sofa", "stoveburner", "toaster", "toilet", "toiletpaperhanger",
    "towelholder", "tvstand",
}

# Static (non-portable) receptacles: never valid as the is-at/found-at *subject*
# (that would mean the direction is reversed -- the subject must be the movable
# object, not the container). Movable receptacles (bowl/box/cup/mug/pan/plate/pot)
# are excluded here since they can legitimately sit inside another receptacle.
_MOVABLE_RECEPTACLES = {"bowl", "box", "cup", "mug", "pan", "plate", "pot"}
NON_PORTABLE_RECEPTACLE_WORDS = RECEPTACLE_WORDS - _MOVABLE_RECEPTACLES

# Closed skill vocabulary for precedes/requires endpoints (ALFWorld action verbs).
# `use` added with HC2, when `use <tool>` became its own skill.
SKILL_WORDS = {"goto", "pickup", "put", "open", "close", "clean", "heat", "cool", "use", "examine",
               "toggle"}


@dataclass(frozen=True)
class Triplet:
    subject: str
    relation: str
    object: str

    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.object}"


def _norm_instance(name: object) -> str:
    """Lower-case + collapse whitespace but KEEP the trailing instance number."""
    return " ".join(str(name or "").strip().lower().split())


def _base_word(name: str) -> str:
    """Drop a trailing instance number, e.g. 'countertop 2' -> 'countertop'."""
    parts = name.split(" ")
    if len(parts) > 1 and parts[-1].isdigit():
        return " ".join(parts[:-1])
    return name


def _has_instance_number(name: str) -> bool:
    parts = name.split(" ")
    return len(parts) > 1 and parts[-1].isdigit()


def validate_triplet(raw: object) -> Triplet | None:
    """Schema + semantic gate for extractor output: bad relation / empty fields / a
    handful of mechanically-detectable extraction errors (audit 2026-08-06) -> None
    (discard). Never raises on malformed LLM output.
    """
    if not isinstance(raw, dict):
        return None
    rel = str(raw.get("relation") or "").strip().lower()
    if rel not in RELATIONS:
        return None
    if rel in EXCLUSIVE_RELATIONS:
        subj, obj = _norm_instance(raw.get("subject")), _norm_instance(raw.get("object"))
    else:
        subj, obj = _norm(raw.get("subject")), _norm(raw.get("object"))
    if not subj or not obj:
        return None

    if rel in ("is-at", "found-at"):
        subj_base, obj_base = _base_word(subj), _base_word(obj)
        # Container/object state words are never a location; agent is never a node.
        if subj_base in STATE_WORDS or obj_base in STATE_WORDS:
            return None
        if subj_base in AGENT_WORDS or obj_base in AGENT_WORDS:
            return None
        # Direction rule: subject = the movable object, object = the receptacle.
        if obj_base not in RECEPTACLE_WORDS:
            return None
        if subj_base in NON_PORTABLE_RECEPTACLE_WORDS:
            return None
        # is-at is instance-level state: both endpoints must carry an instance number.
        if rel == "is-at" and not (_has_instance_number(subj) and _has_instance_number(obj)):
            return None

    if rel == "precedes":
        if subj not in SKILL_WORDS or obj not in SKILL_WORDS:
            return None

    if rel == "requires":
        if subj not in SKILL_WORDS:
            return None
        # A skill, or one type name (the tool). Whether the tool is right is checked
        # against the step's action in `action_attests` (K7).
        if obj not in SKILL_WORDS and (" " in obj or _has_instance_number(obj)):
            return None

    return Triplet(subj, rel, obj)


# The game's commands that name a tool: `heat mug 1 with microwave 1`, `use desklamp 1`.
_TOOL_ACTION = re.compile(r"^(clean|heat|cool)\s+\S+\s+\d+\s+with\s+(\S+)\s+\d+")
_USE_ACTION = re.compile(r"^use\s+(\S+)\s+\d+")


def action_attests(t: Triplet, action: str, result: str) -> bool:
    """K7: a tool fact is kept only when this step's accepted action shows it.

    `<tool> affords <skill>` and `<skill> requires <tool>` state which tool performs a
    skill. The extraction prompt asks for facts "directly evidenced by this step", but
    nothing checked it, and Run 10 learned `butterknife affords examine` and
    `clean requires sinkbasin` on steps with no such action. The step evidences the fact
    only if its action is that skill with that tool and the game did not answer "Nothing
    happens." Every other triplet passes unchanged.
    """
    if t.relation == "affords":
        skill, tool = t.object, t.subject
    elif t.relation == "requires" and t.object not in SKILL_WORDS:
        skill, tool = t.subject, t.object
    else:
        return True
    if (result or "").strip().lower().startswith("nothing happens"):
        return False
    act = (action or "").strip().lower()
    m = _TOOL_ACTION.match(act)
    if m:
        return (m.group(1), m.group(2)) == (skill, tool)
    m = _USE_ACTION.match(act)
    if m:
        return skill == "use" and m.group(1) == tool
    return False

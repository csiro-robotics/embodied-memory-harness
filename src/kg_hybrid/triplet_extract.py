"""qwen3:14b closed-set triplet extraction (AriGraph build step; spec §3.2.1)."""
from __future__ import annotations

import json
import logging
import re

from kg_hybrid.triplets import Triplet, validate_triplet

log = logging.getLogger(__name__)

_SYSTEM = (
    "You extract knowledge-graph triplets from one step of a household text game. "
    "Output ONLY a JSON array of objects with keys subject, relation, object. "
    "No prose, no markdown."
)

# HC1/K12: every example uses placeholders in angle brackets. The earlier examples named
# real objects and tools ("sinkbasin affords clean", "clean requires sinkbasin", "pickup
# precedes clean", a mug on a countertop, a fridge). They are facts the arm must learn,
# and Run 10 showed the model copying them: "clean requires sinkbasin" appeared on 15% of
# steps that involved neither cleaning nor a sink.
_USER_TMPL = """Relations you may use (use NO others):
- is-at: a specific object instance is currently at a specific receptacle instance, e.g. {{"subject": "<object> <n>", "relation": "is-at", "object": "<receptacle> <n>"}}
- found-at: an object TYPE is typically found at a receptacle TYPE (no instance numbers), e.g. {{"subject": "<object>", "relation": "found-at", "object": "<receptacle>"}}
- affords: a receptacle type performs a transformation (clean, heat, cool, use), e.g. {{"subject": "<receptacle>", "relation": "affords", "object": "<transformation>"}}
- requires: a skill requires a transformation/tool, e.g. {{"subject": "<skill>", "relation": "requires", "object": "<receptacle>"}}
- precedes: one skill must happen before another, e.g. {{"subject": "<skill>", "relation": "precedes", "object": "<skill>"}}

DIRECTION RULE for is-at/found-at: the subject is ALWAYS the movable object, the object is
ALWAYS the receptacle. Never put a receptacle in subject position, e.g.
{{"subject": "<receptacle> <n>", "relation": "is-at", "object": "<object> <n>"}} is WRONG
(backwards); the correct triplet is
{{"subject": "<object> <n>", "relation": "is-at", "object": "<receptacle> <n>"}}.
Both the subject and the object of is-at must include an instance number (e.g.
"<object> 1", not "<object>"). Never emit "you" or the agent as a subject or object.

Container/object STATE is NOT a location. Words like open, closed, empty, full, on, off,
locked, unlocked, dirty, clean, nothing, none are states, never a receptacle -- emit nothing
for them. {{"subject": "<receptacle> <n>", "relation": "is-at", "object": "open"}} is WRONG;
{{"subject": "<receptacle> <n>", "relation": "found-at", "object": "nothing"}} is WRONG. If a
step only reports that a receptacle was opened/closed and is empty, emit no is-at/found-at
triplet for it.

INVENTORY RULE: if this step's action picks up/takes an object, that object is now in the
agent's inventory, NOT at the receptacle it came from or at any receptacle the agent walks to
next. Do not assert is-at for a carried object unless the step's own text places it at a new
receptacle (e.g. after "put"). Only emit is-at for objects this step's text actually locates
(the observation/result you were given) -- do not carry forward an object's last-known location
from a previous step.

precedes/requires endpoints MUST be one of these skill verbs, and no others: goto, pickup, put,
open, close, clean, heat, cool, use, examine, toggle. Objects, receptacles and multi-word action
phrases are NOT valid precedes/requires endpoints.
{{"subject": "<object>", "relation": "precedes", "object": "<skill>"}} is WRONG (an object is
not a skill). {{"subject": "<object>", "relation": "precedes", "object": "go to <receptacle>"}}
is WRONG (both endpoints must be single skill verbs). requires may pair a skill with its tool
receptacle, e.g. {{"subject": "<skill>", "relation": "requires", "object": "<receptacle>"}}.

Extract only facts directly evidenced by this step. If nothing is evidenced, output [].

OBSERVATION: {obs}
ACTION TAKEN: {act}
RESULT: {out}

JSON array:"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class TripletExtractor:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.stats = {"calls": 0, "raw": 0, "kept": 0, "discarded": 0, "parse_failures": 0}

    def extract(self, observation: str, action: str, outcome: str) -> list[Triplet]:
        self.stats["calls"] += 1
        user = _USER_TMPL.format(obs=observation, act=action, out=outcome)
        try:
            reply = self.llm.complete(_SYSTEM, user, max_tokens=512)
        except Exception:
            log.exception("extraction LLM call failed; skipping step")
            self.stats["parse_failures"] += 1
            return []
        m = _JSON_ARRAY_RE.search(reply or "")
        if not m:
            self.stats["parse_failures"] += 1
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            self.stats["parse_failures"] += 1
            return []
        out: list[Triplet] = []
        seen: set[str] = set()
        for raw in items if isinstance(items, list) else []:
            self.stats["raw"] += 1
            t = validate_triplet(raw)
            if t is None:
                self.stats["discarded"] += 1
            elif t.key() not in seen:
                seen.add(t.key())
                out.append(t)
        self.stats["kept"] += len(out)
        return out

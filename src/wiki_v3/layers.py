"""The two deterministic layers. Zero LLM, and this is the point.

D08 — the two-channel intake asymmetry, which is the paper's contribution:

    observations   UNGATED         seeing a room is not a claim about the plan
    actions        SUCCESS-GATED   an action is attested only if its episode worked

Measured: 8,505 observations recorded against a success-gated extractor's 312, from the
same 200 episodes (commit 0f105a5).

D56 — `priors.json` is a SHARED DETERMINISTIC SUBSTRATE, not a wiki page. It carries the
signal (everything else ablated to +0.7pp, p=0.90) and contains no LLM. v2 described it as
a wiki layer with a markdown rendering (`likely-object-locations.md`), which muddied both
the design and the claim; the rendering is gone and code serves the table directly. Under
D38 this is a generic mechanism that must be offered to any arm that can use it, which is
what `kg_obs` already does.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from memory.observation_priors import extract_object_locations
from memory.scene import shortlist

# "take mug 1 from countertop 3" / "cool cup 1 with fridge 1" / "heat ..." / "clean ..."
_EFFECTFUL = re.compile(
    r"^(take|put|cool|heat|clean|open|close|toggle|use|move|slice|examine)\b", re.IGNORECASE
)


class PriorsStore:
    """object -> receptacle -> count. UNGATED: every observation of every episode.

    Flat counts, deliberately: memory/scene.shortlist takes {receptacle: count} and does
    the expected-search-cost ranking. Storing probabilities here would bake in a
    normalisation that the ranker needs to redo anyway.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._counts: dict[str, dict[str, int]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._counts = {
                str(o): {str(r): int(c) for r, c in locs.items()}
                for o, locs in (data.get("object_locations") or {}).items()
            }

    # ------------------------------------------------------------------ write
    def ingest_episode(self, trajectory: list[dict]) -> int:
        """UNGATED (D08) — called whether the episode succeeded or failed.

        Returns how many observation pairs were recorded, so intake volume is a reported
        number rather than an assumption. This is the quantity the 8,505-vs-312 comparison
        is made of.
        """
        n = 0
        for obj, recep in extract_object_locations(trajectory):
            bucket = self._counts.setdefault(obj, {})
            bucket[recep] = bucket.get(recep, 0) + 1
            n += 1
        return n

    # ------------------------------------------------------------------- read
    def counts_for(self, obj: str | None) -> dict[str, int]:
        if not obj:
            return {}
        return dict(self._counts.get(str(obj).lower(), {}))

    def has(self, obj: str | None) -> bool:
        return bool(self.counts_for(obj))

    def shortlist(self, obj: str | None, observation: str, k: int = 3) -> list[str]:
        """The k concrete receptacle INSTANCES to check, cheapest-first (D17).

        D17 is the largest single measured gain in the project, and it stays deterministic:
        the LLM never ranks locations. Successful episodes found the object within a median
        of two receptacle instances; failures wandered six in rooms advertising a median of
        22, and the 50-step budget permits roughly six checks.
        """
        return shortlist(self.counts_for(obj), observation, k=k)

    @property
    def n_objects(self) -> int:
        return len(self._counts)

    @property
    def n_observations(self) -> int:
        return sum(sum(locs.values()) for locs in self._counts.values())

    def to_dict(self) -> dict:
        return {"object_locations": self._counts}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class EvidenceStore:
    """Accepted action forms, SUCCESS-GATED (D08). Backs the claim gate (D10).

    The claim gate exists because the outcome gate structurally cannot see a wrong
    procedure written from a SUCCESSFUL episode. That is not hypothetical: a five-step
    open/insert/close/heat/take cooling routine was asserted into 75 of 134 tasks while
    `cool cup 1 with fridge 1` appeared verbatim in 20 of the agent's own successes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._actions: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._actions = dict(data.get("accepted_actions") or {})

    # ------------------------------------------------------------------ write
    def ingest_episode(self, trajectory: list[dict], success: bool, episode_index: int) -> int:
        """SUCCESS-GATED (D08) — a failed episode contributes nothing here."""
        if not success:
            return 0
        n = 0
        for step in trajectory or ():
            action = str(step.get("action", "") or "").strip().lower()
            if not action or not _EFFECTFUL.match(action):
                continue
            rec = self._actions.setdefault(action, {"episodes": [], "count": 0})
            rec["count"] = int(rec.get("count", 0)) + 1
            eps = rec.setdefault("episodes", [])
            if episode_index not in eps:
                eps.append(episode_index)
            n += 1
        return n

    # ------------------------------------------------------------------- read
    def forms(self) -> list[str]:
        return list(self._actions.keys())

    def verb_object_forms(self) -> dict[str, set[str]]:
        """(verb, object_type) -> the accepted action SHAPES observed for it.

        Shapes are instance numbers stripped: `cool cup 1 with fridge 1` becomes
        `cool cup with fridge`. The claim gate compares shapes, because a page asserting a
        procedure is not asserting an instance.
        """
        out: dict[str, set[str]] = {}
        for form in self._actions:
            shape = re.sub(r"\s+\d+\b", "", form).strip()
            parts = shape.split()
            if len(parts) < 2:
                continue
            key = f"{parts[0]} {parts[1]}"
            out.setdefault(key, set()).add(shape)
        return out

    def instrument_forms(self, min_attestations: int = 2) -> list[tuple[str, str, int]]:
        """D60. Attested (verb, instrument) pairs -> [(verb, instrument, n_forms)].

        This is the raw material for the affordance page. `heat egg 1 with microwave 1`
        attests that the microwave affords heating; the environment accepted the action, so
        no gate and no model is involved on the way out either.

        The v3.0 run compiled 500 episodes and never once recorded that the microwave heats
        things — fridge/microwave/sinkbasin appeared in the store ONLY as things to locate.
        That fact is present in every transform episode and is the most reusable
        cross-episode fact in the benchmark, so it is now extracted in code rather than
        hoped for from the consolidator.

        `min_attestations` counts DISTINCT action forms, not occurrences: one accepted
        action could be a fluke of a single odd episode, two distinct forms
        (`heat egg with microwave`, `heat mug with microwave`) is a pattern.
        """
        pairs: dict[tuple[str, str], set[str]] = {}
        for form in self._actions:
            shape = re.sub(r"\s+\d+\b", "", form).strip()
            if " with " not in shape:
                continue
            head, instrument = shape.split(" with ", 1)
            instrument = instrument.split()[0] if instrument.split() else ""
            verb = head.split()[0] if head.split() else ""
            if not verb or not instrument:
                continue
            pairs.setdefault((verb, instrument), set()).add(shape)
        out = [
            (verb, instrument, len(shapes))
            for (verb, instrument), shapes in pairs.items()
            if len(shapes) >= min_attestations
        ]
        # Deterministic: most-attested first, then alphabetical. The page must be
        # byte-identical for the same evidence, or D03 regenerability is a fiction.
        out.sort(key=lambda t: (-t[2], t[0], t[1]))
        return out

    @property
    def n_forms(self) -> int:
        return len(self._actions)

    def to_dict(self) -> dict:
        return {"accepted_actions": self._actions}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

"""Plan store and plan text for the KG arm, modelled on Optimus-1's HDKG (NeurIPS 2024).

This is NOT a faithful port of Optimus-1's HDKG. Optimus-1 stores crafting recipes as a
directed dependency graph, retrieves the sub-graph for a goal, topologically sorts it
into sub-goals, and gives those to an LLM planner. What is kept here:

* a task family is a sub-goal graph built from the most frequent skill sequence of the
  family's successful training episodes (`HybridGraph.to_hdkg`). It is a single chain,
  so the topological sort only reproduces the learned order (K4, K16);
* ``topo_sort_subgoals(family)`` (Kahn's algorithm) returns the ordered sub-goals;
* ``build_context(...)`` writes the plan, the learned transform tool and the ranked
  learned locations as the text the action prompt receives.

What differs: nothing is seeded (Optimus-1's memory also starts empty, but its teacher
phase adds plans taken from the Minecraft wiki; we have no teacher phase), and there is no
reflector or experience pool. The text built here is the knowledge Optimus-1's planner
receives: `kg_hybrid.planner` makes one LLM call per episode that turns it into sub-goals
(DEC-10), and this text is served directly only when that call fails. Every fact in the
text is learned from training episodes, except the object count, which is parsed from
the task, and the sentence templates below.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

# Sub-goal skill -> natural-language plan-line template. Filled with obj/target/tool.
_SKILL_TEMPLATES = {
    "locate": "find the {obj}",
    "locate_tool": "find the {tool}",
    "pickup": "pick up the {obj}",
    "clean": "clean the {obj} at the {tool}",
    "heat": "heat the {obj} using the {tool}",
    "cool": "cool the {obj} using the {tool}",
    "place": "put the {obj} in/on the {target}",
    "use": "use the {tool}",
}


def _norm(name: Any) -> str:
    """Lower-case, collapse whitespace, drop a trailing instance number."""
    s = str(name or "").strip().lower()
    s = " ".join(s.split())
    parts = s.split(" ")
    if len(parts) > 1 and parts[-1].isdigit():
        s = " ".join(parts[:-1])
    return s


class HDKG:
    """Directed dependency graph of ALFWorld task/affordance/location knowledge."""

    def __init__(
        self,
        affordances: dict[str, str],
        task_families: dict[str, Any],
        typical_locations: dict[str, list[str]],
        preferred_tool: dict[str, str] | None = None,
    ) -> None:
        self.affordances = {_norm(k): _norm(v) for k, v in (affordances or {}).items()}
        self.task_families = task_families or {}
        self.typical_locations = {
            _norm(k): [_norm(v) for v in vs] for k, vs in (typical_locations or {}).items()
        }
        self.preferred_tool = {_norm(k): _norm(v) for k, v in (preferred_tool or {}).items()}

        # Reverse index: capability -> [receptacle types that afford it].
        self._tools_for: dict[str, list[str]] = defaultdict(list)
        for recep, cap in self.affordances.items():
            self._tools_for[cap].append(recep)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_backbone(cls, path: str | Path) -> "HDKG":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            affordances=data.get("affordances", {}),
            task_families=data.get("task_families", {}),
            typical_locations=data.get("typical_locations", {}),
            preferred_tool=data.get("preferred_tool", {}),
        )

    # -------------------------------------------------------------- planning
    def topo_sort_subgoals(self, family: str) -> list[dict[str, Any]]:
        """Kahn's algorithm over the family's sub-goal dependency DAG.

        Returns sub-goals in a valid dependency order. Ties are broken alphabetically
        for determinism. Raises ValueError on an unknown family or a cyclic DAG.
        """
        if family not in self.task_families:
            raise ValueError(f"unknown task family: {family!r}")
        subs: dict[str, dict] = self.task_families[family]["subgoals"]

        indeg = {n: 0 for n in subs}
        adj: dict[str, list[str]] = defaultdict(list)
        for name, spec in subs.items():
            for dep in spec.get("after", []) or []:
                if dep not in subs:
                    raise ValueError(f"family {family}: sub-goal {name!r} depends on unknown {dep!r}")
                adj[dep].append(name)
                indeg[name] += 1

        queue = deque(sorted(n for n, d in indeg.items() if d == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in sorted(adj[node]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(subs):
            raise ValueError(f"family {family}: sub-goal DAG has a cycle")
        return [dict(name=n, **subs[n]) for n in order]

    def transform_for(self, family: str) -> str | None:
        return self.task_families.get(family, {}).get("transform")

    def count_for(self, family: str) -> int:
        return int(self.task_families.get(family, {}).get("count", 1) or 1)

    def tool_for_transform(self, transform: str | None) -> str | None:
        """Canonical receptacle that performs a transform (clean/heat/cool/examine)."""
        if not transform:
            return None
        transform = _norm(transform)
        pref = self.preferred_tool.get(transform)
        if pref:
            return pref
        tools = self._tools_for.get(transform)
        return tools[0] if tools else None

    def locations_for(
        self,
        obj_type: str,
        learned: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[str]:
        """Ranked receptacle priors for an object: train-mined counts first, then seeds.

        ``learned`` is ``{object_type: {receptacle_type: count}}`` from build_priors.py.
        Learned receptacles (by descending count) lead; hand-authored seeds fill the
        remainder in their authored order. De-duplicated, truncated to ``top_k``.
        """
        obj_type = _norm(obj_type)
        ranked: list[str] = []

        if learned:
            counts = learned.get(obj_type) or {}
            for recep, _cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                r = _norm(recep)
                if r and r not in ranked:
                    ranked.append(r)

        for recep in self.typical_locations.get(obj_type, []):
            if recep not in ranked:
                ranked.append(recep)

        return ranked[:top_k]

    # ----------------------------------------------------------- serialise
    def build_context(
        self,
        family: str,
        obj: str | None,
        target: str | None,
        tool: str | None = None,
        learned: dict[str, Any] | None = None,
        top_k: int = 5,
        count: int = 1,
    ) -> str:
        """Serialise the retrieved sub-graph into the planner hint string.

        Output: an ordered PLAN, the learned TOOL for a transform, and ranked LIKELY
        LOCATIONS. `count` is the number of objects the task asks for, parsed from the
        task text (K1/K10). Everything else is learned; nothing here is keyed to a family
        (HC2 removed the lamp hint, the desklamp default and the "TWO separate" line).
        """
        obj = _norm(obj) or "the target object"
        target = _norm(target)
        transform = self.transform_for(family)
        steps = self.topo_sort_subgoals(family)

        # The tool: named by the task, else learned for the transform or for `use`.
        resolved_tool = _norm(tool) or None
        if not resolved_tool and transform:
            resolved_tool = self.tool_for_transform(transform)
        if not resolved_tool and any(sg["skill"] == "use" for sg in steps):
            resolved_tool = self.tool_for_transform("use")

        fields = {
            "obj": obj,
            "target": target or "the goal location",
            "tool": resolved_tool or "appliance",
        }

        # Ordered plan from topo-sort. When the task asks for several objects, a repeated
        # step is labelled with the object it belongs to.
        plan_lines: list[str] = []
        for i, sg in enumerate(steps, 1):
            template = _SKILL_TEMPLATES.get(sg["skill"], sg["skill"])
            line = template.format(**fields)
            suffix = sg["name"].rsplit("_", 1)[-1]
            if count > 1 and sg["name"] != sg["skill"] and suffix.isdigit():
                line += f"  (object #{suffix})"
            plan_lines.append(f"  {i}. {line}")

        parts: list[str] = ["PLAN (sub-goals, in order):"]
        parts.extend(plan_lines)

        if transform and resolved_tool:
            parts.append(f"TOOL: use a {resolved_tool} to {transform} the {obj}.")

        locs = self.locations_for(obj, learned=learned, top_k=top_k)
        if locs:
            ranked = ", ".join(f"{i}. {loc}" for i, loc in enumerate(locs, 1))
            parts.append(f"LIKELY LOCATIONS for {obj} (search in this order): {ranked}")

        if resolved_tool:
            tool_locs = self.locations_for(resolved_tool, learned=learned, top_k=3)
            if tool_locs:
                parts.append(f"LIKELY LOCATIONS for the {resolved_tool}: " + ", ".join(tool_locs))

        return "\n".join(parts)


def load_learned_priors(path: str | Path | None) -> dict[str, Any]:
    """Load train-mined object->{receptacle: count} priors, or {} if absent."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8")) or {}
    # Accept either {object_locations: {...}} or a bare mapping.
    return data.get("object_locations", data)

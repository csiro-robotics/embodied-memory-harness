"""AriGraph-style learnable graph with HDKG-typed content (spec §3.1–§3.2).

Starts EMPTY. Type-level edges carry weights and are never deleted by a differing
observation. An edge's weight is the number of training episodes that evidenced it
(K6): a fact seen on every step of a 50-step loop counts once, not 50 times.

Instance-level `is-at` triplets ("mug 1 is at countertop 2") only feed the type-level
`found-at` edge. There is no instance-level state layer (K13). AriGraph replaces an
outdated instance triplet within one game; this graph pools 500 games, where "mug 1"
names a different mug in every household, and nothing ever read the state it kept.
"""
from __future__ import annotations

import json
import re as _re
from collections import defaultdict
from pathlib import Path

from kg_hybrid.hdkg import _norm
from kg_hybrid.triplets import ADDITIVE_RELATIONS, EXCLUSIVE_RELATIONS, SKILL_WORDS, Triplet

# ALFWorld command -> (skill, the object instance the command acts on). This map reads the
# game's command grammar; it is disclosed as hand-written structure (K3). The instance is
# captured so that learning can keep only the actions on the task's object (K1).
_ACTION_SKILL = [
    (_re.compile(r"^take\s+(\S+\s+\d+)\s+from\b"), "pickup"),
    (_re.compile(r"^clean\s+(\S+\s+\d+)"), "clean"),
    (_re.compile(r"^heat\s+(\S+\s+\d+)"), "heat"),
    (_re.compile(r"^cool\s+(\S+\s+\d+)"), "cool"),
    (_re.compile(r"^put\s+(\S+\s+\d+)"), "place"),
    # The unified harness surfaces the game's own phrasing, "move X to Y" (the hybrid
    # branch's env bridged put->move the other way). Without this, successful episodes
    # learn recipes with no `place` step (caught by the first unified smoke run).
    (_re.compile(r"^move\s+(\S+\s+\d+)"), "place"),
    # HC2: `use` is its own skill. It used to map to "examine", so the plan said "examine
    # the X under the lamp" and a hand-written lamp hint filled in how.
    (_re.compile(r"^use\s+(\S+\s+\d+)"), "use"),
]


def _parse_action(act: str) -> tuple[str | None, str | None]:
    for rx, name in _ACTION_SKILL:
        m = rx.match(act)
        if m:
            return name, m.group(1)
    return None, None


def skills_from_trajectory(trajectory: list[dict], task_object: str | None = None) -> list[str]:
    """The ordered skills of a successful episode, reduced to the task-relevant actions.

    K1: the learned heat and cool plans carried detours ("place, pick up" at the fridge
    before cooling), and those detours became the most frequent plan. Two rules, both
    read from the episode itself, remove them:

      * when `task_object` is known (the object type parsed from the task), only actions
        on that type count, so picking up and putting down another object is ignored;
      * an object put down and later picked up again was not delivered, so that place
        and the later pickup are both dropped.

    `use <tool>` names no task object and is always kept. "locate" is a `go to` directly
    before a kept pickup.
    """
    actions = [str(t.get("action", "")).strip().lower() for t in trajectory]
    parsed = [_parse_action(a) for a in actions]
    keep = [False] * len(actions)
    for i, (skill, inst) in enumerate(parsed):
        if skill is None:
            continue
        if skill != "use" and task_object and _norm(inst) != _norm(task_object):
            continue
        keep[i] = True
    # Drop a place that is followed by a pickup of the same instance, and that pickup.
    for i, (skill, inst) in enumerate(parsed):
        if not keep[i] or skill != "place":
            continue
        for j in range(i + 1, len(parsed)):
            if keep[j] and parsed[j] == ("pickup", inst):
                keep[i] = keep[j] = False
                break
    skills: list[str] = []
    for i, act in enumerate(actions):
        skill = None
        if act.startswith("go to"):
            if i + 1 < len(actions) and keep[i + 1] and parsed[i + 1][0] == "pickup":
                skill = "locate"
        elif keep[i]:
            skill = parsed[i][0]
        if skill and (not skills or skills[-1] != skill):
            skills.append(skill)
    return skills


def edge_for(t: Triplet) -> tuple[str, str, str] | None:
    """The graph edge a triplet adds to: `is-at` feeds its type-level `found-at` edge."""
    if t.relation in EXCLUSIVE_RELATIONS:
        return (_norm(t.subject), "found-at", _norm(t.object))
    if t.relation in ADDITIVE_RELATIONS:
        return (t.subject, t.relation, t.object)
    return None


def is_own_placement(t: Triplet, placed: set | None) -> bool:
    """Whether `t` only records where the agent itself put an object this episode (K6).

    `placed` holds (object instance, receptacle instance) pairs from the game's "You move
    the mug 1 to the shelf 1". K18: the extractor also writes the type-level fact directly
    ("mug found-at shelf"), which the instance check never caught; 1.6-2.3% of move steps
    in the final runs recorded the agent's own placement that way. A type-level fact is
    skipped when the agent put an object of that type in a receptacle of that type.
    """
    if not placed:
        return False
    if t.relation in EXCLUSIVE_RELATIONS:
        return (t.subject, t.object) in placed
    if t.relation == "found-at":
        return (_norm(t.subject), _norm(t.object)) in {(_norm(o), _norm(r)) for o, r in placed}
    return False


def edge_key(edge: tuple[str, str, str]) -> str:
    """The string form shared by episodic steps and retrieval (`retrieve.keys_from`)."""
    return "|".join(edge)


class HybridGraph:
    def __init__(self) -> None:
        self.edges: dict[tuple[str, str, str], int] = {}
        self.recipes: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------ reconcile
    def reconcile(self, triplets: list[Triplet], seen: set | None = None,
                  placed: set | None = None) -> dict:
        """Add one step's triplets to the graph.

        `seen` holds the edges already counted in this episode; an edge in it is not
        counted again (K6). `placed` holds the (object instance, receptacle instance)
        pairs the agent itself moved in this episode; an `is-at` for one of them records
        where the agent put the object, not where such objects are found, so it is
        skipped (K6, same rule as the wiki's W14). Both default to "not tracked", so a
        caller without an episode counts every triplet, as before.
        """
        stats = {"inserted": 0, "reinforced": 0, "repeated": 0, "own_placement": 0}
        for t in triplets:
            if is_own_placement(t, placed):
                stats["own_placement"] += 1
                continue
            edge = edge_for(t)
            if edge is None:
                continue
            if seen is not None:
                if edge in seen:
                    stats["repeated"] += 1
                    continue
                seen.add(edge)
            stats["reinforced" if edge in self.edges else "inserted"] += 1
            self.edges[edge] = self.edges.get(edge, 0) + 1
        return stats

    # -------------------------------------------------------------- queries
    def found_at_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for (s, r, o), w in self.edges.items():
            if r == "found-at":
                out[s][o] = w
        return dict(out)

    def affordance_map(self) -> dict[str, str]:
        best: dict[str, tuple[int, str]] = {}
        for (s, r, o), w in self.edges.items():
            if r == "affords" and (s not in best or (w, o) > best[s]):
                best[s] = (w, o)
        return {s: cap for s, (_w, cap) in best.items()}

    def preferred_tool_map(self) -> dict[str, str]:
        """transform -> argmax-weight receptacle over (recep, affords, transform).

        Queried by edge OBJECT so a receptacle's more frequent affordance
        (e.g. microwave->open, w=26) cannot shadow its transform role
        (microwave->heat, w=13) the way affordance_map's per-receptacle argmax
        does. Skill-word subjects (reversed-extraction noise like
        'clean affords sinkbasin') are excluded. Ties break lexicographically.
        """
        best: dict[str, tuple[int, str]] = {}
        for (s, r, o), w in self.edges.items():
            if r != "affords" or o not in ("clean", "heat", "cool", "use", "examine"):
                continue
            if s in SKILL_WORDS:
                continue
            if o not in best or w > best[o][0] or (w == best[o][0] and s < best[o][1]):
                best[o] = (w, s)
        return {t: s for t, (_w, s) in best.items()}

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "edges": [[s, r, o, w] for (s, r, o), w in sorted(self.edges.items())],
            "recipes": self.recipes,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls()
        g.edges = {(s, r, o): int(w) for s, r, o, w in data.get("edges", [])}
        g.recipes = {f: dict(seqs) for f, seqs in (data.get("recipes") or {}).items()}
        return g

    # -------------------------------------------------------- recipe learning
    def reinforce_recipe(self, family: str, skills: list[str]) -> None:
        if not skills:
            return
        seqs = self.recipes.setdefault(family, {})
        key = ">".join(skills)
        seqs[key] = seqs.get(key, 0) + 1

    def modal_recipe(self, family: str) -> list[str] | None:
        """Most-reinforced skill sequence for a family.

        Ties on count break toward the SHORTEST sequence, then lexicographically (K1, K2).
        The previous rule preferred the sequence that contained its tied peers, then the
        longer one. It was chosen after looking at valid_unseen results, and it favoured
        plans that carried detours. Shortest is the rule with no tuning behind it.
        """
        seqs = self.recipes.get(family)
        if not seqs:
            return None
        max_count = max(seqs.values())
        tied = [k for k, c in seqs.items() if c == max_count]
        return min(tied, key=lambda k: (len(k.split(">")), k)).split(">")

    def to_hdkg(self):
        # The package's own planning-capable HDKG (verbatim hybrid-branch class, with
        # topo-sort + PLAN serialisation). Local import avoids a cycle at module load.
        from kg_hybrid.hdkg import HDKG
        task_families: dict[str, dict] = {}
        for family in self.recipes:
            skills = self.modal_recipe(family)
            if not skills:
                continue
            reps = {s: skills.count(s) for s in skills}
            seen: dict[str, int] = {}
            names: list[str] = []
            for s in skills:
                if reps[s] > 1:
                    seen[s] = seen.get(s, 0) + 1
                    names.append(f"{s}_{seen[s]}")
                else:
                    names.append(s)
            subgoals = {}
            for i, (name, skill) in enumerate(zip(names, skills)):
                subgoals[name] = {"skill": skill, "after": [names[i - 1]] if i else []}
            transform = next((s for s in skills if s in ("clean", "heat", "cool")), None)
            # K1/K10: no object count here. It used to be the number of pickups, so a
            # plan with a detour told the agent the task needed TWO objects. The count
            # comes from the task text at retrieval (`build_context(count=...)`).
            fam: dict = {"subgoals": subgoals}
            if transform:
                fam["transform"] = transform
            task_families[family] = fam
        return HDKG(
            affordances=self.affordance_map(),
            task_families=task_families,
            typical_locations={},
            preferred_tool=self.preferred_tool_map(),
        )

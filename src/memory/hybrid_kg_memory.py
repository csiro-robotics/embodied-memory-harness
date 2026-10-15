"""HybridKGMemory — HDKG structure + AriGraph learning/retrieval, unified-protocol port.

Ported from the jeremia-kg-arigraph-hybrid branch (alf-world/kg_pipeline). It no longer
behaves like that branch: the re-run fixes (K1-K18, DEC-10's planner, the per-step AriGraph
search) are listed in docs/rerun_change_log.md. The port's own changes were the unified
protocol's contracts:

  1. update() honours MemoryModule.freeze() (`_read_only`) — no graph/episodic/index
     writes during evaluation, in addition to the `learn_online` switch.
  2. Eval loads its trained artifacts from trained_dir/kg_hybrid/ (factory convention)
     and any index re-save lands under the CURRENT run_dir, never back into trained_dir.
  3. The triplet extractor runs on the harness-provided memory LLM (`memory_llm`), so its
     inference is counted in token_usage.memory / calls_per_episode.memory (SDD N4).

Embedding calls (nomic-embed-text) are NOT in that counter — they are declared in
configs/arm_kg_hybrid.yaml instead.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from kg_hybrid.episodic import EpisodeStep, EpisodicStore
from kg_hybrid.hybrid_graph import (HybridGraph, edge_for, edge_key, is_own_placement,
                                    skills_from_trajectory)
from kg_hybrid.retrieve import OllamaEmbedder, TripletIndex, keys_from, semantic_search
from kg_hybrid.planner import KnowledgePlanner
from kg_hybrid.task_parse import parse_task
from kg_hybrid.triplet_extract import TripletExtractor
from kg_hybrid.triplets import action_attests
from memory.base import EpisodeResult, MemoryContext, MemoryModule

log = logging.getLogger(__name__)


# The simulator's own report of a placement, e.g. "You move the mug 1 to the cabinet 1."
_MOVED = re.compile(r"^you (?:move|put) the (\S+ \d+) (?:to|in/on|in|on) the (\S+ \d+)")


def learn_from_episode(graph, episodic, extractor, *, episode_id, task_family,
                       trajectory, success, task_description="", final_observation="") -> dict:
    """Extract triplets from every step and add them to the graph and the episodic log.

    Each step is (observation, action, result), where the result is the next step's
    observation, or `final_observation` for the last step (K8; before, the last step's
    result was empty, so the step that finished a task taught nothing). An edge counts
    once per episode, and the agent's own placements are not location evidence (K6).
    A tool fact is kept only on a step whose action shows it (K7, `action_attests`).
    The episodic log stores the graph edge keys each step evidenced, the same keys
    retrieval returns, so the episodic search can match them (K9).
    """
    stats = {"steps": 0, "triplets": 0, "inserted": 0, "reinforced": 0, "repeated": 0,
             "own_placement": 0, "unattested": 0, "by_relation": {},
             "recipe_reinforced": False}
    seen: set = set()
    placed: set = set()
    for i, turn in enumerate(trajectory):
        obs = str(turn.get("observation", ""))
        act = str(turn.get("action", ""))
        outcome = (str(trajectory[i + 1].get("observation", "")) if i + 1 < len(trajectory)
                   else str(final_observation or ""))
        moved = _MOVED.match(outcome.strip().lower())
        if moved:
            placed.add((moved.group(1), moved.group(2)))
        extracted = extractor.extract(obs, act, outcome)
        triplets = [t for t in extracted if action_attests(t, act, outcome)]
        stats["unattested"] += len(extracted) - len(triplets)
        rec = graph.reconcile(triplets, seen=seen, placed=placed)
        keys = []
        for t in triplets:
            edge = edge_for(t)
            if edge is not None and not is_own_placement(t, placed):
                keys.append(edge_key(edge))
        episodic.log_step(EpisodeStep(
            episode_id=str(episode_id), step=int(turn.get("step", i)),
            observation=obs, action=act, outcome=outcome,
            triplet_keys=sorted(set(keys)),
        ))
        stats["steps"] += 1
        stats["triplets"] += len(triplets)
        for k in ("inserted", "reinforced", "repeated", "own_placement"):
            stats[k] += rec[k]
        for t in triplets:
            stats["by_relation"][t.relation] = stats["by_relation"].get(t.relation, 0) + 1
    if success:
        # K1: only actions on the task's own object type count towards the recipe.
        task_object = parse_task(task_description).object if task_description else None
        graph.reinforce_recipe(task_family, skills_from_trajectory(trajectory, task_object))
        stats["recipe_reinforced"] = True
    return stats


class _CountingEmbedder:
    """Wraps the embedder to count calls and texts (logging only).

    Embedding runs in this arm's memory path at train and eval but goes to a separate
    model, so it never reached token_usage. Counting it puts the cost in the summary.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.n_calls = 0
        self.n_texts = 0

    def embed(self, texts):
        self.n_calls += 1
        self.n_texts += len(texts)
        return self._inner.embed(texts)


def _default_embed_base_url() -> str:
    # Same precedence as runner.build_llm for the memory role: the job's memory server
    # (H19), then its action server, then the static default.
    return (os.environ.get("OLLAMA_MEMORY_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1")


class HybridKGMemory(MemoryModule):
    def __init__(self, llm, *, graph_path=None, episodic_path=None,
                 learn_online=False, episodic_search=True,
                 depth=2, width=5, k_steps=3, top_k=5,
                 embed_base_url=None,
                 embed_model="nomic-embed-text",
                 persist_dir=None, vectors_path=None, vectors_save_path=None,
                 embedder=None, extractor=None, planner=True) -> None:
        self.graph = HybridGraph.load(graph_path) if graph_path and Path(graph_path).exists() else HybridGraph()
        ep_src = episodic_path if episodic_path and Path(episodic_path).exists() else None
        persist = Path(persist_dir) if persist_dir else None
        # A provided trained episodic log wins (eval: loaded read-only from trained_dir,
        # regardless of the config's learn_online — freeze() suppresses writes anyway);
        # otherwise online training appends to a fresh persistent store.
        if ep_src:
            self.episodic = EpisodicStore.load(ep_src)
        elif learn_online and persist:
            self.episodic = EpisodicStore(persist / "episodic.jsonl")
        else:
            self.episodic = EpisodicStore()
        self.learn_online = learn_online
        self.episodic_search = episodic_search
        self.depth, self.width = depth, width
        self.k_steps, self.top_k = k_steps, top_k
        self.persist_dir = persist
        # Persisted triplet-index LOAD location: explicit arg > alongside the graph
        # file (if graph_path known) > persist_dir > no persistence.
        if vectors_path is not None:
            self.vectors_path = Path(vectors_path)
        elif graph_path is not None:
            self.vectors_path = Path(graph_path).parent / "triplet_vectors.json"
        elif persist is not None:
            self.vectors_path = persist / "triplet_vectors.json"
        else:
            self.vectors_path = None
        # SAVE location may differ from the load location: at eval the load path points
        # into trained_dir, which must stay exactly as training left it. Defaults to the
        # load path when the two coincide (training / standalone use).
        self.vectors_save_path = Path(vectors_save_path) if vectors_save_path else self.vectors_path
        self.embedder = _CountingEmbedder(embedder or OllamaEmbedder(
            embed_base_url or _default_embed_base_url(), embed_model))
        if embedder is None:
            self.embedder.embed(["ping"])  # hard fail at startup if model missing
        self.extractor = extractor or TripletExtractor(llm)
        # DEC-10: Optimus-1's knowledge-guided planner, one memory-side call per episode.
        # `planner` may be True (build one on `llm`), an object with .plan(), or False.
        if planner is True:
            planner = KnowledgePlanner(llm) if llm is not None else None
        self.planner = planner or None
        self._episode_plan: tuple[str, str, dict] | None = None
        self._hdkg = None
        # Eager, persisted build: a lazy first-retrieve build previously re-attempted the
        # full (slow) embedding call on every step whenever it failed/timed out. Building
        # once here — and hard-failing if the embeddings service is unreachable — matches
        # the "hard fail, not silent degrade" policy for a missing memory backend.
        self._index = self._build_index()
        self._plan_cache: dict[str, tuple[str, dict]] = {}
        self._search_cache: dict = {}
        self._found_at: dict = {}
        # Logging only: retrieval counts and the last episode's update report.
        self._n_retrieve_calls = 0
        self._n_retrieve_cache_hits = 0
        self._n_retrieve_errors = 0
        self.last_update_report: dict = {}

    def _build_index(self) -> TripletIndex:
        cache: dict[str, list[float]] = {}
        if self.vectors_path and self.vectors_path.exists():
            idx = TripletIndex.load(self.vectors_path, self.graph, self.embedder)
            if idx is not None:
                return idx
            # Stale index (graph changed since save): reuse its vectors as a
            # cache so only the new/changed texts get embedded.
            try:
                data = json.loads(self.vectors_path.read_text())
                cache = dict(zip(data.get("texts", []), data.get("vectors", [])))
            except Exception:
                cache = {}
        idx = TripletIndex(self.graph, self.embedder, cache=cache or None)
        if self.vectors_save_path:
            idx.save(self.vectors_save_path)
        return idx

    @property
    def name(self) -> str:
        return "kg_hybrid"

    # ------------------------------------------------------------- retrieve
    def retrieve(self, task_description, observation, admissible_commands, step) -> MemoryContext:
        """The memory text for one step: the plan block, then what AriGraph retrieves.

        DEC-09 split. The plan block (Optimus-1's side) is built once per task from the
        graph's knowledge about the task object: learned plan, tool and locations. The
        rest (AriGraph's side, K14) is retrieved every step, with the goal and the current
        observation as queries: semantic search over the graph, then the episodic search
        over the retrieved facts. Before, the whole text was built once per episode from
        the goal alone and never changed.
        """
        self._n_retrieve_calls += 1
        try:
            if not self.graph.edges and not self.graph.recipes:
                return MemoryContext(system_context="",
                                     metadata={"source": "kg_hybrid", "path": "empty_graph"})
            plan_text, meta = self._plan_block(task_description)
            if self.planner is not None:
                plan_text, meta = self._planned(task_description, observation, step,
                                                plan_text, meta)
            key = (task_description, observation)
            found = self._search_cache.get(key)
            if found is None:
                found = self._search(task_description, observation)
                self._search_cache = {key: found}  # consecutive repeats only
            else:
                self._n_retrieve_cache_hits += 1
            facts, steps = found
            parts = [plan_text] if plan_text else []
            if steps:
                parts.append("PAST EXPERIENCE (similar past steps):")
                parts.extend(f"- {s.action} -> {' '.join(s.outcome.split())[:200]}" for s in steps)
            if facts:
                parts.append("FACTS FROM MEMORY:")
                parts.extend(f"- {s} {r} {o}" for (s, r, o), _w in facts)
            meta = dict(meta, n_facts=len(facts),
                        facts=[f"{' '.join(e)} (w={w})" for e, w in facts[:40]],
                        episodic=[f"{s.episode_id}:{s.step}" for s in steps])
            return MemoryContext(system_context="\n".join(parts), metadata=meta)
        except Exception:
            self._n_retrieve_errors += 1
            log.exception("hybrid retrieve failed for task=%r", task_description)
            return MemoryContext(system_context="",
                                 metadata={"source": "kg_hybrid", "path": "error"})

    def _plan_block(self, task_description: str) -> tuple[str, dict]:
        """Plan, tool and ranked locations for the task object, cached per task.

        Locations come from the graph's `found-at` edges for the task object, looked up
        directly (Optimus-1 retrieves the sub-graph of its target). Before, they came from
        whatever the goal-only embedding search happened to return.
        """
        cached = self._plan_cache.get(task_description)
        if cached is not None:
            return cached
        if self._hdkg is None:
            self._hdkg = self.graph.to_hdkg()
            self._found_at = self.graph.found_at_counts()
        parsed = parse_task(task_description)
        learned = self._found_at
        meta = {
            "source": "kg_hybrid", "family": parsed.family, "object": parsed.object,
            "target": parsed.target, "tool": parsed.tool, "count": parsed.count,
            "learned_locations": dict(sorted(
                (learned.get(parsed.object) or {}).items(), key=lambda kv: -kv[1])[: self.top_k]),
        }
        try:
            text = self._hdkg.build_context(
                family=parsed.family, obj=parsed.object, target=parsed.target,
                tool=parsed.tool, learned=learned, top_k=self.top_k, count=parsed.count)
            meta["path"] = "plan"
        except ValueError:  # family recipe never learned -> locations-only fallback
            meta["path"] = "locations_only"
            locs = sorted((learned.get(parsed.object) or {}).items(), key=lambda kv: -kv[1])
            text = ""
            if locs:
                ranked = ", ".join(f"{i}. {r}" for i, (r, _c) in enumerate(locs[: self.top_k], 1))
                text = f"LIKELY LOCATIONS for {parsed.object} (search in this order): {ranked}"
        self._plan_cache[task_description] = (text, meta)
        return text, meta

    def _planned(self, task, observation, step, knowledge, meta):
        """The planner's sub-goals for this episode, made at its first step (DEC-10).

        Optimus-1 plans once per task from the task, the observation and the retrieved
        sub-graph (Eq. 2). The plan block is that sub-graph here. When the call fails or
        returns no numbered line, the plan block is served instead, and the log says so.
        """
        if step == 0 or self._episode_plan is None or self._episode_plan[0] != task:
            goals = self.planner.plan(task, observation, knowledge)
            if goals:
                text = "PLAN (sub-goals, in order):" + "".join(
                    f"\n  {i}. {g}" for i, g in enumerate(goals, 1))
                extra = {"plan_source": "planner"}
            else:
                text = knowledge
                extra = {"plan_source": "plan_block_fallback"}
            extra["planner_reply"] = (getattr(self.planner, "last_reply", "") or "")[:2000]
            self._episode_plan = (task, text, extra)
        _task, text, extra = self._episode_plan
        return text, dict(meta, **extra)

    def _search(self, task_description: str, observation: str):
        """AriGraph Algorithm 1: semantic search on each query, then episodic search."""
        facts = semantic_search([task_description, observation], self.graph, self.embedder,
                                depth=self.depth, width=self.width, index=self._index)
        steps = (self.episodic.rerank(keys_from(facts), k=self.k_steps)
                 if self.episodic_search else [])
        return facts, steps

    # --------------------------------------------------------------- update
    def update(self, episode: EpisodeResult) -> None:
        # Honour the unified eval-freeze in addition to the arm's own switch.
        if self._read_only or not self.learn_online:
            return
        ex_before = dict(getattr(self.extractor, "stats", {}) or {})
        emb_before = (self.embedder.n_calls, self.embedder.n_texts)
        report = learn_from_episode(self.graph, self.episodic, self.extractor,
                                    episode_id=episode.task_id, task_family=episode.task_family,
                                    trajectory=episode.trajectory, success=episode.success,
                                    task_description=episode.task_description,
                                    final_observation=episode.final_observation)
        self._hdkg = None
        self._index = self._build_index()  # graph changed -> rebuild+resave eagerly
        self._hdkg = None
        self._plan_cache, self._search_cache = {}, {}
        if self.persist_dir:
            self.graph.save(self.persist_dir / "graph.json")
        # Logging only: what this episode wrote (extractor counts are deltas).
        ex_after = dict(getattr(self.extractor, "stats", {}) or {})
        report["extractor"] = {k: ex_after.get(k, 0) - ex_before.get(k, 0) for k in ex_after}
        report["embed_calls"] = self.embedder.n_calls - emb_before[0]
        report["embed_texts"] = self.embedder.n_texts - emb_before[1]
        report["graph_edges_after"] = len(self.graph.edges)
        self.last_update_report = report

    def reset(self) -> None:
        self.graph = HybridGraph()
        self.episodic = EpisodicStore()
        self._hdkg = None
        self._index = self._build_index()
        self._plan_cache, self._search_cache = {}, {}
    def stats(self) -> dict:
        """Store description for summary.json / memory_growth.json.

        Read-only; nothing here touches the graph. Added so this arm appears in the
        cross-arm store-growth comparison alongside the wiki and KG arms.
        """
        return {
            "graph_edges": len(self.graph.edges),
            "recipe_families": len(self.graph.recipes),
            "episodic_steps": len(self.episodic.steps),
            "indexed_triplets": len(getattr(self._index, "edge_list", []) or []),
            "edges_by_relation": self._edges_by_relation(),
            "extractor": dict(getattr(self.extractor, "stats", {}) or {}),
            "embed_calls": self.embedder.n_calls,
            "embed_texts": self.embedder.n_texts,
            "retrieve_calls": self._n_retrieve_calls,
            "retrieve_cache_hits": self._n_retrieve_cache_hits,
            "retrieve_errors": self._n_retrieve_errors,
            "planner": dict(getattr(self.planner, "stats", {}) or {}),
        }

    def _edges_by_relation(self) -> dict:
        out: dict = {}
        for edge in self.graph.edges:
            rel = edge[1] if isinstance(edge, tuple) and len(edge) == 3 else "?"
            out[rel] = out.get(rel, 0) + 1
        return out

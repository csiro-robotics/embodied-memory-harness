import json
from pathlib import Path

from kg_hybrid.hybrid_graph import HybridGraph
from kg_hybrid.retrieve import TripletIndex
from kg_hybrid.triplets import Triplet
from memory import create_memory
from memory.base import EpisodeResult
from memory.hybrid_kg_memory import HybridKGMemory, learn_from_episode


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(t))] for t in texts]


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t))] for t in texts]


class FakeExtractor:
    def __init__(self):
        self.stats = {"calls": 0}

    def extract(self, obs, act, out):
        self.stats["calls"] += 1
        if act.startswith("take mug"):
            return [Triplet("mug 1", "is-at", "countertop 1")]
        return []


def make_episode(success=True):
    return dict(
        episode_id="ep1", task_family="pick_and_place", success=success,
        trajectory=[
            {"step": 0, "action": "go to countertop 1", "observation": "start"},
            {"step": 1, "action": "take mug 1 from countertop 1",
             "observation": "You arrive at countertop 1. You see a mug 1."},
            {"step": 2, "action": "put mug 1 in/on shelf 1",
             "observation": "You pick up the mug 1."},
        ],
    )


def test_learn_from_episode_populates_all_layers(tmp_path):
    from kg_hybrid.episodic import EpisodicStore
    g, store, ex = HybridGraph(), EpisodicStore(tmp_path / "e.jsonl"), FakeExtractor()
    learn_from_episode(g, store, ex, **make_episode())
    assert g.found_at_counts()["mug"]["countertop"] == 1     # semantic learned
    assert len(store.steps) == 3                             # episodic logged
    assert g.modal_recipe("pick_and_place") == ["locate", "pickup", "place"]
    assert ex.stats["calls"] == 3


def test_failed_episode_learns_facts_but_not_recipe(tmp_path):
    from kg_hybrid.episodic import EpisodicStore
    g = HybridGraph()
    learn_from_episode(g, EpisodicStore(), FakeExtractor(),
                       **make_episode(success=False))
    assert g.found_at_counts()  # facts still learned
    assert g.modal_recipe("pick_and_place") is None  # recipe only on success


def test_retrieve_builds_hint_and_never_raises(tmp_path):
    g = HybridGraph()
    g.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    for _ in range(2):
        g.reconcile([Triplet("mug", "found-at", "countertop")])
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=FakeExtractor())
    mem.graph = g
    mem._index = mem._build_index()  # graph swapped post-construction: rebuild eagerly
    ctx = mem.retrieve("put a mug in shelf", "obs", [], 0)
    assert "PLAN" in ctx.system_context and "countertop" in ctx.system_context
    empty = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=FakeExtractor())
    assert empty.retrieve("put a mug in shelf", "obs", [], 0).system_context == ""


def test_episodic_at_eval_appends_snippets(tmp_path):
    from kg_hybrid.episodic import EpisodeStep
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=FakeExtractor(),
                         episodic_search=True)
    mem.graph.reinforce_recipe("pick_and_place", ["pickup", "place"])
    mem.graph.reconcile([Triplet("mug", "found-at", "countertop")])
    mem._index = mem._build_index()  # graph mutated post-construction: rebuild eagerly
    mem.episodic.log_step(EpisodeStep("ep9", 4, "o", "take mug 1 from countertop 1",
                                      "You pick up the mug 1.",
                                      # two keys: a one-triplet step scores zero (Eq. 1)
                                      ["mug|found-at|countertop", "cup|found-at|shelf"]))
    ctx = mem.retrieve("put a mug in shelf", "obs", [], 0)
    assert "PAST EXPERIENCE" in ctx.system_context
    assert "- take mug 1 from countertop 1 -> You pick up the mug 1." in ctx.system_context
    assert ctx.metadata["episodic"] == ["ep9:4"]  # the source step is logged, not shown


def test_update_persists_when_learning(tmp_path):
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=FakeExtractor(),
                         learn_online=True, persist_dir=tmp_path)
    ep = make_episode()
    mem.update(EpisodeResult(task_id="ep1", task_family=ep["task_family"],
                             task_description="put a mug in shelf",
                             trajectory=ep["trajectory"], success=True, steps_taken=3))
    assert (tmp_path / "graph.json").exists()
    assert json.loads((tmp_path / "graph.json").read_text())["edges"]


def test_retrieve_caches_per_task_and_reset_clears_cache():
    g = HybridGraph()
    g.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    g.reconcile([Triplet("mug", "found-at", "countertop")])
    embedder = CountingEmbedder()
    mem = HybridKGMemory(llm=None, embedder=embedder, extractor=FakeExtractor())
    mem.graph = g
    mem._index = mem._build_index()  # graph swapped post-construction: rebuild eagerly

    mem.retrieve("put a mug in shelf", "obs", [], 0)
    calls_after_first = embedder.calls
    assert calls_after_first > 0

    mem.retrieve("put a mug in shelf", "obs", [], 1)
    assert embedder.calls == calls_after_first  # cached: no new embedder calls

    mem.reset()
    mem.graph = g
    mem._index = mem._build_index()  # graph swapped post-reset: rebuild eagerly
    mem.retrieve("put a mug in shelf", "obs", [], 2)
    assert embedder.calls > calls_after_first  # cache cleared: recomputed


def test_factory_creates_kg_hybrid(tmp_path, monkeypatch):
    import memory.hybrid_kg_memory as m
    monkeypatch.setattr(m, "OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    mem = create_memory("kg_hybrid", run_dir=tmp_path, llm=None,
                        config={"learn_online": False})
    assert mem.name == "kg_hybrid"


def test_factory_passes_vectors_path_alongside_graph(tmp_path, monkeypatch):
    import memory.hybrid_kg_memory as m
    monkeypatch.setattr(m, "OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    graph_path = tmp_path / "shared" / "graph.json"
    HybridGraph().save(graph_path)

    mem = create_memory("kg_hybrid", run_dir=tmp_path / "run", llm=None,
                        config={"graph_path": str(graph_path), "learn_online": False})

    assert mem.vectors_path == graph_path.parent / "triplet_vectors.json"


def test_index_builds_eagerly_at_init_not_at_retrieve(tmp_path):
    g = HybridGraph()
    g.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    g.reconcile([Triplet("mug", "found-at", "countertop")])
    graph_path = tmp_path / "graph.json"
    g.save(graph_path)

    counting = CountingEmbedder()
    mem = HybridKGMemory(llm=None, embedder=counting, extractor=FakeExtractor(),
                         graph_path=graph_path)
    calls_after_init = counting.calls
    assert calls_after_init > 0  # index embedded during __init__, not lazily

    mem.retrieve("put a mug in shelf", "obs", [], 0)
    # K14: one call for the queries, then at most one per hop for new vertex names;
    # the triplet index itself is not rebuilt.
    assert calls_after_init + 1 <= counting.calls <= calls_after_init + mem.depth
    calls_after_first = counting.calls
    mem.retrieve("put a mug in shelf", "a new observation", [], 1)
    assert counting.calls == calls_after_first + 1  # vertex names cached


def test_index_persisted_and_reused_across_instances(tmp_path):
    g = HybridGraph()
    g.reconcile([Triplet("mug", "found-at", "countertop")])
    graph_path = tmp_path / "graph.json"
    g.save(graph_path)

    counting1 = CountingEmbedder()
    mem1 = HybridKGMemory(llm=None, embedder=counting1, extractor=FakeExtractor(),
                          graph_path=graph_path)
    assert counting1.calls == 1  # builds the index once and persists it
    assert mem1.vectors_path.exists()

    counting2 = CountingEmbedder()
    HybridKGMemory(llm=None, embedder=counting2, extractor=FakeExtractor(),
                   graph_path=graph_path)
    assert counting2.calls == 0  # loaded from the persisted vectors file


class ScriptedExtractor:
    """Returns a fixed triplet list per action and records every (obs, act, result)."""

    def __init__(self, by_action):
        self.by_action = by_action
        self.seen = []
        self.stats = {"calls": 0}

    def extract(self, obs, act, out):
        self.stats["calls"] += 1
        self.seen.append((obs, act, out))
        return list(self.by_action.get(act, []))


def test_last_step_result_is_extracted_and_placements_skipped():
    """K8: the final action's result reaches extraction. K6: the agent's own placement
    is not location evidence, and a fact repeated within the episode counts once."""
    from kg_hybrid.episodic import EpisodicStore
    ex = ScriptedExtractor({
        "take mug 1 from countertop 1": [Triplet("mug 1", "is-at", "countertop 1")],
        "look": [Triplet("mug 1", "is-at", "countertop 1")],
        "move mug 1 to shelf 1": [Triplet("mug 1", "is-at", "shelf 1"),
                                  Triplet("bowl 1", "is-at", "shelf 1")],
    })
    g, store = HybridGraph(), EpisodicStore()
    stats = learn_from_episode(
        g, store, ex, episode_id="ep1", task_family="pick_and_place", success=True,
        task_description="put a mug in shelf",
        trajectory=[
            {"step": 0, "action": "take mug 1 from countertop 1", "observation": "o0"},
            {"step": 1, "action": "look", "observation": "o1"},
            {"step": 2, "action": "move mug 1 to shelf 1", "observation": "o2"},
        ],
        final_observation="You move the mug 1 to the shelf 1.")
    assert ex.seen[-1][2] == "You move the mug 1 to the shelf 1."
    assert g.found_at_counts()["mug"] == {"countertop": 1}
    assert g.found_at_counts()["bowl"] == {"shelf": 1}
    assert stats["repeated"] == 1 and stats["own_placement"] == 1
    # K9: episodic keys are graph edge keys, which is what retrieval returns.
    assert store.steps[0].triplet_keys == ["mug|found-at|countertop"]
    assert store.steps[2].triplet_keys == ["bowl|found-at|shelf"]


class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, system, user, max_tokens=None, **kw):
        self.prompts.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _planner_memory(reply):
    from kg_hybrid.planner import KnowledgePlanner
    llm = ScriptedLLM(reply)
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=FakeExtractor(),
                         planner=KnowledgePlanner(llm))
    mem.graph.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    mem.graph.reconcile([Triplet("mug", "found-at", "countertop")])
    mem._index = mem._build_index()
    return mem, llm


def test_planner_runs_once_per_episode_and_its_subgoals_are_served():
    """DEC-10: one planner call at the episode's first step; its plan replaces the block."""
    mem, llm = _planner_memory("<task planning>:\n1. go to the countertop\n2. take the mug\n"
                               "3. move the mug to the shelf")
    first = mem.retrieve("put a mug in shelf", "You are in a room.", [], 0)
    later = mem.retrieve("put a mug in shelf", "You arrive at countertop 1.", [], 1)
    assert len(llm.prompts) == 1
    assert "put a mug in shelf" in llm.prompts[0] and "You are in a room." in llm.prompts[0]
    assert "find the mug" in llm.prompts[0]        # the plan block is the knowledge
    for ctx in (first, later):
        assert "PLAN (sub-goals, in order):\n  1. go to the countertop" in ctx.system_context
        assert "find the mug" not in ctx.system_context
        assert ctx.metadata["plan_source"] == "planner"
    mem.retrieve("put a mug in shelf", "You are in a room.", [], 0)  # next episode
    assert len(llm.prompts) == 2


def test_planner_failure_falls_back_to_the_plan_block():
    for reply in ("I cannot help with that.", RuntimeError("server down")):
        mem, _llm = _planner_memory(reply)
        ctx = mem.retrieve("put a mug in shelf", "obs", [], 0)
        assert "1. find the mug" in ctx.system_context
        assert ctx.metadata["plan_source"] == "plan_block_fallback"


def test_planner_prompt_names_no_objects_or_tools():
    import re
    from kg_hybrid.planner import SYSTEM, USER, parse_plan
    from kg_hybrid.triplets import RECEPTACLE_WORDS
    text = (SYSTEM + USER).lower()
    names = RECEPTACLE_WORDS | {"desklamp", "floorlamp", "apple", "mug"}
    assert [w for w in names if re.search(rf"\b{w}\b", text)] == []
    assert parse_plan("x\n1. a\n2) b\nnote\n3. c") == ["a", "b", "c"]

"""Unified-protocol contracts for the ported kg_hybrid arm.

What the hybrid branch's own suite (test_hybrid_memory.py etc.) does not cover:
the eval freeze (MemoryModule.freeze()), the trained_dir loading convention, the
"eval never writes into trained_dir" invariant, the arm config passing the frozen-key
guard, and the modal_recipe tie-break (shortest wins, K1/K2).

No ALFWorld / LLM / network. Run: pytest tests/test_hybrid_unified.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kg_hybrid.episodic import EpisodicStore
from kg_hybrid.hybrid_graph import HybridGraph
from kg_hybrid.triplets import Triplet
from memory import create_memory
from memory.base import EpisodeResult
from memory.hybrid_kg_memory import HybridKGMemory, learn_from_episode

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(t))] for t in texts]


class FakeExtractor:
    def __init__(self):
        self.stats = {"calls": 0}

    def extract(self, obs, act, out):
        self.stats["calls"] += 1
        return [Triplet("mug", "found-at", "countertop")]


def _episode(success=True) -> EpisodeResult:
    return EpisodeResult(
        task_id="ep1", task_family="pick_and_place",
        task_description="put a mug in/on the shelf",
        trajectory=[
            {"step": 0, "action": "go to countertop 1", "observation": "start",
             "admissible_commands": [], "score": 0.0},
            {"step": 1, "action": "take mug 1 from countertop 1",
             "observation": "On the countertop 1 you see a mug 1.",
             "admissible_commands": [], "score": 0.0},
        ],
        success=success, steps_taken=2,
    )


def _trained_dir(tmp_path: Path) -> Path:
    """Simulate a completed training phase: graph + episodic under train/kg_hybrid/."""
    art = tmp_path / "train" / "kg_hybrid"
    g = HybridGraph()
    g.reconcile([Triplet("mug", "found-at", "countertop"),
                 Triplet("sinkbasin", "affords", "clean")])
    g.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    g.save(art / "graph.json")
    store = EpisodicStore(art / "episodic.jsonl")
    from kg_hybrid.episodic import EpisodeStep
    store.log_step(EpisodeStep(episode_id="t1", step=0, observation="o", action="a",
                               outcome="r", triplet_keys=["mug|found-at|countertop"]))
    return tmp_path / "train"


# ---------------------------------------------------------------- eval freeze

def test_freeze_blocks_all_learning_writes(tmp_path):
    ex = FakeExtractor()
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=ex,
                         learn_online=True, persist_dir=tmp_path / "eval" / "kg_hybrid")
    mem.freeze()
    edges_before = dict(mem.graph.edges)
    mem.update(_episode(success=True))
    assert ex.stats["calls"] == 0, "frozen memory must not spend extractor inference"
    assert mem.graph.edges == edges_before
    assert not (tmp_path / "eval" / "kg_hybrid" / "graph.json").exists()


def test_unfrozen_online_training_still_learns(tmp_path):
    ex = FakeExtractor()
    mem = HybridKGMemory(llm=None, embedder=FakeEmbedder(), extractor=ex,
                         learn_online=True, persist_dir=tmp_path / "kg_hybrid")
    mem.update(_episode(success=True))
    assert ex.stats["calls"] == 2
    assert mem.graph.found_at_counts()["mug"]["countertop"] >= 1
    assert (tmp_path / "kg_hybrid" / "graph.json").exists()
    assert (tmp_path / "kg_hybrid" / "episodic.jsonl").exists()


# ------------------------------------------------------- trained_dir convention

def test_factory_eval_loads_trained_artifacts(tmp_path, monkeypatch):
    import memory.hybrid_kg_memory as m
    monkeypatch.setattr(m, "OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    train = _trained_dir(tmp_path)

    mem = create_memory("kg_hybrid", run_dir=tmp_path / "eval", llm=None,
                        config={"learn_online": True}, trained_dir=train)

    assert mem.graph.found_at_counts()["mug"]["countertop"] == 1
    assert mem.graph.modal_recipe("pick_and_place") == ["locate", "pickup", "place"]
    assert len(mem.episodic.steps) == 1  # trained episodic loaded despite learn_online

    mem.freeze()
    ctx = mem.retrieve("put a mug in/on the shelf", "obs", [], 0)
    assert "PLAN" in ctx.system_context
    assert "countertop" in ctx.system_context


def test_eval_never_writes_into_trained_dir(tmp_path, monkeypatch):
    import memory.hybrid_kg_memory as m
    monkeypatch.setattr(m, "OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    train = _trained_dir(tmp_path)
    before = sorted(p.name for p in (train / "kg_hybrid").iterdir())

    mem = create_memory("kg_hybrid", run_dir=tmp_path / "eval", llm=None,
                        config={"learn_online": True}, trained_dir=train)
    mem.freeze()
    mem.retrieve("put a mug in/on the shelf", "obs", [], 0)
    mem.update(_episode(success=True))

    after = sorted(p.name for p in (train / "kg_hybrid").iterdir())
    assert after == before, "eval phase must leave trained_dir exactly as training left it"
    # Any index (re)build lands under the eval run_dir instead.
    assert mem.vectors_save_path == tmp_path / "eval" / "kg_hybrid" / "triplet_vectors.json"


def test_factory_train_cold_starts_empty(tmp_path, monkeypatch):
    import memory.hybrid_kg_memory as m
    monkeypatch.setattr(m, "OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    mem = create_memory("kg_hybrid", run_dir=tmp_path / "train", llm=None,
                        config={"learn_online": True}, trained_dir=None)
    assert not mem.graph.edges and not mem.graph.recipes
    assert mem.retrieve("put a mug in/on the shelf", "obs", [], 0).system_context == ""


# --------------------------------------------- harness action-vocabulary port

def test_skills_map_the_unified_move_phrasing():
    """The unified harness surfaces the game's own `move X to Y`; the hybrid branch's
    env bridged it to `put X in/on Y`. Both must map to the `place` skill — the first
    smoke run learned `locate>pickup>locate>pickup` (no place) without this."""
    from kg_hybrid.hybrid_graph import skills_from_trajectory
    traj = [
        {"action": "go to countertop 1"},
        {"action": "take mug 1 from countertop 1"},
        {"action": "move mug 1 to shelf 1"},
    ]
    assert skills_from_trajectory(traj) == ["locate", "pickup", "place"]
    traj[2]["action"] = "put mug 1 in/on shelf 1"
    assert skills_from_trajectory(traj) == ["locate", "pickup", "place"]


# ----------------------------------------------------- modal_recipe tie-break

def test_modal_recipe_tie_prefers_shortest_over_supersequence():
    """K1/K2: the supersequence tie-break was chosen on valid_unseen results and is gone.
    In the old four-way tie the shortest sequence now wins."""
    g = HybridGraph()
    for seq in (["pickup", "cool", "place"],
                ["locate", "pickup", "cool", "place"],
                ["locate", "pickup", "cool"],
                ["cool", "place"]):
        g.reinforce_recipe("pick_cool_then_place", seq)
        g.reinforce_recipe("pick_cool_then_place", seq)
    assert g.modal_recipe("pick_cool_then_place") == ["cool", "place"]


def test_modal_recipe_count_still_dominates():
    g = HybridGraph()
    for _ in range(3):
        g.reinforce_recipe("pick_and_place", ["pickup", "place"])
    g.reinforce_recipe("pick_and_place", ["locate", "pickup", "place"])
    assert g.modal_recipe("pick_and_place") == ["pickup", "place"]


# ----------------------------------------------------------- config guard

def test_arm_config_passes_frozen_key_guard():
    from protocol_config import load_config
    cfg = load_config(_CONFIGS / "arm_kg_hybrid.yaml")
    assert cfg["train"]["memory"]["type"] == "kg_hybrid"
    assert cfg["eval"]["memory"]["type"] == "kg_hybrid"
    assert cfg["train"]["env"]["num_episodes"] == 500
    assert cfg["eval"]["env"]["num_episodes"] == 134
    assert cfg["eval"]["env"]["train_eval"] == "eval_out_of_distribution"

import json

from kg_hybrid.episodic import EpisodeStep, EpisodicStore


def s(ep, st, keys):
    return EpisodeStep(ep, st, f"obs{st}", f"act{st}", f"out{st}", keys)


def test_log_and_jsonl_roundtrip(tmp_path):
    p = tmp_path / "episodic.jsonl"
    store = EpisodicStore(p)
    store.log_step(s("ep1", 0, ["a|found-at|b"]))
    store.log_step(s("ep1", 1, ["c|affords|d"]))
    assert len(p.read_text().splitlines()) == 2
    loaded = EpisodicStore.load(p)
    assert [x.step for x in loaded.steps] == [0, 1]
    assert loaded.steps[1].triplet_keys == ["c|affords|d"]


def test_rerank_follows_arigraph_eq1():
    """K9: rel = n / max(N,1) * log2(max(N,1)) (AriGraph Eq. 1, p.3)."""
    store = EpisodicStore()
    store.log_step(s("ep1", 0, ["k1", "k2"]))                # 2/2 * log2 2 = 1.0
    store.log_step(s("ep1", 1, ["k1", "x1", "x2", "x3"]))     # 1/4 * log2 4 = 0.5
    store.log_step(s("ep1", 2, ["k1", "k2", "x1", "x2"]))     # 2/4 * log2 4 = 1.0
    store.log_step(s("ep1", 3, ["x9"]))                       # n=0 -> excluded
    out = store.rerank({"k1", "k2"}, k=5)
    assert [x.step for x in out] == [0, 2, 1]


def test_rerank_gives_one_triplet_steps_zero_weight():
    """AriGraph: "observations containing exactly one triplet are assigned zero weight"."""
    store = EpisodicStore()
    store.log_step(s("ep1", 0, ["k1"]))
    assert store.rerank({"k1"}, k=1) == []

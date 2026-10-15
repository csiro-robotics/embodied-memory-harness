from kg_hybrid.hybrid_graph import HybridGraph
from kg_hybrid.retrieve import EMBED_BATCH_SIZE, OllamaEmbedder, TripletIndex, found_at_from, keys_from, semantic_search
from kg_hybrid.triplets import Triplet


class FakeEmbedder:
    VOCAB = ["mug", "countertop", "cabinet", "sinkbasin", "clean", "book", "shelf"]

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * len(self.VOCAB)
            for i, w in enumerate(self.VOCAB):
                if w in t.lower():
                    v[i] = 1.0
            out.append(v)
        return out


def build_graph():
    g = HybridGraph()
    for _ in range(3):
        g.reconcile([Triplet("mug", "found-at", "countertop")])
    g.reconcile([Triplet("mug", "found-at", "cabinet"),
                 Triplet("sinkbasin", "affords", "clean"),
                 Triplet("book", "found-at", "shelf")])
    return g


def test_search_picks_query_relevant_edges():
    edges = semantic_search(["clean a mug"], build_graph(), FakeEmbedder(), depth=1, width=2)
    rels = {e for e, _w in edges}
    assert any(s == "mug" for s, r, o in rels)
    assert ("book", "found-at", "shelf") not in rels


def test_search_depth_zero_returns_nothing_and_depth_bounds_hops():
    g = build_graph()
    assert semantic_search(["mug"], g, FakeEmbedder(), depth=0, width=5) == []
    one = semantic_search(["mug"], g, FakeEmbedder(), depth=1, width=1)
    two = semantic_search(["mug"], g, FakeEmbedder(), depth=2, width=1)
    assert len(one) == 1 and len(two) >= len(one)
    assert len({e for e, _ in two}) == len(two)  # deduplicated


def test_search_embeds_at_every_hop_not_by_weight():
    """K14: the hop from a vertex ranks edges by similarity to that vertex (AriGraph
    Alg. 2), so a heavy but unrelated edge does not crowd out a relevant one."""
    vec = {
        "q": [1, 0, 0], "mug": [1, 0, 0], "mug found-at countertop": [1, 0, 0],
        "countertop": [0, 1, 0], "clean": [0, 1, 0], "countertop affords clean": [0, 1, 0],
        "book": [0, 0, 1], "book found-at countertop": [0, 0, 1],
    }

    class Explicit:
        def embed(self, texts):
            return [vec[t] for t in texts]

    g = HybridGraph()
    g.edges[("mug", "found-at", "countertop")] = 1
    g.edges[("book", "found-at", "countertop")] = 50   # heavy, shares the vertex
    g.edges[("countertop", "affords", "clean")] = 1
    found = {e for e, _w in semantic_search(["q"], g, Explicit(), depth=2, width=1)}
    assert found == {("mug", "found-at", "countertop"), ("countertop", "affords", "clean")}


def test_search_pools_queries_and_caches_vertices():
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.batches = []

        def embed(self, texts):
            self.batches.append(list(texts))
            return super().embed(texts)

    g = build_graph()
    emb = CountingEmbedder()
    idx = TripletIndex(g, emb)
    emb.batches.clear()
    semantic_search(["mug", "book"], g, emb, depth=2, width=2, index=idx)
    assert emb.batches[0] == ["mug", "book"]           # both queries in one call
    first = len(emb.batches)
    semantic_search(["mug", "book"], g, emb, depth=2, width=2, index=idx)
    assert len(emb.batches) == first + 1               # only the queries again


def test_found_at_and_keys_shapes():
    g = build_graph()
    edges = semantic_search(["mug"], g, FakeEmbedder(), depth=2, width=5)
    fa = found_at_from(edges)
    assert fa.get("mug", {}).get("countertop") == 3
    assert "mug|found-at|countertop" in keys_from(edges)


def test_index_cached_one_embed_call_per_graph():
    class CountingEmbedder(FakeEmbedder):
        calls = 0
        def embed(self, texts):
            CountingEmbedder.calls += 1
            return super().embed(texts)
    g = build_graph()
    emb = CountingEmbedder()
    idx = TripletIndex(g, emb)          # embeds all edges once
    idx.query("mug", seeds=2)
    idx.query("book", seeds=2)
    assert CountingEmbedder.calls == 3  # 1 index build + 2 goal embeddings


class _FakeEmbeddingsAPI:
    """Stand-in for OpenAI().embeddings that records each call's batch."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def create(self, model, input):  # noqa: A002 - matches OpenAI client signature
        self.batches.append(list(input))

        class _Item:
            def __init__(self, embedding):
                self.embedding = embedding

        class _Resp:
            def __init__(self, data):
                self.data = data

        return _Resp([_Item([float(len(t))]) for t in input])


def _fake_ollama_embedder() -> tuple[OllamaEmbedder, _FakeEmbeddingsAPI]:
    embedder = OllamaEmbedder()  # constructing the OpenAI client makes no network call
    fake_api = _FakeEmbeddingsAPI()
    embedder._client.embeddings = fake_api
    return embedder, fake_api


def test_embed_chunks_into_batches_of_at_most_32():
    embedder, fake_api = _fake_ollama_embedder()
    texts = [f"triplet {i}" for i in range(75)]  # 3 batches: 32, 32, 11

    vectors = embedder.embed(texts)

    assert all(len(batch) <= EMBED_BATCH_SIZE for batch in fake_api.batches)
    assert len(fake_api.batches) == 3
    assert [len(b) for b in fake_api.batches] == [32, 32, 11]
    # order-preserving concatenation across batches
    assert vectors == [[float(len(t))] for t in texts]


def test_embed_empty_input_makes_no_calls():
    embedder, fake_api = _fake_ollama_embedder()
    assert embedder.embed([]) == []
    assert fake_api.batches == []


def test_triplet_index_save_load_roundtrip(tmp_path):
    g = build_graph()
    emb = FakeEmbedder()
    idx = TripletIndex(g, emb)
    path = tmp_path / "vectors.json"

    idx.save(path)
    loaded = TripletIndex.load(path, g, emb)

    assert loaded is not None
    assert loaded.vectors == idx.vectors
    assert loaded.edge_list == idx.edge_list


def test_triplet_index_load_returns_none_when_missing(tmp_path):
    g = build_graph()
    assert TripletIndex.load(tmp_path / "nonexistent.json", g, FakeEmbedder()) is None


def test_triplet_index_load_returns_none_on_stale_mismatch(tmp_path):
    g = build_graph()
    emb = FakeEmbedder()
    idx = TripletIndex(g, emb)
    path = tmp_path / "vectors.json"
    idx.save(path)

    g.reconcile([Triplet("book", "found-at", "cabinet")])  # graph changed since save

    assert TripletIndex.load(path, g, emb) is None


def test_index_cache_embeds_only_missing():
    from kg_hybrid.hybrid_graph import HybridGraph
    from kg_hybrid.retrieve import TripletIndex

    class CountingEmbedder:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    g = HybridGraph()
    g.reconcile_key = None  # no-op guard; edges set directly below
    g.edges[("apple", "found-at", "fridge 1")] = 1
    g.edges[("plate", "found-at", "cabinet 2")] = 1
    emb = CountingEmbedder()
    cache = {"apple found-at fridge 1": [0.5, 0.5]}
    idx = TripletIndex(g, emb, cache=cache)
    # only the uncached text got embedded
    assert emb.calls == [["plate found-at cabinet 2"]]
    assert idx.vectors[sorted(g.edges).index(("apple", "found-at", "fridge 1"))] == [0.5, 0.5]

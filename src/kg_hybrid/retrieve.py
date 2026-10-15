"""AriGraph two-stage retrieval: embedding search -> bounded traversal (spec §3.3)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from openai import OpenAI

from kg_hybrid.hybrid_graph import HybridGraph, edge_key

EMBED_BATCH_SIZE = 32


class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "nomic-embed-text", api_key: str = "ollama") -> None:
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0, max_retries=3)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in batches of at most EMBED_BATCH_SIZE texts per API call.

        Ollama's server-side embedding batch takes >5min for ~455 texts in one
        call, exceeding the client's 300s timeout. Chunking keeps each call fast
        while preserving input order in the concatenated result.
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class TripletIndex:
    """Embeds every additive edge once; reusable across queries on a frozen graph."""

    def __init__(self, graph: HybridGraph, embedder,
                 cache: dict[str, list[float]] | None = None) -> None:
        self.graph = graph
        self.embedder = embedder
        self.edge_list = sorted(graph.edges.items())  # [((s,r,o), w)]
        texts = [f"{s} {r} {o}" for (s, r, o), _w in self.edge_list]
        if cache:
            # Incremental build: reuse known vectors, embed only unseen texts
            # (keeps per-episode online updates cheap — the graph grows by a
            # handful of edges per episode, not hundreds).
            missing = [t for t in texts if t not in cache]
            fresh = dict(zip(missing, embedder.embed(missing) if missing else []))
            self.vectors = [cache[t] if t in cache else fresh[t] for t in texts]
        else:
            self.vectors = embedder.embed(texts) if texts else []

    def query(self, goal: str, seeds: int) -> list[tuple[tuple[str, str, str], int]]:
        if not self.edge_list:
            return []
        return self.top(self.embedder.embed([goal])[0], seeds)

    def top(self, vec: list[float], k: int) -> list[tuple[tuple[str, str, str], int]]:
        """The k edges most similar to `vec`; ties break on the edge text."""
        ranked = sorted(
            zip(self.edge_list, self.vectors),
            key=lambda ev: (-_cos(vec, ev[1]), ev[0][0]),
        )
        return [edge for edge, _v in ranked[:k]]

    def vectors_for(self, texts: list[str]) -> dict[str, list[float]]:
        """Embeddings for `texts` in one call, reusing cached ones (vertex names repeat)."""
        cache = self.__dict__.setdefault("_text_vecs", {})
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            cache.update(zip(missing, self.embedder.embed(missing)))
        return {t: cache[t] for t in texts}

    def _texts(self) -> list[str]:
        return [f"{s} {r} {o}" for (s, r, o), _w in self.edge_list]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"texts": self._texts(), "vectors": self.vectors}))

    @classmethod
    def load(cls, path: str | Path, graph: HybridGraph, embedder) -> "TripletIndex | None":
        """Restore a persisted index iff its texts exactly match the graph's
        current triplet strings (same order); otherwise None so the caller
        rebuilds (e.g. the graph changed since the index was saved)."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        edge_list = sorted(graph.edges.items())
        current_texts = [f"{s} {r} {o}" for (s, r, o), _w in edge_list]
        if data.get("texts") != current_texts:
            return None
        idx = cls.__new__(cls)
        idx.graph = graph
        idx.embedder = embedder
        idx.edge_list = edge_list
        idx.vectors = data.get("vectors", [])
        return idx


def semantic_search(queries: list[str], graph: HybridGraph, embedder, *,
                    depth: int = 2, width: int = 5,
                    index: TripletIndex | None = None) -> list[tuple[tuple[str, str, str], int]]:
    """AriGraph's semantic search (Algorithm 1 and 2, p.3 and App. A).

    For each query: a breadth-first search from the query. Every text taken from the
    queue (the query, then each vertex reached) is embedded, and the `width` edges most
    similar to it (EmbedAndRetrieve) are added; their two vertices join the queue one
    hop further out. A text at distance `depth` or more is not expanded. Results are
    pooled over the queries and deduplicated, in the order found.

    K14: this replaces a search that ran once per episode on the goal alone, took seed
    edges by embedding and then followed the highest-WEIGHT incident edges, so frequent
    edges crowded out relevant ones. AriGraph embeds at every hop. Vertex vectors are
    cached on the index, so after the first episodes a step costs one embedding call
    (the queries) plus one per hop for vertices not seen before.
    """
    idx = index or TripletIndex(graph, embedder)
    if not idx.edge_list:
        return []
    picked: dict[tuple[str, str, str], int] = {}
    queries = [q for q in queries if q]
    # Queries (goal, observation) change every step, so they are embedded together and
    # not cached; vertex names repeat and are cached on the index.
    qvecs = dict(zip(queries, idx.embedder.embed(queries))) if queries else {}
    for q in queries:
        level = [q]
        dist = {q: 0}
        d = 0
        while level and d < depth:
            vecs = qvecs if d == 0 else idx.vectors_for(level)
            nxt: list[str] = []
            for text in level:
                for edge, w in idx.top(vecs[text], width):
                    picked.setdefault(edge, w)
                    for v in (edge[0], edge[2]):
                        if v not in dist:
                            dist[v] = d + 1
                            nxt.append(v)
            level = nxt
            d += 1
    return list(picked.items())


def found_at_from(edges) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (s, r, o), w in edges:
        if r == "found-at":
            out.setdefault(s, {})[o] = w
    return out


def keys_from(edges) -> set[str]:
    return {edge_key(e) for e, _w in edges}

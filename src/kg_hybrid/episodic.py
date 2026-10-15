"""AriGraph episodic layer: raw step vertices linked to their triplets (spec §3.1)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeStep:
    episode_id: str
    step: int
    observation: str
    action: str
    outcome: str
    triplet_keys: list[str]


class EpisodicStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.steps: list[EpisodeStep] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, s: EpisodeStep) -> None:
        self.steps.append(s)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(s)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicStore":
        store = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                store.steps.append(EpisodeStep(**json.loads(line)))
        return store

    def rerank(self, retrieved_keys: set[str], k: int) -> list[EpisodeStep]:
        """AriGraph Eq. (1), p.3: rel = n / max(N, 1) * log2(max(N, 1)).

        n is the number of retrieved triplets the step evidenced, N the number of triplets
        it evidenced in all. The log term gives more weight to steps with more triplets,
        and a step with exactly one triplet scores zero, "as they are unlikely to provide
        information beyond the triplet itself" (AriGraph §2). K9: this used to divide by
        N * ln N, which inverted the weighting and gave one-triplet steps the top score.
        Steps scoring zero are not returned.
        """
        scored = []
        for s in self.steps:
            keys = set(s.triplet_keys)
            n, N = len(keys & retrieved_keys), max(len(keys), 1)
            rel = n / N * math.log2(N)
            if rel <= 0:
                continue
            scored.append((-rel, s.episode_id, s.step, s))
        scored.sort(key=lambda x: x[:3])
        return [s for *_ignore, s in scored[:k]]

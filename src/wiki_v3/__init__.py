"""LLM-Wiki v3 — compile rarely, read cheaply, one model.

Described in Section 4 of the paper. Dnn tags cite internal design decisions (not in this repository).

The three properties that make this a wiki rather than an incrementally-maintained rule
set (design doc Part XVI):

  * consolidation is BATCHED (every K episodes), so the consolidator can see a regularity
    across K episodes that no single episode exhibits. v2 curated after every episode and
    therefore could not cross-synthesise at all.
  * the read path is BOUNDED (2 LLM calls against a deterministic index), so retrieval
    cost does not scale with store size. cochran's H3 was refuted at 21x the wrong
    direction specifically because its wiki ran a 30-turn loop at query time.
  * per-episode writes are DETERMINISTIC (journal + priors + evidence, zero LLM), so the
    only LLM in the write path runs 1/K as often.

Everything measured is kept: two-channel intake (D08), outcome gate (D09), claim gate
(D10), instance naming (D17), abstention (D18), no embeddings (D19).
"""
from __future__ import annotations

from wiki_v3.journal import Journal, document_record, episode_record
from wiki_v3.layers import EvidenceStore, PriorsStore
from wiki_v3.store import PageStore

__all__ = [
    "Journal",
    "episode_record",
    "document_record",
    "PriorsStore",
    "EvidenceStore",
    "PageStore",
]

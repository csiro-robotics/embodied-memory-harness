"""The journal — append-only source record, and the only thing that is not regenerable.

D03/D05: everything else in the store (pages, index, priors, evidence) can be deleted and
rebuilt from this file. That is the auditability claim and `test_regenerable` tests it
directly rather than asserting it.

D57: a *source* is an episode record OR a document record. One interface, so a handbook
document reaches the consolidator by exactly the path an episode does. This is the
property that makes Part XV's handbook study a config change rather than a feature — v2's
curator took an `EpisodeResult` and was episode-shaped, so a document had nowhere to enter.

This file replaces v2's `raw/` directory: one artefact, not two.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

EPISODE = "episode"
DOCUMENT = "document"


def episode_record(
    index: int,
    *,
    task: str,
    request_shown: str,
    family: str,
    trajectory: list[dict],
    success: bool,
    steps: int,
    briefing_served: str,
    scene_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    """One episode, verbatim. The trajectory is never edited or truncated here.

    `request_shown` differs from `task` only in handbook mode (D40), where the agent reads
    an underspecified request while the simulator still scores the original goal. Storing
    both means the journal records what the agent actually saw.
    """
    return {
        "kind": EPISODE,
        "index": int(index),
        "task_id": task_id,
        "task": task,
        "request_shown": request_shown,
        "family": family,
        "outcome": "success" if success else "failure",
        "steps": int(steps),
        "briefing_served": briefing_served or "",
        "scene_id": scene_id,
        "trajectory": trajectory,
    }


def document_record(
    index: int,
    *,
    doc_id: str,
    title: str,
    text: str,
    supersedes: list[str] | None = None,
) -> dict:
    """One handbook document. index=0 means seeded before episode 1.

    A document is *authority*, not evidence — see gates.claim_gate and D58. It is admitted
    to the journal unconditionally; what it is allowed to assert is decided at
    consolidation, where the environment still outranks it.
    """
    return {
        "kind": DOCUMENT,
        "index": int(index),
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "supersedes": list(supersedes or []),
    }


class Journal:
    """Append-only JSONL. Nothing in this class rewrites an existing line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extend(self, records: list[dict]) -> None:
        for r in records:
            self.append(r)

    # ------------------------------------------------------------------- read
    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def all(self) -> list[dict]:
        return list(self)

    def episodes(self) -> list[dict]:
        return [r for r in self if r.get("kind") == EPISODE]

    def since(self, last_index: int) -> list[dict]:
        """Every source recorded after `last_index`, in journal order.

        Documents seeded at index 0 are returned only on the first consolidation (the
        state starts at -1 for exactly this reason): they are compiled once and then live
        in the pages, not re-read every pass. That IS the wiki premise — read the corpus
        once, write it down, do not read it again.
        """
        return [r for r in self if int(r.get("index", 0)) > int(last_index)]

    @property
    def n_records(self) -> int:
        return sum(1 for _ in self)


class ConsolidationState:
    """{last_consolidated_index, pass_count}. Tiny, but it must survive a crash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # -1, not 0. `since()` returns records with index > last_consolidated_index, and
        # documents seeded before episode 1 carry index 0 — with a 0 default they could
        # never be consolidated at all, so a handbook corpus would sit in the journal
        # forever and the whole of Part XV would silently measure nothing.
        self.last_consolidated_index = -1
        self.pass_count = 0
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self.last_consolidated_index = int(data.get("last_consolidated_index", -1))
            self.pass_count = int(data.get("pass_count", 0))

    def advance(self, to_index: int) -> None:
        self.last_consolidated_index = int(to_index)
        self.pass_count += 1
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "last_consolidated_index": self.last_consolidated_index,
                    "pass_count": self.pass_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

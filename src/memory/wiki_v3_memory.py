"""WikiV3Memory — the v3 store behind the unified MemoryModule interface.

Described in Section 4 of the paper. Dnn tags cite internal design decisions (not in this repository).

Two harness details that are easy to get wrong and would silently destroy the cost claim:

  * `retrieve()` is called by agent.run_episode at EVERY step. D20 says one retrieval, at
    episode start — re-querying an unchanged store from a changed context measured worse
    (contmem Cond-Step -1.5% vs bundled +6.5%). The briefing is therefore computed once
    per episode and cached; steps 1..49 return the cached object. Without this the read
    path would cost 100 LLM calls per episode instead of 2.
  * There is no mid-episode refresh. v2 had one; the harness hook was removed for every arm
    (fix plan H9), and `update()` is not called at all once the store is frozen (H10).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from alfworld_env import classify_task_family
from memory.base import EpisodeResult, MemoryContext, MemoryModule
from wiki_v3.consolidator import Consolidator
from wiki_v3.journal import ConsolidationState, Journal, document_record, episode_record
from wiki_v3.layers import EvidenceStore, PriorsStore
from wiki_v3.reader import Reader
from wiki_v3.store import PageStore

log = logging.getLogger(__name__)


class WikiV3Memory(MemoryModule):
    def __init__(
        self,
        wiki_dir: str | Path,
        llm: Any = None,
        config: dict | None = None,
    ) -> None:
        cfg = config or {}
        self.wiki_dir = Path(wiki_dir)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        self.consolidate_every = int(cfg.get("consolidate_every", 25))
        self.sources = [str(s).lower() for s in cfg.get("sources", ["episode"])]
        self.git_enabled = bool(cfg.get("git", False))

        self.journal = Journal(self.wiki_dir / "journal.jsonl")
        self.state = ConsolidationState(self.wiki_dir / ".consolidation_state.json")
        self.priors = PriorsStore(self.wiki_dir / "priors.json")
        self.evidence = EvidenceStore(self.wiki_dir / "evidence.json")
        self.pages = PageStore(self.wiki_dir / "pages")

        self._llm = llm
        self._reader = (Reader(self.pages, self.priors, llm,
                               search_order=bool(cfg.get("search_order", True)))
                        if llm is not None else None)
        self._consolidator = (
            # priors is passed for D62 — the observed object vocabulary that keeps
            # hallucinated entities (cdromdrive, cdplayer) out of the generated index.
            Consolidator(self.pages, self.evidence, llm, priors=self.priors)
            if llm is not None else None
        )

        # Per-episode cache — see the class docstring. Keyed by task so a new episode
        # invalidates it even if reset() is somehow missed.
        self._cached_task: str | None = None
        self._cached_ctx: MemoryContext | None = None

        # Reported telemetry.
        # Restored from the JOURNAL, not from the consolidation state: the state starts at
        # -1, and up to K-1 episodes can sit journalled after the last pass. Reading it
        # from the state would renumber them on resume and corrupt `since()`.
        self._episode_index = max(
            (int(r.get("index", 0)) for r in self.journal if r.get("kind") == "episode"),
            default=0,
        )
        self._n_read_calls = 0
        self._n_read_errors = 0              # read-path model calls that raised (H13)
        self._n_consolidation_calls = 0
        # Episodes this PHASE has seen. _episode_index cannot serve as the per-episode
        # divisor: it is restored from the journal (500 after training) and update() returns
        # early once freeze() is set, so at eval it stays at 500 while the call counters
        # count 134 episodes. That reported eval cost as 268/500 = 0.536 calls/ep when the
        # true rate was 268/134 = 2.0 — understating the wiki's eval spend 3.7x.
        self._episodes_seen = 0
        self._n_consolidations = 0
        self.last_update_report: dict = {}

        # Documents seeded before episode 1 (Part XV handbook mode). Idempotent: seeding
        # twice must not double-count, because a resumed run re-enters __init__.
        docs = cfg.get("documents")
        if docs and "document" in self.sources:
            self._seed_documents(Path(docs))

    # --------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return "wiki_v3"

    # -------------------------------------------------------------- documents
    def _seed_documents(self, doc_dir: Path) -> None:
        """D57 — a document is a source, and enters by the path an episode does."""
        if not doc_dir.exists():
            log.warning("[wiki_v3] documents dir not found: %s", doc_dir)
            return
        already = {
            r.get("doc_id") for r in self.journal if r.get("kind") == "document"
        }
        n = 0
        for path in sorted(doc_dir.glob("*.md")):
            if path.stem in already:
                continue
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
            self.journal.append(
                document_record(0, doc_id=path.stem, title=title, text=text)
            )
            n += 1
        if n:
            log.info("[wiki_v3] seeded %d documents from %s", n, doc_dir)

    def ingest_document(self, doc_id: str, title: str, text: str, supersedes=None) -> None:
        """Mid-run document arrival (Part XV §114 update events).

        Journalled at the current episode index, so it is picked up by the NEXT
        consolidation. That is what gives the supersession curve its staircase shape — an
        architectural property (O16), to be reported as one rather than smoothed over.
        """
        self.journal.append(
            document_record(
                self._episode_index,
                doc_id=doc_id,
                title=title,
                text=text,
                supersedes=supersedes,
            )
        )

    # --------------------------------------------------------------- retrieve
    def retrieve(
        self,
        task_description: str,
        observation: str,
        admissible_commands: list[str],
        step: int,
    ) -> MemoryContext:
        # D20 — one retrieval per episode. See the class docstring.
        if step > 0 and self._cached_ctx is not None and self._cached_task == task_description:
            return self._cached_ctx

        # A cache miss is an episode boundary (D20: one retrieval per episode), and it is
        # the only per-episode signal available during a frozen eval phase.
        self._episodes_seen += 1

        if self._reader is None:
            return MemoryContext()

        family = classify_task_family(task_description)
        res = self._reader.read(task_description, observation, family)
        self._n_read_calls += res.n_llm_calls
        self._n_read_errors += res.errors

        ctx = MemoryContext(
            system_context=res.briefing,
            metadata={
                "source": "wiki_v3",
                "pages_read": list(res.pages_read),
                "abstained": res.abstained,
                "retrieval_fallback": res.fallback,
                "shortlist": res.shortlist,
                "n_llm_calls": res.n_llm_calls,
                "reason": res.reason,
            },
        )
        self._cached_task = task_description
        self._cached_ctx = ctx
        return ctx

    # ------------------------------------------------------------------ update
    def update(self, episode: EpisodeResult) -> None:
        """Per episode: three deterministic writes, ZERO LLM. Then maybe consolidate."""
        self._cached_task = None
        self._cached_ctx = None

        if self._read_only:
            return

        self._episode_index += 1
        idx = self._episode_index

        self.journal.append(
            episode_record(
                idx,
                task=episode.task_description,
                request_shown=episode.task_description,
                family=episode.task_family,
                trajectory=episode.trajectory,
                success=episode.success,
                steps=episode.steps_taken,
                briefing_served=episode.memory_context or "",
                scene_id=episode.scene_id,
                task_id=episode.task_id,
            )
        )

        # D08 — the two-channel asymmetry, and the only gating outside consolidation.
        n_obs = self.priors.ingest_episode(episode.trajectory)      # UNGATED
        n_act = self.evidence.ingest_episode(                       # SUCCESS-GATED
            episode.trajectory, episode.success, idx
        )
        self.priors.save()
        self.evidence.save()

        self.last_update_report = {
            "episode_index": idx,
            "observations_recorded": n_obs,
            "actions_attested": n_act,
            "consolidated": False,
        }

        # max(..., 0): the state starts at -1 so that seeded documents (index 0) are
        # reachable, but episode COUNTING must start from 0 or the first pass fires one
        # episode early and every subsequent interval is off by one.
        since_last = idx - max(self.state.last_consolidated_index, 0)
        if since_last >= self.consolidate_every:
            self.consolidate()

    # ------------------------------------------------------------- consolidate
    def consolidate(self) -> dict:
        """The compilation pass (D51). Every K episodes, not every episode."""
        if self._consolidator is None or self._read_only:
            return {}

        batch = self.journal.since(self.state.last_consolidated_index)
        if not batch:
            return {}

        log.info(
            "[wiki_v3] consolidation %d: %d sources (episodes %d..%d)",
            self.state.pass_count + 1, len(batch),
            self.state.last_consolidated_index + 1, self._episode_index,
        )
        report = self._consolidator.run(
            batch, git_dir=self.wiki_dir if self.git_enabled else None
        )
        self._n_consolidation_calls += report.n_llm_calls
        self._n_consolidations += 1
        self.state.advance(self._episode_index)

        payload = report.as_dict()
        payload["pass"] = self.state.pass_count
        payload["store"] = self.pages.stats()
        (self.wiki_dir / "consolidations.jsonl").open("a", encoding="utf-8").write(
            json.dumps(payload) + "\n"
        )
        self._log_pass_detail(report)
        self.last_update_report = {**self.last_update_report, "consolidated": True, **payload}
        log.info(
            "[wiki_v3] consolidation %d done: %d calls, %d pages written, %d claims rejected",
            self.state.pass_count, report.n_llm_calls,
            len(report.written), len(report.rejected_claims),
        )
        return payload

    def _log_pass_detail(self, report) -> None:
        """Logging only (W9): raw outputs, every removed line, and a copy of the pages.

        consolidations.jsonl keeps counts; this keeps the text, so what a gate rejected
        and what the word budget cut can be read back after the run.
        """
        try:
            detail = {"pass": self.state.pass_count, **report.detail()}
            with (self.wiki_dir / "consolidation_detail.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail) + "\n")
            snap = self.wiki_dir / "pages_history" / f"pass_{self.state.pass_count:03d}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(self.pages.root, snap)
        except Exception:
            log.warning("[wiki_v3] could not write the consolidation detail log", exc_info=True)

    # -------------------------------------------------------------- lifecycle
    def finalize(self) -> None:
        """Flush the tail of the chain.

        Without this, up to K-1 episodes of experience would never be compiled — the store
        would be missing the end of training entirely.
        """
        if self._read_only:
            return
        if self._episode_index > self.state.last_consolidated_index:
            self.consolidate()

    def reset(self) -> None:
        self._cached_task = None
        self._cached_ctx = None

    # -------------------------------------------------------------- telemetry
    def stats(self) -> dict:
        # Divide by episodes seen in THIS phase, not by _episode_index — see
        # _episodes_seen in __init__ for why the two differ at eval.
        eps = max(self._episodes_seen, 1)
        return {
            "episodes": self._episode_index,
            "episodes_this_phase": self._episodes_seen,
            "consolidations": self._n_consolidations,
            "read_llm_calls": self._n_read_calls,
            "read_errors": self._n_read_errors,
            "consolidation_llm_calls": self._n_consolidation_calls,
            "memory_llm_calls_per_episode": round(
                (self._n_read_calls + self._n_consolidation_calls) / eps, 3
            ),
            "priors_objects": self.priors.n_objects,
            "priors_observations": self.priors.n_observations,
            "evidence_forms": self.evidence.n_forms,
            **self.pages.stats(),
        }

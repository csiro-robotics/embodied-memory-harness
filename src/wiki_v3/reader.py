"""The read path — exactly two bounded LLM calls (D54).

v2 ran a 40-round agentic tool loop per episode. cochran's H3 was refuted at 21x the wrong
direction *because* its wiki ran a 30-turn list_pages/read_page loop at query time, and the
preregistered break-even formula returned a negative N — encoding "no crossover exists".
Two calls is what keeps the amortisation question (O10, Part XV P4) answerable at all.

This is not a retrieval downgrade. The selection work moved to consolidation time: the
index IS the retrieval decision, made once over the whole store instead of re-derived every
episode from a changed context. D19 (no embeddings) and D20 (one retrieval, at episode
start) are unchanged and cheaper to honour.

The location shortlist stays DETERMINISTIC (D17). The LLM never ranks locations — it is
handed an ordered list of concrete receptacle instances and told to repeat it verbatim.
That mechanism is the largest single measured gain in the project and it is not being
delegated to a 14B.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from wiki_v3 import prompts
from wiki_v3.digest import target_object
from wiki_v3.layers import PriorsStore
from wiki_v3.store import PageStore

log = logging.getLogger(__name__)

MAX_PAGES = 3
BRIEFING_MAX_WORDS = 90
_NONE = re.compile(r"^\s*none\s*\.?\s*$", re.IGNORECASE)


@dataclass
class ReadResult:
    briefing: str = ""
    pages_read: list[str] = field(default_factory=list)
    shortlist: list[str] = field(default_factory=list)
    n_llm_calls: int = 0
    abstained: bool = False
    fallback: bool = False       # call 1 parsed to nothing; deterministic index used
    errors: int = 0              # model calls that raised (H13); the read carried on
    reason: str = ""


def _parse_selection(text: str, valid: list[str]) -> list[str]:
    """Names only. Tolerant of bullets, numbering, backticks and a stray '.md'.

    Anything that does not resolve to a real page is dropped rather than guessed at — the
    reader must never fail open into reading everything.
    """
    if not text or _NONE.match(text.strip()):
        return []
    valid_set = set(valid)
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line)
        line = line.removesuffix(".md").strip().lower()
        if not line or _NONE.match(line):
            continue
        if line in valid_set and line not in out:
            out.append(line)
            if len(out) >= MAX_PAGES:
                break
    return out


def _trim_words(text: str, limit: int) -> str:
    """Enforce the payload budget by code (D16).

    The A-K payload-length curve ran 1062 chars -> 32.9% and 215 chars -> 58.6%, so an
    over-long briefing is not a cosmetic problem. Trim at a sentence boundary where one is
    available so the last claim is not left half-stated.
    """
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    cut = " ".join(words[:limit])
    for stop in (". ", "; "):
        if stop in cut:
            head, _sep, _tail = cut.rpartition(stop)
            if len(head.split()) >= limit // 2:
                return (head + ".").strip()
    return cut.strip() + "..."


def _format_shortlist(names: list[str]) -> str:
    if not names:
        return ""
    ranked = ", ".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    return f"SEARCH ORDER (already ranked cheapest-first — repeat these names exactly):\n{ranked}"


class Reader:
    """Two calls: SELECT, then COMPOSE. Page bodies are read by code in between."""

    def __init__(self, pages: PageStore, priors: PriorsStore, llm, search_order: bool = True) -> None:
        self._pages = pages
        self._priors = priors
        self._llm = llm
        # Ablation switch (paper Section 7.1): False drops the ranked search order, so the
        # briefing gets no location ranking from the priors. Pages are read as usual.
        self._search_order = search_order

    def _required_pages(self, family: str) -> list[str]:
        """Pages always read for `family`: its own page once it says something, and any
        page generated from attested evidence that applies to it (the affordance page)."""
        out: list[str] = []
        if not family:
            return out
        own = self._pages.read(family.replace("_", "-"))
        if own is not None and own.confidence != "unestablished":
            out.append(own.name)
        # W19: "generated from attested evidence" means built by code from evidence.json. The
        # model writes `confidence: attested` on its own pages too, and those were forced into
        # every read of their families (in the final runs r3's "search-exhaustion" page, which
        # says to stop searching, was always read). Only code-built pages carry that source.
        for pg in self._pages.all_pages():
            if (pg.confidence == "attested" and pg.sources == ["evidence.json"]
                    and family in pg.applies_to and pg.name not in out):
                out.append(pg.name)
        return out

    def read(self, task: str, observation: str, family: str) -> ReadResult:
        res = ReadResult()
        obj = target_object(task)
        res.shortlist = (self._priors.shortlist(obj, observation, k=MAX_PAGES)
                         if self._search_order else [])

        available = self._pages.names()

        # ------------------------------------------------------------ abstain
        # D18: an empty briefing is a legitimate output. On examine_in_light a KG emitting
        # nothing scored 94% against the wiki's 83% — serving nothing beat serving noise.
        # With nothing in the store and nothing in the priors there is nothing to serve,
        # and spending two LLM calls to discover that would be waste.
        if not available and not res.shortlist:
            res.abstained = True
            res.reason = "empty store and no priors for the target object"
            return res

        # --------------------------------------------------- call 1 — SELECT
        picks: list[str] = []
        if available:
            try:
                out = self._llm.complete(
                    system=prompts.SELECT_SYSTEM,
                    user=prompts.SELECT_USER.format(
                        task=task,
                        family=family or "unknown",
                        observation=(observation or "").split("Your task", 1)[0].strip(),
                        descriptions=self._pages.descriptions(),
                    ),
                    max_tokens=128,
                )
                res.n_llm_calls += 1
                picks = _parse_selection(out, available)
            except Exception:
                res.errors += 1
                log.exception("[wiki_v3] select call failed; falling back to index lookup")

            if not picks:
                # Deterministic fallback via the generated index (store.candidates_for).
                # Not an error path: if the model says NONE for a family with pages, the
                # index still knows which pages claim that family.
                picks = self._pages.candidates_for(family, obj)[:MAX_PAGES]
                res.fallback = True

        # W11: the family's own page and the page of attested tools are read whenever they
        # exist and apply, and the model's picks fill the remaining slots. The family was
        # never shown to SELECT, and generic pages took the slots, so the page that states
        # the transform's tool was often not read.
        picks = self._required_pages(family) + [p for p in picks
                                                if p not in self._required_pages(family)]
        picks = picks[:MAX_PAGES]
        res.pages_read = picks
        # W3: the parsed body under the page name, never the raw file. Serving the raw file
        # passed header blocks, a 12-entry sources list and stray fields to the compose call.
        bodies = [b for b in (self._pages.read_body(n) for n in picks) if b]

        if not bodies and not res.shortlist:
            res.abstained = True
            res.reason = "no page selected and no priors for the target object"
            return res

        # -------------------------------------------------- call 2 — COMPOSE
        try:
            out = self._llm.complete(
                system=prompts.COMPOSE_SYSTEM,
                user=prompts.COMPOSE_USER.format(
                    task=task,
                    observation=(observation or "").split("Your task", 1)[0].strip(),
                    search_order=_format_shortlist(res.shortlist),
                    pages="\n\n".join(bodies) if bodies else "(no pages retrieved)",
                ),
                max_tokens=512,
            )
            res.n_llm_calls += 1
        except Exception:
            res.errors += 1
            log.exception("[wiki_v3] compose call failed; serving the shortlist alone")
            out = ""

        briefing = (out or "").strip()
        if not briefing or _NONE.match(briefing):
            # The model declined. If we hold a deterministic shortlist, serve THAT rather
            # than nothing — it is the component that carries the measured signal, and it
            # needs no LLM to be correct.
            if res.shortlist:
                res.briefing = _format_shortlist(res.shortlist)
                res.reason = "compose declined; served deterministic shortlist"
                return res
            res.abstained = True
            res.reason = "compose returned NONE and no shortlist available"
            return res

        res.briefing = _trim_words(briefing, BRIEFING_MAX_WORDS)
        return res

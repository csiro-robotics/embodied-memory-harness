"""Consolidation — the compilation pass. This is the wiki (D51).

Runs every K episodes, not every episode. Two reasons, and the second is the one that
matters:

  * COST. v2 paid an LLM curation pass per episode, which is why O7 ("curation cost does
    not amortise") exists — the amortisation was designed out. Here the write path costs
    1/K as often.
  * CROSS-SYNTHESIS. A curator shown one episode can relate it to the store. It can never
    notice a regularity that only appears across twenty-five episodes. That is the wiki's
    one structural advantage — cochran measures it at +6.625 on inter_paper_mapping,
    P(mu>=2)=0.967, the only strong result in that paper — and v2's write path made it
    impossible by construction.

Stages: ANALYSE (1 call) -> REWRITE (1 call per affected page) -> INDEX (deterministic)
-> housekeeping. Both gates apply at write time and are unchanged.

Note the deliberate departure from the architecture spec: index generation is
deterministic (store.build_index) rather than a third LLM call. The index is a pure
function of the pages, so generating it removes a parse-failure mode and makes it
impossible for the index to disagree with what it indexes.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from wiki_v3 import prompts
from wiki_v3.digest import batch_stats, digest_batch
from wiki_v3.gates import claim_gate, enforce_word_budget, outcome_gate_claims
from wiki_v3.journal import EPISODE
from wiki_v3.layers import EvidenceStore
from wiki_v3.store import DEFAULT_TYPE, WORD_BUDGET, Page, PageStore, parse_page, valid_page_name

log = logging.getLogger(__name__)

MAX_LESSONS = 6
# NOTE [ \t]* rather than \s*: \s matches newlines, so on a mangled block whose PAGE:
# line is empty the engine swallows the newline and reads the NEXT label as the page
# name, inventing a page called "type-procedure". Horizontal whitespace only.
_BLOCK = re.compile(
    r"PAGE:[ \t]*(?P<page>[^\n]+)\n"
    r"(?:[ \t]*TYPE:[ \t]*(?P<type>[^\n]*)\n)?"
    r"(?:[ \t]*FAMILIES:[ \t]*(?P<families>[^\n]*)\n)?"
    r"(?:[ \t]*OBJECTS:[ \t]*(?P<objects>[^\n]*)\n)?"
    r"[ \t]*LESSON:[ \t]*(?P<lesson>[^\n]+)\n"
    r"(?:[ \t]*EVIDENCE:[ \t]*(?P<evidence>[^\n]*))?",
    re.IGNORECASE,
)


@dataclass
class Finding:
    page: str
    type: str = DEFAULT_TYPE
    families: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    lesson: str = ""
    evidence: str = ""


@dataclass
class ConsolidationReport:
    n_sources: int = 0
    n_llm_calls: int = 0
    findings: list[Finding] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    rejected_outcome: list[tuple[str, str]] = field(default_factory=list)
    rejected_claims: list[tuple[str, str]] = field(default_factory=list)
    stubs_created: list[str] = field(default_factory=list)   # D64
    affordances: list[str] = field(default_factory=list)     # D60
    lint: list[str] = field(default_factory=list)
    # Logging only (W9): the raw model outputs and what each gate or cut removed. Written
    # to consolidation_detail.jsonl by the arm; as_dict() below stays compact.
    raw_analyse: str = ""
    rewrites: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_sources": self.n_sources,
            "n_llm_calls": self.n_llm_calls,
            "n_findings": len(self.findings),
            "written": self.written,
            "n_rejected_outcome": len(self.rejected_outcome),
            "n_rejected_claims": len(self.rejected_claims),
            "stubs_created": self.stubs_created,
            "affordances": self.affordances,
            "rejected_claims": self.rejected_claims[:10],
            "lint": self.lint,
        }

    def detail(self) -> dict:
        """Everything needed to see what this pass did and why (W9)."""
        return {
            "n_sources": self.n_sources,
            "n_llm_calls": self.n_llm_calls,
            "raw_analyse": self.raw_analyse,
            "findings": [
                {"page": f.page, "type": f.type, "families": f.families,
                 "objects": f.objects, "lesson": f.lesson, "evidence": f.evidence}
                for f in self.findings
            ],
            "rewrites": self.rewrites,
            "rejected_outcome": self.rejected_outcome,
            "rejected_claims": self.rejected_claims,
            "written": self.written,
            "stubs_created": self.stubs_created,
            "affordances": self.affordances,
            "lint": self.lint,
        }


# D63 — page-name shapes that describe a failure mode rather than a way to succeed.
_POSTMORTEM_NAME = re.compile(
    r"(confusion|exhaustion|failure|issue|problem|mistake|pitfall|cap\b|limit|"
    r"uncertainty|challenge|optimization|optimisation)",
    re.IGNORECASE,
)

# W13: no cap. The cap of 12 kept only the last batch, so the audit trail could not show
# which earlier episodes a page came from. Sources live in the header, which is never
# served or shown to the rewrite call (W3, D61), so a long list costs nothing.


def _merge_sources(prior: list[str], batch_ids: list[str]) -> list[str]:
    """D61 — union of what the page already cited and what this batch contributed.

    Order preserved (oldest first) so the chain reads chronologically, and deduplicated.
    """
    seen: set[str] = set()
    out: list[str] = []
    for src in list(prior) + list(batch_ids):
        key = str(src).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _family(name: str) -> str:
    """One spelling for a task family (W1). The analyse call writes families as the page
    names it sees (`pick-heat-then-place`); episode records use ALFWorld's underscores."""
    return re.sub(r"[\s-]+", "_", (name or "").strip().lower())


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    v = value.strip()
    if v.upper() in {"NONE", "-", "N/A", ""}:
        return []
    v = v.strip("[]")
    return [x.strip().lower() for x in v.split(",") if x.strip()]


def _slug(name: str) -> str:
    s = (name or "").strip().strip("`").lower()
    # W18: the model often writes "PAGE: pick-clean-then-place [procedure]", repeating the
    # TYPE in brackets. The bracket used to become part of the name, so the final runs
    # forked 13 family pages into a second "pick-clean-then-place-procedure" page.
    s = re.sub(r"\s*[\[(][^\])]*[\])]\s*$", "", s)
    s = s.removesuffix(".md")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def parse_findings(text: str) -> list[Finding]:
    """Line-based parsing, deliberately. JSON from a 14B is a coin flip; labelled lines
    degrade gracefully — a mangled block is dropped and the rest survive."""
    out: list[Finding] = []
    for m in _BLOCK.finditer(text or ""):
        page = _slug(m.group("page"))
        if not valid_page_name(page):
            continue
        ptype = (m.group("type") or DEFAULT_TYPE).strip().lower()
        if ptype not in WORD_BUDGET:
            ptype = DEFAULT_TYPE
        lesson = (m.group("lesson") or "").strip()
        if not lesson:
            continue
        out.append(
            Finding(
                page=page,
                type=ptype,
                families=[_family(f) for f in _csv(m.group("families"))],
                objects=_csv(m.group("objects")),
                lesson=lesson,
                evidence=(m.group("evidence") or "").strip(),
            )
        )
        if len(out) >= MAX_LESSONS:
            break
    return out


class Consolidator:
    def __init__(self, pages: PageStore, evidence: EvidenceStore, llm, priors=None) -> None:
        self._pages = pages
        self._evidence = evidence
        self._llm = llm
        # D62 — the observed object vocabulary, used to keep hallucinated entities out of
        # the index. Optional so existing callers and tests keep working; None means
        # "no vocabulary available", and nothing is filtered.
        self._priors = priors

    # ------------------------------------------------------------------ main
    def run(self, batch: list[dict], *, git_dir: Path | None = None) -> ConsolidationReport:
        report = ConsolidationReport(n_sources=len(batch))
        if not batch:
            return report

        stats = batch_stats(batch)
        rendered = digest_batch(batch)

        # ------------------------------------------------- stage A — ANALYSE
        try:
            raw = self._llm.complete(
                system=prompts.ANALYSE_SYSTEM,
                user=prompts.ANALYSE_USER.format(
                    n_episodes=stats["n_episodes"],
                    n_success=stats["n_success"],
                    n_documents=stats["n_documents"],
                    families=", ".join(
                        f"{k} {v['ok']}/{v['n']}" for k, v in sorted(stats["families"].items())
                    )
                    or "none",
                    batch=rendered,
                    descriptions=self._pages.descriptions() or "(the wiki is empty)",
                ),
                max_tokens=1536,
            )
            report.n_llm_calls += 1
        except Exception:
            log.exception("[wiki_v3] analyse call failed; consolidation is a no-op this pass")
            return report

        report.raw_analyse = raw
        report.findings = parse_findings(raw)
        if not report.findings:
            log.warning("[wiki_v3] analyse produced no parseable findings")
            self._pages.write_index()
            return report

        # Which families produced a success in THIS batch — the outcome gate's input (D09).
        ok_families = {
            _family(str(r.get("family", "")))
            for r in batch
            if r.get("kind") == EPISODE and r.get("outcome") == "success"
        }
        evidence_shapes = self._evidence.verb_object_forms()

        # D61 — what this batch contributed, for the computed `sources` chain.
        batch_ids = [
            (f"ep{r.get('index')}" if r.get("kind") == EPISODE else f"doc:{r.get('doc_id')}")
            for r in batch
        ]

        # D64 — every family SEEN in this batch gets a page, successful or not. Without this
        # a family that rarely succeeds never gets one at all: pick_cool_then_place went
        # 6/75 in training and ended the v3.0 run with no page in the index.
        for fam in sorted({
            str(r.get("family", "")).lower()
            for r in batch
            if r.get("kind") == EPISODE and r.get("family")
        }):
            if self._pages.ensure_family_stub(fam):
                report.stubs_created.append(fam)

        # ------------------------------------------------- stage B — REWRITE
        for finding in report.findings:
            page = self._rewrite(finding, rendered, report, batch_ids)
            if page is None:
                continue
            entry = report.rewrites[-1]

            # --- OUTCOME GATE (D09, per-claim since D59) ----------------
            # Was: reject the whole page when the batch held no success for its families.
            # That rejected 48 pages against 53 written on run 59348394 and starved the
            # transform families of any page at all. Now it strips only the lines that
            # assert how to succeed, and keeps lines the ENVIRONMENT has already attested.
            # W1: both sides use one spelling, so a batch success for the family counts.
            # A finding with no families used to skip the gate ("FAMILIES: NONE"); it now
            # takes the families the page already claims, and failing that needs a success
            # of any family in the batch.
            existing = self._pages.read(finding.page)
            families = finding.families or [
                _family(f) for f in (existing.applies_to if existing else [])]
            has_success = (any(f in ok_families for f in families) if families
                           else bool(ok_families))
            page, outcome_report = outcome_gate_claims(page, has_success, evidence_shapes)
            for line, why in outcome_report.rejected:
                report.rejected_outcome.append((page.name, f"{line.strip()} — {why}"))
            entry["outcome_gate"] = {"batch_has_success": has_success,
                                     "rejected": [[l.strip(), w] for l, w in outcome_report.rejected]}

            # --- CLAIM GATE (D10/D58) -----------------------------------
            page, gate_report = claim_gate(page, evidence_shapes)
            for line, why in gate_report.rejected:
                report.rejected_claims.append((page.name, f"{line} — {why}"))
            entry["claim_gate"] = {"rejected": [[l, w] for l, w in gate_report.rejected]}
            if not page.body.strip():
                # D64 — an emptied page is not discarded, it becomes unestablished. The page
                # keeps its identity and its sources chain, and `# Not established` is a
                # legitimate thing to serve (D18). Discarding it would recreate the v3.0
                # hole where a family simply had no page.
                page.body = "# Not established\nNothing has been attested for this family yet."
                page.confidence = "unestablished"

            before = page.body
            page = enforce_word_budget(page)
            entry["word_budget"] = {
                "words_before": len(before.split()),
                "words_after": len(page.body.split()),
                "cut": before[len(page.body):].strip() if before.startswith(page.body) else
                       ("(cut mid-line)" if page.body != before else ""),
            }
            entry["written_body"] = page.body
            self._pages.write(page)
            report.written.append(page.name)

        # ------------------------- stage B2 — AFFORDANCES (D60, no LLM)
        self._write_affordance_page(report)

        # --------------------------------------- stage C — INDEX (no LLM)
        # D62 — objects filtered against what was actually observed.
        self._pages.write_index(known_objects=self._known_objects())

        # ------------------------------------- stage D — housekeeping
        report.lint = self._pages.lint()
        if git_dir is not None:
            self._git_commit(git_dir)

        return report

    # ------------------------------------------------------- D60 / D62 helpers
    def _known_objects(self) -> "set[str] | None":
        """The object types actually observed, or None if no priors layer was supplied."""
        if self._priors is None:
            return None
        try:
            return {str(k).lower() for k in (self._priors.to_dict().get("object_locations") or {})}
        except Exception:
            log.debug("could not read observed object vocabulary", exc_info=True)
            return None

    def _write_affordance_page(self, report: "ConsolidationReport") -> None:
        """D60 — the affordance page, emitted from evidence.json with ZERO LLM involvement.

        v3.0 consumed 500 episodes and never recorded that the microwave is what heats
        things: fridge/microwave/sinkbasin appeared in the store only as things to LOCATE.
        That fact is in every transform episode and is the most reusable cross-episode fact
        in the benchmark, so it is now generated rather than hoped for. Generated means it
        cannot hallucinate, cannot drift on rewrite, and needs no gate — the environment
        already gated it on the way in.
        """
        pairs = self._evidence.instrument_forms()
        if not pairs:
            return
        lines = [
            "Tools the environment has been observed to accept for each transformation.",
            "",
        ]
        verbs: set[str] = set()
        for verb, instrument, n in pairs:
            lines.append(f"- to {verb} something, use the {instrument}  ({n} attested forms)")
            verbs.add(verb)
            report.affordances.append(f"{verb}:{instrument}")

        families = sorted({
            fam for fam in (
                "pick_cool_then_place" if "cool" in verbs else "",
                "pick_heat_then_place" if "heat" in verbs else "",
                "pick_clean_then_place" if "clean" in verbs else "",
            ) if fam
        })
        self._pages.write(Page(
            name="affordances",
            description="Which receptacle performs which transformation, from attested actions.",
            type="entity",
            applies_to=families,
            objects=[],
            sources=["evidence.json"],
            confidence="attested",
            body="\n".join(lines),
        ))

    # ------------------------------------------------------------- internals
    def _rewrite(self, finding: Finding, rendered: str, report: ConsolidationReport,
                 batch_ids: list[str] | None = None):
        # D61 — the model sees the BODY only. It used to be handed the page including its
        # frontmatter, echoed that YAML block back and appended its own, so 3 of 9 pages in
        # run 59348394 carried two frontmatter blocks and served the second to the executor
        # as body text.
        existing = self._pages.read(finding.page)
        current = (existing.body if existing else "") or "(this page does not exist yet)"
        batch_ids = batch_ids or []
        lo, hi = WORD_BUDGET.get(finding.type, WORD_BUDGET[DEFAULT_TYPE])
        try:
            raw = self._llm.complete(
                system=prompts.REWRITE_SYSTEM.format(budget_lo=lo, budget_hi=hi),
                user=prompts.REWRITE_USER.format(
                    name=finding.page,
                    current=current,
                    findings=f"{finding.lesson}\nEvidence: {finding.evidence}",
                    batch=rendered,
                ),
                max_tokens=1536,
            )
            report.n_llm_calls += 1
        except Exception:
            log.exception("[wiki_v3] rewrite call failed for %s", finding.page)
            report.rewrites.append({"page": finding.page, "raw": None, "kept": False,
                                    "reason": "rewrite call failed"})
            return None
        report.rewrites.append({"page": finding.page, "raw": raw, "kept": True})

        page = parse_page(raw, fallback_name=finding.page)

        # The model is asked to echo the name; it will sometimes not. The finding's name is
        # authoritative — otherwise a hallucinated name creates an orphan page.
        page.name = finding.page
        if page.type not in WORD_BUDGET:
            page.type = finding.type
        # Fall back to the finding, then to what the page already claimed. The second
        # fallback matters for D64 stubs: a stub knows its family, and losing that on the
        # first rewrite makes the page unreachable through the index.
        if not page.applies_to:
            page.applies_to = finding.families or (list(existing.applies_to) if existing else [])
        if not page.objects:
            page.objects = finding.objects or (list(existing.objects) if existing else [])
        if not page.description:
            page.description = finding.lesson

        # D63 — a page whose content is a failure post-mortem is a CAUTION, not a procedure.
        # 8 of 9 pages in run 59348394 were typed `procedure` while `object-confusion`,
        # `search-exhaustion`, `step-cap` and `task-optimization` were diagnoses of what went
        # wrong. Mistyping put them under the outcome gate for no benefit.
        if page.type == "procedure" and _POSTMORTEM_NAME.search(page.name):
            page.type = "caution"

        # D61 — `sources` is computed, never parsed. The model was writing this line and got
        # it wrong in both directions: 3 of 9 pages carried `sources: []`, and rewritten
        # pages listed only the newest batch (ep487-500), destroying the audit chain D03/D05
        # rest on. The union of what the page already cited and what this batch contributed
        # is a fact about the store, not a judgement, so code owns it.
        prior_sources = list(existing.sources) if existing else []
        page.sources = _merge_sources(prior_sources, batch_ids)

        if not page.body.strip():
            report.rewrites[-1].update(kept=False, reason="empty body after parsing")
            return None
        return page

    @staticmethod
    def _git_commit(git_dir: Path) -> None:
        """D06 as amended: commit per CONSOLIDATION, not per episode.

        Per-episode commits made sense when there was an LLM pass per episode. There is
        nothing to commit after an episode now — the per-episode writes are append-only and
        deterministic. Failures are logged and swallowed: no run should die because git is
        unavailable on a compute node.
        """
        try:
            subprocess.run(["git", "add", "-A"], cwd=git_dir, check=False, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "wiki_v3 consolidation", "--allow-empty"],
                cwd=git_dir,
                check=False,
                capture_output=True,
            )
        except Exception:
            log.debug("[wiki_v3] git commit skipped", exc_info=True)

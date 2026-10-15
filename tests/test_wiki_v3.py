"""Tests for LLM-Wiki v3. The test plan came from the internal design doc.

Every test here corresponds to a row of that table, or to a decision that would be
silently violated if it regressed. The deterministic layers are tested with no LLM at all;
the LLM-driven paths use a scripted fake so the parsing and the gates are tested rather
than the model.
"""
from __future__ import annotations

import json

import pytest

from memory.wiki_v3_memory import WikiV3Memory
from wiki_v3.consolidator import Consolidator, parse_findings
from wiki_v3.digest import digest_batch, digest_episode
from wiki_v3.gates import claim_gate, enforce_word_budget, outcome_gate
from wiki_v3.journal import ConsolidationState, Journal, document_record, episode_record
from wiki_v3.layers import EvidenceStore, PriorsStore
from wiki_v3.consolidator import _POSTMORTEM_NAME, _merge_sources
from wiki_v3.gates import outcome_gate_claims
from wiki_v3.reader import Reader, _parse_selection, _trim_words
from wiki_v3.store import Page, PageStore, parse_page


# --------------------------------------------------------------------- helpers

def _traj_success():
    return [
        {"step": 0, "action": "go to countertop 1",
         "observation": "You arrive at loc 1. On the countertop 1, you see a tomato 1."},
        {"step": 1, "action": "go to countertop 3",
         "observation": "You arrive at loc 4. On the countertop 3, you see a apple 2, a mug 1."},
        {"step": 2, "action": "take apple 2 from countertop 3",
         "observation": "You pick up the apple 2 from the countertop 3."},
        {"step": 3, "action": "cool apple 2 with fridge 1",
         "observation": "You cool the apple 2 using the fridge 1."},
        {"step": 4, "action": "put apple 2 in/on countertop 1",
         "observation": "You put the apple 2 in/on the countertop 1."},
    ]


def _traj_failure():
    return [
        {"step": 0, "action": "go to towelholder 1",
         "observation": "You arrive at loc 2. On the towelholder 1, you see a handtowel 1."},
        {"step": 1, "action": "take handtowel 1 from towelholder 1",
         "observation": "You pick up the handtowel 1 from the towelholder 1."},
        {"step": 2, "action": "go to cabinet 4",
         "observation": "You arrive at loc 9. On the cabinet 4, you see nothing."},
    ]


def _ep(index, *, success=True, family="pick_cool_then_place",
        task="cool some apple and put it in countertop", traj=None, briefing=""):
    return episode_record(
        index,
        task=task,
        request_shown=task,
        family=family,
        trajectory=traj if traj is not None else _traj_success(),
        success=success,
        steps=len(traj if traj is not None else _traj_success()),
        briefing_served=briefing,
    )


class FakeLLM:
    """Returns scripted responses in order; records every prompt for assertion."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, max_tokens=None, **kw):
        self.calls.append({"system": system, "user": user})
        return self.responses.pop(0) if self.responses else ""


# ================================================================== journal

def test_journal_append_only_and_roundtrip(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    j.append(_ep(1))
    j.append(_ep(2, success=False, traj=_traj_failure()))
    j.append(document_record(0, doc_id="h1", title="Handbook", text="Fruit is served chilled."))

    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert [json.loads(x)["kind"] for x in lines] == ["episode", "episode", "document"]

    # Appending again must not rewrite anything already on disk.
    first = lines[0]
    j.append(_ep(3))
    assert (tmp_path / "journal.jsonl").read_text(encoding="utf-8").split("\n")[0] == first


def test_journal_since_returns_only_new_sources(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.append(document_record(0, doc_id="d", title="t", text="x"))
    for i in range(1, 6):
        j.append(_ep(i))

    # The first pass runs with last_consolidated_index = -1, which is the only way a
    # document seeded at index 0 is ever reachable. Regression guard: with a 0 default a
    # handbook corpus sits in the journal forever and Part XV silently measures nothing.
    assert len(j.since(-1)) == 6
    assert any(r["kind"] == "document" for r in j.since(-1))
    # After that pass, index 0 is behind us and only new episodes are returned.
    assert len(j.since(0)) == 5
    assert len(j.since(2)) == 3


def test_consolidation_state_survives_reload(tmp_path):
    p = tmp_path / "state.json"
    s = ConsolidationState(p)
    s.advance(25)
    assert ConsolidationState(p).last_consolidated_index == 25
    assert ConsolidationState(p).pass_count == 1


# =================================================================== layers

def test_two_channel_intake_asymmetry(tmp_path):
    """D08 — the paper's contribution. Observations ungated, actions success-gated."""
    priors = PriorsStore(tmp_path / "p.json")
    evidence = EvidenceStore(tmp_path / "e.json")

    # A FAILED episode still contributes observations...
    n_obs = priors.ingest_episode(_traj_failure())
    n_act = evidence.ingest_episode(_traj_failure(), success=False, episode_index=1)
    assert n_obs > 0, "observations must be ungated"
    assert n_act == 0, "actions from a failed episode must not be attested"

    # ...and a successful one contributes both.
    assert priors.ingest_episode(_traj_success()) > 0
    assert evidence.ingest_episode(_traj_success(), success=True, episode_index=2) > 0
    assert "cool apple 2 with fridge 1" in evidence.forms()


def test_priors_regenerable_from_journal(tmp_path):
    """D03/D05 — delete everything but the journal, replay, get the same layers back."""
    j = Journal(tmp_path / "j.jsonl")
    records = [_ep(1), _ep(2, success=False, traj=_traj_failure()), _ep(3)]
    for r in records:
        j.append(r)

    def rebuild(tag):
        pr = PriorsStore(tmp_path / f"p_{tag}.json")
        ev = EvidenceStore(tmp_path / f"e_{tag}.json")
        for rec in j.episodes():
            pr.ingest_episode(rec["trajectory"])
            ev.ingest_episode(rec["trajectory"], rec["outcome"] == "success", rec["index"])
        return pr.to_dict(), ev.to_dict()

    assert rebuild("a") == rebuild("b")


def test_shortlist_names_instances_not_types(tmp_path):
    """D17 — the largest single measured gain. Must stay deterministic."""
    priors = PriorsStore(tmp_path / "p.json")
    for _ in range(10):
        priors.ingest_episode(_traj_success())
    obs = "You are in the middle of a room. You see a countertop 1, a countertop 3, a fridge 1."
    got = priors.shortlist("apple", obs, k=3)
    assert got, "priors held apple observations but produced no shortlist"
    assert all(any(ch.isdigit() for ch in name) for name in got), \
        f"shortlist must name INSTANCES, got {got}"


# =================================================================== digest

def test_digest_is_deterministic_and_compact():
    """D55 — no LLM, and small enough that 25 of them fit in one prompt."""
    rec = _ep(47, briefing="check countertop 3, countertop 1")
    a, b = digest_episode(rec), digest_episode(rec)
    assert a == b
    assert len(a.splitlines()) <= 8, a
    assert "ep 47" in a and "SUCCESS" in a


def test_digest_names_the_failure_class_and_wrong_object():
    rec = _ep(51, success=False, family="pick_clean_then_place",
              task="clean some cloth and put it in cabinet", traj=_traj_failure())
    d = digest_episode(rec)
    assert "handtowel" in d, "wrong-object pickup is the largest named failure class"
    assert "failure class" in d


def test_digest_batch_puts_documents_first():
    batch = [_ep(1), document_record(0, doc_id="h", title="Handbook", text="Fruit is chilled.")]
    out = digest_batch(batch)
    assert out.index("=== DOCUMENTS ===") < out.index("=== EPISODES ===")


# ==================================================================== store

def test_page_roundtrip_and_index_generation(tmp_path):
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="cool-before-placement", description="Cool then place.",
                  type="procedure", applies_to=["pick_cool_then_place"], objects=["apple"],
                  body="# Steps\n1. Take the object.\n2. Cool it.\n"))
    got = ps.read("cool-before-placement")
    assert got.applies_to == ["pick_cool_then_place"]
    assert got.objects == ["apple"]

    index = ps.write_index()
    assert "pick_cool_then_place -> cool-before-placement" in index
    # D18 — abstention is structural: a family with no page is listed as a known unknown.
    assert "Known unknowns" in index
    assert "examine_in_light" in index


def test_parse_page_tolerates_missing_frontmatter():
    """A 14B will sometimes omit frontmatter. Dropping the page would lose the lesson."""
    pg = parse_page("# Steps\n1. Do the thing.", fallback_name="my-page")
    assert pg.name == "my-page"
    assert "Do the thing" in pg.body


def test_lint_flags_unreachable_and_overlong_pages(tmp_path):
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="orphan", description="d", type="caution", body="short body here"))
    problems = " ".join(ps.lint())
    assert "unreachable" in problems


# ==================================================================== gates

def test_outcome_gate_blocks_procedure_without_success():
    """D09 — a failure may produce a caution, never a procedure."""
    proc = Page(name="p", type="procedure", body="1. take the apple")
    assert outcome_gate(proc, batch_has_success_for=False)[0] is False
    assert outcome_gate(proc, batch_has_success_for=True)[0] is True
    # A caution is always allowed — that is what a failure is for.
    assert outcome_gate(Page(name="c", type="caution", body="x"), False)[0] is True


def test_claim_gate_rejects_the_canonical_steps_failure():
    """D10 — the exact incident that motivated the gate.

    A five-step cooling routine asserted while `cool cup 1 with fridge 1` sat in 20
    successes. The outcome gate structurally cannot see this: the episode SUCCEEDED.
    """
    evidence = {"cool cup": {"cool cup with fridge"}}
    page = Page(
        name="cool-before-placement", type="procedure",
        body=(
            "# Steps\n"
            "1. Open the fridge 1.\n"
            "2. Put the cup 1 in the fridge 1.\n"
            "3. Cool the cup with the microwave.\n"
            "4. Take the cup 1 from the fridge 1.\n"
        ),
    )
    page, report = claim_gate(page, evidence)
    assert report.n_rejected >= 1
    assert "microwave" not in page.body, "the contradicting line must be removed"
    # The rest of the page survives — rejecting it wholesale would discard correct lessons.
    assert "Open the fridge" in page.body


def test_claim_gate_allows_unattested_pairs():
    """Silence is not evidence of impossibility (ALFWorld.md affordance caveat)."""
    page = Page(name="p", type="procedure", body="1. Slice the bread with the knife.")
    page, report = claim_gate(page, {"cool cup": {"cool cup with fridge"}})
    assert report.n_rejected == 0
    assert "Slice the bread" in page.body


def test_claim_gate_applies_to_document_sourced_claims():
    """D58 — the environment outranks documents too, not just prose."""
    page = Page(
        name="handbook-cooling", type="procedure",
        sources=["doc handbook-p4"],
        body="1. Cool the cup with the icebox.\n",
    )
    page, report = claim_gate(page, {"cool cup": {"cool cup with fridge"}})
    assert report.n_rejected == 1, "a document does not outrank observed evidence"


def test_word_budget_is_enforced():
    """D25 — wicer: an LLM told to compress over-fires by 2-3x and does not report it."""
    page = Page(name="p", type="caution", body="\n".join(f"word{i} " * 20 for i in range(20)))
    before = page.word_count()
    page = enforce_word_budget(page)
    assert before > 120 and page.word_count() <= 120


# =================================================================== reader

def test_reader_uses_exactly_two_llm_calls(tmp_path):
    """D54 — the whole cost argument. v2 budgeted 40 tool rounds here."""
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="cool-before-placement", description="Cool then place.",
                  type="procedure", applies_to=["pick_cool_then_place"], body="1. Cool it."))
    ps.write_index()
    priors = PriorsStore(tmp_path / "p.json")
    priors.ingest_episode(_traj_success())

    llm = FakeLLM(["cool-before-placement", "Take the apple, cool it, then place it."])
    res = Reader(ps, priors, llm).read(
        "cool some apple and put it in countertop",
        "You see a countertop 1, a countertop 3, a fridge 1.",
        "pick_cool_then_place",
    )
    assert res.n_llm_calls == 2
    assert len(llm.calls) == 2
    assert res.pages_read == ["cool-before-placement"]
    assert res.briefing


def test_reader_abstains_on_empty_store_without_calling_the_llm(tmp_path):
    """D18 — and spending two calls to discover there is nothing to say would be waste."""
    llm = FakeLLM([])
    res = Reader(PageStore(tmp_path / "pages"), PriorsStore(tmp_path / "p.json"), llm).read(
        "examine the book with the desklamp", "You see a desk 1.", "examine_in_light"
    )
    assert res.abstained and res.briefing == "" and res.n_llm_calls == 0
    assert llm.calls == []


def test_reader_falls_back_to_the_index_when_selection_fails(tmp_path):
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="cool-before-placement", description="Cool then place.",
                  type="procedure", applies_to=["pick_cool_then_place"], body="1. Cool it."))
    ps.write_index()
    llm = FakeLLM(["I think you should look at the cooling page maybe", "A briefing."])
    res = Reader(ps, PriorsStore(tmp_path / "p.json"), llm).read(
        "cool some apple and put it in countertop", "You see a fridge 1.",
        "pick_cool_then_place",
    )
    assert res.fallback is True
    assert res.pages_read == ["cool-before-placement"]


def test_reader_serves_the_shortlist_when_compose_declines(tmp_path):
    """The deterministic component carries the measured signal; never drop it."""
    priors = PriorsStore(tmp_path / "p.json")
    for _ in range(5):
        priors.ingest_episode(_traj_success())
    llm = FakeLLM(["NONE", "NONE"])
    res = Reader(PageStore(tmp_path / "pages"), priors, llm).read(
        "cool some apple and put it in countertop",
        "You see a countertop 1, a countertop 3, a fridge 1.",
        "pick_cool_then_place",
    )
    assert not res.abstained
    assert "countertop 3" in res.briefing


def test_reader_without_search_order_sends_no_ranking(tmp_path):
    """The paper's ablation: with search_order off, the priors rank nothing and the
    compose prompt carries an empty search order. Pages are still read."""
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="cool-before-placement", description="Cool then place.",
                  type="procedure", applies_to=["pick_cool_then_place"], body="1. Cool it."))
    ps.write_index()
    priors = PriorsStore(tmp_path / "p.json")
    for _ in range(5):
        priors.ingest_episode(_traj_success())
    llm = FakeLLM(["cool-before-placement", "Take the apple, cool it, then place it."])
    res = Reader(ps, priors, llm, search_order=False).read(
        "cool some apple and put it in countertop",
        "You see a countertop 1, a countertop 3, a fridge 1.",
        "pick_cool_then_place",
    )
    assert res.shortlist == []
    assert res.pages_read == ["cool-before-placement"]
    # the room description still lists countertop 3, so check the ranked entry instead
    assert "1. countertop" not in str(llm.calls[1])


def test_selection_parsing_is_tolerant_but_never_fails_open():
    valid = ["a-page", "b-page"]
    assert _parse_selection("- a-page\n2. b-page", valid) == ["a-page", "b-page"]
    assert _parse_selection("`a-page.md`", valid) == ["a-page"]
    assert _parse_selection("NONE", valid) == []
    assert _parse_selection("nonexistent-page", valid) == []


def test_briefing_is_trimmed_to_budget():
    """D16 — the payload-length curve: 1062 chars -> 32.9%, 215 chars -> 58.6%."""
    long = " ".join(["word"] * 300)
    assert len(_trim_words(long, 90).split()) <= 91


# ============================================================= consolidator

FINDINGS_OUT = """\
PAGE: cool-before-placement
TYPE: procedure
FAMILIES: pick_cool_then_place
OBJECTS: apple
LESSON: Cool the object with the fridge in a single action, then place it.
EVIDENCE: ep 1, ep 3
"""

PAGE_OUT = """\
---
name: cool-before-placement
description: Cool an object then place it; for cool-then-place tasks with fruit.
type: procedure
applies_to: [pick_cool_then_place]
objects: [apple]
sources: [ep 1, ep 3]
confidence: attested
---

# Steps
1. Find and take the exact requested object.
2. Cool it with the fridge in one action.
3. Place it at the destination named in the task.

# Not established
- Whether cooling twice has any effect.
"""


def test_parse_findings_is_robust_to_surrounding_prose():
    out = parse_findings("Here are the lessons I found:\n\n" + FINDINGS_OUT + "\nThat is all.")
    assert len(out) == 1
    assert out[0].page == "cool-before-placement"
    assert out[0].families == ["pick_cool_then_place"]
    assert out[0].objects == ["apple"]


def test_parse_findings_drops_mangled_blocks_and_keeps_good_ones():
    bad = "PAGE: \nTYPE: procedure\nLESSON: nothing\n\n"
    out = parse_findings(bad + FINDINGS_OUT)
    assert [f.page for f in out] == ["cool-before-placement"]


def test_consolidation_writes_pages_and_regenerates_the_index(tmp_path):
    ps = PageStore(tmp_path / "pages")
    ev = EvidenceStore(tmp_path / "e.json")
    ev.ingest_episode(_traj_success(), success=True, episode_index=1)

    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    report = Consolidator(ps, ev, llm).run([_ep(1), _ep(2), _ep(3)])

    assert report.n_llm_calls == 2, "1 analyse + 1 rewrite"
    assert report.written == ["cool-before-placement"]
    assert ps.exists("cool-before-placement")
    assert "pick_cool_then_place -> cool-before-placement" in ps.read_index()


def test_consolidation_strips_unattested_claims_not_the_whole_page(tmp_path):
    """D59. A batch with no success may not assert HOW TO SUCCEED — but it still gets a page.

    This replaces the original page-level assertion. On the v3.0 run rejecting whole pages
    cost 48 of 101 proposals and left `pick_cool_then_place` (6/75 train successes) with no
    page at all, so the gate now strips the offending LINES and keeps the page.
    """
    ps = PageStore(tmp_path / "pages")
    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    report = Consolidator(ps, EvidenceStore(tmp_path / "e.json"), llm).run(
        [_ep(1, success=False, traj=_traj_failure())]
    )
    # The page survives; the unattested how-to-succeed steps inside it do not.
    assert ps.exists("cool-before-placement")
    assert report.rejected_outcome, "unattested procedure steps must still be rejected"


def test_outcome_gate_admits_an_attested_fact_from_a_failed_batch(tmp_path):
    """D59, the half that matters: the environment attests, the outcome does not veto it."""
    ev_shapes = {"cool apple": {"cool apple with fridge"}}
    page = Page(
        name="p", type="procedure",
        body="- cool the apple with the fridge\n- open the fridge and wait ten minutes",
    )
    out, rep = outcome_gate_claims(page, batch_has_success_for=False, evidence_shapes=ev_shapes)
    assert "cool the apple with the fridge" in out.body, "attested shape must survive"
    assert "wait ten minutes" not in out.body, "unattested how-to-succeed must be stripped"
    assert rep.n_rejected == 1


def test_family_stub_is_created_on_first_sighting(tmp_path):
    """D64. Coverage, not success, decides whether a family has a page."""
    ps = PageStore(tmp_path / "pages")
    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    report = Consolidator(ps, EvidenceStore(tmp_path / "e.json"), llm).run(
        [_ep(1, success=False, family="pick_heat_then_place", traj=_traj_failure())]
    )
    assert "pick_heat_then_place" in report.stubs_created
    assert ps.exists("pick-heat-then-place")
    stub = ps.read("pick-heat-then-place")
    assert stub.confidence == "unestablished"
    # A stub must not be reported as a lint problem, or the lint becomes unreadable.
    assert not any("pick-heat-then-place" in issue for issue in ps.lint())


def test_affordance_page_is_generated_from_evidence_without_the_llm(tmp_path):
    """D60. The fact v3.0 never wrote down in 500 episodes."""
    ev = EvidenceStore(tmp_path / "e.json")
    ev.ingest_episode(
        [{"action": "heat egg 1 with microwave 1"}, {"action": "heat mug 2 with microwave 1"},
         {"action": "cool cup 1 with fridge 1"}, {"action": "cool pot 2 with fridge 1"}],
        True, 1,
    )
    ps = PageStore(tmp_path / "pages")
    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    report = Consolidator(ps, ev, llm).run([_ep(1)])

    assert ps.exists("affordances")
    body = ps.read("affordances").body
    assert "microwave" in body and "heat" in body
    assert "fridge" in body and "cool" in body
    assert set(report.affordances) == {"heat:microwave", "cool:fridge"}
    # Generated, not written: no extra model call was spent on it.
    assert report.n_llm_calls == 2, "1 analyse + 1 rewrite, and nothing for affordances"


def test_index_drops_objects_never_actually_observed(tmp_path):
    """D62. `cdromdrive` is not an ALFWorld object type; the model invented it."""
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="p", type="procedure", applies_to=["pick_and_place"],
                  objects=["mug", "cdromdrive"]))
    assert "cdromdrive" in ps.build_index()
    filtered = ps.build_index(known_objects={"mug"})
    assert "cdromdrive" not in filtered
    assert "mug" in filtered


def test_parse_page_repairs_a_doubled_frontmatter_block(tmp_path):
    """D61. 3 of 9 pages in run 59348394 served YAML to the executor as body text."""
    doubled = (
        "---\nname: pick-and-place\ntype: procedure\nsources: []\n---\n\n"
        "---\nname: pick-and-place\ntype: procedure\n---\nThe real body."
    )
    pg = parse_page(doubled)
    assert pg.name == "pick-and-place"
    assert pg.body == "The real body."
    assert not pg.body.lstrip().startswith("---")


def test_sources_are_merged_by_code_not_taken_from_the_model(tmp_path):
    """D61. Rewritten pages listed only the newest batch, destroying the D03 audit chain."""
    assert _merge_sources(["ep1", "ep2"], ["ep2", "ep3"]) == ["ep1", "ep2", "ep3"]
    # W13: not capped, so the chain keeps the earliest sources too.
    many = _merge_sources([f"ep{i}" for i in range(40)], ["ep99"])
    assert len(many) == 41
    assert many[0] == "ep0" and many[-1] == "ep99"


def test_applies_to_is_normalised_to_real_family_names(tmp_path):
    """The v3.1 run's regression: hyphenated families made 4 of 6 pages unroutable.

    D64 names stub pages in kebab-case, the rewrite prompt shows the model the page name,
    and the model echoed `pick-and-place` into `applies_to`. ALFWorld families use
    underscores, so build_index matched nothing and the index read "(nothing recorded)"
    for families that had a perfectly good page sitting beside them.
    """
    pg = parse_page(
        "---\nname: pick-and-place\ntype: procedure\n"
        "applies_to: [pick-and-place, pick-two-obj-and-place, not_a_real_family]\n---\nBody."
    )
    assert pg.applies_to == ["pick_and_place", "pick_two_obj_and_place"]

    ps = PageStore(tmp_path / "pages")
    ps.write(pg)
    assert "pick_and_place -> pick-and-place" in ps.build_index()


def test_stub_keeps_its_family_through_a_rewrite(tmp_path):
    """D64 + D61. A stub that loses applies_to on first rewrite becomes unreachable."""
    ps = PageStore(tmp_path / "pages")
    ps.ensure_family_stub("pick_cool_then_place")
    assert ps.read("pick-cool-then-place").applies_to == ["pick_cool_then_place"]
    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    Consolidator(ps, EvidenceStore(tmp_path / "e.json"), llm).run([_ep(1)])
    assert ps.read("pick-cool-then-place").applies_to == ["pick_cool_then_place"]


def test_postmortem_pages_are_typed_caution(tmp_path):
    """D63. Mistyping a diagnosis as a procedure put it under the outcome gate."""
    for name in ("search-exhaustion", "object-confusion", "step-cap", "task-optimization"):
        assert _POSTMORTEM_NAME.search(name), name
    for name in ("pick-and-place", "affordances", "examine-in-light"):
        assert not _POSTMORTEM_NAME.search(name), name


# ================================================================== adapter

def test_memory_retrieves_once_per_episode_not_once_per_step(tmp_path):
    """D20, and the single easiest way to destroy the cost claim by accident.

    agent.run_episode calls retrieve() at EVERY step. Without the per-episode cache the
    read path would cost 100 LLM calls per episode instead of 2.
    """
    llm = FakeLLM(["cool-before-placement", "A briefing about the apple."] + ["x"] * 100)
    mem = WikiV3Memory(tmp_path / "store", llm=llm, config={"consolidate_every": 1000})
    mem.priors.ingest_episode(_traj_success())
    # A page must exist, otherwise SELECT is correctly skipped and the full 2-call path
    # never runs — the test would pass for the wrong reason.
    mem.pages.write(Page(name="cool-before-placement", description="Cool then place.",
                         type="procedure", applies_to=["pick_cool_then_place"],
                         body="1. Cool it."))
    mem.pages.write_index()

    task = "cool some apple and put it in countertop"
    obs = "You see a countertop 1, a countertop 3, a fridge 1."
    for step in range(50):
        mem.retrieve(task, obs, [], step)

    assert len(llm.calls) == 2, f"expected 2 calls for 50 steps, got {len(llm.calls)}"


def test_memory_update_is_zero_llm_until_the_interval_is_reached(tmp_path):
    """D51 — per-episode writes are deterministic; the LLM runs 1/K as often."""
    from memory.base import EpisodeResult

    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    mem = WikiV3Memory(tmp_path / "store", llm=llm, config={"consolidate_every": 3})

    def ep(i, success=True):
        return EpisodeResult(
            task_id=f"t{i}", task_family="pick_cool_then_place",
            task_description="cool some apple and put it in countertop",
            trajectory=_traj_success(), success=success, steps_taken=5,
            memory_context="check countertop 3",
        )

    mem.update(ep(1))
    mem.update(ep(2))
    assert llm.calls == [], "no LLM may run in the per-episode write path"
    assert mem.priors.n_observations > 0, "but the deterministic layers must have grown"

    mem.update(ep(3))
    assert len(llm.calls) == 2, "consolidation runs on the K-th episode"
    assert mem.pages.exists("cool-before-placement")


def test_frozen_memory_writes_nothing(tmp_path):
    from memory.base import EpisodeResult

    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    mem = WikiV3Memory(tmp_path / "store", llm=llm, config={"consolidate_every": 1})
    mem.freeze()
    mem.update(EpisodeResult(
        task_id="t", task_family="pick_cool_then_place", task_description="cool some apple",
        trajectory=_traj_success(), success=True, steps_taken=5,
    ))
    assert mem.journal.n_records == 0
    assert llm.calls == []


def test_finalize_flushes_the_tail_of_the_chain(tmp_path):
    """Without this, up to K-1 episodes would never be compiled."""
    from memory.base import EpisodeResult

    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    mem = WikiV3Memory(tmp_path / "store", llm=llm, config={"consolidate_every": 100})
    mem.update(EpisodeResult(
        task_id="t", task_family="pick_cool_then_place",
        task_description="cool some apple and put it in countertop",
        trajectory=_traj_success(), success=True, steps_taken=5,
    ))
    assert llm.calls == []
    mem.finalize()
    assert len(llm.calls) == 2


def test_document_source_reaches_consolidation_like_an_episode(tmp_path):
    """D57 — the property that makes the handbook study a config change."""
    docs = tmp_path / "handbook"
    docs.mkdir()
    (docs / "handbook-p7.md").write_text(
        "# Serving\nFruit is served chilled. Prepared food goes on the kitchen countertop.",
        encoding="utf-8",
    )
    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    mem = WikiV3Memory(
        tmp_path / "store", llm=llm,
        config={"consolidate_every": 25, "sources": ["episode", "document"],
                "documents": str(docs)},
    )
    assert any(r.get("kind") == "document" for r in mem.journal)

    mem.consolidate()
    analyse_prompt = llm.calls[0]["user"]
    assert "=== DOCUMENTS ===" in analyse_prompt
    assert "served chilled" in analyse_prompt


def test_document_seeding_is_idempotent(tmp_path):
    """A resumed run re-enters __init__; seeding twice must not double-count."""
    docs = tmp_path / "handbook"
    docs.mkdir()
    (docs / "d1.md").write_text("# T\nbody", encoding="utf-8")
    cfg = {"sources": ["episode", "document"], "documents": str(docs)}
    WikiV3Memory(tmp_path / "store", llm=FakeLLM([]), config=cfg)
    mem = WikiV3Memory(tmp_path / "store", llm=FakeLLM([]), config=cfg)
    assert sum(1 for r in mem.journal if r.get("kind") == "document") == 1


def test_factory_builds_the_arm():
    from memory import create_memory
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        mem = create_memory("wiki_v3", run_dir=Path(td), llm=None, config={})
        assert mem.name == "wiki_v3"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_eval_calls_per_episode_divides_by_this_phase_not_the_journal(tmp_path):
    """Run 10 reported eval at 0.536 calls/ep when the true rate was 2.0.

    _episode_index is restored from the journal (500 after training) and update() returns
    early under freeze(), so it stays at 500 while the call counters count the 134 eval
    episodes. The divisor must be episodes seen in THIS phase.
    """
    from memory.wiki_v3_memory import WikiV3Memory

    mem = WikiV3Memory(tmp_path / "store", llm=None)
    mem._episode_index = 500          # as restored from a trained journal
    mem.freeze()
    mem._n_read_calls = 268           # what a 134-episode eval actually spent
    mem._episodes_seen = 134

    st = mem.stats()
    assert st["episodes_this_phase"] == 134
    assert st["memory_llm_calls_per_episode"] == 2.0
    assert st["episodes"] == 500      # the journal figure is still reported, unchanged


def test_consolidation_detail_log_keeps_raw_outputs_and_a_page_snapshot(tmp_path):
    """W9 — the text a pass produced and removed is written down, not only counted."""
    import json

    from memory.base import EpisodeResult

    llm = FakeLLM([FINDINGS_OUT, PAGE_OUT])
    mem = WikiV3Memory(tmp_path / "store", llm=llm, config={"consolidate_every": 1})
    mem.update(EpisodeResult(
        task_id="t1", task_family="pick_cool_then_place",
        task_description="cool some apple and put it in countertop",
        trajectory=_traj_success(), success=True, steps_taken=5,
    ))

    lines = (tmp_path / "store" / "consolidation_detail.jsonl").read_text().splitlines()
    detail = json.loads(lines[0])
    assert detail["pass"] == 1
    assert detail["raw_analyse"] == FINDINGS_OUT
    assert detail["rewrites"][0]["raw"] == PAGE_OUT
    for key in ("outcome_gate", "claim_gate", "word_budget", "written_body"):
        assert key in detail["rewrites"][0]
    assert (tmp_path / "store" / "pages_history" / "pass_001" / "cool-before-placement.md").exists()


# =================================================================== Stage 3 (W1, W2)

def test_gates_attest_a_transform_by_verb_and_tool_not_by_object():
    """W2: 'Heat the object with the microwave' never matched `heat mug with microwave`,
    so the claim gate was inert and the outcome gate stripped the step as unattested."""
    ev = {"heat mug": {"heat mug with microwave"}, "cool cup": {"cool cup with fridge"}}
    page = Page(name="p", type="procedure", body=(
        "1. Heat the object with the microwave.\n"
        "2. Heat the object with the stoveburner.\n"
        "3. Clean the plate with the sink basin."))
    out, rep = claim_gate(page, ev)
    assert "stoveburner" not in out.body and rep.n_rejected == 1
    assert "Heat the object with the microwave" in out.body
    assert "sink basin" in out.body, "clean was never attested, so nothing contradicts it"

    kept, rep2 = outcome_gate_claims(
        Page(name="p", type="procedure", body="1. Heat the object with the microwave."),
        batch_has_success_for=False, evidence_shapes=ev)
    assert "microwave" in kept.body and rep2.n_rejected == 0


def test_gates_keep_transform_steps_that_name_no_tool():
    """W15: the final runs' claim gate removed 28 steps like 'heat <obj> with <tool>' and
    'Heat the object with the heating tool.', reading 'tool' or 'heating' as a tool other
    than the attested microwave. 6 of 9 transform pages lost their step."""
    ev = {"heat mug": {"heat mug with microwave"}, "cool cup": {"cool cup with fridge"},
          "clean plate": {"clean plate with sinkbasin"}}
    generic = [
        "1. heat <obj> with <tool>",
        "2. Heat the object with the heating tool.",
        "3. Cool the object with the designated cooling tool.",
        "4. clean <obj> with <cleaning tool>",
        "5. Cool the object using the appropriate cooling tool (e.g., fridge).",
    ]
    out, rep = claim_gate(Page(name="p", type="procedure", body="\n".join(generic)), ev)
    assert rep.n_rejected == 0, rep.rejected
    kept, rep2 = outcome_gate_claims(Page(name="p", type="procedure", body="\n".join(generic)),
                                     batch_has_success_for=False, evidence_shapes=ev)
    assert rep2.n_rejected == 0, rep2.rejected
    # A named wrong tool is still a contradiction, also when the right one is in brackets.
    bad = "1. Heat the object with the stoveburner.\n2. Cool the cup with the tool (e.g., microwave)."
    out, rep = claim_gate(Page(name="p", type="procedure", body=bad), ev)
    assert rep.n_rejected == 2


def test_put_and_place_read_as_move():
    """W2: ALFWorld 0.4.2 places with `move ... to`; 'put' or 'place' lines can be attested."""
    from wiki_v3.gates import _parts

    assert _parts("Put the mug 1 in the shelf 1")[:2] == ("move", "mug")
    assert _parts("Place the mug on the shelf")[:2] == ("move", "mug")
    assert _parts("Pick up the mug")[:2] == ("take", "mug")
    ev = {"move mug": {"move mug to shelf"}}
    out, rep = outcome_gate_claims(Page(name="p", type="procedure", body="1. Put the mug on the shelf."),
                                   batch_has_success_for=False, evidence_shapes=ev)
    assert rep.n_rejected == 0


def test_admission_gate_matches_families_across_spellings(tmp_path):
    """W1: the analyse call wrote `pick-heat-then-place`, episode records say
    `pick_heat_then_place`; a batch success in that family must count."""
    from wiki_v3.consolidator import parse_findings, _family

    assert _family("pick-heat-then-place") == "pick_heat_then_place"
    f = parse_findings("PAGE: heat-page\nTYPE: procedure\nFAMILIES: pick-heat-then-place\n"
                       "OBJECTS: mug\nLESSON: heat it.\nEVIDENCE: ep1\n")
    assert f[0].families == ["pick_heat_then_place"]


# =================================================================== Stage 3 (W3, W12)

def test_an_unclosed_second_header_is_not_served(tmp_path):
    """W3: 5 of 11 Run 10 pages carried a second header, several without a closing fence,
    and the reader served it to the agent."""
    text = (
        "---\nname: pick-heat-then-place\ntype: procedure\nsources:\n- ep464\n---\n\n"
        "---\nname: pick-heat-then-place\ndescription: A procedure for heating.\n"
        "type: procedure\napplies_to: [pick-heat-then-place]\nconfidence: hedged\n\n"
        "# Steps\n1. Heat the mug with the microwave.\n"
    )
    pg = parse_page(text)
    assert pg.body.startswith("# Steps")
    assert "applies_to" not in pg.body and "confidence" not in pg.body

    ps = PageStore(tmp_path / "pages")
    ps.path_for("pick-heat-then-place").write_text(text, encoding="utf-8")
    served = ps.read_body("pick-heat-then-place")
    assert served.startswith("## pick-heat-then-place\n# Steps")
    assert "sources" not in served


def test_only_an_unclosed_header_keeps_its_fields():
    pg = parse_page("---\nname: p\ndescription: About p.\ntype: caution\n\n# Pitfalls\n- x")
    assert pg.description == "About p." and pg.type == "caution"
    assert pg.body.startswith("# Pitfalls")


def test_lint_finds_header_leftovers_step_gaps_and_empty_sections():
    from wiki_v3.store import body_problems

    probs = body_problems("# Steps\n1. a\n2. b\n4. d\n# Pitfalls\n\n# Not established\n- x\nname: p")
    assert any("not consecutive" in p for p in probs)
    assert any("empty section '# Pitfalls'" in p for p in probs)
    assert any("header fields" in p for p in probs)
    assert body_problems("# Steps\n1. a\n2. b\n# Pitfalls\n- c") == []


def test_word_budget_drops_whole_sections_and_keeps_steps():
    """W4: the old cut took the page's tail, leaving empty headings and losing whatever came last."""
    body = ("# Preconditions\n" + "- long precondition text here " * 30 + "\n"
            "# Steps\n1. Take the mug.\n2. Heat the mug with the microwave.\n"
            "# Pitfalls\n- Do not take the cup.\n"
            "# Not established\n" + "- unknown thing " * 30)
    page = enforce_word_budget(Page(name="p", type="caution", body=body))
    assert page.word_count() <= 120
    assert "Heat the mug with the microwave" in page.body
    assert "# Pitfalls" in page.body
    assert page.body.index("# Steps") < page.body.index("# Pitfalls")
    from wiki_v3.store import body_problems
    assert not any("empty section" in p for p in body_problems(page.body))


# =================================================================== Stage 3 (W5, W11)

def test_reader_always_reads_the_family_page_and_attested_tools(tmp_path):
    """W11: SELECT never saw the family, and generic pages took the three slots."""
    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="pick-heat-then-place", description="Heating.", type="procedure",
                  applies_to=["pick_heat_then_place"], confidence="hedged",
                  body="# Steps\n1. heat mug with microwave"))
    ps.write(Page(name="affordances", description="Tools.", type="entity",
                  applies_to=["pick_heat_then_place"], confidence="attested",
                  sources=["evidence.json"],  # as consolidator writes it (W19)
                  body="- to heat something, use the microwave"))
    for n in ("generic-a", "generic-b", "generic-c"):
        ps.write(Page(name=n, description="Generic.", type="caution", applies_to=[], body="x"))
    llm = FakeLLM(["generic-a\ngeneric-b\ngeneric-c", "Heat the mug with the microwave."])
    res = Reader(ps, PriorsStore(tmp_path / "p.json"), llm).read(
        "heat some mug and put it in cabinet", "You are in a room.", "pick_heat_then_place")
    assert res.pages_read[:2] == ["pick-heat-then-place", "affordances"]
    assert len(res.pages_read) == 3
    assert "TASK FAMILY: pick_heat_then_place" in llm.calls[0]["user"]


def test_compose_prompt_asks_for_the_transform_and_drops_hedging_rules():
    from wiki_v3 import prompts

    assert "done to the object before it is placed" in prompts.COMPOSE_SYSTEM
    assert "Put warnings first" not in prompts.COMPOSE_SYSTEM
    assert "uncertain as uncertain" not in prompts.COMPOSE_SYSTEM
    assert "fridge" not in prompts.REWRITE_SYSTEM


# =================================================================== Stage 3 (W6, W7)

def _rec(family, task, steps):
    """steps: [(action, observation seen before it)]"""
    return {"kind": "episode", "index": 1, "family": family, "task": task, "outcome": "failure",
            "steps": 50, "trajectory": [{"action": a, "observation": o} for a, o in steps]}


def test_digest_names_a_missing_transform():
    from wiki_v3.digest import failure_class, target_object, _found_at

    r = _rec("pick_heat_then_place", "heat some mug and put it in cabinet", [
        ("go to countertop 1", "You are in a room."),
        ("take mug 1 from countertop 1", "You arrive at countertop 1. On it you see a mug 1."),
        ("go to cabinet 1", "You pick up the mug 1 from the countertop 1."),
        ("move mug 1 to cabinet 1", "You arrive at cabinet 1."),
        ("look", "You move the mug 1 to the cabinet 1."),
    ])
    t = target_object(r["task"]); fw, _ = _found_at(r["trajectory"], t)
    assert "no-transform-issued" in failure_class(r, t, fw)

    r["trajectory"].insert(3, {"action": "heat mug 1 with microwave 1", "observation": "You arrive at microwave 1."})
    r["trajectory"][4]["observation"] = "You heat the mug 1 using the microwave 1."
    assert "no-transform-issued" not in failure_class(r, t, fw)


def test_digest_names_retaking_from_the_destination():
    from wiki_v3.digest import failure_class

    r = _rec("pick_two_obj_and_place", "put two cd in safe", [
        ("move cd 1 to safe 1", "You pick up the cd 1 from the desk 1."),
        ("take cd 1 from safe 1", "You move the cd 1 to the safe 1."),
        ("look", "You pick up the cd 1 from the safe 1."),
    ])
    assert "retook-from-destination" in failure_class(r, "cd", "desk 1")


# =================================================================== Stage 3 (W14)

def test_the_agents_own_placements_are_not_location_evidence():
    """W14: after 'You move the mug 1 to the shelf 1', seeing mug 1 on shelf 1 says nothing
    about where mugs are found. Other mugs seen there still count."""
    from memory.observation_priors import extract_object_locations

    traj = [
        {"observation": "On the countertop 1, you see a mug 1."},
        {"observation": "You move the mug 1 to the shelf 1."},
        {"observation": "On the shelf 1, you see a mug 1, and a mug 2."},
        {"observation": "You open the fridge 1. In it, you see a apple 1."},
    ]
    assert extract_object_locations(traj) == [
        ("mug", "countertop"), ("mug", "shelf"), ("apple", "fridge")]


# =================================================================== W16, W17

def test_objects_inside_opened_receptacles_are_recorded():
    """W16: ALFWorld says 'The fridge 1 is open. In it, you see ...', which the old pattern
    never matched, so no object was ever recorded in a fridge, microwave, safe or drawer."""
    from memory.observation_priors import extract_object_locations

    traj = [
        {"observation": "You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, "
                        "and a egg 1."},
        {"observation": "You arrive at cabinet 1. The cabinet 1 is open. In it, you see a cup 1."},
        {"observation": "You open the drawer 2. The drawer 2 is open. In it, you see nothing."},
        {"observation": "You move the cup 1 to the microwave 1."},
        {"observation": "You open the microwave 1. The microwave 1 is open. In it, you see a cup 1."},
    ]
    assert extract_object_locations(traj) == [
        ("apple", "fridge"), ("egg", "fridge"), ("cup", "cabinet")]


def test_target_object_is_found_in_examine_and_cd_tasks():
    """W17: 'examine the X with the desklamp', 'look at X under the desklamp' and 'cd'
    returned [], so the wiki named no locations for any examine task."""
    from alfworld_env import extract_target_objects

    assert extract_target_objects("examine the bowl with the desklamp") == ["bowl"]
    assert extract_target_objects("look at mug under the desklamp") == ["mug"]
    assert extract_target_objects("put some cd in safe") == ["cd"]
    assert extract_target_objects("heat some apple and put it in fridge") == ["apple"]
    assert extract_target_objects("put a clean cloth in bathtubbasin") == ["cloth"]


# =================================================================== W18, W19, W20

def test_page_name_drops_a_bracketed_type():
    """W18: 'PAGE: pick-clean-then-place [procedure]' forked the family page."""
    from wiki_v3.consolidator import _slug

    assert _slug("pick-clean-then-place [procedure]") == "pick-clean-then-place"
    assert _slug("search-exhaustion (caution)") == "search-exhaustion"
    assert _slug("pick-clean-then-place") == "pick-clean-then-place"


def test_only_code_built_attested_pages_are_forced_into_a_read(tmp_path):
    """W19: a page the model labelled `confidence: attested` was read for every task."""
    from wiki_v3.reader import Reader

    ps = PageStore(tmp_path / "pages")
    ps.write(Page(name="affordances", type="entity", applies_to=["pick_heat_then_place"],
                  sources=["evidence.json"], confidence="attested", body="- heat: microwave"))
    ps.write(Page(name="search-exhaustion", type="caution", applies_to=["pick_heat_then_place"],
                  sources=["ep1", "ep2"], confidence="attested", body="- stop searching"))
    req = Reader(ps, None, None)._required_pages("pick_heat_then_place")
    assert "affordances" in req and "search-exhaustion" not in req


def test_gates_leave_pitfall_and_precondition_bullets_alone():
    """W20: the outcome gate stripped cautions that happened to contain a verb."""
    ev = {"heat mug": {"heat mug with microwave"}}
    body = ("# Preconditions\n- Whether the object is already clean.\n"
            "# Steps\n1. Take the mug and wait ten minutes.\n"
            "# Pitfalls\n- The robot may pick up the wrong object.\n"
            "- Heating with the stoveburner fails.")
    out, rep = outcome_gate_claims(Page(name="p", type="procedure", body=body),
                                   batch_has_success_for=False, evidence_shapes=ev)
    assert "already clean" in out.body and "wrong object" in out.body
    assert "wait ten minutes" not in out.body and rep.n_rejected == 1
    out, rep = claim_gate(Page(name="p", type="procedure", body=body), ev)
    assert "stoveburner" in out.body and rep.n_rejected == 0

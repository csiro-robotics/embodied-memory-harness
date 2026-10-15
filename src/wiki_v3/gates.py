"""The two gates. Unchanged from v2 — they are the contribution, not the part being redesigned.

D09 OUTCOME GATE (episode-level). A positive procedure may not be created from a batch
containing no successful episode for that family. A failure may produce a caution, a
correction, an observation, or nothing.

D10 CLAIM GATE (claim-level). No written line may contradict an action form the
environment was observed to accept. This catches what the outcome gate structurally
cannot: a wrong procedure distilled from a SUCCESSFUL episode. The five-step
open/insert/close/heat/take cooling routine reached 75 of 134 tasks while
`cool cup 1 with fridge 1` sat in 20 of the agent's own successes.

D11/D58 PRECEDENCE. The environment outranks prose, and now also outranks DOCUMENTS. A
handbook page is authority, not evidence; if the environment was observed to contradict
it, the environment still wins. Recency never breaks ties — provenance does. Recency-first
(the v2 gist's proposal) would not have caught CANONICAL_STEPS, because the wrong routine
was the most recently written claim on that page and was being reinforced every episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from wiki_v3.store import Page

# A line that reads as an imperative procedure step rather than prose.
_STEP_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")
# Verbs whose action shape the environment can attest. ALFWorld 0.4.2 places an object
# with `move <obj> to <recep>`; a page that writes "put" means the same action, so "put" is
# read as "move" (W2). Before, "put" was checked but could never be attested.
_ATTESTABLE = re.compile(
    r"\b(cool|heat|clean|take|put|move|open|close|toggle|use|slice)\b", re.IGNORECASE
)
_SYNONYM = {"put": "move"}
# W15: words that describe a tool or object without naming one. "heat <obj> with <tool>",
# "the heating tool" and "the appropriate tool (e.g., fridge)" were read as naming a tool
# called "tool", "heating" or "appropriate", so the claim gate removed them as contradicting
# the attested tool (28 steps in the final runs, 6 of 9 transform pages left without their
# step). A placeholder names nothing; a real tool named later in the phrase still counts.
_GENERIC = {
    "tool", "tools", "appliance", "device", "receptacle", "appropriate", "designated",
    "correct", "right", "suitable", "proper", "required", "relevant", "available",
    "heating", "cooling", "cleaning", "lighting", "obj", "object", "objects", "item",
    "target", "e", "g", "i", "eg", "ie", "such", "as", "like", "to", "in", "on",
    "of", "first", "second", "instance", "cleaned", "heated", "cooled", "chilled",
    "warmed", "destination",
}
_PREPOSITIONS = {"from", "to", "in", "into", "on", "onto", "at", "with", "using", "under"}
_PLACEHOLDER = re.compile(r"<[^>]*>")
_HEADING = re.compile(r"^\s*#+\s*(.*\S)?\s*$")


def _step_lines(body: str):
    """(line, match) for each line, match set only on a line the gates check.

    W20: the rewrite prompt tells the model that both gates parse the procedure lines and
    only those. They parsed every bullet, so a pitfall such as "The robot may pick up the
    wrong object" or "Whether the object is already clean" was read as a step and stripped
    by the outcome gate (its docstring says cautions survive). Lines under a heading other
    than Steps are prose now; lines under Steps, or under no heading, are still checked.
    """
    section = ""
    for raw in body.splitlines():
        h = _HEADING.match(raw)
        if h:
            section = (h.group(1) or "").strip().lower()
            yield raw, None
            continue
        if section and not section.startswith("steps"):
            yield raw, None
            continue
        yield raw, _STEP_LINE.match(raw)


@dataclass
class GateReport:
    kept: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (line, reason)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _parts(line: str) -> tuple[str, str, list[str]] | None:
    """A written step as (verb, object, instrument candidates), or None if not an action.

    Instance numbers are stripped because a page asserting a procedure is not asserting an
    instance — `cool the cup with the fridge` and `cool cup 1 with fridge 1` are the same
    claim about how cooling works. The instrument is what follows "with" or "using". It is
    returned as candidates (one word, then two words joined) so that "sink basin" matches
    the environment's `sinkbasin`.
    """
    s = re.sub(r"[`*_]", " ", (line or "").lower())
    s = _PLACEHOLDER.sub(" object ", s)  # W15: "<obj>", "<tool>" name no instance or tool
    s = re.sub(r"\s+\d+\b", " ", s)
    # Plain-English forms of ALFWorld's commands, as the pages tend to write them.
    s = re.sub(r"\bpick(?:s|ed)? up\b", "take", s)
    s = re.sub(r"\bplac(?:e|es|ed|ing)\b", "move", s)
    s = re.sub(r"\b(the|a|an|your|then|first|next|finally|it|that|same|requested|exact)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    m = _ATTESTABLE.search(s)
    if not m:
        return None
    words = s[m.start():].split()
    if len(words) < 2:
        return None
    verb = _SYNONYM.get(words[0], words[0])
    # W15: the object is the first word that names something, up to the next preposition;
    # "the first instance of the target object" and "the cleaned object" name no object.
    phrase = []
    for w in words[1:]:
        if w in _PREPOSITIONS:
            break
        phrase.append(w)
    named = [w for w in phrase if w not in _GENERIC]
    obj = named[0] if named else ("object" if phrase else words[1])
    instruments: list[str] = []
    tail = re.split(r"\b(?:with|using)\b", s[m.start():], maxsplit=1)
    if len(tail) == 2:
        # W15: the tool phrase ends at a sentence stop or a following clause; within it,
        # only words that are not generic can name the tool.
        phrase = re.sub(r"\b(?:e\.g|i\.e)\b\.?", " ", tail[1])
        phrase = re.split(r"[.;:]|\b(?:and|then|until|before|after)\b", phrase, maxsplit=1)[0]
        t = [w for w in re.findall(r"[a-z]+", phrase) if w not in _GENERIC]
        if t:
            instruments.append(t[0])
            if len(t) > 1:
                instruments.append(t[0] + t[1])
    return verb, obj, instruments


def _shape(line: str) -> str | None:
    """The comparable shape of a written step, `verb object [with instrument]`."""
    p = _parts(line)
    if p is None:
        return None
    verb, obj, inst = p
    return f"{verb} {obj}" + (f" with {inst[0]}" if inst else "")


def _instruments(evidence_shapes: dict[str, set[str]]) -> dict[str, set[str]]:
    """verb -> instruments the environment accepted it with, from the attested shapes.

    W2: the gates used to compare whole shapes, so "Heat the object with the microwave"
    (object "object") never matched `heat mug with microwave`, and the claim gate rejected
    nothing in 20 passes. What a transformation step claims is its verb and its tool, so
    that is what is attested.
    """
    out: dict[str, set[str]] = {}
    for shapes in evidence_shapes.values():
        for shape in shapes:
            if " with " in shape:
                verb = shape.split()[0]
                inst = shape.split(" with ", 1)[1].split()
                if inst:
                    out.setdefault(verb, set()).add(inst[0])
    return out


def _is_attested(parts, evidence_shapes, instruments) -> bool:
    verb, obj, inst = parts
    if inst:
        return any(i in instruments.get(verb, ()) for i in inst)
    if obj in _GENERIC:
        # W15: "heat <obj> with <tool>" names no object and no tool; what it claims is that
        # the verb is a step, which any accepted form of that verb attests.
        return any(key.split()[0] == verb for key in evidence_shapes)
    return f"{verb} {obj}" in evidence_shapes


def _contradicts(parts, instruments) -> bool:
    """A step names a tool for a verb the environment accepted only with other tools."""
    verb, _obj, inst = parts
    known = instruments.get(verb)
    return bool(inst and known and not any(i in known for i in inst))


def outcome_gate(page: Page, batch_has_success_for: bool) -> tuple[bool, str]:
    """D09, as originally written: page-level allow/reject. Returns (allowed, reason).

    RETAINED for the tests that pin the original behaviour and for any caller that wants the
    old semantics, but the consolidator no longer uses it — see `outcome_gate_claims` (D59)
    for why rejecting the whole page was the wrong granularity.
    """
    if page.type != "procedure":
        return True, ""
    if batch_has_success_for:
        return True, ""
    return False, "outcome gate: no successful episode in this batch for this page's families"


def outcome_gate_claims(
    page: Page,
    batch_has_success_for: bool,
    evidence_shapes: dict[str, set[str]],
) -> tuple[Page, GateReport]:
    """D59 — D09 applied per CLAIM instead of per PAGE. Returns (page, report).

    The original gate asked "does this batch contain a success for this page's families?"
    and threw the whole page away if not. On the v3.0 run that rejected 48 pages against 53
    written, and it rejected hardest exactly where knowledge was scarcest: with
    `pick_cool_then_place` succeeding 6/75 in training, a 25-episode batch typically held no
    cool success at all, so no cool page could ever be written, so cool scored 5% at eval.
    The gate forbade knowledge precisely where it was most needed.

    The principle survives, the scope narrows. Without a success in the batch:

      * a step line asserting HOW TO SUCCEED is still inadmissible and is stripped;
      * a step line whose action shape the ENVIRONMENT has already attested survives,
        because the simulator accepting `heat egg 1 with microwave 1` attests that the
        microwave affords heating whether or not that episode went on to succeed;
      * prose, cautions and observations survive, as they always did.

    That is D08's two-channel asymmetry — observations ungated, actions success-gated —
    applied one level up, so it continues the project's argument rather than retreating from
    it. Mechanically it is the same per-line walk `claim_gate` already performs.

    A page emptied of every step line is NOT rejected: it keeps its prose and its identity,
    which is what D64's stubs rely on.
    """
    report = GateReport()
    if page.type != "procedure" or batch_has_success_for:
        return page, report

    instruments = _instruments(evidence_shapes)
    out_lines: list[str] = []
    for raw, m in _step_lines(page.body):
        if not m:
            out_lines.append(raw)
            continue
        parts = _parts(m.group(1))
        if parts is None:
            # Not an attestable action at all — prose in list clothing. Keep it.
            out_lines.append(raw)
            continue
        if _is_attested(parts, evidence_shapes, instruments):
            # The environment already accepted this exact shape. Admissible on its own
            # authority; the outcome of the episode it came from is irrelevant.
            out_lines.append(raw)
            report.kept.append(raw)
            continue
        report.rejected.append(
            (raw, "outcome gate (D59): asserts how to succeed, no success in batch and "
                  "no attested action form")
        )

    page.body = "\n".join(out_lines).strip()
    return page, report


def claim_gate(page: Page, evidence_shapes: dict[str, set[str]]) -> tuple[Page, GateReport]:
    """D10/D58. Strip any step line whose action shape contradicts attested evidence.

    Contradiction means: the environment attested tools for this verb, and the written line
    names a tool it never accepted for it (W2: compared by verb and tool, not by the whole
    shape, which never matched a written step). An unattested verb is not a contradiction — silence is not evidence of impossibility (see ALFWorld.md: an action
    absent from admissible_commands may simply never have had its preconditions met).

    The rest of the page survives. Rejecting the whole page would discard correct lessons
    alongside the wrong claim, which is the failure D14 was adopted to avoid.
    """
    report = GateReport()
    instruments = _instruments(evidence_shapes)
    out_lines: list[str] = []

    for raw, m in _step_lines(page.body):
        if not m:
            out_lines.append(raw)
            continue
        parts = _parts(m.group(1))
        if parts is None:
            out_lines.append(raw)
            report.kept.append(raw)
            continue
        if _contradicts(parts, instruments):
            verb = parts[0]
            report.rejected.append(
                (raw.strip(), f"contradicts attested tools {sorted(instruments[verb])!r} "
                              f"for {verb!r}")
            )
            continue
        out_lines.append(raw)
        report.kept.append(raw)

    page.body = "\n".join(out_lines).strip()
    return page, report


# W4: when a page is over budget, sections are dropped whole, least useful first, and the
# rest keep their order. Cutting the page's tail used to leave empty headings, and cut
# whatever came last, whatever it was.
_SECTION_PRIORITY = ("steps", "pitfalls", "preconditions")


def enforce_word_budget(page: Page) -> Page:
    """D25. wicer: an LLM asked for 75%/50%/25% retention produced 35.4%/12.2%/8.2%.

    An unbounded page defeats D16 (the payload-length curve ran 1062 chars -> 32.9% and
    215 chars -> 58.6%), so the budget is enforced by code. Sections are kept whole in the
    order Steps, Pitfalls, Preconditions, then any other section, until the budget is used
    (W4). A section that alone exceeds what is left is cut at a line boundary only if it is
    the first one kept, so the page is never emptied.
    """
    from wiki_v3.store import DEFAULT_TYPE, WORD_BUDGET

    _lo, hi = WORD_BUDGET.get(page.type, WORD_BUDGET[DEFAULT_TYPE])
    if len(page.body.split()) <= hi:
        return page

    sections: list[list[str]] = [[]]
    for line in page.body.splitlines():
        if line.lstrip().startswith("#") and sections[-1]:
            sections.append([])
        sections[-1].append(line)

    def rank(sec: list[str]) -> int:
        head = sec[0].lstrip("# ").strip().lower() if sec[0].lstrip().startswith("#") else ""
        for i, name in enumerate(_SECTION_PRIORITY):
            if head.startswith(name):
                return i
        return len(_SECTION_PRIORITY)

    keep: set[int] = set()
    used = 0
    for idx in sorted(range(len(sections)), key=lambda i: (rank(sections[i]), i)):
        n = sum(len(l.split()) for l in sections[idx])
        if used + n <= hi:
            keep.add(idx)
            used += n
        elif not keep:
            lines, total = [], 0
            for l in sections[idx]:
                if total + len(l.split()) > hi:
                    break
                lines.append(l)
                total += len(l.split())
            sections[idx] = lines
            keep.add(idx)
            used = total
    page.body = "\n".join(l for i, sec in enumerate(sections) if i in keep for l in sec).strip()
    return page

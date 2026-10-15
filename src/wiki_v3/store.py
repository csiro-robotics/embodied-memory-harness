"""Pages and the index.

D52 — ONE page type. Karpathy specifies one page per concept or topic; v2's skill/note
split existed because a per-episode curator could not decide where a precondition
belonged, and a consolidator reading K episodes can simply put it in the procedure.
`type` survives as a descriptive tag so page kinds can still be counted and reported, but
nothing in the read path branches on it. Entity CONTENT is kept on Part XIII's evidence
(13 of 23 failures in run 3 were wrong-object pickups); only the separate kind is gone.

D02 — flat, no subdirectories. Internal null on tiered variants, plus contmem's finding
that finer organisation measures worse (bundled +6.5% FWT vs individual -2.0%).

`index.md` is generated DETERMINISTICALLY from page frontmatter, not by an LLM. The
architecture spec originally had it as a third consolidation call; making it a pure
function of the pages removes a parse-failure mode, guarantees the index can never
disagree with the pages it indexes, and saves a call per pass. See build_index().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

INDEX_NAME = "index"

# Page-name hygiene: kebab-case stem, no episode/room/instance identifiers (D01 rules).
_VALID_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BANNED_NAME = re.compile(r"(^|-)(ep|episode|room|floorplan|trial)\d", re.IGNORECASE)

# Word budgets (D25) — the curator gets budgets, not adjectives. wicer: an LLM told to
# "compress" over-fires by 2-3x and does not report it, so the budget is enforced here.
# W10: the procedure minimum was 120 words, which asked the model to pad a short procedure;
# it is now 40, the same as the other types.
WORD_BUDGET = {"procedure": (40, 250), "entity": (40, 120), "caution": (40, 120)}

# A line of a YAML header: a `---` fence, `key: value`, a `- item`, or an indented
# continuation of a folded value.
_HEADER_LINE = re.compile(r"^(?:---\s*|[A-Za-z_]+:(?:\s.*)?|\s*-\s+\S.*|\s{2,}\S.*)$")


def _split_header(body: str) -> tuple[dict, str]:
    """Remove a YAML-like header at the top of `body`, closed or not (W3).

    The model sometimes wrote a second header and did not close it with `---`, so the old
    repair (which needed both fences) left it in the body, and the reader served it to the
    agent. Everything before the first markdown heading is removed when it looks like a
    header; with no heading, the header lines at the top are removed. Returns the header's
    fields when they parse, so a page whose only header was unclosed keeps its metadata.
    """
    lines = body.splitlines()
    first = next((l for l in lines if l.strip()), "")
    if not (first.strip() == "---" or re.match(r"^[A-Za-z_]+:\s", first)):
        return {}, body
    cut = next((i for i, l in enumerate(lines) if l.lstrip().startswith("#")), None)
    if cut is None:
        cut = 0
        while cut < len(lines) and (not lines[cut].strip() or _HEADER_LINE.match(lines[cut])):
            cut += 1
    header = "\n".join(l for l in lines[:cut] if l.strip() != "---")
    fields: dict = {}
    try:
        loaded = yaml.safe_load(header) if header.strip() else None
        if isinstance(loaded, dict):
            fields = loaded
    except yaml.YAMLError:
        pass
    return fields, "\n".join(lines[cut:]).strip()
DEFAULT_TYPE = "procedure"


@dataclass
class Page:
    name: str
    description: str = ""
    type: str = DEFAULT_TYPE
    applies_to: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: str = "hedged"
    body: str = ""

    def word_count(self) -> int:
        return len(self.body.split())

    def to_markdown(self) -> str:
        fm = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "applies_to": self.applies_to,
            "objects": self.objects,
            "sources": self.sources,
            "confidence": self.confidence,
        }
        head = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return f"---\n{head}---\n\n{self.body.strip()}\n"


def parse_page(text: str, fallback_name: str = "") -> Page:
    """Parse a markdown page. Tolerant: a 14B will sometimes omit or mangle frontmatter.

    Missing frontmatter is synthesised rather than rejected, because dropping the page
    would silently lose the lesson — the failure mode D14 was adopted to avoid ("weak
    evidence is hedged, not suppressed").
    """
    text = (text or "").strip()
    fm: dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                if isinstance(loaded, dict):
                    fm = loaded
                body = parts[2]
            except yaml.YAMLError:
                body = parts[2] if len(parts) >= 3 else text

    # D61, W3 — strip any further header block, closed or not. `_rewrite` used to hand the
    # model the page including its frontmatter, so the model echoed that block back and
    # appended its own; 5 of 11 Run 10 pages carried a second block, several unclosed, and
    # the reader served them to the agent. Repeated, because a page can carry more than one.
    body = body.strip()
    for _ in range(3):
        extra, body = _split_header(body)
        if not extra and not body.startswith("---"):
            break
        if not fm and extra:
            fm = extra

    def _list(key: str) -> list[str]:
        v = fm.get(key)
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    def _families(key: str) -> list[str]:
        """Normalise `applies_to` to real ALFWorld family names.

        D64 named the family stub pages in kebab-case (`pick-cool-then-place`), and the
        rewrite prompt shows the model the page name, so it duly echoed the HYPHENATED form
        into `applies_to`. ALFWorld families use underscores, `build_index` groups on exact
        matches, and the result was an index reading "(nothing recorded)" for four of six
        families that each had a perfectly good page sitting next to it — the reader could
        not route to them at all.

        Same principle as D62: normalise and validate against reality rather than asking the
        model to be careful. An entry that is not a real family after normalisation is
        dropped, because a page claiming to apply to a family that does not exist is
        unreachable anyway.
        """
        from alfworld_env import _FAMILY_PATTERNS  # noqa: PLC0415

        known = {fam for _pat, fam in _FAMILY_PATTERNS}
        out: list[str] = []
        for raw in _list(key):
            fam = raw.strip().lower().replace("-", "_")
            if fam in known and fam not in out:
                out.append(fam)
        return out

    name = str(fm.get("name") or fallback_name or "").strip().lower()
    ptype = str(fm.get("type") or DEFAULT_TYPE).strip().lower()
    if ptype not in WORD_BUDGET:
        ptype = DEFAULT_TYPE

    return Page(
        name=name,
        description=str(fm.get("description") or "").strip(),
        type=ptype,
        applies_to=_families("applies_to"),
        objects=[x.lower() for x in _list("objects")],
        sources=_list("sources"),
        confidence=str(fm.get("confidence") or "hedged").strip().lower(),
        body=body.strip(),
    )


def body_problems(body: str) -> list[str]:
    """W12: what the old lint missed — header lines left in the body, gaps in the step
    numbering (a step removed by a gate or the budget), and headings with nothing under them."""
    out: list[str] = []
    lines = body.splitlines()
    if any(re.match(r"^(name|description|applies_to|sources|confidence):", l) for l in lines):
        out.append("header fields inside the body")
    nums = [int(m.group(1)) for l in lines if (m := re.match(r"^\s*(\d+)[.)]\s", l))]
    if nums and nums != list(range(nums[0], nums[0] + len(nums))):
        out.append(f"step numbers not consecutive: {nums}")
    for i, l in enumerate(lines):
        if l.lstrip().startswith("#"):
            nxt = next((x for x in lines[i + 1:] if x.strip()), "")
            if not nxt or nxt.lstrip().startswith("#"):
                out.append(f"empty section {l.strip()!r}")
    return out


def valid_page_name(name: str) -> bool:
    return bool(name) and bool(_VALID_NAME.match(name)) and not _BANNED_NAME.search(name)


class PageStore:
    """A flat directory of markdown pages plus a generated index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- read
    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.md"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def read(self, name: str) -> Page | None:
        p = self.path_for(name)
        if not p.exists():
            return None
        return parse_page(p.read_text(encoding="utf-8"), fallback_name=name)

    def read_text(self, name: str) -> str:
        p = self.path_for(name)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def read_body(self, name: str) -> str:
        """The page as the reader serves it (W3): its name and parsed body, no header."""
        pg = self.read(name)
        if pg is None:
            return ""
        return f"## {pg.name}\n{pg.body}".strip()

    def names(self) -> list[str]:
        return sorted(
            p.stem for p in self.root.glob("*.md") if p.stem != INDEX_NAME
        )

    def all_pages(self) -> list[Page]:
        out = []
        for n in self.names():
            pg = self.read(n)
            if pg is not None:
                out.append(pg)
        return out

    def descriptions(self) -> str:
        """The survey surface. `description` is the primary retrieval surface (D01)."""
        lines = []
        for pg in self.all_pages():
            lines.append(f"- {pg.name} [{pg.type}]: {pg.description or '(no description)'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ write
    def write(self, page: Page) -> None:
        if not valid_page_name(page.name):
            raise ValueError(f"invalid page name: {page.name!r}")
        self.path_for(page.name).write_text(page.to_markdown(), encoding="utf-8")

    def delete(self, name: str) -> bool:
        p = self.path_for(name)
        if p.exists():
            p.unlink()
            return True
        return False

    # ------------------------------------------------------------------ index
    def build_index(self, known_objects: set[str] | None = None) -> str:
        """Generate index.md deterministically from page frontmatter.

        Why deterministic rather than a third LLM call: the index is a pure function of the
        pages, so generating it removes a parse-failure mode and makes it impossible for
        the index to disagree with what it indexes. It also makes ABSTENTION structural
        (D18) — a family with no page gets listed under "known unknowns", so the reader
        serves nothing by construction rather than by judgement. On examine_in_light a KG
        that emitted nothing scored 94% against the wiki's 83%.
        """
        pages = self.all_pages()

        by_family: dict[str, list[str]] = {}
        by_object: dict[str, list[str]] = {}
        for pg in pages:
            for fam in pg.applies_to:
                by_family.setdefault(fam, []).append(pg.name)
            for obj in pg.objects:
                # D62 — an object type the environment was never observed to contain does
                # not get an index entry. `objects:` is filled freely by the model, and in
                # run 59348394 that put `cdrom`, `cdromdrive`, `cdromplayer`, `cdcase` and
                # `cdplayer` into the index; none are ALFWorld object types. Filtering
                # against what `priors.json` actually saw removes the model's discretion
                # rather than asking it to behave. `known_objects=None` means "no observed
                # vocabulary available" (a cold store), in which case nothing is filtered —
                # filtering everything out would be worse than a few bad entries.
                if known_objects is not None and obj not in known_objects:
                    continue
                by_object.setdefault(obj, []).append(pg.name)

        from alfworld_env import _FAMILY_PATTERNS  # noqa: PLC0415

        known_families = sorted({fam for _pat, fam in _FAMILY_PATTERNS})

        lines = [
            "---",
            f"name: {INDEX_NAME}",
            "description: Which pages to read for which task. Regenerated at every consolidation.",
            "---",
            "",
            "# By task family",
        ]
        for fam in known_families:
            got = sorted(set(by_family.get(fam, [])))
            lines.append(f"- {fam} -> {', '.join(got) if got else '(nothing recorded)'}")

        if by_object:
            lines.append("")
            lines.append("# By object")
            for obj in sorted(by_object):
                lines.append(f"- {obj} -> {', '.join(sorted(set(by_object[obj])))}")

        unknown = [f for f in known_families if not by_family.get(f)]
        if unknown:
            lines.append("")
            lines.append("# Known unknowns — serve nothing for these")
            for fam in unknown:
                lines.append(f"- {fam}")

        return "\n".join(lines) + "\n"

    def write_index(self, known_objects: set[str] | None = None) -> str:
        text = self.build_index(known_objects)
        (self.root / f"{INDEX_NAME}.md").write_text(text, encoding="utf-8")
        return text

    def read_index(self) -> str:
        p = self.root / f"{INDEX_NAME}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def ensure_family_stub(self, family: str) -> bool:
        """D64 — a family gets a page the first time it is seen. Returns True if created.

        v3.0 wrote a page only when consolidation proposed one AND the gates admitted it, so
        `pick_cool_then_place` — 6/75 successes in training — never got a page at all and the
        store saturated at 9 pages by pass 2. Seeding on first sighting makes the store grow
        with COVERAGE rather than only with SUCCESS, and gives the page a continuous identity
        and `sources` chain from the moment the family is first encountered.

        The stub is inert but not dishonest: `# Not established` is a legitimate thing to
        serve, because D18 already makes abstention a valid output, and telling the executor
        that nothing is known about cooling beats silently routing it to a page about search
        exhaustion.
        """
        name = family.replace("_", "-")
        if name in self.names():
            return False
        self.write(Page(
            name=name,
            description=f"What has been established about {family.replace('_', ' ')} tasks.",
            type="procedure",
            applies_to=[family],
            objects=[],
            sources=[],
            confidence="unestablished",
            body="# Not established\nNothing has been attested for this family yet.",
        ))
        return True

    def candidates_for(self, family: str, obj: str | None) -> list[str]:
        """Deterministic index lookup — the reader's fallback when call 1 parses to nothing.

        The reader must never fail open into "read everything": that is the 40-round loop
        v3 exists to remove.
        """
        out: list[str] = []
        for pg in self.all_pages():
            if family and family in pg.applies_to:
                out.append(pg.name)
            elif obj and str(obj).lower() in pg.objects:
                out.append(pg.name)
        return sorted(set(out))

    # -------------------------------------------------------------------- lint
    def lint(self) -> list[str]:
        """Deterministic hygiene sweep. Problems are reported, not silently repaired."""
        problems: list[str] = []
        for pg in self.all_pages():
            if not valid_page_name(pg.name):
                problems.append(f"{pg.name}: invalid page name")
            # D64 — an unestablished stub is SUPPOSED to be short and empty. Reporting it as
            # under budget every pass would bury the real problems in noise, which is how a
            # lint that nobody reads happens.
            if pg.confidence == "unestablished":
                continue
            if not pg.description:
                problems.append(f"{pg.name}: empty description (primary retrieval surface)")
            lo, hi = WORD_BUDGET.get(pg.type, WORD_BUDGET[DEFAULT_TYPE])
            wc = pg.word_count()
            if wc > hi:
                problems.append(f"{pg.name}: {wc} words over budget {hi} ({pg.type})")
            elif wc < lo:
                problems.append(f"{pg.name}: {wc} words under budget {lo} ({pg.type})")
            if not pg.applies_to and not pg.objects:
                problems.append(f"{pg.name}: unreachable — no applies_to and no objects")
            problems.extend(f"{pg.name}: {p}" for p in body_problems(pg.body))
        return problems

    def stats(self) -> dict[str, Any]:
        pages = self.all_pages()
        by_type: dict[str, int] = {}
        for pg in pages:
            by_type[pg.type] = by_type.get(pg.type, 0) + 1
        return {
            "n_pages": len(pages),
            "by_type": by_type,
            "total_words": sum(p.word_count() for p in pages),
        }

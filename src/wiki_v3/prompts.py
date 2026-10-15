"""Prompts for the three v3 roles.

D13 — NO HAND-FED DOMAIN KNOWLEDGE. Nothing here may pair a transform verb with an
appliance name, name a receptacle where an object "usually" is, or otherwise hand the wiki
what other arms must earn. `tests/test_no_handfed_knowledge.py` enforces this, and it
exists because a seeded mechanics table had 3 of its 6 entries factually wrong and reached
75 of 134 tasks. Read that test before editing any string in this file.

D25 — budgets, not adjectives. "Be concise" makes an LLM over-compress by 2-3x without
reporting it (wicer: 75%/50%/25% targets produced 35.4%/12.2%/8.2%). Every prompt that
produces text states a word count.
"""
from __future__ import annotations

# --------------------------------------------------------------------- READER

SELECT_SYSTEM = """\
You choose which memory pages are worth reading for one task. You do not answer the task.

Output ONLY page names, one per line, at most 3. No prose, no numbering, no explanation.
If none of the pages is relevant, output exactly: NONE

Choose a page only if its description suggests it bears on THIS task. Choosing an
irrelevant page wastes the budget; choosing none is a valid answer."""

SELECT_USER = """\
TASK: {task}
TASK FAMILY: {family}

ROOM (first look):
{observation}

AVAILABLE PAGES:
{descriptions}

Page names to read (max 3, one per line, or NONE):"""


COMPOSE_SYSTEM = """\
You write a short briefing for a household robot about to attempt one task. The robot
cannot see your notes — the briefing is the only thing that reaches it.

Rules:
- 40 to 90 words. Nothing longer.
- Write about THIS task: the exact object named in the task, and the exact destination
  named in the task.
- If a SEARCH ORDER is given below, repeat those location names exactly as written,
  in the same order. They are already ranked. Do not reorder, rename or generalise them.
- If the task needs something done to the object before it is placed, say what to do
  and with which receptacle or tool, in the robot's own command words, when the notes
  say so.
- Invent nothing. If the notes do not support a claim, leave it out.
- If nothing you were given applies to this task, output exactly: NONE

Do not explain your reasoning. Output the briefing text only."""

COMPOSE_USER = """\
TASK: {task}

ROOM (first look):
{observation}

{search_order}

NOTES RETRIEVED:
{pages}

Briefing (40-90 words, or NONE):"""


# --------------------------------------------------------------- CONSOLIDATOR

ANALYSE_SYSTEM = """\
You maintain a small wiki of notes for a household robot. You are shown a BATCH of recent
experience and the wiki's current page list. Your job is to decide what the wiki should
learn from the batch as a whole.

Look for regularities ACROSS the batch, not facts about one episode. A thing that happened
once is weak evidence; a thing that happened five times is a lesson. This is the only
point at which you can see many episodes together — use it.

Output one block per lesson, at most 6 blocks, in exactly this format:

PAGE: <kebab-case-page-name>
TYPE: procedure | entity | caution
FAMILIES: <comma-separated task families this applies to, or NONE>
OBJECTS: <comma-separated object types this is about, or NONE>
LESSON: <one sentence, what the wiki should say>
EVIDENCE: <which episodes or documents support it>

Prefer editing an existing page over creating a new one. Use an existing page name when
the lesson belongs there. Do not create a page for a single episode unless it names a
specific, repeatable trap.

No prose outside the blocks."""

ANALYSE_USER = """\
BATCH SUMMARY: {n_episodes} episodes ({n_success} succeeded), {n_documents} documents.
Per family: {families}

BATCH:
{batch}

CURRENT PAGES:
{descriptions}

Lessons:"""


REWRITE_SYSTEM = """\
You write one page of a wiki used by a household robot. Output the complete page and
nothing else — no commentary before or after.

Format exactly:

---
name: <kebab-case, matches the page name you were given>
description: <one sentence: what this page holds and when to use it. This is how the page
  gets found later, so name the task type and the objects it concerns.>
type: <procedure | entity | caution>
applies_to: [<task families, or empty>]
objects: [<object types, or empty>]
confidence: <attested | hedged>
---

# Preconditions
- <what must already be true, or omit this section>

# Steps
1. <one imperative action per line, in order>
2. <write the ACTION as the robot would issue it: `<verb> <obj> with <tool>`>

# Pitfalls
- <what goes wrong, or omit this section>

# Not established
- <what the evidence does not settle, or omit this section>

Rules:
- {budget_lo} to {budget_hi} words in the body. Nothing longer.
- Write what generalises. Do not transcribe a trajectory, and do not name specific
  receptacle instances (like "cabinet 4") in the steps — those differ every room.
- If the evidence is one episode, set confidence: hedged and say so in the body.
- Record what is NOT established as well as what is. An explicit unknown is useful.
- Keep any existing content that the new evidence does not contradict.
- Write each step in the robot's own command words, as the actions appear in the batch
  (for example `take <obj> from <place>`), not as a paraphrase like "navigate" or
  "obtain". Only command forms can be checked against what the environment accepted.
- Put every procedural claim in `# Steps`, as a NUMBERED line naming the action. Both
  gates parse that form and only that form; a procedure written as a prose paragraph is
  unverifiable and will be admitted without ever being checked (see D65).
- A caution or entity page may have no `# Steps` section at all. A procedure page without
  one is asserting nothing.
- Do NOT write a `sources:` line. Provenance is recorded by the system (D61); a
  sources line you invent is discarded, and inventing one wastes your budget.
- Name a task family in `applies_to` only if the page really is about it, and an
  object in `objects` only if you saw that object in this batch (D62)."""

REWRITE_USER = """\
PAGE TO WRITE: {name}

CURRENT CONTENT OF THIS PAGE (body only — the system owns the frontmatter):
{current}

WHAT THE BATCH SHOWED:
{findings}

SUPPORTING BATCH DETAIL:
{batch}

Write the complete page:"""

# Qwen3 + Ollama 0.9.5 — Known Gotchas, Root Causes, and Fixes

This document captures every concrete issue we hit running qwen3:14b via Ollama's
OpenAI-compatible endpoint. Each section covers what the symptom looks like in logs,
exactly why it happens, and what was changed to fix it.

---

## 1. `<think>` blocks in content (truncated or complete)

### Symptom
Log lines like:
```
[postmortem] parse failed for ep=25 — raw: "<think>\nOkay, let's try to figure out..."
[synthesis] PARSE FAILED ep=12 — raw (first 200 chars): "<think>\nLet me think about..."
```
Or raw LLM output contains `<think>` but no closing `</think>` and is otherwise empty.

### Root cause
Qwen3 (and other thinking models) emit reasoning inside `<think>...</think>` tags
before producing their final answer.  Via Ollama's OpenAI-compatible endpoint the
thinking tokens appear in `message.content` — not in a separate field.

Two failure modes:

**A — Complete block:** `complete()` strips them with a regex, but the regex is
`<think>.*?</think>` (lazy). If the model's answer starts immediately after `</think>`
with no whitespace, stripping works.  Harmless.

**B — Truncated block (the real problem):** When `max_tokens` is too small the model
hits the limit mid-reasoning and the completion ends *before* `</think>` is emitted.
The regex `<think>.*?</think>` does **not** match a block with no close tag — the raw
`<think>...` leaks through into the return value.  Any downstream JSON search on that
string finds no `{`, so parse fails.

### Fix
`src/llm.py` — `complete()`:
```python
content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
# NEW: strip truncated <think> block (max_tokens hit mid-thought)
if "<think>" in content and "</think>" not in content:
    content = content[:content.index("<think>")].strip()
return content.strip()
```

### Prevention
- For any call that outputs JSON or structured text, prefer `think=False`.
  The `/no_think` prefix sent by `complete(think=False)` is more reliable than the
  `think=False` API parameter alone on Ollama 0.9.5.
- If you do use `think=True`, set `max_tokens` to at least 3× the expected output
  length, because the thinking block consumes a large portion of the budget.

---

## 2. `budget_tokens` is not enforced on the OpenAI-compat endpoint

### Symptom
Setting `budget_tokens=150` (or any value) in `extra_body` has no observable effect
— qwen3:14b still uses 400-800 tokens on reasoning before producing output.

### Root cause
`budget_tokens` is an Ollama-native parameter supported on the **native** `/api/chat`
endpoint.  The **OpenAI-compatible** endpoint (`/v1/chat/completions`) silently ignores
unknown `extra_body` fields.

### Fix
Do not rely on `budget_tokens` as a budget cap.  Instead:
- Use `think=False` whenever the task is deterministic/structural (JSON formatting,
  section cleanup, classification).
- Reserve `think=True` only where deliberate multi-step reasoning genuinely helps
  (e.g. the retrieval pre-plan, main synthesis).
- Size `max_tokens` generously: if output is ~200 tokens and thinking is enabled,
  budget at least 1000-1500 tokens total.

---

## 3. `think=True` + `tools` returns HTTP 400

### Symptom
```
openai.BadRequestError: 400 Bad Request — think: true cannot be used with tools
```

### Root cause
Ollama 0.9.5 does not support extended thinking alongside tool calls in the same
request. The two features are mutually exclusive on the server side.

### Fix
`src/llm.py` — `run_tool_loop()` always forces `think=False` in its `extra_body`:
```python
extra_body={"think": False, "options": {"num_ctx": self._num_ctx}},
```
If you want the model to reason before a tool loop, use a separate `complete(think=True)`
call *before* starting the loop (see `_plan_retrieval()` in `agent_retriever.py`).

---

## 4. Postmortem parse failures (~57% failure rate)

### Symptom
```
DEBUG wiki.episode_postmortem [postmortem] parse failed for ep=N — raw: "<think>..."
```
98 out of 172 episodes in run 55618117.

### Root cause
`max_tokens=500, think=True, budget_tokens=150` — qwen3:14b uses 350-500 tokens on
the `<think>` block, exhausts the budget, and emits no JSON.  `budget_tokens` is
ignored (see issue 2 above).

### Fix
`src/wiki/episode_postmortem.py`:
```python
# Before:
raw = llm.complete(_SYSTEM, user, max_tokens=500, think=True, budget_tokens=150)
# After:
raw = llm.complete(_SYSTEM, user, max_tokens=600, think=False)
```
Diagnosis is a classification task (6 failure categories + JSON fields). It does not
benefit from extended reasoning.

---

## 5. Maintenance passes produce 0/N parseable responses

### Symptom
```
[wiki] maintenance: unparseable LLM response for pick_and_place_simple
[wiki] maintenance complete: 0/6 pages updated
```
Every maintenance pass at every checkpoint produced 0 updates in run 55618117.

### Root cause
Same as issue 4: `max_tokens=1000, think=True, budget_tokens=300` in `_clean_page()`.
The three-section output (Tips, Reflections, Failure Modes) requires ~400-500 tokens,
but the thinking block consumes everything first.

`_clean_object_page()` had the same problem: `max_tokens=1200, think=True, budget_tokens=400`.

### Fix
`src/wiki/agent_maintenance.py`:
```python
# _clean_page():
# Before:
response = self._llm.complete(system=_SYSTEM, user=user, max_tokens=1000, think=True, budget_tokens=300)
# After:
response = self._llm.complete(system=_SYSTEM, user=user, max_tokens=1500, think=False)

# _clean_object_page():
# Before:
response = self._llm.complete(system=_CONTRADICTION_SYSTEM, user=user, max_tokens=1200, think=True, budget_tokens=400)
# After:
response = self._llm.complete(system=_CONTRADICTION_SYSTEM, user=user, max_tokens=1500, think=False)
```
Both are formatting/pruning tasks with explicit rules — they do not need deliberation.

---

## 6. FALLBACK tier: empty tool loop → raw wiki dump injected as context

### Symptom
```
[retriever] pick_and_place_simple | tier=FALLBACK(direct) | tool_calls=0 | 2847 chars
```
The agent receives the entire raw task wiki page instead of a curated 150-word context.
A 2847-char dump is ~3× the intended context size and includes stale/verbose content
that confuses the action model.

### Root cause
When the retrieval tool loop exits without `finish_retrieval()` being called,
`compiled_context` is empty.  The old fallback code then called `_direct_context()` —
a function that reads the raw task wiki page and location priors directly.

This was designed as a safety net but became a source of noise: the model got
500-2800 chars of unstructured context instead of a clean retrieval failure signal.

### Fix
`src/wiki/agent_retriever.py` — `build_context()` and `build_midep_context()`:
```python
# Before:
if not context:
    context = self._direct_context(task_family)
    self.last_fallback = True

# After:
if not context:
    log.warning("[wiki] retrieval: no context — returning empty")
    self.last_fallback = True
    return ""
```
An empty return means the `build_action_prompt()` in `prompts/act.py` omits the
`=== WIKI CONTEXT ===` block entirely (guarded by `if memory_context:`).  The agent
acts without wiki guidance rather than with misleading guidance.

---

## 7. Agent outputs step count instead of action index

### Symptom
```
LLM action fallback: output '17' not in admissible; using nearest match 'look'
```
71 occurrences in run 55618117.  The model outputs a number like 17, 25, or 31 that
is out of range for the admissible actions list (typically 4-8 items).

### Root cause
The LOOP WARNING message in `src/prompts/act.py`:
```python
f"you have repeated {loop_actions} for {loop_len} steps"
```
`loop_len` is the number of steps the agent has been looping (e.g. 17).  The action
prompt immediately follows with "Your choice (a single integer from 1 to N):" — the
model conflates the step count with the requested integer.

### Fix
`src/llm.py` — `pick_action()`:
```python
# After finding a digit that's out of range, retry once with an explicit constraint.
_out_of_range_digit: str | None = None
for token in raw.split():
    token = token.strip(".,;:")
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(admissible):
            return admissible[idx]
        _out_of_range_digit = token
        break

if _out_of_range_digit is not None:
    n = len(admissible)
    log.warning("LLM action: digit %r out of range [1-%d] — retrying", _out_of_range_digit, n)
    raw = self.complete(
        system,
        user + f"\n\nIMPORTANT: Output a single integer from 1 to {n}. Nothing else.",
    ).strip()
    for token in raw.split():
        token = token.strip(".,;:")
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(admissible):
                return admissible[idx]
# Falls through to text matching if retry also fails.
```

---

## 8. Per-episode reflections (new feature)

### Feature
After each episode, the synthesis LLM now produces a `reflection` field alongside
`tip`, `failure_mode`, `object_name`, and `object_location`.

The reflection is 1-2 sentences answering:
- **Why** did this episode succeed or fail?
- What wiki knowledge helped (or was missing)?
- What to try differently on a retry?

Reflections accumulate in the "Reflections" section of each task wiki page and are
used by the maintenance agent to prune redundant lessons and by the retriever's
context (after a wiki refresh).

### Schema addition (`src/wiki/simple_updater.py`)
```json
"reflection": "[Ep N] <1-2 sentences: WHY this episode succeeded or failed. What wiki knowledge helped. What to try differently. null if nothing noteworthy.>"
```

---

## Quick reference: when to use `think=True` vs `think=False`

| Task | think= | Reason |
|------|--------|--------|
| Action selection (`pick_action`) | False | Fast, deterministic choice; thinking wastes latency |
| Retrieval pre-plan (`_plan_retrieval`) | True | Benefits from deliberation; output is plain text |
| Main synthesis (`update_all`) | True | Judgment call on abstract lessons; worth deliberation |
| Episode postmortem (`generate_postmortem`) | **False** | Classification task; thinking exhausts budget |
| Maintenance cleanup (`_clean_page`) | **False** | Formatting task; explicit rules; thinking exhausts budget |
| Object page contradiction scan | **False** | Explicit rules; thinking exhausts budget |
| Tool loops (`run_tool_loop`) | **Always False** | `think=True` + tools = HTTP 400 on Ollama 0.9.5 |

---

## Ollama API endpoint notes (v0.9.5)

- **OpenAI-compat** (`/v1/chat/completions`): Use this for all LLM calls.
  `extra_body={"think": True/False}` controls thinking mode.
  `budget_tokens` in `extra_body` is **silently ignored** (use `max_tokens` + `think=False` instead).
  `tools` parameter is supported but **requires `think=False`** (implicit).
- **Native** (`/api/chat`): Supports `budget_tokens` natively.  Not used in this codebase
  because the OpenAI SDK wraps the compat endpoint.
- **Thinking tokens in content**: Via OpenAI-compat, `<think>...</think>` appears in
  `message.content`.  The `LLMClient.complete()` method strips them before returning.
  Via native API, thinking tokens appear in `message.thinking` (separate field).
- **`parallel_tool_calls=False`**: Critical for retrieval.  Without this, qwen3 batches
  `read_priors` + `finish_retrieval` in one turn, filling `finish_retrieval` from model
  memory before the wiki data is read — pure hallucination.  Ollama 0.9.5 respects this flag.
- **`/no_think` prefix**: Qwen3-specific directive prepended to user messages when
  `think=False`.  More reliable than the `think=False` API parameter alone, which
  qwen3 sometimes ignores.

"""LLM client - wraps OpenAI-compatible API, defaults to Ollama."""
from __future__ import annotations

import logging
import re
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

log = logging.getLogger(__name__)

# Transport retries (H11). The OpenAI SDK retried failed requests twice by default and did
# not report it. The SDK's own retries are now off and complete() retries instead, so each
# retry is counted in `transport_retries`. A retry resends the identical request, so at
# temperature 0 it cannot change which action is chosen; it only costs time.
_MAX_ATTEMPTS = 3
_RETRY_WAIT_S = (2.0, 8.0)
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API (Ollama-compatible)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: str | None = "http://localhost:11434/v1",
        api_key: str | None = "ollama",
        num_ctx: int = 32768,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Recorded only. The server's OLLAMA_CONTEXT_LENGTH sets the window (H1).
        self.num_ctx = num_ctx
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=600.0, max_retries=0)
        # Requests resent after a connection error, timeout or retryable server status (H11).
        self.transport_retries = 0
        # Replies cut off at max_tokens (finish_reason "length"). Such a reply may be an
        # incomplete page or JSON object, so the count is reported per client.
        self.length_stops = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        # Inference-call count (SDD N4). Tokens alone understate how differently the arms
        # use a model: some call one inside their memory path and some do not. Counting
        # calls makes that a reported number, per client and per phase.
        self._n_calls = 0
        # Logging only (H14): how each action was chosen, counted per path. pick_action's
        # retry and fallback paths are part of the protocol, so how often each one fires
        # is reported per arm instead of assumed to be rare.
        self.pick_paths: dict[str, int] = {}
        # The most recent pick_action: {path, calls, prompt_tokens, completion_tokens, raw}.
        self.last_pick: dict = {}
        # Server token counts of the most recent complete() call (None if not reported).
        self._last_usage: tuple[int, int] | None = None

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        think: bool = False,
        budget_tokens: int | None = None,
    ) -> str:
        """Single-turn completion.

        think=True enables extended reasoning before the response.  Thinking
        tokens appear as <think>...</think> in the content and are stripped
        before returning — the caller only sees the final answer.

        think=False (default) prepends /no_think and sets think=False in the
        API body — belt-and-suspenders for Qwen3 which sometimes ignores the
        API param alone.

        budget_tokens: when think=True, caps how many tokens the model may spend
        on the <think> block, leaving the remainder of max_tokens for the actual
        output.  Passed as extra_body["budget_tokens"] (Ollama 0.9.5+ / Qwen3).
        If None, no cap is applied.
        """
        # No num_ctx here: the /v1 endpoint ignores it, so the server sets the window (H1).
        if think:
            user_msg = user
            extra: dict = {"think": True}
            if budget_tokens is not None:
                extra["budget_tokens"] = budget_tokens
        else:
            # /no_think is Qwen3's own directive — more reliable than the API param.
            user_msg = f"/no_think\n{user}"
            extra = {"think": False}

        self._n_calls += 1
        request = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            # An empty system prompt sends no system message, so a caller whose own code sent
            # only a user message (MACLA) gets the same chat template it was written for.
            messages=([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": user_msg}],
            extra_body=extra,
        )
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.chat.completions.create(**request)
                break
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is None or status in _RETRYABLE_STATUS
                if not retryable or attempt == _MAX_ATTEMPTS - 1:
                    raise
                self.transport_retries += 1
                log.warning("LLM request failed (%s); retry %d", exc.__class__.__name__,
                            attempt + 1)
                time.sleep(_RETRY_WAIT_S[min(attempt, len(_RETRY_WAIT_S) - 1)])
        if resp.usage:
            self._total_prompt_tokens += resp.usage.prompt_tokens
            self._total_completion_tokens += resp.usage.completion_tokens
            self._last_usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
        else:
            self._last_usage = None
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            self.length_stops += 1
        content = resp.choices[0].message.content or ""
        # Strip complete <think>...</think> blocks (thinking mode leaks these into content).
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # Strip truncated <think> block — model hit max_tokens mid-thought so </think>
        # was never emitted.  Everything after the open tag is pure reasoning, not output.
        if "<think>" in content and "</think>" not in content:
            content = content[:content.index("<think>")].strip()
        return content.strip()

    def pick_action(self, system: str, user: str, admissible: list[str]) -> str:
        """Call the LLM and constrain the output to an admissible command.

        The prompt asks for a number (1-indexed from the admissible list).
        We parse the number first; fall back to text matching for robustness.

        Records which path chose the action in `last_pick` and `pick_paths` (H14). The
        recording does not change which action is chosen.
        """
        self._pick = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "raw": ""}
        action, path = self._pick_action(system, user, admissible)
        if self._pick.pop("empty_retry", False):
            path = "empty_retry+" + path
        self._pick["path"] = path
        self.last_pick = self._pick
        self.pick_paths[path] = self.pick_paths.get(path, 0) + 1
        return action

    def _complete_counted(self, system: str, user: str) -> str:
        """complete() for pick_action, adding the call to the current step's record."""
        out = self.complete(system, user)
        self._pick["calls"] += 1
        if self._last_usage is not None:
            self._pick["prompt_tokens"] += self._last_usage[0]
            self._pick["completion_tokens"] += self._last_usage[1]
        self._pick["raw"] = (out or "")[:200]
        return out

    def _pick_action(self, system: str, user: str, admissible: list[str]) -> tuple[str, str]:
        """Choose an admissible action from the model's reply.

        The order matters (H18). A reply can be a number ("3"), a numbered line ("3. go to
        desk 1") or an action's text ("go to desk 1"). Action text often contains an
        instance number, so a number is read as the list index only when it is the whole
        reply or leads it. An action's text is matched before any other number is used.
        """
        raw = self._complete_counted(system, user).strip()
        # Retry once if the model returned nothing (thinking stripped, or empty gen).
        if not raw:
            log.warning("LLM action: empty response, retrying once")
            self._pick["empty_retry"] = True
            raw = self._complete_counted(system, user).strip()

        n = len(admissible)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        # 1. The reply is a number, or starts with one ("3", "3.", "3. go to desk 1").
        lead = _leading_int(raw)
        if lead is not None:
            if 1 <= lead <= n:
                bare = raw.strip().strip(".,;:)") == str(lead)
                return admissible[lead - 1], ("digit" if bare else "digit_leading")
            return self._range_retry(system, user, admissible, lead)

        # 2. The reply is an action's text.
        for candidate in [raw] + lines:
            if candidate in admissible:
                return candidate, "text_exact"
        for candidate in [raw] + lines:
            norm = _normalize(candidate)
            for cmd in admissible:
                if _normalize(cmd) == norm:
                    return cmd, "text_normalized"
        # Longest first, so "go to desk 1" is not taken for "go to desk 10".
        by_length = sorted(admissible, key=len, reverse=True)
        for line in (reversed(lines) if lines else [raw]):
            line_lower = line.lower()
            for cmd in by_length:
                if cmd.lower() in line_lower:
                    return cmd, "text_substring"

        # 3. A number somewhere inside a longer reply ("I choose 3").
        for token in raw.split():
            token = token.strip(".,;:()")
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= n:
                    return admissible[idx - 1], "digit_in_text"
                return self._range_retry(system, user, admissible, idx)

        # 4. Last resort: edit distance on the last non-empty line.
        # Exclude 'help' from candidates unless it is the only option —
        # it is never a productive action and short strings like '' match it trivially.
        last_line = lines[-1] if lines else raw
        candidates = [c for c in admissible if c != "help"] or admissible
        best = min(candidates, key=lambda c: _edit_distance(_normalize(c), _normalize(last_line)))
        log.warning(
            "LLM action fallback: output %r not in admissible; using nearest match %r",
            raw[:120], best,
        )
        return best, "edit_distance"

    def _range_retry(
        self, system: str, user: str, admissible: list[str], number: int
    ) -> tuple[str, str]:
        """The reply named a number outside [1, N]: ask once more, then clamp."""
        n = len(admissible)
        log.warning(
            "LLM action: number %d out of range [1-%d], retrying with explicit constraint",
            number, n,
        )
        raw = self._complete_counted(
            system,
            user + f"\n\nIMPORTANT: Output a single integer from 1 to {n}. Nothing else.",
        ).strip()
        retry = _leading_int(raw)
        if retry is None:
            for token in raw.split():
                token = token.strip(".,;:()")
                if token.isdigit():
                    retry = int(token)
                    break
        if retry is not None:
            if 1 <= retry <= n:
                return admissible[retry - 1], "range_retry_digit"
            # Retry also out of range: clamp. N+1 -> N is almost always the intent.
            return admissible[min(max(retry, 1), n) - 1], "range_retry_clamped"
        return admissible[min(max(number, 1), n) - 1], "range_clamped_no_digit"

    @property
    def token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "n_calls": self._n_calls,
        }

    @property
    def n_calls(self) -> int:
        """Inference calls made by this client (SDD N4)."""
        return self._n_calls

def _leading_int(s: str) -> int | None:
    """The integer a reply starts with ("3", "3.", "3) take ..."), or None."""
    m = re.match(r"\s*(\d+)(?![\w])", s)
    return int(m.group(1)) if m else None


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]

"""Token counts with the action model's own tokenizer (fix plan H6).

The memory budget, the injection meter and every token figure the harness reports use
these functions. Before H6 they used words / 0.75, which undercounts ALFWorld text: Qwen3
gives about 1.6 tokens per word, so a "1,024-token" block could hold about 1,600 real
tokens, and more for arms whose text has many instance numbers and punctuation.

The tokenizer is Qwen3's, pinned in assets/qwen3_tokenizer/ (see its README). It is the
vocabulary qwen3:14b uses in Ollama, so a count here matches what the model reads, apart
from the chat template's few special tokens. QWEN3_TOKENIZER overrides the path.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "assets" / "qwen3_tokenizer" / "tokenizer.json"
TRUNCATION_MARKER = " […memory truncated to protocol budget]"


def tokenizer_path() -> Path:
    return Path(os.environ.get("QWEN3_TOKENIZER") or _DEFAULT)


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    path = tokenizer_path()
    if not path.exists():
        # No fallback estimate: a different counter would silently change the budget.
        raise FileNotFoundError(
            f"Qwen3 tokenizer not found at {path}. It is committed under "
            f"assets/qwen3_tokenizer/; set QWEN3_TOKENIZER if it lives elsewhere."
        )
    return Tokenizer.from_file(str(path))


def count_tokens(text: str) -> int:
    """Tokens in `text`, without special tokens."""
    if not text:
        return 0
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def truncate_to_tokens(text: str, max_tokens: int | None) -> str:
    """Cut `text` so that it and the truncation marker fit in `max_tokens` tokens.

    The cut falls at the last whitespace inside the budget, so no word is split. Text that
    already fits is returned unchanged.
    """
    if not text or not max_tokens or max_tokens <= 0:
        return text
    enc = _tokenizer().encode(text, add_special_tokens=False)
    if len(enc.ids) <= max_tokens:
        return text
    room = max(1, max_tokens - count_tokens(TRUNCATION_MARKER))
    end = enc.offsets[room - 1][1]
    cut = text[:end]
    # Back off to a word boundary, unless the budget holds less than one word.
    space = max(cut.rfind(" "), cut.rfind("\n"))
    if space > 0 and text[end:end + 1] not in ("", " ", "\n"):
        cut = cut[:space]
    return cut.rstrip() + TRUNCATION_MARKER

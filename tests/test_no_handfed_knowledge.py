"""Guard: no code may hand an arm, or every arm, task knowledge the arms must learn.

A transform->tool binding ("heat with the microwave", "clean requires sinkbasin"), a
literal appliance command sequence, or a default tool written into code gives that arm
the answer for a whole task family. Every arm must learn such facts from its own
training episodes, and the shared harness must not supply them to any arm.

The scan covers every Python file under src/ (the harness, every arm, and the copied
MACLA engine) and every string literal in them except docstrings, which never reach a
model. Four checks:

  * binding   a string line that names a transform verb and a tool,
  * sequence  a string line that chains two or more arrows around a tool name,
  * mapping   a dict entry pairing a transform verb with a tool,
  * default   `x or "<tool>"`: a tool used when nothing was learned.

Reading the task text is allowed: a classifier that maps "examine ... under the desklamp"
to a family parses what the task itself says, and every arm receives the task.

Known offenders are listed in `KNOWN` with their entry in docs/hardcoded_inventory.md
and the stage that fixes them. Each check fails if an unlisted file offends, and also if
a listed file no longer offends, so the list is updated the moment an item is fixed.

Run: pytest tests/test_no_handfed_knowledge.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

# ALFWorld's transform tools and verbs.
TOOLS = ("microwave", "fridge", "sinkbasin", "desklamp", "floorlamp", "faucet")
TRANSFORMS = ("heat", "cool", "clean")

# check -> {file relative to src/: reason}. Each reason names the inventory entry and stage.
KNOWN: dict[str, dict[str, str]] = {
    "binding": {},
    "sequence": {},
    "mapping": {},
    "default": {},
}


def _files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _string_lines(tree: ast.AST) -> list[str]:
    """Every line of every non-docstring string literal (f-string parts included)."""
    skip = _docstring_ids(tree)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            lines.extend(node.value.splitlines())
    return lines


def _strings_in(node: ast.AST) -> str:
    return " ".join(
        c.value.lower() for c in ast.walk(node)
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    )


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{w}", text) for w in words)


def _binding(tree: ast.AST) -> list[str]:
    return [l.strip() for l in _string_lines(tree)
            if _has(l.lower(), TOOLS) and _has(l.lower(), TRANSFORMS)]


def _sequence(tree: ast.AST) -> list[str]:
    return [l.strip() for l in _string_lines(tree)
            if l.count("→") + l.count("->") >= 2 and _has(l.lower(), TOOLS)]


def _mapping(tree: ast.AST) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None:
                    continue
                ks, vs = _strings_in(k), _strings_in(v)
                if ((_has(ks, TRANSFORMS) and _has(vs, TOOLS))
                        or (_has(ks, TOOLS) and _has(vs, TRANSFORMS))):
                    hits.append(f"line {k.lineno}: {ks!r}: {vs!r}")
    return hits


def _default(tree: ast.AST) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for operand in node.values:
                if (isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                        and _has(operand.value.lower(), TOOLS)):
                    hits.append(f"line {node.lineno}: ... or {operand.value!r}")
    return hits


CHECKS = {"binding": _binding, "sequence": _sequence, "mapping": _mapping, "default": _default}


@pytest.mark.parametrize("check", list(CHECKS))
def test_no_handfed_knowledge(check):
    offenders: dict[str, list[str]] = {}
    for path in _files():
        hits = CHECKS[check](ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[_rel(path)] = hits
    known = KNOWN[check]
    new = {f: h for f, h in offenders.items() if f not in known}
    fixed = [f for f in known if f not in offenders]
    msg = []
    for f, h in new.items():
        msg.append(f"{f} hands the arms task knowledge ({check}):\n    " + "\n    ".join(h[:5]))
    for f in fixed:
        msg.append(f"{f} no longer offends ({check}); remove it from KNOWN: {known[f]}")
    assert not msg, "\n".join(msg)


def test_known_files_exist():
    missing = [f for entries in KNOWN.values() for f in entries if not (SRC / f).exists()]
    assert not missing, f"KNOWN lists files that do not exist: {missing}"


def test_detectors_catch_what_they_describe():
    """The checks themselves, on small examples, so a broken detector cannot pass silently."""
    tree = ast.parse(
        'A = "then heat it with the microwave"\n'
        'B = "go to fridge -> open fridge -> cool mug"\n'
        'C = {"heat": "microwave"}\n'
        'D = tool or "desklamp"\n'
        'def f():\n    """heat with the microwave (docstring, ignored)"""\n'
    )
    assert _binding(tree) == ["then heat it with the microwave", "go to fridge -> open fridge -> cool mug"]
    assert len(_sequence(tree)) == 1
    assert len(_mapping(tree)) == 1
    assert len(_default(tree)) == 1


def test_never_return_policy_is_absent():
    """The old v2 search policy ("one look per location ... never return") must not come back."""
    policy = re.compile(r"one look per location.*never return", re.IGNORECASE | re.DOTALL)
    for path in _files():
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if policy.search(line):
                pytest.fail(f"{path.name}:{i} still emits the 'Never return' policy: {line.strip()}")

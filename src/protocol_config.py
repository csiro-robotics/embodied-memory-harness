"""Unified-protocol config loading: _base.yaml + per-arm override with a frozen-key guard.

SDD F2/F6: under the unified evaluation protocol a per-arm config may differ from the
shared baseline ONLY in its `memory:` block. Everything that defines the protocol (model,
step budget, history window, seed, splits, episode counts, context budgets) lives in
configs/_base.yaml and is frozen. This module deep-merges a thin per-arm config onto the
base and HARD-FAILS if the arm tries to override a frozen key to a different value, so a
silent protocol drift can never sneak into a run.

Pure functions, no I/O beyond reading the declared `extends` file — unit-testable.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Leaf key names that define the protocol. If a per-arm config sets any of these to a
# value different from _base.yaml (at any nesting depth), loading fails.
FROZEN_KEYS: frozenset[str] = frozenset({
    "model",
    "temperature",
    "num_ctx",
    "max_tokens",
    "max_steps",
    "history_window",
    "seed",
    "num_episodes",
    "train_eval",
    "memory_block_max_tokens",
    "domain_randomization",
    "alfworld_version",
    "fewshot",
})


class ConfigGuardError(ValueError):
    """Raised when a per-arm config overrides a frozen protocol key."""


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base, returning a new dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _collect_frozen(node: dict, acc: dict[str, list] | None = None) -> dict[str, list]:
    """Collect every value the base assigns to each frozen key NAME, at any depth.

    Keyed by name rather than path so that a frozen key appearing in a subtree the base
    does not have (e.g. `memory_llm.num_ctx` when base only defines `action_llm.num_ctx`)
    is still measured against the protocol value.
    """
    if acc is None:
        acc = {}
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict):
                _collect_frozen(v, acc)
            elif k in FROZEN_KEYS:
                acc.setdefault(k, []).append(v)
    return acc


def _frozen_violations(
    base: dict, override: dict, declared: set[str] | None = None
) -> list[str]:
    """Return violations where the override changes a frozen key's value.

    Walks the ENTIRE override tree, including subtrees absent from the base. The earlier
    implementation only checked a key when the same path already existed in base, so an
    arm could introduce a whole new block (`memory_llm:`) and set frozen keys inside it
    with no check at all. Anything intentional must be declared via `deviations:`.
    """
    declared = declared or set()
    base_frozen = _collect_frozen(base)
    violations: list[str] = []

    def walk(node: dict, path: str = "") -> None:
        for k, v in node.items():
            here = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                walk(v, here)
            elif k in FROZEN_KEYS:
                if here in declared:
                    continue  # explicitly declared deviation (recorded in the summary)
                allowed = base_frozen.get(k)
                if allowed is None:
                    violations.append(
                        f"{here}: frozen key {k!r} is not defined in the base protocol "
                        f"(arm sets {v!r}); declare it under `deviations:` if intended"
                    )
                elif v not in allowed:
                    violations.append(f"{here}: base={allowed[0]!r} arm={v!r}")

    walk(override)
    return violations


def merge_and_guard(
    base: dict, override: dict, declared: set[str] | None = None
) -> dict:
    """Deep-merge override onto base; raise if it changes an undeclared frozen key."""
    violations = _frozen_violations(base, override, declared)
    if violations:
        raise ConfigGuardError(
            "per-arm config overrides frozen protocol keys (only `memory:` may differ, "
            "anything else must be declared under `deviations:`):\n  "
            + "\n  ".join(violations)
        )
    return deep_merge(base, override)


def load_config(config_path: str | Path) -> dict:
    """Load a config, applying _base.yaml inheritance + the frozen-key guard if declared.

    A per-arm config opts into the unified protocol with a top-level
    `extends: _base.yaml` (path relative to the config file's directory). Without it, the
    config is returned as-is (backward compatible with the legacy self-contained
    pipeline.yaml).
    """
    config_path = Path(config_path)
    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    # Declared deviations (SDD F9): an arm may knowingly depart from a frozen protocol
    # value, but only by naming the dotted path and a reason. Declared deviations are
    # carried into the merged config so compute_summary can report them per arm, which
    # turns a silent advantage into a results column.
    #   deviations:
    #     - key: train.memory_llm.num_ctx
    #       reason: "prose retrieval needs more context than single-action selection"
    deviations = cfg.pop("deviations", []) or []
    declared = {d["key"] for d in deviations if isinstance(d, dict) and "key" in d}

    extends = cfg.pop("extends", None)
    if not extends:
        if deviations:
            cfg["deviations"] = deviations
        return cfg

    base_path = (config_path.parent / extends).resolve()
    with base_path.open() as f:
        base = yaml.safe_load(f) or {}

    merged = merge_and_guard(base, cfg, declared)
    merged["deviations"] = deviations
    return merged

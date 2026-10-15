"""Unit tests for the unified-protocol M1 harness pieces (SDD §8).

All pure — no ALFWorld, no LLM, no GPU. Run: pytest tests/test_protocol.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alfworld_env import game_id, select_and_order_games
from tokens import count_tokens, truncate_to_tokens
from protocol_config import ConfigGuardError, deep_merge, merge_and_guard
from memory.base import EpisodeResult, MemoryContext, MemoryModule


# ---------------------------------------------------------------------------
# game_id + deterministic selection/ordering (F4/F5/N1)
# ---------------------------------------------------------------------------

def _mk(trial: str) -> str:
    return f"/data/json_2.1.1/train/{trial}/game.tw-pddl"

_FILES = [_mk(f"task-{i:03d}") for i in range(20)]


def test_game_id_is_trial_dir_name():
    assert game_id(_mk("pick_and_place_simple-Cup-X")) == "pick_and_place_simple-Cup-X"


def test_order_is_deterministic_for_same_seed():
    a = select_and_order_games(_FILES, "eval_out_of_distribution", seed=42, num_episodes=10)
    b = select_and_order_games(_FILES, "eval_out_of_distribution", seed=42, num_episodes=10)
    assert a == b
    assert len(a) == 10


def test_different_seed_gives_different_order():
    a = select_and_order_games(_FILES, "eval_out_of_distribution", seed=42, num_episodes=20)
    b = select_and_order_games(_FILES, "eval_out_of_distribution", seed=7, num_episodes=20)
    assert sorted(a) == sorted(b)  # same set
    assert a != b                  # different order


def test_train_intersection_filters_by_id():
    wanted = [game_id(_mk(f"task-{i:03d}")) for i in range(5)]
    out = select_and_order_games(_FILES, "train", seed=42, num_episodes=5, train500_ids=wanted)
    assert sorted(game_id(f) for f in out) == sorted(wanted)


def test_train_intersection_shortfall_hard_fails():
    wanted = [game_id(_mk(f"task-{i:03d}")) for i in range(5)] + ["does-not-exist"]
    with pytest.raises(RuntimeError, match="intersection produced"):
        select_and_order_games(_FILES, "train", seed=42, num_episodes=6, train500_ids=wanted)


def test_eval_split_ignores_train500_list():
    # train500_ids must not filter a non-train split.
    out = select_and_order_games(
        _FILES, "eval_out_of_distribution", seed=42, num_episodes=20, train500_ids=["x"]
    )
    assert len(out) == 20


# ---------------------------------------------------------------------------
# Cross-arm determinism (F4): the harness ordering does not depend on the arm
# ---------------------------------------------------------------------------

def test_order_independent_of_memory_arm():
    # The selection helper has no arm input; same (split, seed) -> same order for all
    # arms by construction. This guards against a future regression that sneaks an
    # arm-dependent input into ordering.
    orders = [
        select_and_order_games(_FILES, "train", seed=42, num_episodes=20)
        for _ in range(4)
    ]
    assert all(o == orders[0] for o in orders)


def test_order_independent_of_input_ordering():
    # The same SET of games shuffled with the same seed must yield the same ORDER even if
    # the input arrives in a different order (e.g. a different machine's os.walk). This is
    # what lets every arm — across machines/reruns — train on an identical sequence.
    forward = list(_FILES)
    reversed_in = list(reversed(_FILES))
    a = select_and_order_games(forward, "eval_out_of_distribution", seed=42, num_episodes=20)
    b = select_and_order_games(reversed_in, "eval_out_of_distribution", seed=42, num_episodes=20)
    assert a == b


# ---------------------------------------------------------------------------
# Config guard (F6)
# ---------------------------------------------------------------------------

_BASE = {
    "seed": 42,
    "train": {"env": {"max_steps": 50, "num_episodes": 500}, "agent": {"history_window": 20}},
}


def test_memory_only_override_is_allowed():
    arm = {"train": {"memory": {"type": "kg_hybrid", "top_k": 5}}}
    merged = merge_and_guard(_BASE, arm)
    assert merged["train"]["memory"]["type"] == "kg_hybrid"
    assert merged["train"]["env"]["max_steps"] == 50  # base preserved


def test_overriding_frozen_key_raises():
    arm = {"train": {"env": {"max_steps": 20}}}  # P-RAG's old 20-step budget
    with pytest.raises(ConfigGuardError, match="max_steps"):
        merge_and_guard(_BASE, arm)


def test_setting_frozen_key_to_same_value_is_ok():
    arm = {"seed": 42, "train": {"memory": {"type": "wiki_v3"}}}
    merged = merge_and_guard(_BASE, arm)
    assert merged["seed"] == 42


def test_deep_merge_does_not_mutate_base():
    arm = {"train": {"memory": {"type": "macla"}}}
    _ = deep_merge(_BASE, arm)
    assert "memory" not in _BASE["train"]


# --- the leak: frozen keys inside a subtree the base does not define -------------

_BASE_LLM = {
    "seed": 42,
    "train": {"action_llm": {"num_ctx": 8192, "max_tokens": 512}},
}


def test_frozen_key_in_new_subtree_is_caught():
    # An arm config can introduce a `memory_llm:` block absent from _base.yaml. The old
    # guard only checked a key when the same PATH existed in base, so num_ctx 16384 and
    # max_tokens 2048 passed silently: the arm ran its memory LLM at 2x context and
    # 4x output, unchecked.
    arm = {"train": {"memory_llm": {"num_ctx": 16384, "max_tokens": 2048}}}
    with pytest.raises(ConfigGuardError) as exc:
        merge_and_guard(_BASE_LLM, arm)
    msg = str(exc.value)
    assert "memory_llm.num_ctx" in msg and "memory_llm.max_tokens" in msg


def test_declared_deviation_is_permitted():
    arm = {"train": {"memory_llm": {"num_ctx": 16384}}}
    merged = merge_and_guard(_BASE_LLM, arm, declared={"train.memory_llm.num_ctx"})
    assert merged["train"]["memory_llm"]["num_ctx"] == 16384


def test_declaring_one_key_does_not_excuse_another():
    arm = {"train": {"memory_llm": {"num_ctx": 16384, "max_tokens": 2048}}}
    with pytest.raises(ConfigGuardError) as exc:
        merge_and_guard(_BASE_LLM, arm, declared={"train.memory_llm.num_ctx"})
    assert "max_tokens" in str(exc.value)
    assert "num_ctx" not in str(exc.value)


def test_frozen_key_unknown_to_base_is_flagged():
    # A frozen key the base never defines anywhere cannot be silently introduced.
    arm = {"train": {"env": {"domain_randomization": True}}}
    with pytest.raises(ConfigGuardError, match="not defined in the base protocol"):
        merge_and_guard({"seed": 42}, arm)


# ---------------------------------------------------------------------------
# Memory-block budget (§3.2)
# ---------------------------------------------------------------------------

def test_truncate_caps_long_block_at_the_budget_in_real_tokens():
    """H6: the kept text plus the marker is at most the budget, counted by the tokenizer."""
    text = " ".join(f"countertop {i}," for i in range(2000))
    out = truncate_to_tokens(text, max_tokens=100)
    assert "truncated" in out
    assert count_tokens(out) <= 100
    assert count_tokens(out) >= 90          # the budget is used, not wasted


def test_truncate_cuts_at_a_word_boundary():
    text = "alpha " * 50 + "supercalifragilistic " * 50
    out = truncate_to_tokens(text, max_tokens=80)
    body = out.split(" […memory")[0]
    assert all(w in ("alpha", "supercalifragilistic") for w in body.split())


def test_truncate_noop_when_under_budget_or_disabled():
    short = "a few words only"
    assert truncate_to_tokens(short, 1024) == short
    assert truncate_to_tokens(short, None) == short
    assert truncate_to_tokens(short, 0) == short


def test_count_tokens_uses_the_qwen3_vocabulary():
    """Pinned numbers from the committed tokenizer: ALFWorld text is about 1.6 tokens a word,
    not the 1.33 the old words/0.75 estimate assumed."""
    s = "You arrive at countertop 1. On the countertop 1, you see a mug 1, and a soapbottle 2."
    assert count_tokens(s) == 30
    assert count_tokens("") == 0


# ---------------------------------------------------------------------------
# freeze() contract (F8, §3.1)
# ---------------------------------------------------------------------------

class _DummyMem(MemoryModule):
    name = "dummy"

    def __init__(self):
        self.writes = 0

    def retrieve(self, *a, **k):
        return MemoryContext()

    def update(self, episode):
        if self.read_only:
            return
        self.writes += 1

    def reset(self):
        pass


def _ep():
    return EpisodeResult("id", "fam", "desc", [], True, 1)


def test_freeze_suppresses_writes():
    m = _DummyMem()
    m.update(_ep())
    assert m.writes == 1
    m.freeze()
    assert m.read_only is True
    m.update(_ep())
    assert m.writes == 1  # no further write after freeze


# ---------------------------------------------------------------------------
# Memory-injection telemetry (retrieval rate + tokens offered vs injected)
# ---------------------------------------------------------------------------

def _summary_with(inj: list[dict]) -> dict:
    """compute_summary over episodes carrying the given memory_injection blocks."""
    from runner import compute_summary

    class _Mem:
        name = "test_arm"

    class _LLM:
        model = "test-model"
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                       "total_tokens": 0, "n_calls": 0}

    results = [
        {"success": False, "steps_taken": 10, "task_family": "f",
         "memory_injection": block}
        for block in inj
    ]
    return compute_summary(
        results=results, memory=_Mem(), action_llm=_LLM(), elapsed=1.0,
    )


def _block(**kw) -> dict:
    base = {
        "steps": 10, "retrieval_steps": 10, "offered_tokens": 1000.0,
        "injected_tokens": 1000.0, "peak_offered_tokens": 100.0,
        "truncated_steps": 0,
    }
    base.update(kw)
    return base


def test_injection_aggregate_reports_per_step_rates():
    s = _summary_with([_block(), _block()])["memory_injection"]
    assert s["episodes_measured"] == 2
    assert s["steps_total"] == 20
    assert s["retrieval_rate"] == 1.0
    assert s["offered_tokens_per_step"] == pytest.approx(100.0)
    assert s["discarded_fraction"] == 0.0


def test_injection_aggregate_exposes_truncation_loss():
    """The P-RAG case: retrieves on every step, most of it never reaches the prompt."""
    s = _summary_with([
        _block(offered_tokens=4000.0, injected_tokens=1000.0,
               truncated_steps=10, peak_offered_tokens=4476.0),
    ])["memory_injection"]
    assert s["discarded_fraction"] == pytest.approx(0.75)
    assert s["truncated_step_rate"] == 1.0
    assert s["peak_offered_tokens"] == pytest.approx(4476.0)


def test_injection_aggregate_flags_a_silent_memory():
    """An arm that never injects anything must be distinguishable from one that does."""
    s = _summary_with([
        _block(retrieval_steps=0, offered_tokens=0.0, injected_tokens=0.0),
    ])["memory_injection"]
    assert s["retrieval_rate"] == 0.0
    assert s["injected_tokens_per_step"] == 0.0
    # No division by zero when nothing was offered.
    assert s["discarded_fraction"] == 0.0


def test_injection_absent_for_runs_without_the_telemetry():
    """Old episodes.jsonl has no memory_injection; the summary must not invent one."""
    from runner import compute_summary

    class _Mem:
        name = "test_arm"

    class _LLM:
        model = "m"
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                       "total_tokens": 0, "n_calls": 0}

    s = compute_summary(
        results=[{"success": True, "steps_taken": 5, "task_family": "f"}],
        memory=_Mem(), action_llm=_LLM(), elapsed=1.0,
    )
    assert s["memory_injection"] is None


# ---------------------------------------------------------------------------
# Trained store must be complete at eval (H13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm, files", [
    ("kg_hybrid", ["graph.json"]),                      # episodic.jsonl missing
    ("macla", []),                                      # state.pkl missing
    ("wiki_v3", ["journal.jsonl", "priors.json"]),      # evidence.json and pages/ missing
])
def test_eval_stops_on_an_incomplete_trained_store(tmp_path, arm, files):
    from memory import check_trained_store

    (tmp_path / arm).mkdir()
    for f in files:
        (tmp_path / arm / f).write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing"):
        check_trained_store(arm, tmp_path)


def test_complete_trained_store_passes(tmp_path):
    from memory import TRAINED_STORE_FILES, check_trained_store

    for arm, files in TRAINED_STORE_FILES.items():
        (tmp_path / arm).mkdir()
        for f in files:
            (tmp_path / arm / f).mkdir() if f == "pages" else (tmp_path / arm / f).write_text("")
        check_trained_store(arm, tmp_path)
    check_trained_store("no_memory", tmp_path)


# ---------------------------------------------------------------------------
# Frozen keys nothing else reads (H15) and the store hash (H17)
# ---------------------------------------------------------------------------

def test_protocol_keys_are_checked():
    import importlib.metadata as md

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from run_pipeline import check_protocol_keys

    assert check_protocol_keys({"fewshot": "none"}) == []
    assert any("fewshot" in p for p in check_protocol_keys({"fewshot": "three"}))
    try:
        md.version("alfworld")
    except md.PackageNotFoundError:
        pytest.skip("alfworld not installed here")
    assert any("alfworld_version" in p
               for p in check_protocol_keys({"alfworld_version": "0.0.1"}))


def test_store_digest_detects_any_change(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from run_pipeline import store_digest

    assert store_digest(tmp_path / "absent") is None
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "a.md").write_text("x")
    first = store_digest(tmp_path)
    assert store_digest(tmp_path) == first
    (tmp_path / "pages" / "a.md").write_text("y")
    assert store_digest(tmp_path) != first

"""Stage 1 logging (fix plan H12, H14): what each episode and each action records.

The logging must not change behaviour: pick_action must choose the same action as
before, and the trajectory the arms receive must keep exactly its old keys.

Run: pytest tests/test_logging.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_episode  # noqa: E402
from llm import LLMClient  # noqa: E402
from memory.base import MemoryContext, MemoryModule  # noqa: E402

ADM = ["go to shelf 1", "go to desk 1", "take mug 1 from desk 1", "look"]


def _client(replies: list[str]) -> LLMClient:
    """A real LLMClient whose complete() returns scripted replies (no server needed)."""
    llm = LLMClient(model="fake")
    it = iter(replies)
    llm.complete = lambda system, user, **kw: next(it)
    return llm


@pytest.mark.parametrize("replies, action, path, calls", [
    (["2"], "go to desk 1", "digit", 1),
    (["", "3"], "take mug 1 from desk 1", "empty_retry+digit", 2),
    (["9", "1"], "go to shelf 1", "range_retry_digit", 2),
    (["9", "7"], "look", "range_retry_clamped", 2),
    (["9", "none"], "look", "range_clamped_no_digit", 2),
    (["look"], "look", "text_exact", 1),
    (["Go to desk 1!"], "go to desk 1", "text_normalized", 1),
    (["I will: go to shelf 1!"], "go to shelf 1", "text_substring", 1),
    (["go to dsk"], "go to desk 1", "edit_distance", 1),
    (["3. take mug 1 from desk 1"], "take mug 1 from desk 1", "digit_leading", 1),
    (["I choose option 2"], "go to desk 1", "digit_in_text", 1),
    (["I choose option 9", "4"], "look", "range_retry_digit", 2),
])
def test_pick_action_records_its_path(replies, action, path, calls):
    llm = _client(replies)
    assert llm.pick_action("sys", "user", ADM) == action
    assert llm.last_pick["path"] == path
    assert llm.last_pick["calls"] == calls
    assert llm.pick_paths == {path: 1}


def test_instance_number_in_a_text_reply_is_not_read_as_the_index():
    """H18: "go to desk 1" names an action. Its instance number used to be read as action
    index 1, which picked the shelf."""
    llm = _client(["go to desk 1"])
    assert llm.pick_action("sys", "user", ADM) == "go to desk 1"
    assert llm.last_pick["path"] == "text_exact"


def test_text_match_prefers_the_longest_action():
    """"go to desk 10" contains "go to desk 1"; the longer, exact action must win."""
    adm = ["go to desk 1", "go to desk 10"]
    llm = _client(["I will go to desk 10 now"])
    assert llm.pick_action("sys", "user", adm) == "go to desk 10"


def test_transport_retries_are_counted():
    """H11: a failed request is resent by complete() itself and counted."""
    import httpx
    from openai import APIConnectionError

    import llm as llm_mod

    client = LLMClient(model="fake")
    calls = {"n": 0}

    class _Resp:
        usage = None

        class _Choice:
            class message:
                content = "2"
        choices = [_Choice]

    def create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIConnectionError(request=httpx.Request("POST", "http://x"))
        return _Resp

    client._client.chat.completions.create = create
    old = llm_mod._RETRY_WAIT_S
    llm_mod._RETRY_WAIT_S = (0.0,)
    try:
        assert client.complete("s", "u") == "2"
    finally:
        llm_mod._RETRY_WAIT_S = old
    assert client.transport_retries == 1
    assert client.n_calls == 1


def test_pick_paths_accumulate():
    llm = _client(["1", "2", "look"])
    for _ in range(3):
        llm.pick_action("sys", "user", ADM)
    assert llm.pick_paths == {"digit": 2, "text_exact": 1}


# --------------------------------------------------------------- episode logs

class _Env:
    """Two steps: go to desk, take mug (success)."""

    task_description = "put a mug in shelf"
    task_family = "pick_and_place"
    task_id = "episode_1"
    game_file = "/data/json_2.1.1/train/pick_and_place-Mug-None-Shelf-1/trial_1/game.tw-pddl"

    def reset(self):
        self._t = 0
        return "You are in a room.", list(ADM)

    def step(self, action):
        self._t += 1
        done = self._t >= 2
        return f"obs {self._t}", (1.0 if done else 0.0), done, list(ADM)


class _Mem(MemoryModule):
    """Serves 'A' at step 0, then 'B' from step 1 on."""

    name = "test"

    def __init__(self):
        self.seen = None

    def retrieve(self, task, obs, adm, step):
        return MemoryContext(system_context="A" if step == 0 else "B", metadata={"s": step >= 1})

    def update(self, episode):
        self.seen = episode

    def reset(self):
        pass


def test_episode_records_game_file_memory_log_and_step_log():
    mem = _Mem()
    r = run_episode(env=_Env(), memory=mem, llm=_client(["2", "3"]), max_steps=5)

    assert r.game_file.endswith("trial_1/game.tw-pddl")
    # Logged at step 0 and again when the text changed, not on every step.
    assert [(e["step"], e["text"]) for e in r.memory_log] == [(0, "A"), (1, "B")]
    assert r.memory_log[1]["metadata"] == {"s": True}
    assert [s["step"] for s in r.step_log] == [0, 1]
    assert r.step_log[0]["pick_path"] == "digit"
    assert r.step_log[0]["action_calls"] == 1


def test_trajectory_keys_are_unchanged():
    """Arms serialise the trajectory into their own prompts; logging must not touch it."""
    mem = _Mem()
    run_episode(env=_Env(), memory=mem, llm=_client(["2", "3"]), max_steps=5)
    for entry in mem.seen.trajectory:
        assert set(entry) == {"step", "action", "observation", "admissible_commands", "score"}


def test_runner_writes_gzip_log_and_reports_paths(tmp_path):
    from runner import compute_summary, run_phase

    mem = _Mem()
    llm = _client(["2", "3"])
    results, elapsed = run_phase(
        env=_Env(), memory=mem, action_llm=llm, num_episodes=1, max_steps=5,
        history_window=5, out_dir=tmp_path, phase="TEST",
    )
    with gzip.open(tmp_path / "episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["memory_log"][0]["text"] == "A"
    assert rows[0]["game_file"].endswith("game.tw-pddl")

    diag = json.loads((tmp_path / "diagnostics.jsonl").read_text().splitlines()[0])
    assert diag["game_file"].endswith("game.tw-pddl")

    s = compute_summary(results=results, memory=mem, action_llm=llm, elapsed=elapsed)
    assert s["token_usage"]["action_pick_paths"] == {"digit": 2}


def test_unencodable_metadata_does_not_fail_the_episode(tmp_path):
    """An arm's metadata is logged as-is; a value JSON cannot encode must not turn a
    finished episode into an errored failure."""
    from runner import run_phase

    class _SetMem(_Mem):
        def retrieve(self, task, obs, adm, step):
            return MemoryContext(system_context="A", metadata={"ids": {1, 2}})

    results, _ = run_phase(
        env=_Env(), memory=_SetMem(), action_llm=_client(["2", "3"]), num_episodes=1,
        max_steps=5, history_window=5, out_dir=tmp_path, phase="TEST",
    )
    assert len(results) == 1, "a failed log write added a second, errored record"
    assert results[0]["success"] is True
    assert not results[0].get("errored")


def test_episode_records_prompts_timing_and_update_report():
    class _ReportingMem(_Mem):
        def update(self, episode):
            self.last_update_report = {"written": 2}

    r = run_episode(env=_Env(), memory=_ReportingMem(), llm=_client(["2", "3"]), max_steps=5)
    assert r.system_prompt
    assert "put a mug in shelf" in r.step_log[0]["user_prompt"]
    assert set(r.memory_timing) == {"retrieve_s", "action_s", "update_s"}
    assert r.memory_update == {"written": 2}


def test_update_reports_are_summed_per_key():
    from runner import _sum_reports

    out = _sum_reports([
        {"inserted": 2, "recipe": True, "meta": "created", "by_rel": {"is-at": 3},
         "n_after": 5},
        {"inserted": 1, "recipe": False, "meta": "none", "by_rel": {"is-at": 1, "affords": 2},
         "n_after": 7},
    ])
    assert out == {"inserted": 3, "recipe": 1, "meta": {"created": 1, "none": 1},
                   "by_rel": {"is-at": 4, "affords": 2}, "n_after": 7}


# --------------------------------------------------------------- H9, H10

def test_frozen_store_is_not_offered_the_episode():
    """H10: after freeze() the harness does not call update() at all."""
    mem = _Mem()
    mem.freeze()
    r = run_episode(env=_Env(), memory=mem, llm=_client(["2", "3"]), max_steps=5)
    assert mem.seen is None
    assert r.success is True
    assert r.memory_update == {}


def test_update_failure_keeps_the_episode_result():
    """H10: an exception in update() is recorded, not raised, so a success stays a success."""
    class _BrokenMem(_Mem):
        def update(self, episode):
            raise RuntimeError("disk full")

    r = run_episode(env=_Env(), memory=_BrokenMem(), llm=_client(["2", "3"]), max_steps=5)
    assert r.success is True
    assert r.memory_update["update_errors"] == 1
    assert "disk full" in r.memory_update["update_error"]


def test_no_refresh_hook_remains():
    """H9: the refresh mechanism did nothing for any arm and is gone."""
    import agent
    from memory.base import EpisodeResult

    assert not hasattr(MemoryModule, "refresh_context")
    assert not hasattr(agent, "_is_phase_transition")
    assert "refreshes" not in EpisodeResult.__dataclass_fields__


def test_memory_clients_use_the_memory_server(monkeypatch):
    """H19: memory-side calls go to the job's second server; action calls never do."""
    from runner import build_llm

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1111/v1")
    monkeypatch.setenv("OLLAMA_MEMORY_BASE_URL", "http://127.0.0.1:2222/v1")
    assert "1111" in str(build_llm({"model": "m"})._client.base_url)
    assert "2222" in str(build_llm({"model": "m"}, role="memory")._client.base_url)
    monkeypatch.delenv("OLLAMA_MEMORY_BASE_URL")
    assert "1111" in str(build_llm({"model": "m"}, role="memory")._client.base_url)

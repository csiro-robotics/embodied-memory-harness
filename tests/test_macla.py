"""Tests for the MACLA arm after Stage 5 (paper-faithful rewrite, DEC-11/DEC-12).

Each test names the paper section or change-log item it checks. The model is a scripted
client, and the sentence embedder is switched off, so the arm falls back to word-set
similarity; no test depends on MiniLM's numbers.
Run: pytest tests/test_macla.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytest.importorskip("sklearn", reason="MACLA.py needs scikit-learn")
pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("MACLA", reason="download MACLA.py into src/ (see README)")

import MACLA  # noqa: E402
MACLA._EMBED_AVAILABLE = False

from memory.base import EpisodeResult, MemoryContext  # noqa: E402
from memory.macla_memory import (  # noqa: E402
    REFINE_SYSTEM, SEGMENT_SYSTEM, MACLAMemory, _beta_entropy,
)


class _Client:
    """Scripted harness client: a segmentation reply and a refinement reply."""

    def __init__(self, segments=None, refine=None):
        self.segments = segments
        self.refine = refine
        self.calls = []
        self.token_usage = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, system, user, **kw):
        self.calls.append((system, user, kw))
        u = self.token_usage
        self.token_usage = {"n_calls": u["n_calls"] + 1,
                            "prompt_tokens": u["prompt_tokens"] + 7,
                            "completion_tokens": u["completion_tokens"] + 3}
        if system == SEGMENT_SYSTEM:
            return self.segments if isinstance(self.segments, str) else json.dumps(self.segments)
        if system == REFINE_SYSTEM:
            return json.dumps(self.refine) if self.refine is not None else "no idea"
        return ""


def _arm(tmp_path, client=None, **kw):
    return MACLAMemory(run_dir=tmp_path, llm=client, **kw)


def _episode(i, task, family, steps, success):
    """ALFWorld-style trajectory: steps are (action, reply). The trajectory holds each
    step's observation BEFORE its action; the last reply goes to final_observation."""
    traj = []
    adm = ["go to countertop 1", "go to microwave 1", "go to cabinet 3",
           "go to sinkbasin 1", "go to fridge 1", "go to diningtable 1"]
    obs = ("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
           "Looking quickly around you, you see a cabinet 3, a countertop 1, a microwave 1, "
           f"a sinkbasin 1, a fridge 1, and a diningtable 1.\n\nYour task is to: {task}.")
    for s, (action, nxt) in enumerate(steps):
        traj.append({"step": s, "action": action, "observation": obs,
                     "admissible_commands": list(adm), "score": 0.0})
        obs = nxt
        if nxt.lower().startswith("you ") and not nxt.lower().startswith("you are"):
            adm = adm + [f"marker {s}"]
    return EpisodeResult(task_id=f"r{i}", task_family=family, task_description=task,
                         trajectory=traj, success=success, steps_taken=len(traj),
                         final_observation=steps[-1][1] if steps else "")


HEAT_TASK = "heat some plate and put it in cabinet"
HEAT = [
    ("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a plate 2."),
    ("take plate 2 from countertop 1", "You pick up the plate 2 from the countertop 1."),
    ("go to microwave 1", "You arrive at microwave 1. The microwave 1 is closed."),
    ("heat plate 2 with microwave 1", "You heat the plate 2 using the microwave 1."),
    ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is closed."),
    ("open cabinet 3", "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing."),
    ("move plate 2 to cabinet 3", "You move the plate 2 to the cabinet 3."),
]
# ALFWorld's first observation ends with the task, as the real one does.
FIRST_OBS = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 3, "
             "a countertop 1 and a microwave 1.\n\nYour task is to: " + HEAT_TASK + ".")
HEAT_SEGMENTS = [
    {"start": 0, "end": 1, "goal": "find and take the plate",
     "preconditions": ["hands empty"], "postconditions": ["holding the plate 2"]},
    {"start": 2, "end": 3, "goal": "heat the plate with the microwave",
     "preconditions": ["holding the plate"], "postconditions": ["plate is hot"]},
    {"start": 4, "end": 6, "goal": "put the plate in the cabinet",
     "preconditions": ["holding a hot plate"], "postconditions": ["plate 2 is in cabinet 3"]},
]


def _train_heat(arm, n=1, start=0):
    for i in range(start, start + n):
        arm.update(_episode(i, HEAT_TASK, "pick_heat_then_place_in_recep", HEAT, True))


def _procs(arm):
    return arm.agent.memory_system.procedural_memory


def _metas(arm):
    return arm.agent.memory_system.meta_procedural_memory


# ---------------------------------------------------------------- interface

def test_constructs_and_names(tmp_path):
    assert _arm(tmp_path).name == "macla"


def test_empty_store_is_zero_shot(tmp_path):
    ctx = _arm(tmp_path).retrieve(HEAT_TASK, "You are in a room.", ["go to cabinet 3"], 0)
    assert isinstance(ctx, MemoryContext)
    assert ctx.system_context == "" and ctx.metadata["path"] == "zero_shot"


def test_freeze_blocks_update_and_finalize(tmp_path):
    client = _Client(HEAT_SEGMENTS)
    arm = _arm(tmp_path, client)
    arm.freeze()
    _train_heat(arm)
    arm.finalize()
    assert arm._n_episodes == 0 and not _procs(arm) and not client.calls
    assert not (tmp_path / "macla" / "state.pkl").exists()


# ---------------------------------------------------------------- §4.1 extraction

def test_llm_segments_become_placeholder_procedures(tmp_path):
    """DEC-11: one call per successful episode; each segment is a procedure whose π is
    its own cleaned actions, with the task's object and target as placeholders."""
    client = _Client(HEAT_SEGMENTS)
    arm = _arm(tmp_path, client)
    _train_heat(arm)
    assert len(client.calls) == 1 and client.calls[0][0] == SEGMENT_SYSTEM
    by_verb = {arm._proc_verb[k]: e.procedure for k, e in _procs(arm).items()}
    assert set(by_verb) == {"take", "heat", "move"}
    assert by_verb["take"].steps == ["go to <source>", "take <object> from <source>"]
    assert by_verb["heat"].steps == ["go to microwave", "heat <object> with microwave"]
    assert by_verb["move"].steps[-1] == "move <object> to <target>"
    assert by_verb["move"].postconditions == ["<object> is in <target>"]
    assert by_verb["heat"].goal == "heat the <object> with the microwave"
    assert arm.last_update_report["segment_llm"] == 1


def test_unusable_reply_falls_back_to_navigation_cuts(tmp_path):
    arm = _arm(tmp_path, _Client("I cannot do that"))
    _train_heat(arm)
    assert {arm._proc_verb[k] for k in _procs(arm)} == {"take", "heat", "move"}
    assert arm.last_update_report["segment_fallback"] == 1


def test_single_object_and_wrapped_replies_are_accepted(tmp_path):
    """Seen on M3 (job 60494819): a short episode's reply can be one object, not a list."""
    one = {"start": 0, "end": 6, "goal": "put the plate in the cabinet"}
    for reply in (json.dumps(one), json.dumps({"segments": HEAT_SEGMENTS}),
                  json.dumps(HEAT_SEGMENTS) + "]",               # stray bracket (run 60495244)
                  "Here is the split:\n" + json.dumps(HEAT_SEGMENTS) + "\nDone."):
        arm = _arm(tmp_path / str(len(reply)), _Client(reply))
        _train_heat(arm)
        assert arm.last_update_report.get("segment_llm") == 1, reply


def test_segmentation_replies_are_logged(tmp_path):
    arm = _arm(tmp_path, _Client("no json here"))
    _train_heat(arm)
    rows = [json.loads(l) for l in (tmp_path / "macla" / "llm_calls.jsonl").read_text().splitlines()]
    assert rows == [{"episode": 1, "kind": "segment", "ref": "r0", "usable": False,
                     "reply": "no json here"}]


def test_failed_episodes_keep_their_verified_substeps(tmp_path):
    """MACLA §1/§B.1: failed episodes contain correct substeps. A failure that took and
    heated the plate but never placed it yields the take and heat procedures, each with
    one success; it does not count toward meta-procedure formation."""
    client = _Client(HEAT_SEGMENTS)
    arm = _arm(tmp_path, client)
    arm.update(_episode(0, HEAT_TASK, "x", HEAT[:4], False))
    assert len(client.calls) == 1 and "did not complete the task" in client.calls[0][1]
    assert sorted(arm._proc_verb[k] for k in _procs(arm)) == ["heat", "take"]
    assert all((e.procedure.alpha, e.procedure.beta) == (2, 1) for e in _procs(arm).values())
    assert list(arm._history["pick_heat_then_place"]) == [()]


def test_failed_episodes_drop_unverified_and_off_target_substeps(tmp_path):
    """A heat the game rejected, and a placement onto a receptacle that is not the task's
    destination, are not skills (P1)."""
    steps = [HEAT[0], HEAT[1], HEAT[2],
             ("heat plate 2 with microwave 1", "Nothing happens."),
             ("go to diningtable 1", "You arrive at diningtable 1. On the diningtable 1, you see nothing."),
             ("move plate 2 to diningtable 1", "You move the plate 2 to the diningtable 1.")]
    segs = [{"start": 0, "end": 1, "goal": "take"}, {"start": 2, "end": 3, "goal": "heat"},
            {"start": 4, "end": 5, "goal": "put it down"}]
    arm = _arm(tmp_path, _Client(segs))
    arm.update(_episode(0, HEAT_TASK, "x", steps, False))
    assert [arm._proc_verb[k] for k in _procs(arm)] == ["take"]
    assert arm.last_update_report.get("failure_segment_off_target") == 1


def test_oscillation_split_across_segments_is_cancelled(tmp_path):
    """π is cleaned over the whole trajectory before it is cut, so a take/move pair the
    model put into different segments does not survive as its own procedure."""
    steps = [HEAT[0], HEAT[1],
             ("move plate 2 to countertop 1", "You move the plate 2 to the countertop 1."),
             ("take plate 2 from countertop 1", "You pick up the plate 2 from the countertop 1."),
             *HEAT[2:]]
    segs = [{"start": 0, "end": 1, "goal": "take the plate"},
            {"start": 2, "end": 3, "goal": "put it down and take it again"},
            {"start": 4, "end": 5, "goal": "heat the plate"},
            {"start": 6, "end": 8, "goal": "put the plate in the cabinet"}]
    arm = _arm(tmp_path, _Client(segs))
    arm.update(_episode(0, HEAT_TASK, "x", steps, True))
    places = [e.procedure.steps for k, e in _procs(arm).items() if arm._proc_verb[k] == "move"]
    assert places == [["go to <target>", "open <target>", "move <object> to <target>"]]



def test_swapping_copies_of_the_object_is_cancelled(tmp_path):
    """P28: taking plate 2 from the fridge, putting it down on the countertop and taking
    plate 1 from there is one pick-up once both are <object>, so no procedure stores a
    put-down followed by a pick-up (episode 1088 of f1500-macla, with tomatoes)."""
    one = [(a.replace("plate 2", "plate 1"), r.replace("plate 2", "plate 1")) for a, r in HEAT]
    steps = [("go to fridge 1", "You arrive at fridge 1. The fridge 1 is open. In it, you see a plate 2."),
             ("take plate 2 from fridge 1", "You pick up the plate 2 from the fridge 1."),
             ("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a plate 1."),
             ("move plate 2 to countertop 1", "You move the plate 2 to the countertop 1."),
             ("take plate 1 from countertop 1", "You pick up the plate 1 from the countertop 1."),
             *one[2:]]
    segs = [{"start": 0, "end": 1, "goal": "take the plate"},
            {"start": 2, "end": 4, "goal": "swap the plate"},
            {"start": 5, "end": 6, "goal": "heat the plate"},
            {"start": 7, "end": 9, "goal": "put the plate in the cabinet"}]
    arm = _arm(tmp_path, _Client(segs))
    arm.update(_episode(0, HEAT_TASK, "x", steps, True))
    for e in _procs(arm).values():
        verbs = [x.split()[0] for x in e.procedure.steps]
        assert not any(a == "move" and b == "take" for a, b in zip(verbs, verbs[1:])),             e.procedure.steps

def test_identical_procedures_merge_and_accumulate_evidence(tmp_path):
    """P1/P15: the same sub-task seen again merges; its Beta counts one success per
    episode (P19: a new procedure starts at Beta(1,1) plus one success, not two)."""
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm)
    assert all((e.procedure.alpha, e.procedure.beta) == (2, 1) for e in _procs(arm).values())
    _train_heat(arm, 2, start=1)
    assert len(_procs(arm)) == 3
    assert all((e.procedure.alpha, e.procedure.beta) == (4, 1) for e in _procs(arm).values())


def test_different_key_verbs_never_merge(tmp_path):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS), theta_dup=-1.0)   # merge anything similar
    _train_heat(arm)
    assert len(_procs(arm)) == 3


def test_cool_then_place_never_merges_into_a_plain_placement(tmp_path):
    """Stage 6: grouping by the last action let "cool then place" merge into "place" and
    lose its cool step. Procedures now merge only within the same action group."""
    cool = [HEAT[0], HEAT[1],
            ("go to fridge 1", "You arrive at fridge 1. The fridge 1 is closed."),
            ("cool plate 2 with fridge 1", "You cool the plate 2 using the fridge 1."),
            ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is open."),
            ("move plate 2 to cabinet 3", "You move the plate 2 to the cabinet 3.")]
    place = [HEAT[0], HEAT[1],
             ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is open."),
             ("move plate 2 to cabinet 3", "You move the plate 2 to the cabinet 3.")]
    one = lambda n: [{"start": 0, "end": n - 1, "goal": "put the plate in the cabinet",
                      "preconditions": ["plate is on countertop"],
                      "postconditions": ["plate is in cabinet"]}]
    arm = _arm(tmp_path, _Client(one(4)), theta_dup=-1.0)     # merge anything similar
    arm.update(_episode(0, "put some plate in cabinet", "x", place, True))
    arm.llm.segments = one(6)
    arm.update(_episode(1, "cool some plate and put it in cabinet", "x", cool, True))
    groups = sorted(arm._action_group(e.procedure.steps) for e in _procs(arm).values())
    assert groups == ["take+cool+move", "take+move"]


def test_merge_by_similarity_expands_conditions(tmp_path):
    """§4.1: a near-duplicate (same verb, similarity above θ_dup) merges by expanding
    its condition sets."""
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS), theta_dup=0.3)
    _train_heat(arm)
    other = [dict(s) for s in HEAT_SEGMENTS]
    other[1] = dict(other[1], preconditions=["holding the plate", "microwave is closed"])
    arm.llm.segments = other
    steps = list(HEAT)
    steps[2] = ("go to microwave 1", "You arrive at microwave 1. The microwave 1 is open.")
    arm.update(_episode(1, HEAT_TASK, "x", steps, True))
    heat = [e.procedure for k, e in _procs(arm).items() if arm._proc_verb[k] == "heat"]
    assert len(heat) == 1 and "microwave is closed" in heat[0].preconditions


def test_procedure_keys_are_deterministic(tmp_path):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm)
    assert sorted(_procs(arm)) == ["proc_000001", "proc_000002", "proc_000003"]


def test_tool_keeps_its_name_when_it_is_the_target(tmp_path):
    steps = [HEAT[0], HEAT[1],
             ("go to microwave 1", "You arrive at microwave 1. The microwave 1 is closed."),
             ("heat plate 2 with microwave 1", "You heat the plate 2 using the microwave 1."),
             ("move plate 2 to microwave 1", "You move the plate 2 to the microwave 1.")]
    segs = [{"start": 0, "end": 1, "goal": "take"}, {"start": 2, "end": 3, "goal": "heat"},
            {"start": 4, "end": 4, "goal": "place"}]
    arm = _arm(tmp_path, _Client(segs))
    arm.update(_episode(0, "heat some plate and put it in microwave", "x", steps, True))
    heat = [e.procedure.steps for k, e in _procs(arm).items() if arm._proc_verb[k] == "heat"]
    assert heat == [["go to microwave", "heat <object> with microwave"]]


# ---------------------------------------------------------------- §4.4 meta-procedures

def test_meta_procedure_forms_at_three_of_fifteen(tmp_path):
    """App. C: a sequence in 3 of the family's last 15 episodes becomes a
    meta-procedure, with Beta(1,1) plus the successes that formed it."""
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm, 2)
    assert not _metas(arm)
    _train_heat(arm, 1, start=2)
    assert len(_metas(arm)) == 1
    mp = next(iter(_metas(arm).values()))
    assert (mp.alpha, mp.beta) == (4, 1) and len(mp.sub_procedures) == 3
    assert set(mp.composition_policy["branches"]) == {"continue", "skip", "repeat", "abort"}
    _train_heat(arm, 1, start=3)
    assert len(_metas(arm)) == 1, "the same sequence must not form a second meta"


def _trained(tmp_path):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm, 3)
    arm._zero_shot = [0, 10]          # low zero-shot success -> low θ_conf
    return arm, next(iter(_metas(arm)))


def _run_meta(arm, mk, replies):
    """Start an episode, put meta-procedure `mk` in charge, then feed the replies."""
    task = "heat some plate and put it in cabinet"
    adm = ["go to countertop 1", "go to microwave 1", "go to cabinet 3"]
    arm.retrieve(task, "You are in the middle of a room.", adm, 0)
    arm._ep_meta = mk
    ctx = None
    for i, r in enumerate(replies, 1):
        ctx = arm.retrieve(task, r, adm, i)
    return ctx


def test_policy_skips_finished_stages_and_continues_the_next(tmp_path):
    arm, mk = _trained(tmp_path)
    ctx = _run_meta(arm, mk, ["You pick up the plate 1 from the countertop 1."])
    assert ctx.metadata["policy"] == ["skip", "continue", "pending"]
    assert "[skip: done]" in ctx.system_context and "[continue]" in ctx.system_context
    assert "heat plate with microwave" in ctx.system_context


def test_finished_stages_stay_finished(tmp_path):
    """P2: once the heat stage is done it stays done after the plate is put down."""
    arm, mk = _trained(tmp_path)
    ctx = _run_meta(arm, mk, ["You pick up the plate 1 from the countertop 1.",
                              "You heat the plate 1 using the microwave 1.",
                              "You move the plate 1 to the countertop 1."])
    assert ctx.metadata["policy"][1] == "skip"


def test_policy_repeats_retrieval_when_the_object_was_put_down(tmp_path):
    arm, mk = _trained(tmp_path)
    ctx = _run_meta(arm, mk, ["You pick up the plate 1 from the countertop 1.",
                              "You move the plate 1 to the countertop 1."])
    assert ctx.metadata["policy"] == ["repeat", "continue", "pending"]


def _stall(arm, mk, n):
    """Run meta `mk` for n steps in which nothing the running sub-procedure expects happens."""
    adm = ["go to countertop 1", "go to microwave 1", "go to cabinet 3"]
    arm.retrieve(HEAT_TASK, "You are in the middle of a room.", adm, 0)
    arm._ep_meta = mk
    return [arm.retrieve(HEAT_TASK, f"You arrive at cabinet 3. Step {i}.", adm, i)
            for i in range(1, n + 1)]


def test_policy_aborts_when_sub_procedure_failures_recur(tmp_path):
    """P25: a sub-procedure fails when its expected result has not appeared after as many
    steps as its sketch has; three failures in a row abort the meta-procedure."""
    arm, mk = _trained(tmp_path)
    first = _metas(arm)[mk].sub_procedures[0]
    budget = len(_procs(arm)[first].procedure.steps)
    ctxs = _stall(arm, mk, 3 * budget + 1)
    events = [c.metadata.get("policy_event") for c in ctxs]
    assert events.count("abort") == 1 and events[-1] == "abort"
    assert "abort" not in events[:-1]
    assert arm._ep_meta != mk
    later = arm.retrieve(HEAT_TASK, "You arrive at countertop 1.", [], 99)
    assert later.metadata.get("key") != mk, "an aborted meta is not chosen again"


def test_progress_resets_the_failure_count(tmp_path):
    arm, mk = _trained(tmp_path)
    first = _metas(arm)[mk].sub_procedures[0]
    budget = len(_procs(arm)[first].procedure.steps)
    _stall(arm, mk, 2 * budget + 1)
    assert arm._ep_meta_fails == 2
    arm.retrieve(HEAT_TASK, "You pick up the plate 1 from the countertop 1.", [], 50)
    assert arm._ep_meta_fails == 0 and arm._ep_meta == mk


def test_an_aborted_meta_records_a_failure(tmp_path):
    """P24/P26, Alg. 1 lines 11-12: the outcome updates the meta's posterior after an abort
    too, and its initial context joins the meta's failure set, which Eq. 4's Risk reads."""
    arm, mk = _trained(tmp_path)
    before = _metas(arm)[mk]
    a, b, n = before.alpha, before.beta, before.execution_count
    budget = len(_procs(arm)[_metas(arm)[mk].sub_procedures[0]].procedure.steps)
    _stall(arm, mk, 3 * budget + 1)
    arm.update(_episode(9, HEAT_TASK, "x", HEAT, True))   # aborted, even though it succeeded
    mp = _metas(arm)[mk]
    assert (mp.alpha, mp.beta, mp.execution_count) == (a, b + 1, n + 1)
    fails = arm._meta_ctx[mk]["failure"]
    assert len(fails) == 1
    assert arm._risk_of(fails, fails[0].observation_init) == 1.0


def test_meta_credit_uses_the_episode_outcome(tmp_path):
    arm, mk = _trained(tmp_path)
    _run_meta(arm, mk, ["You pick up the plate 1 from the countertop 1."])
    arm.update(_episode(9, HEAT_TASK, "x", HEAT[:2], False))
    mp = _metas(arm)[mk]
    assert (mp.alpha, mp.beta) == (4, 2)


def test_pick_two_counts_instances(tmp_path):
    """The n-th retrieval and placement stages need n objects taken and placed."""
    two = [
        ("go to countertop 1", "You arrive at countertop 1. You see a plate 1 and a plate 2."),
        ("take plate 1 from countertop 1", "You pick up the plate 1 from the countertop 1."),
        ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is open."),
        ("move plate 1 to cabinet 3", "You move the plate 1 to the cabinet 3."),
        ("go to countertop 1", "You arrive at countertop 1. You see a plate 2."),
        ("take plate 2 from countertop 1", "You pick up the plate 2 from the countertop 1."),
        ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is open."),
        ("move plate 2 to cabinet 3", "You move the plate 2 to the cabinet 3."),
    ]
    segs = [{"start": 0, "end": 1, "goal": "take a plate"}, {"start": 2, "end": 3, "goal": "place it"},
            {"start": 4, "end": 5, "goal": "take a plate"}, {"start": 6, "end": 7, "goal": "place it"}]
    arm = _arm(tmp_path, _Client(segs))
    task = "put two plate in cabinet"
    for i in range(3):
        arm.update(_episode(i, task, "pick_two_obj_and_place", two, True))
    mk = next(iter(_metas(arm)))
    assert len(_metas(arm)[mk].sub_procedures) == 4
    arm.retrieve(task, "You are in the middle of a room.", ["go to cabinet 3"], 0)
    arm._ep_meta = mk
    arm.retrieve(task, "You pick up the plate 1 from the countertop 1.", [], 1)
    ctx = arm.retrieve(task, "You move the plate 1 to the cabinet 3.", [], 2)
    assert ctx.metadata["policy"] == ["skip", "skip", "continue", "pending"]


# ---------------------------------------------------------------- §4.2 selection

def test_theta_conf_follows_eq_13(tmp_path):
    """P16: θ = ρ·R_max − (1−ρ)·C_fail − c_LLM with ρ the zero-shot success rate."""
    arm = _arm(tmp_path)
    arm._zero_shot = [3, 8]                      # ρ = 4/10 under Beta(1,1)
    assert arm.theta_conf == pytest.approx(0.4 - 0.6 * 0.5 - 0.15)


def test_zero_shot_rate_counts_episodes_without_memory(tmp_path):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm)
    arm.update(_episode(1, HEAT_TASK, "x", HEAT[:2], False))
    assert arm._zero_shot == [1, 2]


def test_eu_matches_eq_4(tmp_path):
    arm = _arm(tmp_path)
    eu = arm._eu(rel=0.8, alpha=3, beta=1, risk=0.5)
    assert eu == pytest.approx(0.8 * 0.75 - 0.5 * 0.25 * 0.5 + 0.1 * _beta_entropy(3, 1))


def test_retrieve_selects_by_expected_utility(tmp_path):
    arm, _mk = _trained(tmp_path)
    ctx = arm.retrieve(HEAT_TASK, FIRST_OBS,
                       ["go to cabinet 3"], 0)
    assert ctx.metadata["path"] in ("meta_procedure", "procedure")
    assert ctx.metadata["candidates"] and ctx.system_context
    assert "plate" in ctx.system_context and "<object>" not in ctx.system_context


def test_below_threshold_is_zero_shot(tmp_path):
    arm, _mk = _trained(tmp_path)
    arm._zero_shot = [1000, 1000]                 # θ_conf near R_max − c_LLM
    ctx = arm.retrieve(HEAT_TASK, "You are in the middle of a room.", [], 0)
    assert ctx.metadata["path"] == "zero_shot" and ctx.system_context == ""


def test_precondition_check_blocks_a_procedure(tmp_path):
    """CheckPre: a procedure learned while holding the object is not shown otherwise."""
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm)
    heat = next(k for k in _procs(arm) if arm._proc_verb[k] == "heat")
    arm.retrieve(HEAT_TASK, "You are in the middle of a room.", [], 0)
    from memory.macla_memory import _State
    assert not arm._pre_ok(heat, _State(), arm._ep_bind)
    assert arm._pre_ok(heat, _State(held=("plate", "1")), arm._ep_bind)


# ---------------------------------------------------------------- §4.3 refinement

def _refine_setup(tmp_path, refine):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS, refine))
    _train_heat(arm, 3)
    heat = next(k for k in _procs(arm) if arm._proc_verb[k] == "heat")
    for i in range(3):          # three failed uses of the heat procedure
        arm.retrieve(HEAT_TASK, "You are in the middle of a room.", [], 0)
        arm._ep_shown[heat] = 0
        arm.update(_episode(10 + i, HEAT_TASK, "x", HEAT[:2], False))
    return arm, heat


def test_refinement_edits_conditions_and_steps(tmp_path):
    """P5/P17: the comparison's output changes Ψ, π and Φ."""
    refine = {"preconditions_positive": ["microwave is closed"],
              "preconditions_negative": ["hands empty"],
              "steps": ["go to microwave", "open microwave", "heat <object> with microwave"],
              "postconditions": ["<object> is hot"]}
    arm, heat = _refine_setup(tmp_path, refine)
    p = _procs(arm)[heat].procedure
    assert "microwave is closed" in p.preconditions
    assert "<object> is hot" in p.postconditions
    assert p.steps == refine["steps"]
    assert sum(1 for c in arm.llm.calls if c[0] == REFINE_SYSTEM) == 1


def test_refinement_rejects_steps_nobody_took(tmp_path):
    refine = {"preconditions_positive": [], "preconditions_negative": [],
              "steps": ["go to microwave", "microwave <object>"], "postconditions": []}
    arm, heat = _refine_setup(tmp_path, refine)
    assert _procs(arm)[heat].procedure.steps == ["go to microwave", "heat <object> with microwave"]


def test_refinement_waits_for_new_evidence(tmp_path):
    """P12: no second call until three more uses, counted on α+β."""
    arm, heat = _refine_setup(tmp_path, {"preconditions_positive": []})
    n = sum(1 for c in arm.llm.calls if c[0] == REFINE_SYSTEM)
    arm.update(_episode(20, HEAT_TASK, "x", HEAT[:2], False))
    assert sum(1 for c in arm.llm.calls if c[0] == REFINE_SYSTEM) == n


# ---------------------------------------------------------------- §4.6 pruning

def test_pruning_keeps_procedures_a_meta_uses(tmp_path):
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS), N_p=3)
    _train_heat(arm, 3)
    used = set(next(iter(_metas(arm).values())).sub_procedures)
    arm._prune()
    assert used <= set(_procs(arm))


# ---------------------------------------------------------------- persistence, metering

def test_save_load_roundtrip(tmp_path):
    arm, mk = _trained(tmp_path)
    arm.finalize()
    revived = _arm(tmp_path / "eval")
    revived.load(tmp_path / "macla" / "state.pkl")
    revived.freeze()
    assert set(_procs(revived)) == set(_procs(arm)) and mk in _metas(revived)
    assert revived.theta_conf == pytest.approx(arm.theta_conf)
    ctx = revived.retrieve(HEAT_TASK, FIRST_OBS,
                           ["go to cabinet 3"], 0)
    assert ctx.system_context


def test_model_calls_are_metered(tmp_path):
    """H11: calls go through the harness client and are counted."""
    client = _Client(HEAT_SEGMENTS)
    arm = _arm(tmp_path, client)
    _train_heat(arm)
    s = arm.stats()
    assert s["memory_llm_calls"] == 1
    assert s["memory_prompt_tokens"] == 7 and s["memory_completion_tokens"] == 3


def test_segmentation_prompt_carries_no_task_knowledge(tmp_path):
    """HC: the prompt names no object, receptacle or appliance of its own."""
    from memory.macla_memory import REFINE_USER, SEGMENT_USER
    for text in (SEGMENT_USER, REFINE_USER):
        for word in ("microwave", "fridge", "sinkbasin", "apple", "mug", "cabinet", "lamp"):
            assert word not in text.lower()


def test_failed_episode_about_another_object_stores_nothing(tmp_path):
    """P22: the object is bound from the task text, so a failure that only handled another
    object teaches nothing, even when the task's object type was never picked up."""
    steps = [("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a pen 1."),
             ("take pen 1 from countertop 1", "You pick up the pen 1 from the countertop 1."),
             ("go to cabinet 3", "You arrive at cabinet 3. The cabinet 3 is closed.")]
    arm = _arm(tmp_path, _Client([{"start": 0, "end": 1, "goal": "take the pen"}]))
    arm.update(_episode(0, "put some cd in safe", "pick_and_place", steps, False))
    assert not _procs(arm)
    assert arm._bind("put some cd in safe", {"safe", "cabinet"}).obj == "cd"


def test_refinement_splits_joined_steps(tmp_path):
    """P23: a step returned as one joined string is split, then checked like any other."""
    refine = {"preconditions_positive": [], "preconditions_negative": [],
              "steps": ["go to microwave -> open microwave", "heat <object> with microwave"],
              "postconditions": []}
    arm, heat = _refine_setup(tmp_path, refine)
    assert _procs(arm)[heat].procedure.steps == [
        "go to microwave", "open microwave", "heat <object> with microwave"]


def test_meta_contexts_survive_save_and_load(tmp_path):
    arm, mk = _trained(tmp_path)
    budget = len(_procs(arm)[_metas(arm)[mk].sub_procedures[0]].procedure.steps)
    _stall(arm, mk, 3 * budget + 1)
    arm.update(_episode(9, HEAT_TASK, "x", HEAT[:2], False))
    arm.save(tmp_path / "s.pkl")
    other = _arm(tmp_path / "o")
    other.load(tmp_path / "s.pkl")
    assert len(other._meta_ctx[mk]["failure"]) == 1


def _credit_take(tmp_path, shown_at):
    """A failed episode that takes the plate at step 1 (its reply is observation 2).

    Extraction is switched off for that episode, so the only credit is the one for the
    procedure having been shown; a segment the agent repeats is credited separately
    (§4.1, P19).
    """
    arm = _arm(tmp_path, _Client(HEAT_SEGMENTS))
    _train_heat(arm)
    take = next(k for k in _procs(arm) if arm._proc_verb[k] == "take")
    p = _procs(arm)[take].procedure
    before = (p.alpha, p.beta)
    arm.retrieve(HEAT_TASK, "You are in the middle of a room.", [], 0)
    arm._ep_shown[take] = shown_at
    arm._extract = lambda result, bind: []
    arm.update(_episode(9, HEAT_TASK, "x", HEAT[:3], False))
    return (p.alpha - before[0], p.beta - before[1])


def test_credit_counts_only_effects_after_the_procedure_was_shown(tmp_path):
    """P27: CheckPost judges what followed the procedure. A take-procedure first shown
    after the object was already picked up is not credited with that pick-up."""
    assert _credit_take(tmp_path / "a", 0) == (1, 0)
    assert _credit_take(tmp_path / "b", 2) == (0, 1)

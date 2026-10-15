"""Unit tests for prompts/act.py.

Tests:
  - _build_progress(): progress summary from trajectory + observation
  - build_action_prompt(): full prompt construction with all optional parts
"""
from prompts.act import _build_progress, build_action_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(action: str, observation: str = "You are in the kitchen.") -> dict:
    return {"action": action, "observation": observation}


# ---------------------------------------------------------------------------
# _build_progress
# ---------------------------------------------------------------------------


class TestBuildProgressEmpty:
    def test_empty_trajectory_returns_empty_string(self):
        assert _build_progress([], "obs") == ""

    def test_header_and_defaults(self):
        out = _build_progress([_step("look")], "You are in the middle of a room.")
        assert out.startswith("PROGRESS:")
        assert "Holding: nothing" in out
        assert "Visited: none yet" in out


# The replies below are the exact ALFWorld 0.4.2 forms seen in Run 10's logs.
class TestBuildProgressHolding:
    def test_pickup_counts_only_when_the_simulator_confirms_it(self):
        # The reply to an action is the NEXT entry's observation (or the current one).
        traj = [_step("take mug 1 from countertop 1", "You arrive at countertop 1.")]
        out = _build_progress(traj, "You pick up the mug 1 from the countertop 1.")
        assert "Holding: mug 1" in out

    def test_failed_pickup_is_not_holding(self):
        traj = [_step("take mug 1 from countertop 1", "You arrive at countertop 1.")]
        assert "Holding: nothing" in _build_progress(traj, "Nothing happens.")

    def test_placement_clears_holding(self):
        traj = [
            _step("take mug 1 from countertop 1", "You arrive at countertop 1."),
            _step("move mug 1 to shelf 1", "You pick up the mug 1 from the countertop 1."),
        ]
        out = _build_progress(traj, "You move the mug 1 to the shelf 1.")
        assert "Holding: nothing" in out

    def test_older_put_phrasing_also_clears_holding(self):
        traj = [
            _step("take mug 1 from countertop 1", "x"),
            _step("put mug 1 in/on shelf 1", "You pick up the mug 1 from the countertop 1."),
        ]
        assert "Holding: nothing" in _build_progress(traj, "You put the mug 1 in/on the shelf 1.")

    def test_no_task_specific_flags(self):
        traj = [
            _step("heat mug 1 with microwave 1", "x"),
            _step("examine mug 1", "You heat the mug 1 using the microwave 1."),
        ]
        out = _build_progress(traj, "This is a normal mug 1.").lower()
        for word in ("heated", "cooled", "cleaned", "examined", "placed"):
            assert word not in out


class TestBuildProgressVisited:
    def test_confirmed_arrival_is_listed(self):
        traj = [_step("go to countertop 1", "You are in the middle of a room.")]
        out = _build_progress(traj, "You arrive at countertop 1. On it you see a mug 1.")
        assert "Visited: countertop 1" in out

    def test_current_location_is_listed_without_a_not_there_claim(self):
        # H3: the old list called the current location "object not there".
        traj = [_step("go to countertop 1", "x")]
        out = _build_progress(traj, "You arrive at countertop 1.")
        assert "not there" not in out.lower()
        assert "revisit" not in out.lower()

    def test_arrival_not_confirmed_is_not_listed(self):
        traj = [_step("go to countertop 1", "x")]
        assert "Visited: none yet" in _build_progress(traj, "Nothing happens.")

    def test_visits_are_deduplicated_in_order_and_not_capped(self):
        places = [f"drawer {i}" for i in range(1, 9)] + ["drawer 1"]
        traj = [_step(f"go to {p}", "x") for p in places]
        replies = [f"You arrive at {p}." for p in places]
        for t, r in zip(traj[1:], replies[:-1]):
            t["observation"] = r
        out = _build_progress(traj, replies[-1])
        assert "Visited: " + ", ".join(places[:-1]) in out


class TestBuildProgressLoop:
    def test_loop_warning_after_four_actions_on_two_commands(self):
        traj = [_step(a) for a in ["go to b 1", "go to a 1", "go to b 1", "go to a 1"]]
        out = _build_progress(traj, "obs")
        assert "LOOP WARNING: your last 4 actions repeated only 'go to a 1', 'go to b 1'." in out

    def test_loop_warning_text_is_sorted(self):
        # H5: set order depends on the hash seed; the text must not.
        a = _build_progress([_step(x) for x in ["z", "a", "z", "a"]], "o")
        b = _build_progress([_step(x) for x in ["a", "z", "a", "z"]], "o")
        assert a == b

    def test_no_loop_warning_with_three_commands(self):
        traj = [_step(a) for a in ["a", "b", "c", "a"]]
        assert "LOOP" not in _build_progress(traj, "obs")

    def test_no_loop_warning_under_four_steps(self):
        assert "LOOP" not in _build_progress([_step("a")] * 3, "obs")


# ---------------------------------------------------------------------------
# build_action_prompt
# ---------------------------------------------------------------------------


class TestBuildActionPromptStructure:
    def test_returns_two_strings(self):
        system, user = build_action_prompt(
            task="put apple on shelf",
            observation="You are in the kitchen.",
            admissible_commands=["go to shelf 1", "look"],
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_task_in_user_prompt(self):
        _, user = build_action_prompt(
            task="put apple on shelf",
            observation="You are in the kitchen.",
            admissible_commands=["go to shelf 1"],
        )
        assert "put apple on shelf" in user

    def test_observation_in_user_prompt(self):
        _, user = build_action_prompt(
            task="put apple on shelf",
            observation="You see a mug on the countertop.",
            admissible_commands=["go to shelf 1"],
        )
        assert "You see a mug on the countertop." in user

    def test_admissible_commands_listed(self):
        _, user = build_action_prompt(
            task="put apple on shelf",
            observation="You are in the kitchen.",
            admissible_commands=["go to shelf 1", "take apple 1 from countertop 1"],
        )
        assert "go to shelf 1" in user
        assert "take apple 1 from countertop 1" in user

    def test_your_action_marker_present(self):
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
        )
        assert "Your choice (a single integer" in user

    def test_system_prompt_is_nonempty(self):
        system, _ = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
        )
        assert len(system) > 50


class TestBuildActionPromptMemoryContext:
    def test_memory_context_passed_through_unchanged(self):
        memory_context = "VERIFIED:\n- countertop: 5x\n\nBACKGROUND:\n- fridge, maybe"
        _, user = build_action_prompt(
            task="put apple on shelf",
            observation="You are in the kitchen.",
            admissible_commands=["go to shelf 1"],
            memory_context=memory_context,
        )
        assert memory_context in user

    def test_knowledge_header_present_when_context_given(self):
        _, user = build_action_prompt(
            task="task", observation="obs", admissible_commands=["look"],
            memory_context="some hint",
        )
        assert "TASK KNOWLEDGE" in user

    def test_no_knowledge_header_when_no_context(self):
        _, user = build_action_prompt(task="task", observation="obs", admissible_commands=["look"])
        assert "TASK KNOWLEDGE" not in user

    def test_no_legend_for_any_arm_format(self):
        # H7: the harness no longer explains one arm's tags.
        _, user = build_action_prompt(
            task="task", observation="obs", admissible_commands=["look"],
            memory_context="VERIFIED:\n- tip\n",
        )
        assert "VERIFIED entries" not in user
        assert "BACKGROUND entries" not in user


class TestBuildActionPromptHistory:
    def test_recent_history_included(self):
        history = [{"action": "go to countertop 1", "observation": "You see nothing."}]
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
            recent_history=history,
        )
        assert "go to countertop 1" in user

    def test_history_shows_each_action_with_its_reply(self):
        # H2: the reply to an action is the observation stored with the NEXT action.
        history = [
            {"action": "go to countertop 1", "observation": "START"},
            {"action": "take mug 1 from countertop 1", "observation": "You arrive at countertop 1."},
        ]
        _, user = build_action_prompt(
            task="task",
            observation="You pick up the mug 1 from the countertop 1.",
            admissible_commands=["look"],
            recent_history=history,
        )
        assert "  > go to countertop 1\n    You arrive at countertop 1." in user
        assert ("  > take mug 1 from countertop 1\n"
                "    You pick up the mug 1 from the countertop 1.") in user
        assert "START" not in user

    def test_history_replies_are_not_truncated(self):
        # H16: the old 200-character cut hid the target in 5.3% of observations.
        long_reply = "You open the drawer 1. " + "x" * 300 + " mug 1."
        history = [{"action": "open drawer 1", "observation": "before"}]
        _, user = build_action_prompt(
            task="task", observation=long_reply, admissible_commands=["look"],
            recent_history=history,
        )
        assert f"    {long_reply}" in user

    def test_history_window_limits_shown_entries(self):
        history = [
            {"action": f"go to countertop {i}", "observation": f"step {i}"}
            for i in range(10)
        ]
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
            recent_history=history,
            history_window=3,
        )
        assert "go to countertop 9" in user   # last 3 visible
        assert "go to countertop 0" not in user  # older ones hidden

    def test_no_recent_actions_section_when_no_history(self):
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
        )
        assert "RECENT ACTIONS" not in user

    def test_progress_section_included_with_history(self):
        history = [{"action": "go to countertop 1", "observation": "Nothing here."}]
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
            recent_history=history,
        )
        assert "PROGRESS" in user


class TestBuildActionPromptNoExemplarChannel:
    def test_prompt_has_no_past_examples_section(self):
        # The v2 wiki's exemplar channel was removed from the shared prompt builder:
        # only one arm ever used it, so it gave that arm a second channel into the prompt.
        _, user = build_action_prompt(
            task="task",
            observation="obs",
            admissible_commands=["look"],
        )
        assert "PAST EXAMPLES" not in user


class TestSystemPromptIsInterfaceOnly:
    """DEC-16: the shared prompt keeps interface rules only; task guidance is the arms' job."""

    def test_removed_rules_are_gone(self):
        from prompts.act import SYSTEM_PROMPT
        low = SYSTEM_PROMPT.lower()
        for phrase in ("soapbar", "open containers", "put down held", "verified",
                       "ranked search", "already searched", "revisit", "unexplored",
                       "historical averages", "loop warning", "fridge"):
            assert phrase not in low, phrase

    def test_kept_rules_are_present(self):
        from prompts.act import SYSTEM_PROMPT
        assert "single integer" in SYSTEM_PROMPT
        assert "most progress" in SYSTEM_PROMPT
        assert "matches the object named in the TASK exactly" in SYSTEM_PROMPT
        assert "It may be wrong. Treat it as a hint." in SYSTEM_PROMPT

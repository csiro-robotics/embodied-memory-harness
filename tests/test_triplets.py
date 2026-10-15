from kg_hybrid.triplets import (ADDITIVE_RELATIONS, EXCLUSIVE_RELATIONS, RELATIONS,
                         Triplet, validate_triplet)


def test_vocab():
    assert ADDITIVE_RELATIONS == {"found-at", "affords", "requires", "precedes"}
    assert EXCLUSIVE_RELATIONS == {"is-at"}
    assert RELATIONS == ADDITIVE_RELATIONS | EXCLUSIVE_RELATIONS


def test_validate_good_additive_normalizes_types():
    t = validate_triplet({"subject": "Mug 1", "relation": "found-at", "object": "CounterTop 2"})
    assert t == Triplet("mug", "found-at", "countertop")  # type-level: instance nums dropped


def test_validate_is_at_keeps_instance_numbers():
    t = validate_triplet({"subject": "Mug 1", "relation": "is-at", "object": "countertop 2"})
    assert t == Triplet("mug 1", "is-at", "countertop 2")


def test_validate_rejects_bad_relation_and_empty():
    assert validate_triplet({"subject": "mug", "relation": "near", "object": "sink"}) is None
    assert validate_triplet({"subject": "", "relation": "found-at", "object": "sink"}) is None
    assert validate_triplet({"relation": "found-at", "object": "sink"}) is None
    assert validate_triplet("not a dict") is None


def test_key_roundtrip():
    assert Triplet("mug", "found-at", "countertop").key() == "mug|found-at|countertop"


# --- extraction-precision hardening (audit 2026-08-06, gate iteration 1) -----------------
# Class 1: container/object state encoded as a location.
def test_validate_rejects_state_word_as_is_at_object():
    assert validate_triplet({"subject": "drawer 1", "relation": "is-at", "object": "open"}) is None


def test_validate_rejects_state_word_as_found_at_object():
    assert validate_triplet({"subject": "drawer", "relation": "found-at", "object": "nothing"}) is None


def test_validate_rejects_state_word_closed():
    assert validate_triplet({"subject": "cabinet 2", "relation": "is-at", "object": "closed"}) is None


# Class 2: reversed subject/object (a static receptacle can never be the movable subject).
def test_validate_rejects_reversed_direction_static_receptacle_subject():
    assert validate_triplet({"subject": "fridge 1", "relation": "is-at", "object": "mug 2"}) is None


def test_validate_rejects_reversed_direction_microwave_subject():
    assert validate_triplet({"subject": "microwave 1", "relation": "is-at", "object": "egg 3"}) is None


def test_validate_allows_movable_receptacle_as_subject():
    # A bowl (movable receptacle) legitimately sitting inside a cabinet is not a reversal.
    t = validate_triplet({"subject": "bowl 1", "relation": "is-at", "object": "cabinet 1"})
    assert t == Triplet("bowl 1", "is-at", "cabinet 1")


# Class 3: precedes/requires with non-skill endpoints.
def test_validate_rejects_precedes_object_as_subject():
    assert validate_triplet({"subject": "apple", "relation": "precedes", "object": "pickup"}) is None


def test_validate_rejects_precedes_multiword_phrase():
    assert validate_triplet(
        {"subject": "cd", "relation": "precedes", "object": "go to garbagecan"}
    ) is None


def test_validate_accepts_precedes_between_skills():
    t = validate_triplet({"subject": "pickup", "relation": "precedes", "object": "clean"})
    assert t == Triplet("pickup", "precedes", "clean")


def test_requires_object_endpoint_is_rejected_by_the_action_check():
    """K7: the tool of a requires fact is checked against the step's action, not a list,
    so lamps can be tools. "pickup requires cd" passes the shape check and is dropped
    because no action names cd as a tool."""
    from kg_hybrid.triplets import action_attests
    t = validate_triplet({"subject": "pickup", "relation": "requires", "object": "cd"})
    assert t is not None
    assert not action_attests(t, "take cd 1 from drawer 1", "You pick up the cd 1.")
    assert validate_triplet({"subject": "pickup", "relation": "requires", "object": "go to cd"}) is None


def test_validate_accepts_requires_skill_and_receptacle_tool():
    t = validate_triplet({"subject": "clean", "relation": "requires", "object": "sinkbasin"})
    assert t == Triplet("clean", "requires", "sinkbasin")


# Class 4/5: object must be receptacle-like; agent is never a node; is-at needs instance ids.
def test_validate_rejects_non_receptacle_is_at_object():
    assert validate_triplet({"subject": "mug 1", "relation": "is-at", "object": "kitchen"}) is None


def test_validate_rejects_agent_as_subject():
    assert validate_triplet({"subject": "you", "relation": "is-at", "object": "cabinet 1"}) is None


def test_validate_rejects_is_at_missing_instance_number_on_subject():
    # instance id dropped on the subject (type-level node on an instance-level relation)
    assert validate_triplet({"subject": "sink", "relation": "is-at", "object": "sinkbasin 1"}) is None


def test_validate_rejects_is_at_missing_instance_number_on_object():
    assert validate_triplet({"subject": "mug 1", "relation": "is-at", "object": "countertop"}) is None


# --- prompt constraints (mechanically testable via key phrases in the template) -----------
def test_prompt_states_direction_rule():
    from kg_hybrid.triplet_extract import _USER_TMPL
    assert "subject is ALWAYS the movable object" in _USER_TMPL
    assert "ALWAYS the receptacle" in _USER_TMPL


def test_prompt_forbids_state_words_as_locations():
    from kg_hybrid.triplet_extract import _USER_TMPL
    assert "STATE is NOT a location" in _USER_TMPL
    assert "open, closed, empty" in _USER_TMPL


def test_prompt_has_inventory_rule():
    from kg_hybrid.triplet_extract import _USER_TMPL
    assert "INVENTORY RULE" in _USER_TMPL
    assert "in the\nagent's inventory" in _USER_TMPL or "agent's inventory" in _USER_TMPL


def test_prompt_restricts_precedes_requires_to_skill_vocab():
    from kg_hybrid.triplet_extract import _USER_TMPL
    assert "goto, pickup, put,\nopen, close, clean, heat, cool, examine, toggle" in _USER_TMPL \
        or "goto, pickup, put" in _USER_TMPL


def test_prompt_requires_instance_number_and_forbids_agent():
    from kg_hybrid.triplet_extract import _USER_TMPL
    assert "instance number" in _USER_TMPL
    assert 'Never emit "you" or the agent' in _USER_TMPL


def test_receptacle_words_match_alfworld_and_lamps_can_be_located():
    """K7: lamps are ALFWorld objects, so "desklamp 1 is-at sidetable 1" is a location fact."""
    from kg_hybrid.triplets import RECEPTACLE_WORDS
    assert not {"desklamp", "floorlamp", "sink"} & RECEPTACLE_WORDS
    t = validate_triplet({"subject": "desklamp 1", "relation": "is-at", "object": "sidetable 1"})
    assert t is not None and t.subject == "desklamp 1"


def test_tool_facts_need_the_steps_action():
    """K7: affords/requires tool facts are kept only when the step's action shows them."""
    from kg_hybrid.triplets import Triplet, action_attests
    heat = Triplet("microwave", "affords", "heat")
    act, ok = "heat mug 1 with microwave 1", "You heat the mug 1 using the microwave 1."
    assert action_attests(heat, act, ok)
    assert not action_attests(heat, act, "Nothing happens.")
    assert not action_attests(heat, "go to microwave 1", "You arrive at microwave 1.")
    assert not action_attests(Triplet("butterknife", "affords", "examine"),
                              "take butterknife 1 from drawer 1", "You pick up the butterknife 1.")
    assert action_attests(Triplet("desklamp", "affords", "use"), "use desklamp 1",
                          "You turn on the desklamp 1.")
    assert not action_attests(Triplet("clean", "requires", "sinkbasin"), "go to sinkbasin 1", "")
    assert action_attests(Triplet("clean", "requires", "sinkbasin"),
                          "clean apple 1 with sinkbasin 1", "You clean the apple 1 using the sinkbasin 1.")
    # Other triplets pass unchanged.
    assert action_attests(Triplet("pickup", "precedes", "clean"), "look", "")
    assert action_attests(Triplet("mug", "found-at", "countertop"), "look", "")

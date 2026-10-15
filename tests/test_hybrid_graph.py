from kg_hybrid.hybrid_graph import HybridGraph, skills_from_trajectory
from kg_hybrid.triplets import Triplet


def t(s, r, o):
    return Triplet(s, r, o)


def test_additive_weight_increments():
    g = HybridGraph()
    assert g.reconcile([t("mug", "found-at", "countertop")])["inserted"] == 1
    assert g.reconcile([t("mug", "found-at", "countertop")])["reinforced"] == 1
    assert g.edges[("mug", "found-at", "countertop")] == 2


def test_additive_never_deleted_on_differing_observation():
    g = HybridGraph()
    g.reconcile([t("mug", "found-at", "countertop")])
    g.reconcile([t("mug", "found-at", "cabinet")])
    assert ("mug", "found-at", "countertop") in g.edges  # weighted multi-edge
    assert ("mug", "found-at", "cabinet") in g.edges


def test_no_instance_state_layer():
    """K13: is-at only feeds found-at; nothing instance-level is kept across households."""
    g = HybridGraph()
    g.reconcile([t("mug 1", "is-at", "countertop 2")])
    g.reconcile([t("mug 1", "is-at", "fridge 1")])
    assert not hasattr(g, "state")
    assert g.found_at_counts()["mug"] == {"countertop": 1, "fridge": 1}


def test_an_edge_counts_once_per_episode():
    """K6: a fact seen on every step of a loop is one episode's evidence, not fifty."""
    g = HybridGraph()
    seen = set()
    for _ in range(5):
        g.reconcile([t("mug 1", "is-at", "countertop 2"), t("mug", "found-at", "countertop")],
                    seen=seen)
    assert g.edges[("mug", "found-at", "countertop")] == 1
    g.reconcile([t("mug 2", "is-at", "countertop 1")], seen=set())  # next episode
    assert g.edges[("mug", "found-at", "countertop")] == 2


def test_own_placements_are_not_location_evidence():
    g = HybridGraph()
    stats = g.reconcile([t("mug 1", "is-at", "cabinet 1"), t("cup 1", "is-at", "cabinet 1")],
                        seen=set(), placed={("mug 1", "cabinet 1")})
    assert stats["own_placement"] == 1
    assert "mug" not in g.found_at_counts()
    assert g.found_at_counts()["cup"] == {"cabinet": 1}


def test_own_placement_written_as_found_at_is_not_evidence():
    """K18: after "You move the mug 1 to the cabinet 1" the extractor sometimes writes
    "mug found-at cabinet" directly; that bypassed the instance check."""
    g = HybridGraph()
    stats = g.reconcile([t("mug", "found-at", "cabinet"), t("cup", "found-at", "cabinet"),
                         t("mug", "found-at", "countertop")],
                        seen=set(), placed={("mug 1", "cabinet 1")})
    assert stats["own_placement"] == 1
    assert g.found_at_counts()["mug"] == {"countertop": 1}
    assert g.found_at_counts()["cup"] == {"cabinet": 1}


def test_is_at_feeds_type_level_found_at():
    g = HybridGraph()
    g.reconcile([t("mug 1", "is-at", "countertop 2")])
    g.reconcile([t("mug 2", "is-at", "countertop 1")])
    assert g.found_at_counts()["mug"]["countertop"] == 2


def test_affordance_map_majority():
    g = HybridGraph()
    g.reconcile([t("sinkbasin", "affords", "clean")] * 3 + [t("sinkbasin", "affords", "heat")])
    assert g.affordance_map()["sinkbasin"] == "clean"


def test_save_load_roundtrip(tmp_path):
    g = HybridGraph()
    g.reconcile([t("mug", "found-at", "shelf"), t("mug 1", "is-at", "shelf 2")])
    g.recipes["pick_and_place"] = {"locate>pickup>place": 4}
    p = tmp_path / "graph.json"
    g.save(p)
    g2 = HybridGraph.load(p)
    assert g2.edges == g.edges and g2.recipes == g.recipes


def test_skills_from_trajectory_maps_and_collapses():
    traj = [
        {"action": "go to countertop 1"},          # followed by take -> locate
        {"action": "take mug 1 from countertop 1"},
        {"action": "go to sinkbasin 1"},           # not followed by take -> dropped
        {"action": "clean mug 1 with sinkbasin 1"},
        {"action": "go to coffeemachine 1"},
        {"action": "put mug 1 in/on coffeemachine 1"},
    ]
    assert skills_from_trajectory(traj) == ["locate", "pickup", "clean", "place"]


def test_modal_recipe_and_to_hdkg_topo():
    g = HybridGraph()
    for _ in range(3):
        g.reinforce_recipe("pick_clean_then_place", ["locate", "pickup", "clean", "place"])
    g.reinforce_recipe("pick_clean_then_place", ["pickup", "place"])  # minority
    g.reconcile([t("sinkbasin", "affords", "clean")])
    assert g.modal_recipe("pick_clean_then_place") == ["locate", "pickup", "clean", "place"]
    hdkg = g.to_hdkg()
    order = [sg["name"] for sg in hdkg.topo_sort_subgoals("pick_clean_then_place")]
    assert order == ["locate", "pickup", "clean", "place"]
    assert hdkg.task_families["pick_clean_then_place"]["transform"] == "clean"
    assert hdkg.typical_locations == {}  # no seeds, ever


def test_to_hdkg_counted_family():
    """K1/K10: the object count is not inferred from pickups; repeated steps are numbered."""
    g = HybridGraph()
    g.reinforce_recipe("pick_two_obj_and_place", ["pickup", "place", "pickup", "place"])
    fam = g.to_hdkg().task_families["pick_two_obj_and_place"]
    assert "count" not in fam
    assert set(fam["subgoals"]) == {"pickup_1", "place_1", "pickup_2", "place_2"}


def test_skills_drop_put_down_and_retaken_detour():
    """K1: the Run 10 cool plan was locate>pickup>place>pickup>cool>place>pickup>place."""
    traj = [
        {"action": "go to countertop 1"},
        {"action": "take mug 1 from countertop 1"},
        {"action": "go to fridge 1"},
        {"action": "move mug 1 to fridge 1"},
        {"action": "take mug 1 from fridge 1"},
        {"action": "cool mug 1 with fridge 1"},
        {"action": "move mug 1 to countertop 2"},
        {"action": "take mug 1 from countertop 2"},
        {"action": "go to cabinet 1"},
        {"action": "move mug 1 to cabinet 1"},
    ]
    assert skills_from_trajectory(traj, "mug") == ["locate", "pickup", "cool", "place"]


def test_skills_ignore_other_objects():
    traj = [
        {"action": "go to countertop 1"},
        {"action": "take knife 1 from countertop 1"},
        {"action": "move knife 1 to countertop 1"},
        {"action": "go to countertop 2"},
        {"action": "take mug 1 from countertop 2"},
        {"action": "move mug 1 to shelf 1"},
    ]
    assert skills_from_trajectory(traj, "mug") == ["locate", "pickup", "place"]


def test_skills_keep_two_deliveries_and_use():
    two = [
        {"action": "go to drawer 1"}, {"action": "take cd 1 from drawer 1"},
        {"action": "move cd 1 to safe 1"},
        {"action": "go to shelf 1"}, {"action": "take cd 2 from shelf 1"},
        {"action": "move cd 2 to safe 1"},
    ]
    assert skills_from_trajectory(two, "cd") == [
        "locate", "pickup", "place", "locate", "pickup", "place"]
    lamp = [
        {"action": "go to desk 1"}, {"action": "take book 1 from desk 1"},
        {"action": "use desklamp 1"},
    ]
    assert skills_from_trajectory(lamp, "book") == ["locate", "pickup", "use"]


def test_modal_recipe_tie_prefers_shortest():
    """K1/K2: ties break toward the shortest sequence, then lexicographically."""
    g = HybridGraph()
    g.reinforce_recipe("pick_heat_then_place", ["locate", "pickup", "place", "pickup", "heat", "place"])
    g.reinforce_recipe("pick_heat_then_place", ["locate", "pickup", "heat", "place"])
    assert g.modal_recipe("pick_heat_then_place") == ["locate", "pickup", "heat", "place"]


def test_modal_recipe_unseen_family():
    assert HybridGraph().modal_recipe("pick_and_place") is None


def test_preferred_tool_map_not_shadowed_by_frequent_affordance():
    g = HybridGraph()
    g.edges[("microwave", "affords", "open")] = 26
    g.edges[("microwave", "affords", "heat")] = 13
    g.edges[("stove", "affords", "heat")] = 3
    g.edges[("fridge", "affords", "cool")] = 8
    g.edges[("mug", "affords", "cool")] = 2
    g.edges[("clean", "affords", "sinkbasin")] = 31  # reversed noise: skill subject
    g.edges[("sinkbasin", "affords", "clean")] = 33
    m = g.preferred_tool_map()
    assert m["heat"] == "microwave"   # 13 > 3 despite open(26) shadowing
    assert m["cool"] == "fridge"
    assert m["clean"] == "sinkbasin"


def test_to_hdkg_uses_learned_preferred_tool():
    g = HybridGraph()
    g.edges[("microwave", "affords", "open")] = 26
    g.edges[("microwave", "affords", "heat")] = 13
    g.edges[("stove", "affords", "heat")] = 3
    for _ in range(2):
        g.reinforce_recipe("pick_heat_then_place", ["locate", "pickup", "heat", "place"])
    hdkg = g.to_hdkg()
    assert hdkg.tool_for_transform("heat") == "microwave"


def test_build_context_uses_task_count_and_no_handwritten_lines():
    """K1/K10/HC2: a one-object heat task is never told it needs two objects, and the
    examine plan carries no lamp hint or desklamp default."""
    g = HybridGraph()
    g.reinforce_recipe("pick_heat_then_place", ["locate", "pickup", "heat", "place"])
    g.reinforce_recipe("pick_two_obj_and_place",
                       ["locate", "pickup", "place", "locate", "pickup", "place"])
    g.reinforce_recipe("examine_in_light", ["locate", "pickup", "use"])
    g.edges[("microwave", "affords", "heat")] = 3
    hdkg = g.to_hdkg()

    heat = hdkg.build_context("pick_heat_then_place", "cup", "cabinet", count=1)
    assert "TWO" not in heat and "object #" not in heat
    assert "heat the cup using the microwave" in heat

    two = hdkg.build_context("pick_two_obj_and_place", "cd", "safe", count=2)
    assert "TWO" not in two and "placed one at a time" not in two
    assert "(object #1)" in two and "(object #2)" in two

    lamp = hdkg.build_context("examine_in_light", "book", None, tool="desklamp")
    assert "use the desklamp" in lamp
    assert "turn on the lamp" not in lamp
    lamp_untold = hdkg.build_context("examine_in_light", "book", None)
    assert "desklamp" not in lamp_untold

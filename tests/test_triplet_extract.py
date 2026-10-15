from kg_hybrid.triplet_extract import TripletExtractor
from kg_hybrid.triplets import Triplet


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user, max_tokens=None):
        self.calls.append((system, user))
        return self.reply


def test_extract_parses_and_validates():
    reply = ('[{"subject": "mug 1", "relation": "is-at", "object": "countertop 2"},'
             ' {"subject": "sinkbasin", "relation": "affords", "object": "clean"},'
             ' {"subject": "mug", "relation": "near", "object": "sink"}]')  # bad rel -> dropped
    ex = TripletExtractor(FakeLLM(reply))
    out = ex.extract("You arrive at countertop 2.", "go to countertop 2",
                     "On the countertop 2, you see a mug 1.")
    assert Triplet("mug 1", "is-at", "countertop 2") in out
    assert Triplet("sinkbasin", "affords", "clean") in out
    assert len(out) == 2
    assert ex.stats["kept"] == 2 and ex.stats["discarded"] == 1


def test_extract_handles_fenced_json():
    ex = TripletExtractor(FakeLLM('```json\n[{"subject":"mug","relation":"found-at","object":"shelf"}]\n```'))
    assert ex.extract("o", "a", "r") == [Triplet("mug", "found-at", "shelf")]


def test_extract_garbage_returns_empty_and_counts():
    ex = TripletExtractor(FakeLLM("I think the mug is on the shelf."))
    assert ex.extract("o", "a", "r") == []
    assert ex.stats["parse_failures"] == 1


def test_prompt_mentions_step_and_vocab():
    fake = FakeLLM("[]")
    TripletExtractor(fake).extract("OBS_TEXT", "ACT_TEXT", "OUT_TEXT")
    _, user = fake.calls[0]
    for needle in ("OBS_TEXT", "ACT_TEXT", "OUT_TEXT", "is-at", "found-at", "affords"):
        assert needle in user


def test_prompt_examples_name_no_real_objects_or_tools():
    """HC1/K12: examples are placeholders, so the model cannot copy a fact from them."""
    import re
    from kg_hybrid.triplet_extract import _USER_TMPL
    from kg_hybrid.triplets import RECEPTACLE_WORDS
    names = RECEPTACLE_WORDS | {"apple", "cd", "desklamp", "floorlamp", "faucet"}
    found = sorted(w for w in names if re.search(rf"\b{w}\b", _USER_TMPL.lower()))
    assert found == [], found

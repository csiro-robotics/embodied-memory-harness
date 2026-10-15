# ALFWorld — Dataset Reference

## What is ALFWorld

ALFWorld is a text-based household robot benchmark. The agent navigates rooms, picks up
and manipulates objects, and completes goal-directed tasks described in natural language.
It runs entirely as text — no vision. The agent reads observations (what it sees) and
selects actions from a list of what is currently possible.

---

## What an episode looks like

### Start of episode

The agent receives an initial observation containing the starting room description and
a task instruction:

```
You are in the middle of a room. Looking quickly around you, you see a cabinet 1,
a cabinet 2, a coffeemachine 1, a countertop 1, a fridge 1, a microwave 1,
a sinkbasin 1, a stoveburner 1, and a toaster 1.

Your task is to: cool some apple and put it in countertop.
```

### At every step

The agent receives:
1. An **observation** — what just happened and what the agent now sees
2. A list of **admissible_commands** — every action that can be executed right now

```python
observation = "You arrive at loc 12. On the countertop 1, you see an apple 1 and a plate 2."

admissible_commands = [
    "go to fridge 1",
    "go to microwave 1",
    "go to sinkbasin 1",
    "take apple 1 from countertop 1",
    "take plate 2 from countertop 1",
    "inventory",
    "look",
]
```

After taking the apple:

```python
observation = "You pick up the apple 1 from the countertop 1."

admissible_commands = [
    "go to fridge 1",
    "go to microwave 1",
    "put apple 1 in/on countertop 1",
    "put apple 1 in/on sinkbasin 1",
    "inventory",
    "look",
]
# No "cool apple" yet — fridge is not open
```

After opening the fridge:

```python
observation = "You open the fridge 1. The fridge 1 is open. In it, you see a bowl 1."

admissible_commands = [
    "cool apple 1 with fridge 1",
    "put apple 1 in/on fridge 1",
    "close fridge 1",
    "inventory",
    "look",
]
# Now "cool apple" appears — fridge is open and agent holds the apple
```

---

## Admissible commands as affordance signals

The admissible_commands list is generated from the scene's planning model at every step.
**Each command that appears has all its preconditions satisfied right now.**

This makes admissible_commands a real-time affordance oracle. If `"cool apple 1 with fridge 1"`
appears, it means: the apple can be cooled, the agent is holding it, and the fridge is open.
Every confirmed action reveals something the agent can do with that object in this moment.

**Critical caveat:** An action NOT appearing does not mean the affordance is impossible.
It may simply mean the preconditions weren't met during this episode — the agent never held
the apple while the fridge was open, so `"cool apple"` never appeared. The absence is a gap
in evidence, not proof of impossibility.

---

## Task families

ALFWorld has 6 task types. Each generates natural-language task descriptions:

| Family | Example task description |
|---|---|
| `pick_and_place` | "Put a mug in the fridge." |
| `pick_two_obj_and_place` | "Put two plates on the dining table." |
| `examine_in_light` | "Examine the newspaper with the desk lamp." |
| `pick_clean_then_place` | "Clean a fork and put it in a drawer." |
| `pick_heat_then_place` | "Heat a potato and put it on the counter." |
| `pick_cool_then_place` | "Cool an apple and put it on the counter." |

---

## Observation patterns

Common observation text patterns the agent will see:

```
# Arriving at a location
"You arrive at loc 22. On the shelf 1, you see a vase 1."

# Opening a container
"You open the cabinet 2. The cabinet 2 is open. In it, you see a pan 1."

# Picking up an object
"You pick up the knife 1 from the countertop 1."

# Placing an object
"You put the knife 1 in/on the drawer 1."

# Cleaning
"You clean the mug 1 using the sinkbasin 1."

# Heating
"You heat the potato 1 using the microwave 1."

# Cooling
"You cool the apple 1 using the fridge 1."

# Nothing here
"You arrive at loc 5. On the shelf 2, you see nothing."

# Task complete
"You put a hot potato on the counter. Task success."
```

---

## What's in a trajectory step

Each step stored in `EpisodeResult.trajectory`:

```python
{
    "step": 4,
    "action": "take apple 1 from countertop 1",
    "observation": "You pick up the apple 1 from the countertop 1.",
    "admissible_commands": [
        "go to fridge 1",
        "put apple 1 in/on countertop 1",
        "inventory",
        "look",
        # ... more options
    ],
    "score": 0.0,
}
```

The `admissible_commands` here are for the state AFTER the action was taken — they show
what the agent can do next given the result of `"take apple 1 from countertop 1"`.

---

## Affordances the agent can discover from an episode

By watching what appears in `admissible_commands` throughout the trajectory, the agent can
discover what each object is capable of — empirically, from this episode, without consulting
any external knowledge source.

**What can be learned:**
- Which objects can be picked up (they appear in `take X from Y` actions)
- Which appliances/tools can process each object (they appear in `verb X with instrument`)
- Which containers accept which objects (they appear in `put X in/on Y`)
- Which containers can be opened (they appear in `open Y`)
- Which objects serve as light sources (they appear in `use Y`)

**What cannot be reliably learned from a single episode:**
- Affordances that require preconditions that were never satisfied (e.g. object never held
  while the relevant appliance was open)
- Properties of objects that never appeared in the scene
- Affordances for objects in rooms the agent didn't visit

Over many episodes, the evidence accumulates: seeing `"heat apple with microwave"` in
episode 5 confirms it even if it was absent in episodes 1-4.

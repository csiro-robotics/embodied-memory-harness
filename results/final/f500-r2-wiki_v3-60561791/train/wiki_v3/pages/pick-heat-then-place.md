---
name: pick-heat-then-place
description: A procedure for picking up an object, heating it, and placing it in a
  new location. Used when an object must be heated before being moved.
type: procedure
applies_to:
- pick_heat_then_place
objects:
- plate
- mug
- cup
sources:
- ep26
- ep27
- ep28
- ep29
- ep30
- ep31
- ep32
- ep33
- ep34
- ep35
- ep36
- ep37
- ep38
- ep39
- ep40
- ep41
- ep42
- ep43
- ep44
- ep45
- ep46
- ep47
- ep48
- ep49
- ep50
- ep101
- ep102
- ep103
- ep104
- ep105
- ep106
- ep107
- ep108
- ep109
- ep110
- ep111
- ep112
- ep113
- ep114
- ep115
- ep116
- ep117
- ep118
- ep119
- ep120
- ep121
- ep122
- ep123
- ep124
- ep125
- ep476
- ep477
- ep478
- ep479
- ep480
- ep481
- ep482
- ep483
- ep484
- ep485
- ep486
- ep487
- ep488
- ep489
- ep490
- ep491
- ep492
- ep493
- ep494
- ep495
- ep496
- ep497
- ep498
- ep499
- ep500
confidence: hedged
---

# Preconditions
- The target object must be located in a reachable receptacle.
- The heating tool (e.g., microwave) must be accessible.

# Steps
1. take <obj> from <place>
2. heat <obj> with microwave
3. move <obj> to <new place>

# Pitfalls
- Picking up the wrong object may lead to confusion and wasted steps.
- If the object is not found in the expected location, the robot should pause and ask for clarification.

# Not established
- The exact location of the object in different environments.
- Whether the object is already in the destination location before heating.
- Whether the object can be heated directly in its current location without being picked up first.

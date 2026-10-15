---
name: pick-cool-then-place
description: Procedure for picking up an object, cooling it with a fridge, and placing
  it in a destination. Used when an object needs to be cooled and relocated.
type: procedure
applies_to:
- pick_cool_then_place
objects:
- apple
- mug
- lettuce
- tomato
- bread
- bowl
sources:
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
- ep276
- ep277
- ep278
- ep279
- ep280
- ep281
- ep282
- ep283
- ep284
- ep285
- ep286
- ep287
- ep288
- ep289
- ep290
- ep291
- ep292
- ep293
- ep294
- ep295
- ep296
- ep297
- ep298
- ep299
- ep300
confidence: hedged
---

# Preconditions
- The object to be cooled is located in a known place.
- The fridge is accessible and functional.

# Steps
1. take <obj> from <place>
2. cool <obj> with fridge
3. move <obj> to <destination>

# Pitfalls
- The robot may pick up the wrong object if the target is not clearly identified.
- Cooling may not be successful if the fridge is not the correct tool or is not accessible.
- The destination may not be clearly defined, leading to incorrect placement.

# Not established
- The exact destination for the cooled object may vary depending on the task.
- The robot may need to search for the object if it is not initially found in the expected location.
- The robot may need to retry cooling if the first attempt fails.

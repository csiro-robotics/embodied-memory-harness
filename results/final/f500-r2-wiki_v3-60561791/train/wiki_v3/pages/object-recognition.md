---
name: object-recognition
description: How to recognize objects and avoid confusion when picking them up or
  placing them.
type: procedure
applies_to:
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_and_place
- pick_two_obj_and_place
objects:
- bowl
- tomato
- plate
- soapbottle
- egg
- butterknife
- creditcard
- newspaper
- bread
- mug
- apple
sources:
- ep1
- ep2
- ep3
- ep4
- ep5
- ep6
- ep7
- ep8
- ep9
- ep10
- ep11
- ep12
- ep13
- ep14
- ep15
- ep16
- ep17
- ep18
- ep19
- ep20
- ep21
- ep22
- ep23
- ep24
- ep25
- ep126
- ep127
- ep128
- ep129
- ep130
- ep131
- ep132
- ep133
- ep134
- ep135
- ep136
- ep137
- ep138
- ep139
- ep140
- ep141
- ep142
- ep143
- ep144
- ep145
- ep146
- ep147
- ep148
- ep149
- ep150
confidence: hedged
---

# Preconditions
- The robot must be attempting to pick up or place an object.
- The object must be visually or functionally similar to another object in the environment.

# Steps
1. Identify the object visually and by function.
2. Confirm the object's identity by checking its location and purpose.
3. If the object is not the target, re-evaluate and search for the correct object.

# Pitfalls
- The robot may confuse objects with similar shapes or purposes, leading to incorrect pickups or placements.
- The robot may fail to recognize the correct object even after multiple attempts.

# Not established
- The exact conditions under which object recognition fails consistently.
- The effectiveness of specific tools or methods for disambiguating visually similar objects.

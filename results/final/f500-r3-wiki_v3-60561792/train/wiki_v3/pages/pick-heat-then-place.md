---
name: pick-heat-then-place
description: Procedure for picking up an object, heating it, and placing it in a destination.
  Used when heating an object and moving it to a new location.
type: procedure
applies_to:
- pick_heat_then_place
objects:
- egg
- mug
sources:
- ep201
- ep202
- ep203
- ep204
- ep205
- ep206
- ep207
- ep208
- ep209
- ep210
- ep211
- ep212
- ep213
- ep214
- ep215
- ep216
- ep217
- ep218
- ep219
- ep220
- ep221
- ep222
- ep223
- ep224
- ep225
- ep376
- ep377
- ep378
- ep379
- ep380
- ep381
- ep382
- ep383
- ep384
- ep385
- ep386
- ep387
- ep388
- ep389
- ep390
- ep391
- ep392
- ep393
- ep394
- ep395
- ep396
- ep397
- ep398
- ep399
- ep400
confidence: hedged
---

# Preconditions
- The object to be heated must be located in a known receptacle.
- The heat source (e.g., microwave, stove) must be accessible and functional.

# Steps
1. take <obj> from <receptacle>
2. move <obj> to <heat source>
3. heat <obj> with <heat source>
4. move <obj> to <destination>

# Pitfalls
- Picking up the wrong object (e.g., a bowl instead of a cup) can lead to incorrect heating and placement.
- If the object is not found in the expected location, the task may fail due to search exhaustion.

# Not established
- Whether the object is already at the heat source and can be directly heated without prior movement.
- Whether the heat source is the correct one for the object (e.g., some objects may require a stove instead of a microwave).
- Whether the object needs to be cleaned or prepared before heating.

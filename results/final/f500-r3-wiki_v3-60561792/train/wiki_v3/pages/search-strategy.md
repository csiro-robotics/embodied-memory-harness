---
name: search-strategy
description: General procedure for systematically searching for an object in a room,
  avoiding repeated checks and ensuring efficient exploration.
type: procedure
applies_to:
- pick_and_place
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_two_obj_and_place
objects: []
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
- ep151
- ep152
- ep153
- ep154
- ep155
- ep156
- ep157
- ep158
- ep159
- ep160
- ep161
- ep162
- ep163
- ep164
- ep165
- ep166
- ep167
- ep168
- ep169
- ep170
- ep171
- ep172
- ep173
- ep174
- ep175
confidence: hedged
---

# Preconditions
- The robot must be searching for an object that is not immediately visible.

# Steps
1. Check the most likely receptacles first, based on object type and task context.
2. Move to a new receptacle after each check, avoiding revisiting the same location.
3. If the object is not found, continue checking other receptacles systematically.
4. If the object is found, retrieve it and proceed with the task.
5. If the object is not found after checking multiple receptacles, stop searching and report failure.

# Pitfalls
- Revisiting the same receptacle repeatedly may lead to failure due to step limits.
- Failing to check multiple receptacles may result in missing the object entirely.

# Not established
- The exact order of receptacles to check based on object type.
- Whether the robot should prioritize certain receptacles over others in specific contexts.

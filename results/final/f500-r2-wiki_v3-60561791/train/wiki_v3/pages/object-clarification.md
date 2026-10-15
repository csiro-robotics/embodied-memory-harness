---
name: object-clarification
description: Procedure for clarifying object identity when the robot encounters an
  object that does not match the requested item.
type: procedure
applies_to:
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_and_place
- pick_two_obj_and_place
objects:
- egg
- apple
- bread
- pan
- cup
- potato
- mug
- glassbottle
- knife
- butterknife
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
- ep426
- ep427
- ep428
- ep429
- ep430
- ep431
- ep432
- ep433
- ep434
- ep435
- ep436
- ep437
- ep438
- ep439
- ep440
- ep441
- ep442
- ep443
- ep444
- ep445
- ep446
- ep447
- ep448
- ep449
- ep450
confidence: hedged
---

# Preconditions
- The robot has encountered an object that does not match the requested item.

# Steps
1. Pause the current task.
2. Request clarification from the user about the correct object to proceed.
3. Wait for user input before continuing.

# Pitfalls
- Continuing without clarification may lead to further confusion and wasted steps.
- Misidentifying the object may result in incorrect actions being taken.

# Not established
- The exact phrasing or method of requesting clarification.
- Whether the user will always provide clarification when needed.

---
name: pick-heat-then-place
description: Procedure for picking up an object, heating it, and placing it in a receptacle.
  Used when the task requires heating before placement.
type: procedure
applies_to:
- pick_heat_then_place
objects:
- plate
- cup
- potato
- mug
- apple
- lettuce
sources:
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
- ep301
- ep302
- ep303
- ep304
- ep305
- ep306
- ep307
- ep308
- ep309
- ep310
- ep311
- ep312
- ep313
- ep314
- ep315
- ep316
- ep317
- ep318
- ep319
- ep320
- ep321
- ep322
- ep323
- ep324
- ep325
- ep451
- ep452
- ep453
- ep454
- ep455
- ep456
- ep457
- ep458
- ep459
- ep460
- ep461
- ep462
- ep463
- ep464
- ep465
- ep466
- ep467
- ep468
- ep469
- ep470
- ep471
- ep472
- ep473
- ep474
- ep475
confidence: hedged
---

# Preconditions
- The target object must be present in the environment and identifiable.
- The heating tool (e.g., microwave) must be accessible and functional.
- The destination receptacle must be known and reachable.

# Steps
1. take object from location
2. heat object with microwave
3. move object to receptacle

# Pitfalls
- Picking up the wrong object (e.g., taking a cup instead of a mug) can lead to failure.
- Failing to heat the object before placing it may result in incorrect task completion.
- Repeatedly taking the object from the destination receptacle without completing the task can cause infinite loops.
- Avoid picking up unrelated items from the same receptacle, as this can delay the task and lead to confusion.

# Not established
- Whether the object must be cooled before being placed in certain receptacles.
- Whether the destination receptacle must be checked for the presence of the object before placing it.
- Whether the object must be heated to a specific temperature before placement.
- Whether the object must be uniquely identifiable to avoid confusion with visually similar items before and after heating.

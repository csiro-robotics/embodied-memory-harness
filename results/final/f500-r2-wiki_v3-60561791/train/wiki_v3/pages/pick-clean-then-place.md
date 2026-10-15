---
name: pick-clean-then-place
description: A procedure for picking up an object, cleaning it, and placing it in
  a receptacle. Used when an object needs to be cleaned before being moved.
type: procedure
applies_to:
- pick_clean_then_place
objects:
- cup
- kettle
- soapbar
- spoon
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
- ep76
- ep77
- ep78
- ep79
- ep80
- ep81
- ep82
- ep83
- ep84
- ep85
- ep86
- ep87
- ep88
- ep89
- ep90
- ep91
- ep92
- ep93
- ep94
- ep95
- ep96
- ep97
- ep98
- ep99
- ep100
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
- ep176
- ep177
- ep178
- ep179
- ep180
- ep181
- ep182
- ep183
- ep184
- ep185
- ep186
- ep187
- ep188
- ep189
- ep190
- ep191
- ep192
- ep193
- ep194
- ep195
- ep196
- ep197
- ep198
- ep199
- ep200
- ep251
- ep252
- ep253
- ep254
- ep255
- ep256
- ep257
- ep258
- ep259
- ep260
- ep261
- ep262
- ep263
- ep264
- ep265
- ep266
- ep267
- ep268
- ep269
- ep270
- ep271
- ep272
- ep273
- ep274
- ep275
confidence: attested
---

# Preconditions
- The target object must be present in a location that can be accessed by the robot.
- The robot must be able to identify the correct object to avoid confusion.

# Steps
1. take <obj> from <place>
2. clean <obj> with <tool>
3. move <obj> to <receptacle>

# Pitfalls
- If the robot picks up the wrong object, it may waste time cleaning and placing it incorrectly. The robot should pause and verify the object before and after cleaning.
- If the object is not found in the expected location, the robot should pause and ask for clarification to avoid confusion and wasted steps.

# Not established
- The exact location of the object may vary, and the robot may need to search multiple receptacles before finding it.
- The robot may need to handle multiple objects in some tasks, but this is not consistently established across episodes.

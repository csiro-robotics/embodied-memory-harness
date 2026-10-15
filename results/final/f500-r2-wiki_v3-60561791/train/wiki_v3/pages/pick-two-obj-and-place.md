---
name: pick-two-obj-and-place
description: Procedure for retrieving two identical objects and placing them in a
  receptacle, used when multiple objects are needed in a single task.
type: procedure
applies_to:
- pick_two_obj_and_place
objects:
- cd
- statue
- candle
- spoon
- remotecontrol
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
- ep226
- ep227
- ep228
- ep229
- ep230
- ep231
- ep232
- ep233
- ep234
- ep235
- ep236
- ep237
- ep238
- ep239
- ep240
- ep241
- ep242
- ep243
- ep244
- ep245
- ep246
- ep247
- ep248
- ep249
- ep250
confidence: attested
---

# Preconditions
- The target object must be present in at least one location.
- The robot must be able to distinguish between two identical objects in the same location.

# Steps
1. take <obj> from <place>
2. move <obj> to <receptacle>
3. take <obj> from <place>
4. move <obj> to <receptacle>

# Pitfalls
- Robot may pick the wrong object or re-pick from the destination, leading to failure.
- When multiple identical objects are present, the robot may fail to distinguish between them, especially when more than two are involved.

# Not established
- Whether the robot can reliably distinguish between multiple identical objects in the same location when the task involves more than two objects.
- Whether the robot can consistently avoid re-picking from the destination when placing multiple objects.

---
name: pick-clean-then-place
description: A procedure for picking up a dirty object, cleaning it, and placing it
  in a destination. Use when the object is dirty and needs to be cleaned before being
  moved.
type: procedure
applies_to:
- pick_clean_then_place
objects:
- ladle
- spoon
- butterknife
- dishsponge
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
confidence: hedged
---

# Preconditions
- The object is dirty and located in a reachable place.
- A sinkbasin is available for cleaning.

# Steps
1. take <obj> from <place>
2. verify <obj> is the correct object
3. clean <obj> with sinkbasin
4. move <obj> to <destination>

# Pitfalls
- The robot may attempt to place the object before cleaning it, leading to failure.
- If the sinkbasin is not available or not in reach, the cleaning step cannot be completed.
- The robot may pick up the wrong object, leading to failure.

# Not established
- Whether the object must be placed in a specific type of receptacle after cleaning.
- Whether the object must be placed in a specific location relative to the sinkbasin.
- Whether the object must be placed in a specific location after cleaning.

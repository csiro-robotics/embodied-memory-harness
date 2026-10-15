---
name: object-identification
description: How the robot identifies objects when they are not in the expected location
  or are similar to other objects.
type: procedure
applies_to:
- pick_and_place
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_two_obj_and_place
objects:
- apple
- bowl
- cd
- cloth
- cup
- dishsponge
- keychain
- kettle
- mug
- pan
- potato
- soapbottle
- tomato
- towel
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
- ep51
- ep52
- ep53
- ep54
- ep55
- ep56
- ep57
- ep58
- ep59
- ep60
- ep61
- ep62
- ep63
- ep64
- ep65
- ep66
- ep67
- ep68
- ep69
- ep70
- ep71
- ep72
- ep73
- ep74
- ep75
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
confidence: hedged
---

# Preconditions
- The robot is attempting to locate or manipulate an object.
- The object may be obscured, not immediately visible, or similar in appearance to other objects.

# Steps
1. Look at the object to confirm its identity.
2. Pause and verify the object's identity if multiple similar-looking objects are present.
3. Systematically check other receptacles if the object is not found in the expected location.
4. Avoid picking up objects that are not the target.
5. Continue searching until the correct object is located.
6. Verify the object's identity again after moving it.

# Pitfalls
- Confusing similar-looking objects, such as cups, bowls, plates, mugs, and glassbottles.
- Failing to identify objects when they are not in the expected location.
- Repeatedly picking up the wrong object due to poor visual recognition.

# Not established
- Whether the robot can distinguish objects based on texture or other non-visual cues.
- How the robot handles objects that are completely hidden or not in the field of view.
- Whether the robot can reliably pause and verify identity in all situations involving similar objects.

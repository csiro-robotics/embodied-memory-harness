---
name: pick-and-place
description: A procedure for moving an object from its current location to a target
  receptacle, used when the object must be relocated.
type: procedure
applies_to:
- pick_and_place
- pick_two_obj_and_place
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
objects:
- toiletpaper
- cellphone
- candle
- cup
- egg
- plate
- statue
- alarmclock
- pot
- lettuce
- apple
- tomato
- mug
- potato
- remotecontrol
- spraybottle
- creditcard
- keychain
- book
- pillow
- desklamp
- cup
- lettuce
- mug
- apple
- remotecontrol
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
- The object must be identified and located before attempting to move it.
- The target receptacle must be accessible and known.

# Steps
1. Search for the object systematically, checking multiple receptacles if necessary.
2. Verify the object's identity before attempting to move it.
3. Take the object from its current location.
4. Move the object to the target receptacle.
5. Confirm the object is correctly placed.

# Pitfalls
- Picking up the wrong object can lead to confusion and wasted steps.
- Failing to locate the object before attempting to move it can result in failure.
- If the object is not found in the expected location, the robot should systematically search nearby receptacles rather than re-taking objects from the destination.

# Not established
- Whether the robot can reliably distinguish between similar objects in all lighting and environmental conditions.
- Whether the robot can handle objects that are not in their expected locations.
- Whether the robot can handle multiple objects in the same task without confusion.

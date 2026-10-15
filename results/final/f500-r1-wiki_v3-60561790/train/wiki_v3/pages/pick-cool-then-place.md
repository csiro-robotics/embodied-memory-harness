---
name: pick-cool-then-place
description: Procedure for picking up an object, cooling it, and placing it in a specified
  location. Used when an object must be cooled before being moved.
type: procedure
applies_to:
- pick_cool_then_place
objects:
- cup
- plate
- apple
- potato
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
- ep401
- ep402
- ep403
- ep404
- ep405
- ep406
- ep407
- ep408
- ep409
- ep410
- ep411
- ep412
- ep413
- ep414
- ep415
- ep416
- ep417
- ep418
- ep419
- ep420
- ep421
- ep422
- ep423
- ep424
- ep425
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
- The object to be cooled must be present in a known location.
- The cooling tool (e.g., fridge) must be accessible.
- The destination location must be specified and accessible.

# Steps
1. Take the object from its current location.
2. Verify the object's identity before cooling to avoid confusion with visually similar objects.
3. Cool the object with the fridge.
4. Move the cooled object to the specified destination.

# Pitfalls
- Picking up the wrong object may lead to confusion and re-taking.
- Failing to cool the object before placing it may result in incorrect placement.
- If the destination is not clearly specified, the object may be placed in the wrong location.

# Not established
- The exact number of receptacles to check before concluding the object is missing.
- Whether the robot should attempt to re-check locations if the object is not found initially.
- Whether the robot should verify that the object has been successfully cooled before moving it.

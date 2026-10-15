---
name: object-handling
description: Procedures for handling objects that require transformation (cleaning,
  heating, cooling) and placing them in a destination receptacle.
type: procedure
applies_to:
- pick_and_place
- pick_heat_then_place
- pick_cool_then_place
- pick_clean_then_place
objects:
- mug
- bowl
- pan
- bread
- cup
- tomato
- egg
- lettuce
- potato
- soapbar
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
- ep326
- ep327
- ep328
- ep329
- ep330
- ep331
- ep332
- ep333
- ep334
- ep335
- ep336
- ep337
- ep338
- ep339
- ep340
- ep341
- ep342
- ep343
- ep344
- ep345
- ep346
- ep347
- ep348
- ep349
- ep350
- ep351
- ep352
- ep353
- ep354
- ep355
- ep356
- ep357
- ep358
- ep359
- ep360
- ep361
- ep362
- ep363
- ep364
- ep365
- ep366
- ep367
- ep368
- ep369
- ep370
- ep371
- ep372
- ep373
- ep374
- ep375
- ep476
- ep477
- ep478
- ep479
- ep480
- ep481
- ep482
- ep483
- ep484
- ep485
- ep486
- ep487
- ep488
- ep489
- ep490
- ep491
- ep492
- ep493
- ep494
- ep495
- ep496
- ep497
- ep498
- ep499
- ep500
confidence: attested
---

# Preconditions
- The object must be located in a receptacle.
- The transformation (cleaning, cooling, heating) must be applicable to the object.

# Steps
1. Take the object from its current location.
2. Move the object to the appropriate transformation receptacle (e.g., sinkbasin for cleaning, fridge for cooling, microwave for heating).
3. Apply the required transformation to the object using the appropriate tool (e.g., clean with sinkbasin, cool with fridge, heat with microwave).
4. Move the transformed object to the destination receptacle.

# Pitfalls
- Failing to apply the transformation before placing the object may result in failure.
- Retaking the object from the destination after transformation can cause confusion and failure.
- Incorrect identification of the object or its location may lead to failure.

# Not established
- The exact tool required for transformation may vary depending on the object and environment.
- The robot may not always correctly identify the object's location or the appropriate transformation.
- The robot may sometimes fail to distinguish between similar-looking objects, leading to incorrect transformations or placements.

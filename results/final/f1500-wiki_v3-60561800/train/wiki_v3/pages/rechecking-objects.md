---
name: rechecking-objects
description: A procedure for verifying that an object remains in its destination after
  placement, and whether it is still needed.
type: procedure
applies_to:
- pick_and_place
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_two_obj_and_place
objects:
- pencil
- plate
- box
- pot
- cd
- alarmclock
- egg
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
- ep576
- ep577
- ep578
- ep579
- ep580
- ep581
- ep582
- ep583
- ep584
- ep585
- ep586
- ep587
- ep588
- ep589
- ep590
- ep591
- ep592
- ep593
- ep594
- ep595
- ep596
- ep597
- ep598
- ep599
- ep600
- ep626
- ep627
- ep628
- ep629
- ep630
- ep631
- ep632
- ep633
- ep634
- ep635
- ep636
- ep637
- ep638
- ep639
- ep640
- ep641
- ep642
- ep643
- ep644
- ep645
- ep646
- ep647
- ep648
- ep649
- ep650
- ep676
- ep677
- ep678
- ep679
- ep680
- ep681
- ep682
- ep683
- ep684
- ep685
- ep686
- ep687
- ep688
- ep689
- ep690
- ep691
- ep692
- ep693
- ep694
- ep695
- ep696
- ep697
- ep698
- ep699
- ep700
- ep1401
- ep1402
- ep1403
- ep1404
- ep1405
- ep1406
- ep1407
- ep1408
- ep1409
- ep1410
- ep1411
- ep1412
- ep1413
- ep1414
- ep1415
- ep1416
- ep1417
- ep1418
- ep1419
- ep1420
- ep1421
- ep1422
- ep1423
- ep1424
- ep1425
confidence: hedged
---

# Preconditions
- An object has been placed in a destination receptacle.
- The task is not yet confirmed as complete.

# Steps
1. Take the object from the destination receptacle.
2. Inspect the object to confirm it is intact and still needed.
3. If the object is still needed, return it to the destination or move it to the next required location.
4. If the task is complete, confirm completion and stop further action.

# Pitfalls
- The robot may fail to recheck the object, leading to incorrect task completion.
- The robot may not recognize that the object has been moved or is no longer needed.

# Not established
- Whether the object was originally needed or if the task was already complete before placement.
- The exact conditions under which rechecking is necessary versus optional.

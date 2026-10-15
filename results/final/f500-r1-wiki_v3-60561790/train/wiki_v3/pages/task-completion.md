---
name: task-completion
description: Procedures for completing tasks involving object pickup, transformation,
  and placement.
type: procedure
applies_to:
- pick_and_place
- pick_clean_then_place
- pick_heat_then_place
- pick_cool_then_place
- pick_two_obj_and_place
objects:
- alarmclock
- cloth
- tomato
- spoon
- box
- pot
- cup
- spraybottle
- statue
- creditcard
- keychain
- saltshaker
- soapbottle
- bowl
- kettle
- candle
- newspaper
- handtowel
- egg
- apple
- dishsponge
- remotecontrol
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
confidence: hedged
---

# Preconditions
- The target object must be present in the environment.
- The robot must be able to recognize the object and distinguish it from other objects.
- The robot must have access to the correct tool or receptacle to complete the task.

# Steps
1. `take <obj> from <place>` — locate and pick up the target object from its current location.
3. `move <obj> to <place>` — place the object in the designated receptacle or location.

# Pitfalls
- Confusing the target object with visually similar objects (e.g., picking up a potato instead of a peppershaker).
- Failing to search thoroughly, leading to search exhaustion (e.g., not finding a target object because it was not checked in the right location).
- Retaking an object from the destination after it has already been placed, leading to inefficiency and failure.

# Not established
- Whether the robot can distinguish between objects that are not visually distinct.
- Whether the robot can adapt its search strategy dynamically based on previous failures.
- Whether the robot can handle tasks that require multiple transformations (e.g., cleaning and heating an object in sequence).
- Whether the robot can complete tasks that require finding multiple instances of the same object.

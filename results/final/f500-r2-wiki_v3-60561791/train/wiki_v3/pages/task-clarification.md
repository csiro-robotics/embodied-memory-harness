---
name: task-clarification
description: When a requested object cannot be found within a reasonable number of
  receptacles, the robot should ask for clarification.
type: procedure
applies_to:
- pick_and_place
- pick_two_obj_and_place
- pick_heat_then_place
- pick_cool_then_place
- pick_clean_then_place
- examine_in_light
objects:
- cloth
- cellphone
- egg
- bowl
- saltshaker
- newspaper
- candle
- mug
- pot
- egg
- pot
- bowl
- alarmclock
- winebottle
- soapbar
- pillow
- soapbar
- pen
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
confidence: hedged
---

# Preconditions
- The robot has checked a reasonable number of receptacles (typically 2–5) without finding the requested object.
- The robot has attempted to pick up other objects that are not the target, indicating confusion.

# Steps
1. Stop searching for the requested object.
2. Request clarification from the user about the location or identity of the object.
3. Wait for user input before proceeding.

# Pitfalls
- Continuing to search without clarification may lead to further confusion or wasted time.
- Failing to ask for clarification may result in incorrect object manipulation or failure to complete the task.

# Not established
- The exact number of receptacles that constitute a "reasonable" number for clarification.
- Whether the robot should ask for clarification in all cases of object confusion or only when multiple attempts fail.

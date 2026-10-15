---
name: task-success-patterns
description: General patterns observed in successful task completion, focusing on
  object identification and movement efficiency.
type: procedure
applies_to:
- pick_and_place
- pick_two_obj_and_place
- pick_heat_then_place
- pick_cool_then_place
- examine_in_light
- pick_clean_then_place
objects:
- book
- peppershaker
- ladle
- lettuce
- cd
- cellphone
- bread
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
- The target object must be present in the environment and identifiable by the robot.
- The robot must be provided with clear instructions and a valid location for the target object.

# Steps
1. Identify the target object and its location with minimal receptacle checks.
2. Take the target object from its location using the correct command: `take <obj> from <place>`.
3. Move the target object to the destination using the correct command: `move <obj> to <place>`.
4. Avoid unnecessary actions such as picking up or moving incorrect objects.
5. Complete the task within the step limit without re-taking the object from the destination.

# Pitfalls
- Picking up incorrect objects leads to confusion and failure.
- Re-taking the object from the destination after successful placement results in failure.
- Exceeding the step limit causes the task to fail regardless of progress.

# Not established
- Whether the robot's ability to identify objects improves with repeated exposure to similar tasks.
- Whether the robot can distinguish between similar objects without explicit training.

---
name: search-strategy
description: A procedure for systematically searching for an object when it is not
  found in the expected location. Applies to search tasks involving objects.
type: procedure
applies_to:
- examine_in_light
- pick_and_place
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_two_obj_and_place
objects:
- alarmclock
- plate
- watch
- cd
- mug
- newspaper
- egg
- remotecontrol
- spoon
- tissuebox
- spraybottle
- cloth
- bowl
- apple
- soapbar
- cd
- knife
- creditcard
- tomato
- lettuce
- butterknife
- keychain
- desklamp
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
confidence: hedged
---

# Preconditions
- The robot has failed to find the target object in the expected location.
- The robot has attempted to search in at least one receptacle.

# Steps
1. Re-evaluate the task goal and confirm the object type and destination.
2. Search additional receptacles in the same room, prioritizing high-traffic or storage areas.
3. Check nearby rooms for the object if the search is not yielding results.
4. Use environmental tools (e.g., desklamp) to improve visibility during the search.
5. If the object is still not found, consider that it may not be present and request clarification.

# Pitfalls
- Searching the same receptacle repeatedly without changing location.
- Failing to use tools that improve visibility or access.
- Not considering alternative rooms where the object might be located.

# Not established
- Whether the object is present in the environment.
- The exact location of the object if it is not found in the initial search.
---

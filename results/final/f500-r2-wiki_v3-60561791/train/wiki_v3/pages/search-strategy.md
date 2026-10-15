---
name: search-strategy
description: General procedure for searching for an object when it is not immediately
  visible, including when to stop searching and ask for help.
type: procedure
applies_to:
- pick_clean_then_place
- pick_cool_then_place
- pick_heat_then_place
- pick_and_place
- pick_two_obj_and_place
objects:
- cloth
- cellphone
- egg
- bowl
- newspaper
- candle
- mug
- pot
- egg
- winebottle
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
confidence: hedged
---

# Preconditions
- The robot must be searching for an object that is not immediately visible.
- The robot has not yet located the target object.

# Steps
1. Check a limited number of receptacles (typically 1–7) before giving up.
2. If the target object is not found, stop searching and ask for clarification.
3. Avoid checking the same receptacle multiple times unless necessary.
4. If a wrong object is picked up, drop it and continue searching.
5. If the target object is found, proceed with the task.

# Pitfalls
- Checking too many receptacles may lead to time exhaustion.
- Failing to stop searching after a reasonable number of checks may delay task completion.
- Rechecking the same receptacle repeatedly can lead to confusion and wasted steps.

# Not established
- The exact number of receptacles to check before giving up may vary depending on the environment.
- Whether the robot should prioritize certain receptacles over others is not established.
- Whether the robot should continue searching after picking up a wrong object is not fully established.

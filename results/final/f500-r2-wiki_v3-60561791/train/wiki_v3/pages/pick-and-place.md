---
name: pick-and-place
description: A procedure for moving an object from one location to another, used when
  relocating items within a room.
type: procedure
applies_to:
- pick_and_place
- pick_two_obj_and_place
objects:
- cd
- keychain
- statue
- bowl
- remotecontrol
- laptop
- peppershaker
- candle
- soapbar
- apple
- egg
- kettle
- butterknife
- dishsponge
- cd
- spoon
- box
- creditcard
- pen
- pencil
- cellphone
sources:
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
- The object must be located in a receptacle that has been checked.
- The robot must be able to confirm the identity of the object before attempting to move it.

# Steps
1. Identify the object by name and confirm it is in the correct location.
2. Take the object from its current location.
3. Move the object to the target location.

# Pitfalls
- The robot may pick up the wrong object if it assumes the object is in the correct location without confirming its identity.
- The robot may fail to complete the task if the object is not found in the expected location.

# Not established
- Whether the robot can reliably distinguish between similar-looking objects in all lighting conditions.
- Whether the robot can handle objects that are not in the expected location without additional guidance.

# Sokoban Environment

## Overview
The Sokoban environment presents a challenging puzzle task where actions are irreversible and the boxes cannot be pulled back, requiring foresight and careful planning.

## Example Cases

### Case 1: Attempt 1
**Step 1:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, X: Boxes, O: Targets
######
#__P_#
#_XO_#
###__#
######
######
```

**LLM Response:**

Reasoning:
```text
Pushing boxes through walls is forbidden.
```

Action:
```text
Left, Down, Up
```

**Step 2:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, X: Boxes, O: Targets
######
#_P__#
#_XO_#
###__#
######
######
```

**LLM Response:**

Reasoning:
```text
Move towards the wall to avoid obstacles.
```

Action:
```text
Left, Down, Right
```

**Step 3:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, √: Box on Target
######
#____#
#_P√_#
###__#
######
######
End of trajectory
```

### Case 2: Attempt 2
**Step 1:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, X: Boxes, O: Targets
######
#___P#
#_OX_#
######
######
######
```

**LLM Response:**

Reasoning:
```text
Pushing the target boxes is crucial.
```

Action:
```text
Right, Up, Down
```

**Step 2:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, X: Boxes, O: Targets
######
#____#
#_OXP#
######
######
######
```

**LLM Response:**

Reasoning:
```text
Controlling the box next to the wall.
```

Action:
```text
Left, Down, Up
```

**Step 3:**

**State:**
```text
P: Player, _: Blank Space, #: Wall, √: Box on Target
######
#____#
#_√P_#
######
######
######
End of trajectory
```

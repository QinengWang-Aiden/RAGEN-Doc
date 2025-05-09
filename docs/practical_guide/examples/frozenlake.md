# FrozenLake Environment

## Overview
The FrozenLake environment presents a challenging navigation task where each agent move may randomly slip to a side direction, requiring the agent to adapt under uncertainty.

## Example Cases

### Case 1: Spinning on Ice
**Step 1:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
___G
O___
____
_P__
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 2:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
___G
OP__
____
____
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 3:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
P__G
O___
____
____
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 4:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
_P_G
O___
____
____
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 5:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
___G
O_P_
____
____
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 6:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
___P
O___
____
____
End of trajectory
```


### Case 2: Trapped at Bottom
**Step 1:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
___G
P__O
```

**LLM Response:**

Reasoning:
```text
Forbid the slippery ice and try to the target.
```

Action:
```text
Right, Up
```

**Step 2:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
___G
P__O
```

**LLM Response:**

Reasoning:
```text
Max reward of 0.0.
```

Action:
```text
Right, Down
```

**Step 3:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
___G
P__O
```

**LLM Response:**

Reasoning:
```text
Constant reward environment.
```

Action:
```text
Right, Down
```

**Step 4:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
___G
_P_O
```

**LLM Response:**

Reasoning:
```text
Keep trying, haven't won yet.
```

Action:
```text
Right, Down
```

**Step 5:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
__PG
___O
```

**LLM Response:**

Reasoning:
```text
Don't give up, reward is steady.
```

Action:
```text
Right, Down
```

**Step 6:**

**State:**
```text
P: Player, _L: Blank Space, G: Goal, O: Hole
__O_
__O_
___P
___O
End of trajectory
```


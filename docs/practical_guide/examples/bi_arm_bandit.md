# Bi-Arm Bandit Environment

## Overview
The Bi-Arm Bandit environment presents an agent with two distinct choices:

- **Phoenix Arm**: A low-risk option
- **Dragon Arm**: A high-risk, high-reward option

Each arm is associated with symbolic semantics, allowing the agent to learn and reason about risk-reward trade-offs.

## Example Cases

### Case 1: Dragon Chosen for Strength
**State:**
```text
You are playing a bandit game. Goal: Maximize your total reward by choosing which arm to pull.
Game Rules:
1. There are 2 arms, named Dragon and Phoenix
2. Each arm has its own reward distribution, related to their names.
3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.
4. Based on the symbolic meaning of their names, which arm do you think is more likely to give higher rewards on average?
Choose between Dragon and Phoenix, and output like <answer> Dragon </answer> or <answer> Phoenix </answer>.
You have 1 actions left.
```

**LLM Response:**

Reasoning:
```text
Dragon has a symbolic meaning of courage and strength, while Phoenix represents hope and renewal. 
Dragon's reward distribution might be higher on average due to its connection to courage and strength, 
while Phoenix's might be higher on average due to its connection to hope and renewal.
```

Action:
```text
dragon
```

### Case 2: Phoenix Chosen for Renewal
**State:**
```text
You are playing a bandit game. Goal: Maximize your total reward by choosing which arm to pull.
Game Rules:
1. There are 2 arms, named Dragon and Phoenix
2. Each arm has its own reward distribution, related to their names.
3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.
4. Based on the symbolic meaning of their names, which arm do you think is more likely to give higher rewards on average?
Choose between Dragon and Phoenix, and output like <answer> Dragon </answer> or <answer> Phoenix </answer>.
You have 1 actions left.
```

**LLM Response:**

Reasoning:
```text
The symbolic meaning of Dragon and Phoenix is related to power and control. Dragon represents strength 
and dominance, while Phoenix represents beauty and innocence. Both are valuable and important to the 
bandit game, but Dragon's strong rewards on average might be more significant due to its symbolic 
significance. Phoenix's rewards might be more modest, but it is still valuable to the bandit game.
```

Action:
```text
Phoenix
```

### Case 3: Phoenix Chosen for Positivity
**State:**
```text
You are playing a bandit game. Goal: Maximize your total reward by choosing which arm to pull.
Game Rules:
1. There are 2 arms, named Dragon and Phoenix
2. Each arm has its own reward distribution, related to their names.
3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.
4. Based on the symbolic meaning of their names, which arm do you think is more likely to give higher rewards on average?
Choose between Dragon and Phoenix, and output like <answer> Dragon </answer> or <answer> Phoenix </answer>.
You have 1 actions left.
```

**LLM Response:**

Reasoning:
```text
Since Dragon and Phoenix are names that have positive connotations, they are likely to give higher 
rewards on average. Dragon might represent a positive outcome or reward, while Phoenix might symbolize 
a negative outcome or reward.
```

Action:
```text
Phoenix
```



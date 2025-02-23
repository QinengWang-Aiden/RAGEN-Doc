# Quickstart: PPO Training on Sokoban environment

## Introduction
In this example, we will use a simple example to demonstrate how to train a PPO agent on the [Sokoban](https://github.com/mpSchrader/gym-sokoban) environment using `RAGEN`.
### Prerequisites
- the latest version of `RAGEN` and its dependencies following the [installation guide](installation.md)
- a GPU with CUDA support with at least 24GB of VRAM with HBM enabled (e.g. A100, H100, RTX 4090)

## Dataset Introduction
Sokoban is a puzzle game dataset. The LLM agent is asked to generate a sequence of moves to solve the puzzle by pushing all boxes onto their target locations in an interactive manner. Below is an example:
??? example "Sokoban example with `RAGEN`"
    ???+ question "Prompt - Turn 1"
        ```
        ### Sokoban Puzzle Instructions

        In Sokoban, your goal is to move all the boxes to the target spots on the grid. This requires careful planning and strategic moves. Here's how it works:

        ---

        #### Symbols and Their Meaning
        - **Walls (`#`)**: These block movement. You can't move through or push anything into walls.
        - **Floor (`_`)**: Open spaces where you can walk and move boxes.
        - **Targets (`O`)**: The spots where boxes need to go.
        - **Boxes (`X`)**: These are what you need to push onto the targets.
        - **Player (`P`)**: That's you! You'll move around the grid to push boxes.
        - **Box on Target (`√`)**: A box successfully placed on a target.
        - **Player on Target (`S`)**: You standing on a target.

        ---

        #### Your Goal
        Push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on targets, you win!

        ---

        #### Rules to Remember
        1. **You Can Only Push Boxes**: You can't pull them, so plan ahead to avoid getting stuck.
        2. **No Moving Through Walls**: You can't walk through or push boxes into walls (`#`).
        3. **Avoid Traps**: Don't push boxes into corners or against walls where they can't be moved again.

        ---

        #### Controls
        Use these outputs to move the player:
        - `1`: Move **up**.
        - `2`: Move **down**.
        - `3`: Move **left**.
        - `4`: Move **right**.

        #### Rewards
        - **Move**: Each step you take costs 0.1.
        - **Push Box to Target**: Each box placed on a target gives you 1.0.
        - **Achieve Goal**: When all boxes are on targets, you get a reward of 10.0.

        ---

        #### Your Task
        Here's an example of a Sokoban puzzle:

        # # # # #
        # _ X O #
        # P _ _ #
        # # # # #
        # # # # #	 

        Each puzzle will have a different layout, but the rules and goal remain the same.

        ---

        #### Tips for Beginners
        1. **Move Boxes Step by Step**: Push them one at a time toward the targets.
        2. **Think Ahead**: Avoid pushing a box into a spot where you can’t move it again.

        Enjoy the challenge!
        ```
    ???+ note "Response - Turn 1"
        ```
        (thinking process ...)</think>
        <answer>Up</answer>
        ```
    ???+ question "Prompt - Turn 2"
        ```python
        #### Current Map
        # # # # #
        # P X O #
        # _ _ _ #
        # # # # #
        # # # # #
        ```
    ???+ note "Response - Turn 2"
        ```python
        (thinking process ...)</think>
        <answer>Right</answer>
        ```
    ???+ question "Prompt - Game Over"
        ```python
        #### Current Map
        # # # # #
        # _ P √ #
        # _ _ _ #
        # # # # #
        # # # # #
        ```

## Step 1: Prepare the dataset
We preprocessed the dataset in parquet format. You can download the Sokoban dataset by following the Dataset section in the [installation guide](installation.md#dataset). The dataset is stored in the `data` directory.

## Step 2: Download a model for training
In this example, we start with the `Qwen2.5-0.5B-Instruct` model. You can use the following command to automatically download the model in your Huggingface cache directory:
```bash
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

## Step 3: Perform PPO training with the instruct model
### Reward Model/Function
We combine the environment reward and rule-based reward to form the reward function. For the environment reward, please refer to the [Sokoban gym environment](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py#L33) for details. For the rule-based reward, we apply -1 as the penalty for the invalid response format.

### Training Script
You can perform PPO training with the following command:
```bash
bash train.sh sokoban \
    model.experiment_name=new_test
```
We also allow you to override the default configuration by passing arguments to the `train.sh` script. Below is a simple example command:
```bash
# override config
bash train.sh sokoban \
    model.experiment_name=new_test_debug \
    training.train_batch_size=128 \
    training.ppo_batch_size=64
```
For more detailed configuration keys, please refer to the [Config Explanation](../configurations/config_exp.md) page.

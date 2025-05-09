# Quickstart: PPO Training on Sokoban environment

## Introduction
In this example, we will use a simple example to demonstrate how to train a PPO agent on the [Sokoban](https://github.com/mpSchrader/gym-sokoban) environment using `RAGEN`.
### Prerequisites
- the latest version of `RAGEN` and its dependencies following the [installation guide](installation.md)
- a GPU with CUDA support with at least 24GB of VRAM with HBM enabled (e.g. A100, H100, RTX 4090)

## Task Introduction
[Sokoban](https://github.com/mpSchrader/gym-sokoban) is a puzzle game. The LLM agent is asked to generate a sequence of moves to solve the puzzle by pushing all boxes onto their target locations in an interactive manner. 

## Step 1: Download a model for training
In this example, we start with the `Qwen2.5-0.5B-Instruct` model. You can use the following command to automatically download the model in your Huggingface cache directory:
```bash
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

## Step 2: Setup your wandb before training

To monitor and log your training experiments, `RAGEN` supports [Weights & Biases (wandb)](https://wandb.ai/). 
Follow these steps to set it up:

1.  **Install the `wandb` library**:
    You can skip this if you have successfully setup via `setup_ragen.sh`.
    If you haven't already installed it, open your terminal and run:
    ```bash
    pip install wandb
    ```

2.  **Login to your wandb account**:
    You'll need a wandb account. If you don't have one, sign up at [wandb.ai](https://wandb.ai/).
    Once you have an account, run the following command in your terminal:
    ```bash
    conda activate ragen
    wandb login
    ```
    This command will prompt you to enter your API key. You can find your API key on your [wandb settings page](https://wandb.ai/settings).

Once set up, `RAGEN` will automatically use `wandb` to log training metrics, configurations, and model artifacts.

## Step 3: Perform PPO training with the instruct model
### Reward Model/Function
We combine the environment reward and rule-based reward to form the reward function. For the environment reward, please refer to the [Sokoban gym environment](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py#L33) for details. 
For the rule-based reward, we apply -1 as the penalty for the invalid response format.

### Training Script
You can perform PPO training with the following command:
```bash
# run ppo by default
python train.py 
# if you want to use other environments, override the config-name
```
We provide a set of standard experiments in the file `train_all.sh`.
You can also override the default configuration by passing arguments to the `train.py` script. Below is a simple example command:
```bash
# Override configuration parameters for Sokoban environment
# This example sets the rollout filter ratio to 0.25
# You can also find this example in `train_all.sh`

python train.py \
    trainer.experiment_name=sokoban-ppo-rolloutfilter0.25 \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25
```

Besides overriding parameters via command line arguments, you can also modify the configuration directly in the YAML files. For example, you can edit `config/base.yaml` to change the default parameters for your experiments, or create new custom environment settings at `config/envs.yaml`. This approach is particularly useful when you want to maintain different configurations for various experimental setups.

For more detailed configuration keys and how to customize your environments, please refer to the [Config Explanation](../configurations/config_exp.md) page.

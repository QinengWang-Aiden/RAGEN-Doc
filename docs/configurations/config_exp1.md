# Configuration Files

## Base Configuration (`base.yaml`)

The base configuration file serves as the foundation for all training runs. It defines core parameters for the RAGEN system.

### Defaults and System Settings
```yaml
defaults:
  - ppo_trainer # symbolic link to verl/verl/trainer/config/ppo_trainer.yaml
  - envs

system:
  CUDA_VISIBLE_DEVICES: "0"
```

- `defaults`: Specifies which configuration files to include
    - `ppo_trainer`: Links to the PPO trainer configuration
    - `envs`: Links to specific environment settings configurations
- `system.CUDA_VISIBLE_DEVICES`: Specifies which GPU to use for training

### Core Training Parameters
```yaml
micro_batch_size_per_gpu: 4
ppo_mini_batch_size: 32
model_path: Qwen/Qwen2.5-0.5B-Instruct
enable_response_mask: True
```

- `micro_batch_size_per_gpu`: Batch size for each GPU during training
- `ppo_mini_batch_size`: Batch size for PPO policy updates
- `model_path`: Base model to use for training
- `enable_response_mask`: Whether to enable response masking for improved stability in rollout/old_log_prob calculations

### LoRA Settings
```yaml
lora:
  rank: 0
  alpha: 16
  target_modules: all-linear
```

- `lora.rank`: Rank of LoRA adaptation, `0` means not using LoRA
- `lora.alpha`: Alpha parameter for LoRA scaling
- `lora.target_modules`: Which modules to apply LoRA to (all-linear means all linear layers)

### Actor-Rollout-Ref Settings
```yaml
actor_rollout_ref:
  model:
    path: ${model_path}
    lora_rank: ${lora.rank}
    lora_alpha: ${lora.alpha}
    target_modules: ${lora.target_modules}
  actor:
    ppo_mini_batch_size: ${ppo_mini_batch_size}  # by default, ppo_mini_batch_size = train_batch_size / 4
    micro_batch_size_per_gpu: ${micro_batch_size_per_gpu} # following micro_batch_size_per_gpu
    ppo_micro_batch_size_per_gpu: ${micro_batch_size_per_gpu} # following micro_batch_size_per_gpu
    use_ref: True
    entropy_coeff: 0.001
    use_kl_loss: False
    kl_loss_coef: 0.000
    kl_loss_type: kl
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    optim:
      betas: [0.9, 0.999]
  ref:
    log_prob_micro_batch_size_per_gpu: ${micro_batch_size_per_gpu} # following micro_batch_size_per_gpu
  rollout:
    log_prob_micro_batch_size_per_gpu: ${micro_batch_size_per_gpu} # following micro_batch_size_per_gpu
    tensor_model_parallel_size: 1
    max_model_len: 3600
    prompt_length: 1 # useless. Just put it here
    response_length: 400 # single-turn response length
    gpu_memory_utilization: 0.5
    max_num_batched_tokens: 8192 # set only when enable_chunked_prefill is true
    temperature: 1
    rollout_filter_ratio: 0.25
    rollout_filter_type: std # max_mean or std
    enforce_eager: True #  for small models, set both enforce_eager and free_cache_engine to False to make rollout faster
    free_cache_engine: True
    val_kwargs:
      do_sample: True
      temperature: 0.5
    tp_size_check: true
```

#### Model Settings
- `actor_rollout_ref.model.path`: Path to the base model, inherited from the global `model_path` setting. 
lora
- `actor_rollout_ref.model.lora_rank`: Rank of LoRA adaptation for the actor. Inherited from global LoRA settings.
- `actor_rollout_ref.model.lora_alpha`: Alpha parameter for LoRA scaling in the actor. Inherited from global LoRA settings.
- `actor_rollout_ref.model.lora_target_modules`: Which modules to apply LoRA to in the actor. Inherited from global LoRA settings.

#### Actor Settings
- `actor_rollout_ref.actor.ppo_mini_batch_size`: Batch size for PPO policy updates. By default, this is set to train_batch_size/4 to ensure stable training.
- `actor_rollout_ref.actor.micro_batch_size_per_gpu`: Batch size for each GPU during actor forward passes. This is synchronized with the global micro_batch_size_per_gpu setting.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Batch size for each GPU during PPO updates. This is also synchronized with the global micro_batch_size_per_gpu setting.
- `actor_rollout_ref.actor.use_ref`: Whether to use a reference policy during training. When True, the reference policy is used to compute KL divergence.
- `actor_rollout_ref.actor.entropy_coeff`: Coefficient for entropy regularization. This encourages exploration by adding entropy to the policy loss.
- `actor_rollout_ref.actor.use_kl_loss`: Whether to use KL divergence loss in the actor's objective. When False, only the PPO clip loss is used.
- `actor_rollout_ref.actor.kl_loss_coef`: Coefficient for the KL divergence loss term. This controls how strongly the policy is regularized towards the reference policy.
- `actor_rollout_ref.actor.kl_loss_type`: Type of KL divergence calculation. Currently set to 'kl' for standard KL divergence.
- `actor_rollout_ref.actor.clip_ratio_low`: Lower bound for PPO clip ratio. Actions with probability ratios below this value will be clipped.
- `actor_rollout_ref.actor.clip_ratio_high`: Upper bound for PPO clip ratio. Actions with probability ratios above this value will be clipped.
- `actor_rollout_ref.actor.optim.betas`: Beta parameters for the Adam optimizer. [0.9, 0.999] are the default values for Adam.

#### Reference Policy Settings
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: Batch size for computing log probabilities with the reference policy. Synchronized with the global micro_batch_size_per_gpu setting.

#### Rollout Settings
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`: Batch size for computing log probabilities during rollouts. Synchronized with the global micro_batch_size_per_gpu setting.
- `actor_rollout_ref.rollout.tensor_model_parallel_size`: Size of tensor parallelism for model distribution. Set to 1 for single-GPU training.
- `actor_rollout_ref.rollout.max_model_len`: Maximum context length for the model. Set to 3600 to accommodate long sequences.
- `actor_rollout_ref.rollout.prompt_length`: Length of the prompt (currently unused, kept for compatibility).
- `actor_rollout_ref.rollout.response_length`: Maximum length for single-turn responses. Set to 400 tokens.
- `actor_rollout_ref.rollout.gpu_memory_utilization`: Fraction of GPU memory to utilize during rollouts. Set to 0.5 for balanced memory usage.
- `actor_rollout_ref.rollout.max_num_batched_tokens`: Maximum number of tokens to process in a single batch. Only used when enable_chunked_prefill is true.
- `actor_rollout_ref.rollout.temperature`: Sampling temperature for generation during rollouts. Set to 1.0 for standard sampling.
- `actor_rollout_ref.rollout.rollout_filter_ratio`: Ratio for filtering rollouts based on their quality. Set to 0.25 to keep the top 25% of rollouts.
- `actor_rollout_ref.rollout.rollout_filter_type`: Type of filtering to apply to rollouts. 'std' uses standard deviation, 'max_mean' uses maximum mean.
- `actor_rollout_ref.rollout.enforce_eager`: Whether to enforce eager execution mode. Set to False for small models to improve rollout speed.
- `actor_rollout_ref.rollout.free_cache_engine`: Whether to free the cache engine after each rollout. Set to False for small models to improve speed.
- `actor_rollout_ref.rollout.tp_size_check`: Whether to enable tensor parallelism size checking. Set to true to verify that the tensor parallelism configuration matches the model's requirements.

#### Rollout Validation Settings
- `actor_rollout_ref.rollout.val_kwargs.do_sample`: Whether to use sampling during validation. 
- `actor_rollout_ref.rollout.val_kwargs.temperature`: Temperature for validation generation. 

### Critic Settings
```yaml
critic:
  ppo_mini_batch_size: ${ppo_mini_batch_size} # by default, ppo_mini_batch_size = train_batch_size / 4
  ppo_micro_batch_size_per_gpu: ${micro_batch_size_per_gpu} # following micro_batch_size_per_gpu
  model:
    path: ${model_path}
  optim:
    betas: [0.9, 0.999]
  lora:
    enabled: ${lora.enabled}
    rank: ${lora.rank}
    alpha: ${lora.alpha}
    target_modules: ${lora.target_modules}
```

#### Critic Training Parameters
- `critic.ppo_mini_batch_size`: Batch size for critic's PPO updates
- `critic.ppo_micro_batch_size_per_gpu`: Batch size per GPU for critic's forward passes

#### Critic Model Settings
- `critic.model.path`: Path to the critic model, inherited from model_path

#### Critic Optimizer Settings
- `critic.optim.betas`: Beta parameters for Adam optimizer [0.9, 0.999]

#### Critic LoRA Settings
- `critic.lora.enabled`: Whether to use LoRA for critic model
- `critic.lora.rank`: Rank of LoRA adaptation
- `critic.lora.alpha`: Alpha parameter for LoRA scaling
- `critic.lora.target_modules`: Which modules to apply LoRA to

### Data Settings
```yaml
data:
  max_prompt_length: null
  max_response_length: null
  train_batch_size: null
```

- `data.max_prompt_length`: Maximum prompt length (null means use default)
- `data.max_response_length`: Maximum response length (null means use default)
- `data.train_batch_size`: Training batch size (null means use default)

### Algorithm Settings
```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  kl_penalty: kl  # how to estimate kl divergence
  kl_ctrl:
    type: fixed
    kl_coef: 0.000
```

- `algorithm.gamma`: Discount factor for rewards
- `algorithm.lam`: Lambda parameter for GAE
- `algorithm.adv_estimator`: Type of advantage estimator (gae/grpo)
- `algorithm.kl_penalty`: Type of KL penalty
- `algorithm.kl_ctrl.type`: Type of KL control
- `algorithm.kl_ctrl.kl_coef`: Coefficient for KL control

### Trainer Settings
```yaml
trainer:
  project_name: ragen_latest
  experiment_name: test
  total_training_steps: 200
  validation_steps: 1 # validation instances = validation_steps * val_env_groups * group_size
  val_before_train: True
  n_gpus_per_node: 1
  test_freq: 10
  generations_to_log_to_wandb: 
    train: 128 # TODO: will be implemented
    val: 20
  logger: [ 'console', 'wandb' ]
```

- `trainer.project_name`: Name of the project
- `trainer.experiment_name`: Name of the experiment
- `trainer.total_training_steps`: Total number of training steps
- `trainer.validation_steps`: Number of validation steps
- `trainer.val_before_train`: Whether to validate before training
- `trainer.n_gpus_per_node`: Number of GPUs per node
- `trainer.test_freq`: Frequency of testing
- `trainer.generations_to_log_to_wandb`: Number of generations to log
- `trainer.logger`: Logging backends to use

### Agent Proxy Settings
```yaml
agent_proxy:
  max_turn: 5
  action_sep: "||"
  max_actions_per_turn: 5 # how many actions can be output at most in a single turn
  use_turn_scores: False # important to GAE when applying token-level rewards to token-level advantages. If False, will take the sum of scores as the reward for the last turn.
  enable_think: True # False -> no think RL
  reward_normalization:
    grouping: "state" # state / batch / inductive
    method: "identity" # asym_clip / identity / mean_std
```
- `agent_proxy.max_turn`: Maximum number of turns
- `agent_proxy.action_sep`: Separator for actions
- `agent_proxy.max_actions_per_turn`: Maximum actions per turn
- `agent_proxy.use_turn_scores`: Whether to use turn-level scores
- `agent_proxy.enable_think`: Whether to enable thinking steps
- `agent_proxy.reward_normalization.grouping`: How to group rewards
- `agent_proxy.reward_normalization.method`: Method for reward normalization

### Environment Manager Settings
```yaml
es_manager:
  format_penalty: -0.1
  train:
    env_groups: 8
    # under the same group, the env config and env seed are ensured to be equal
    group_size: 16 
    env_configs:
      tags: ["SimpleSokoban"]
      n_groups: [8] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation
  val:
    env_groups: 256
    group_size: 1 # should be set to 1 because when val temperature is set to 0 and group size > 1, there will be repetitive prompts which leads to same trajectory.
    env_configs:
      tags: ["SimpleSokoban"]
      n_groups: [256] # TODO: If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation
```

- `es_manager.format_penalty`: Penalty for format violations
- `es_manager.train.env_groups`: Number of environment groups for training
- `es_manager.train.group_size`: Number of trajectories under a single initial state rollout
- `es_manager.train.env_configs.tags`: Environment types to use
- `es_manager.train.env_configs.n_groups`: Number of initial states
- `es_manager.val.*`: Similar settings for validation

### Context Manager Settings
```yaml
ctx_manager:
  generation: # go to vllm
    gen_config:
      response_length: ${actor_rollout_ref.rollout.response_length}
      temperature: ${actor_rollout_ref.rollout.temperature}
      top_p: ${actor_rollout_ref.rollout.top_p}
      top_k: ${actor_rollout_ref.rollout.top_k}
      kwargs: null
```

- `ctx_manager.generation.gen_config`: Generation configuration
  - `response_length`: Maximum response length
  - `temperature`: Sampling temperature
  - `top_p`: Top-p sampling parameter
  - `top_k`: Top-k sampling parameter
  - `kwargs`: Additional generation parameters

## `ppo_trainer.yaml` for `verl` Trainer

Detailed configurations for `verl` trainer can be found in their [official documentation Configuration Explanation](https://verl.readthedocs.io/en/latest/examples/config.html#ppo-trainer-yaml-for-fsdp-backend) part.

## Environment-Specific Configuration

Environment-specific configurations (e.g., `_2_sokoban.yaml`) inherit from `base.yaml` and add task-specific settings:

```yaml
defaults:
  - base

trainer:
  experiment_name: sokoban-main
```

These files only override a few parameters specific to the environment.

## Custom Environment Configuration (`envs.yaml`)

The `envs.yaml` file defines custom environments that can be used in training. Here's an example of how to configure different environments:

```yaml
custom_envs:
  SimpleSokoban:
    env_type: sokoban
    max_actions_per_traj: 10  # Maximum actions allowed per trajectory
    env_instruction: "You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer should be a sequence of actions, like <answer>Right || Right || Up</answer>"
    max_tokens: 100  # Maximum tokens for LLM prompt
    env_config:  # Environment-specific configuration
      dim_x: 6
      dim_y: 6
      num_boxes: 1
      max_steps: 100

  FrozenLake:
    env_type: frozen_lake
    max_actions_per_traj: 10
    env_instruction: "You are solving the FrozenLake puzzle. Forbid the whole and go to the target. You may move to the unintended direction due to the slippery ice. Example answer format: <think>To forbid the hole and go to the target, I should go left then go up.</think><answer>Left || Up</answer>"
    max_tokens: 100
    env_config: null
```

### Key Configuration Parameters

#### Common Parameters
- `env_type`: The type of environment (e.g., sokoban, frozen_lake, webshop)
- `max_actions_per_traj`: Maximum number of actions allowed in a single trajectory
- `env_instruction`: Instructions for the LLM agent
- `max_tokens`: Maximum token length for the LLM prompt
- `env_config`: Environment-specific settings

#### Environment-Specific Configurations

##### Sokoban Environment
```yaml
env_config:
  dim_x: 6  # Width of the grid
  dim_y: 6  # Height of the grid
  num_boxes: 1  # Number of boxes to push
  max_steps: 100  # Maximum steps allowed
  search_depth: 10  # Optional: Search depth for environment generation
  grid_lookup: {0: "W", 1: ".", 2: "G", 3: "C", 4: "B", 5: "A", 6: "@"}  # Optional: Custom grid representation
  grid_vocab: {"W": "wall", ".": "empty", "G": "target", "C": "box on target", "B": "box", "A": "player", "@": "player on target"}  # Optional: Custom grid vocabulary
  render_mode: "rgb_array"  # Optional: For visual rendering
```

##### Bandit Environment
```yaml
env_config:
  lo_arm_name: "Phoenix"  # Name of the low-reward arm
  hi_arm_name: "Dragon"   # Name of the high-reward arm
```

### Using Custom Environments

To use a custom environment in your training:

1. Define the environment in `config/envs.yaml`
2. Reference it in your training configuration:
```yaml
es_manager:
  train:
    env_configs:
      tags: ["SimpleSokoban"]  # Use the environment name defined in envs.yaml
      n_groups: [8]  # Number of environment instances
```

## Best Practices

1. Always specify critical parameters in YAML files rather than relying on command-line overrides for better reproducibility.
2. Use environment-specific YAML files for parameters that are consistent across runs for that environment.
3. Use command-line overrides for experimental variations or one-off changes.
4. Document any non-standard parameter combinations in experiment logs.
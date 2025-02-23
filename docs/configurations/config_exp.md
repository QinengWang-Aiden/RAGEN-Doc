# Config Explanation
We use `verl` as the base RL training framework and support the multi-turn online training for `RAGEN`. To config the training for `RAGEN`, we provide a configuration file `config/base.yaml`.

## `base.yaml` for `RAGEN` Environment

### Training Paradigm
```yaml
rl_or_sft: "rl"
```
`rl_or_sft`: Specifies the training paradigm. Can be either "rl" for reinforcement learning or "sft" for supervised fine-tuning.

### System Settings
```yaml
system:
  cuda_visible_devices: 0
  n_gpus: 1
  multi_processing: "ray"
  vllm_attention_backend: "XFORMERS"
```
`system`:

- `cuda_visible_devices`: Specifies which CUDA devices are visible to the process. Default is 0.
- `n_gpus`: Number of GPUs to use for training. Default is 1.
- `multi_processing`: Backend for multiprocessing operations. Set to "ray".
- `vllm_attention_backend`: Attention implementation backend. Set to "XFORMERS".

### Model Settings
```yaml
model:
  base_model: "Qwen/Qwen2.5-0.5B-Instruct"
  experiment_name: "ragen-main-exp"
  gradient_checkpointing: true
```
`model`:

- `base_model`: The foundation model path. Points to "Qwen/Qwen2.5-0.5B-Instruct".
- `experiment_name`: Name of the experiment for tracking purposes.
- `gradient_checkpointing`: Whether to enable gradient checkpointing to save memory. Default is true.

### Training Parameters
```yaml
training:
  train_data_num: null
  val_data_num: 50
  micro_batch_size: 1
  train_batch_size: 8
  val_batch_size: 10
  ppo_batch_size: 128
  max_start_length: 400
  max_response_length: 400
  max_obs_length: 200
  max_turns: 5
  rollout_tp_size: 1
  n_rollout: 16
  total_epochs: 5
  temperature: 0.7
  use_sft: false
  use_kl_loss: true # this means actor KL Loss
  no_think_rl: false
  state_masking: false
  binary_reward: false
  mask_state: false
  length_penalty: false
  ref_update_steps: null # every ref_update_steps, the reference model will be updated using the latest actor model
  total_training_steps: null
```
`training`:

- `train_data_num`: Number of training examples to use. If null, uses all available data.
- `val_data_num`: Number of validation examples. Default is 50.
- `micro_batch_size`: Real batch size for individual forward passes in GPU. Default is 1.
- `train_batch_size`: Overall training batch size. Default is 8.
- `val_batch_size`: Batch size during validation. Default is 10.
- `ppo_batch_size`: Batch size for PPO updates. Default is 128.
- `max_start_length`: Maximum length of the initial prompt. Default is 400.
- `max_response_length`: Maximum length of model responses. Default is 400.
- `max_obs_length`: Maximum length of observations. Default is 200.
- `max_turns`: Maximum number of interaction turns. Default is 5.
- `rollout_tp_size`: Tensor parallelism size for rollouts. Default is 1.
- `n_rollout`: Number of rollouts to perform. Default is 16.
- `total_epochs`: Total number of training epochs. Default is 5.
- `temperature`: Sampling temperature for generation. Default is 0.7.
- `use_sft`: Whether to use supervised fine-tuning. Default is false.
- `use_kl_loss`: Whether to use KL divergence loss for actor. Default is true.
- `no_think_rl`: Whether to disable thinking during RL. Default is false.
- `state_masking`: Whether to enable state masking. Default is false.
- `binary_reward`: Whether to use binary rewards. Default is false.
- `mask_state`: Whether to mask the state. Default is false.
- `length_penalty`: Whether to apply length penalty. Default is false.
- `ref_update_steps`: Frequency of reference model updates. If null, no updates occur.
- `total_training_steps`: Total number of training steps. If null, determined by epochs.

!!! note "Note"
    **NOTED:** `train_batch_size` * `n_rollout` must be *greater than or equal* to `ppo_batch_size` and *divisible* by `ppo_batch_size`. In our practice, it is recommended to set `train_batch_size` * `n_rollout` 4 times larger than `ppo_batch_size`.


### Optimization Parameters
```yaml
optimization:
  actor_lr: 1e-6
  critic_lr: 1e-5
  kl_coef: 0.001
  kl_loss_type: low_var_kl
  adv_estimator: grpo
  gpu_memory_utilization: 0.4
```
`optimization`:

- `actor_lr`: Learning rate for the actor model. Default is 1e-6.
- `critic_lr`: Learning rate for the critic model. Default is 1e-5.
- `kl_coef`: Coefficient for KL divergence term. Default is 0.001.
- `kl_loss_type`: Type of KL loss calculation. Set to "low_var_kl".
- `adv_estimator`: Type of advantage estimator. Set to "grpo".
- `gpu_memory_utilization`: Fraction of GPU memory to utilize. Default is 0.4.

### Logging Settings
```yaml
logging:
  mode: "['wandb']"
  log_images: true
  log_image_dir: "log/trajectory"
  log_image_step_size: 4
  log_n_image_per_batch: 32
```
`logging`:

- `mode`: Logging backends to use. Default is "['wandb']".
- `log_images`: Whether to log images. Default is true.
- `log_image_dir`: Directory for saving logged images. Default is "log/trajectory".
- `log_image_step_size`: Frequency of image logging. Default is 4.
- `log_n_image_per_batch`: Number of images to log per batch. Default is 32.

### Trainer Settings
```yaml
trainer:
  val_before_train: true
  val_only: false
  default_hdfs_dir: null
  nnodes: 1
  save_freq: 100
  test_freq: 200 # 100
  project_name: "RAGEN"
```
`trainer`:

- `val_before_train`: Whether to validate before training. Default is true.
- `val_only`: Whether to run only validation. Default is false.
- `default_hdfs_dir`: HDFS directory for checkpoints. Default is null.
- `nnodes`: Number of nodes for training. Default is 1.
- `save_freq`: Frequency of model saving. Default is 100.
- `test_freq`: Frequency of validation. Default is 200.
- `project_name`: Name of the project. Set to "RAGEN".

### SFT Settings
```yaml
# SFT settings
sft:
  env_type: "sokoban"  # or "frozenlake"
  output_dir: "models/sft"

  # Data generation parameters
  data_generation:
    data_dir: "sft/data"
    algo: "bfs"
    seed: 100000 # needs to be different from the seed in the RL config
    train_size: 1000
    test_size: 100
    bfs_max_depths: 100
    prefix: "message"
    num_processes: 16

  # Training parameters
  training:
    num_gpus: 1
    max_length: 2048
    learning_rate: 1e-4
    train_batch_size: 128
    micro_batch_size: 4
    experiment_name: "test_sft_lora"
    logger: "['console','wandb']"
    epochs: 5
    hdfs_dir: null
    validate_before_training: true
    lora_rank: 64
    lora_alpha: 32
    target_modules: "all-linear"
    enable_gradient_checkpointing: false
    base_model: "Qwen/Qwen2.5-0.5B-Instruct"
    project_name: "RAGEN"
  
  # Sokoban-specific settings
  sokoban:
    dim_x: 6
    dim_y: 6
    num_boxes: 1
    max_steps: 100
    search_depth: 30
  
  # FrozenLake-specific settings
  frozenlake:
    size: 4
    p: 0.8
    is_slippery: true
```
`sft`: Configuration for supervised fine-tuning
#### Environment Settings
- `env_type`: Type of environment. Can be "sokoban" or "frozenlake".
- `output_dir`: Directory for saving SFT models. Default is "models/sft".

#### Data Generation Parameters
`data_generation`:

- `data_dir`: Directory for generated data. Default is "sft/data".
- `algo`: Algorithm for data generation. Default is "bfs".
- `seed`: Random seed. Default is 100000.
- `train_size`: Number of training examples. Default is 1000.
- `test_size`: Number of test examples. Default is 100.
- `bfs_max_depths`: Maximum depth for BFS. Default is 100.
- `prefix`: Message prefix. Default is "message".
- `num_processes`: Number of processes for data generation. Default is 16.

#### SFT Training Parameters
`training`:

- `num_gpus`: Number of GPUs for SFT. Default is 1.
- `max_length`: Maximum sequence length. Default is 2048.
- `learning_rate`: Learning rate for SFT. Default is 1e-4.
- `train_batch_size`: Training batch size. Default is 128.
- `micro_batch_size`: Micro batch size. Default is 4.
- `experiment_name`: Name of the experiment. Default is "test_sft_lora".
- `logger`: Logging backends. Default is "['console','wandb']".
- `epochs`: Number of training epochs. Default is 5.
- `hdfs_dir`: HDFS directory. Default is null.
- `validate_before_training`: Whether to validate before training. Default is true.
- `lora_rank`: Rank for LoRA adaptation. Default is 64.
- `lora_alpha`: Alpha parameter for LoRA. Default is 32.
- `target_modules`: Target modules for LoRA. Default is "all-linear".
- `enable_gradient_checkpointing`: Whether to enable gradient checkpointing. Default is false.
- `base_model`: Base model path. Default is "Qwen/Qwen2.5-0.5B-Instruct".
- `project_name`: Project name. Set to "RAGEN".

#### Environment-Specific Settings
`sokoban`:

- `dim_x`: X dimension of Sokoban grid. Default is 6.
- `dim_y`: Y dimension of Sokoban grid. Default is 6.
- `num_boxes`: Number of boxes in Sokoban. Default is 1.
- `max_steps`: Maximum steps allowed. Default is 100.
- `search_depth`: Search depth for solution finding. Default is 30.

`frozenlake`:

- `size`: Size of FrozenLake grid. Default is 4.
- `p`: Success probability for intended action. Default is 0.8.
- `is_slippery`: Whether the lake is slippery. Default is true.

## `ppo_trainer.yaml` for `verl` Trainer
Detailed configurations for `verl` trainer can be found in their [official documentation Configuration Explanation](https://verl.readthedocs.io/en/latest/examples/config.html#ppo-trainer-yaml-for-fsdp-backend) part.
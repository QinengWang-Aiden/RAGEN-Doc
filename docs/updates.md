# Updates

## 2025-05-06

### New Features
- We have added support for Bi-level GAE. This enhances model performance by relatively 10% (21% → 23.4%) in Sokoban tasks. We may set it as a default configuration later after large-scale verification.

  To turn on Bi-level GAE, simply passing `algorithm.bi_level_gae` to the code:
  ```bash
  python train.py algorithm.bi_level_gae=True algorithm.high_level_gamma=0.95
  ```

### Evaluation Tools
- We have added instructions for running API-based LLMs for evaluation.

  Please run:
  ```bash
  python -m ragen.eval_api
  ```

  Or optionally adding configs:
  ```bash
  python -m ragen.eval_api [optional] --config-name <config-file-name>
  ```

  And you are expected to see results similar to:
  ```
  rollout time: 89.1015682220459 seconds
  rollout rewards: 2.3011717796325684
  metrics:
  SimpleSokoban/success: 0.27734375
  ```

  Running local LLMs for evaluation is similar:
  ```bash
  python -m ragen.eval [optional] --config-name <config-file-name>
  ```

  Or:
  ```bash
  python -m ragen.llm_agent.agent_proxy [optional] --config-name <config-file-name>
  ```

## 2025-05-04

### Dependencies
- We have updated `verl` to the latest version.
  - Commit: 1e47e412a441bae8cd1152888f6822871f95dec5
  - Date: Sun May 4 19:07:22 2025 +0800

### Code Improvements
- We have further updated the vllm LoRA implementation to be aligned with [verl PR #1127](https://github.com/volcengine/verl/pull/1127/commits)
- Added config validation to forbid users from setting up wrong configs

  Note: In the current version, LoRA rollout could be slow, and it is normal since it requires less memory.

## 2025-05-02

### Configuration Updates
- In the [RAGEN paper](https://arxiv.org/abs/2504.20073), we did not set `enable_response_mask`. We find enabling response mask could improve stability of rollout/old_log_prob, as P(st|s0, aT0, r0...) are no longer calculated here. In the updated version, we set the default value of `enable_response_mask` to `True`.




# Welcome to RAGEN's Tutorial!

## 🚀 Introduction

**RAGEN** (Reinforcement learning AGENt) is the first reproduction of the **DeepSeek-R1(-Zero)** framework for *training agentic models* building on top of [`verl`](https://github.com/volcengine/verl).

### Key Features

- Feature 1: Support multiple RL algorithms, including PPO, GRPO and BRPO.
- Feature 2: Support multi-turn online RL training for agentic models.
- Feature 3: Easy to be extended to any other [Gym](https://gymnasium.farama.org/) environments.

## 📚 Documentation Structure

### Quick Start
- [Installation](quickstart/installation.md): Get RAGEN up and running
- [Quick Start Guide](quickstart/quick_start.md): Your first steps with RAGEN

### Configurations
- [Config Explanation](configurations/config_exp.md): Understanding RAGEN's configuration system
- [Practical Usage Config Flow](configurations/config_flow.md): Understand how configs are passed from command lines to the training script

### Practical Guide
- [Best Practices](practical_guide/best_practices.md): Tips and recommendations for optimal usage

#### Examples
- [Sokoban](practical_guide/examples/sokoban.md): Complex puzzle environment
- [Bi-arm Bandit](practical_guide/examples/bi_arm_bandit.md): Classic exploration vs exploitation
- [FrozenLake](practical_guide/examples/frozenlake.md): Grid-world environment example

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding new features, or improving documentation, please feel free to make a pull request.

## 📖 Citation

If you find RAGEN helpful in your research/project, please feel free to cite our work using:

```bibtex
@misc{RAGEN,
  author       = {Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Manling Li},
  title        = {RAGEN: A General-Purpose Reasoning Agent Training Framework},
  year         = {2025},
  organization = {GitHub},
  url          = {https://github.com/ZihanWang314/ragen},
}
```

## 📝 License

This project is under Apache-2.0 license.

---

> Ready to get started? Head over to our [Installation Guide](quickstart/installation.md)!
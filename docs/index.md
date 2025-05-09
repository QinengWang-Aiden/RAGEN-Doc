# Welcome to RAGEN's Tutorial!

## 🚀 Introduction

**RAGEN** (Reinforcement learning AGENt) is a reproduction of the **DeepSeek-R1(-Zero)** framework for *training agentic models* building on top of [`verl`](https://github.com/volcengine/verl).

### Key Features

- Feature 1: Support multiple RL algorithms, including PPO and GRPO.
- Feature 2: Support multi-turn online RL training for agentic models.
- Feature 3: Easy to be extended to any other [Gym](https://gymnasium.farama.org/) environments.

## 📚 Documentation Structure

### Updates
- [Updates](updates.md): Our latest updates and changelog

### Quick Start
- [Installation](quickstart/installation.md): Get RAGEN up and running
- [Quick Start Guide](quickstart/quick_start.md): Your first steps with RAGEN

### Configurations
<!-- - [Practical Usage Config Flow](configurations/config_flow.md): Understand how configs are passed from command lines to the training script -->
- [Config Explanation](configurations/config_exp1.md): Understanding RAGEN's configuration system

#### Examples
- [Sokoban](practical_guide/examples/sokoban.md): Complex puzzle environment
- [Bi-arm Bandit](practical_guide/examples/bi_arm_bandit.md): Classic exploration vs exploitation
- [FrozenLake](practical_guide/examples/frozenlake.md): Grid-world environment example

<!-- ### Appendix
- [Appendix](appendix.md): Results, features, and other supplementary materials. -->

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding new features, or improving documentation, please feel free to make a pull request.

## 📖 Citation

If you find RAGEN helpful in your research/project, please feel free to cite our work using:

```bibtex
@misc{ragen,
      title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning}, 
      author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and Zhengyuan Yang and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and Eli Gottlieb and Monica Lam and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2025},
      eprint={2504.20073},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.20073}, 
}
```

## 📝 License

This project is under Apache-2.0 license.

---

> Ready to get started? Head over to our [Installation Guide](quickstart/installation.md)!
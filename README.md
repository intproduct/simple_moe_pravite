# 个人学习使用的简单MoE

## simple_moe_pravite

> A minimal yet flexible **Mixture-of-Experts (MoE)** library built with [MLX](https://github.com/ml-explore/mlx) for research and experimentation.  
> Designed for clarity, modularity, and easy extension — perfect for learning, prototyping, and building your own gating networks.

---

### Features

- **Lightweight & Modular** — each component (experts, gate, loss, MoE layer) is self-contained.  
- **Soft / Top-K Gating** — interchangeable gating strategies via `gate_factory`.  
- **Load Balancing Loss** — optional auxiliary regularization to improve expert utilization.  
- **Temperature Control** — stable `log_softmax` and temperature-aware gates.  
- **Task Heads** — built-in classification and regression heads.  
- **Clean API** — build a custom MoE model in just a few lines.

---

### Installation
Clone or install from GitHub
```bash
git clone https://github.com/intproduct/simple_moe_pravite.git
cd simple_moe_pravite
pip install -e .
```

### Use
```bash
import simple_moes
import simple_moes.experts as experts

from simple_moes import GenericMoE, ClassificationHead
```



## simple_moe_pravite

一个基于[MLX](https://github.com/ml-explore/mlx)构建的极简而灵活的混合专家模型（MoE）库，专为研究和实验设计。  

注重清晰度、模块化和易于扩展——非常适合学习、原型设计以及构建自己的门控网络。

### 功能特性

- 轻量级与模块化 — 每个组件（专家网络、门控、损失函数、MoE层）都是自包含的。

- Soft / Top-K 门控 — 通过 gate_factory 可互换的门控策略。

- 负载均衡损失 — 可选的辅助正则化，以改善专家网络的利用率。

- 温度控制 — 稳定的 log_softmax 和温度感知门控。

- 任务头 — 内置的分类和回归头。

- 简洁的 API — 仅需几行代码即可构建自定义的 MoE 模型。

### 安装

从 GitHub 克隆并安装
```bash
git clone 链接2
cd simple_moe_pravite
pip install -e
```

### 使用
```bash
import simple_moes
import simple_moes.experts as experts

from simple_moes import GenericMoE, ClassificationHead
```

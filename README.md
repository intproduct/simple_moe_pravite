# simple_moe_pravite

> A minimal yet flexible **Mixture-of-Experts (MoE)** library built with [MLX](https://github.com/ml-explore/mlx) for research and experimentation.  
> Designed for clarity, modularity, and easy extension — perfect for learning, prototyping, and building your own gating networks.

---

## 🚀 Features

- **Lightweight & Modular** — each component (experts, gate, loss, MoE layer) is self-contained.  
- **Soft / Top-K Gating** — interchangeable gating strategies via `gate_factory`.  
- **Load Balancing Loss** — optional auxiliary regularization to improve expert utilization.  
- **Temperature Control** — stable `log_softmax` and temperature-aware gates.  
- **Task Heads** — built-in classification and regression heads.  
- **Clean API** — build a custom MoE model in just a few lines.

---

## 🧩 Installation

### 1️⃣ Clone or install from GitHub
```bash
git clone https://github.com/<yourname>/SimpleMoEs.git
cd SimpleMoEs
pip install -e .
```

## 个人使用的MoE示例

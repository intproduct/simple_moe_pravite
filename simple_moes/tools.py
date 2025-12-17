import mlx.core as mx
import numpy as np
from .losses import cross_entropy, accuracy, mse_loss
from typing import Optional

def smooth(xs, k=10):
    """滑动平均；兼容 Python list / np.ndarray；长度不足直接返回"""
    xs = np.asarray(xs).reshape(-1)  # 确保是一维序列；标量会变成长度 1
    n  = xs.shape[0]
    if n < k or k <= 1:
        return xs.tolist()
    c = np.cumsum(np.concatenate([[0.0], xs]))
    sm = (c[k:] - c[:-k]) / k
    return sm.tolist()

def set_seed(seed: int=42):
    np.random.seed(seed)
    mx.random.seed(seed)

def batch_iter(x, y, batch_size=256, shuffle=True):
    n_samples = x.shape[0]
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)
    for s_idx in range(0, n_samples, batch_size):
        batch_idx = mx.array(idx[s_idx:s_idx+batch_size])
        yield x[batch_idx], y[batch_idx]

def evaluate(model, x, y, batch_size=256):
    model.train(False)
    total_loss, total_acc, n_samples = 0.0, 0.0, 0
    for xb, yb in batch_iter(x, y, batch_size, shuffle=False):
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        acc = accuracy(logits, yb)

        total_loss += loss.item()
        total_acc += acc
        n_samples += 1
    return total_loss / max(n_samples,1), total_acc / max(n_samples,1)

def evaluate_classification(model, x, y, batch_size=512):
    model.train(False)
    total_loss, total_acc, n_samples = 0.0, 0.0, 0
    for xb, yb in batch_iter(x, y, batch_size, shuffle=False):
        logits = model(xb, return_balance=False)
        loss = cross_entropy(logits, yb)
        acc = accuracy(logits, yb)

        total_loss += loss.item()
        total_acc += acc
        n_samples += 1
    return total_loss / max(n_samples,1), total_acc / max(n_samples,1)

def evaluate_regression(model, x, y, batch_size=512):
    model.train(False)
    total_mse, total_rmse, n_batches = 0.0, 0.0, 0
    for xb, yb in batch_iter(x, y, batch_size, shuffle=False):
        preds = model(xb, return_balance = False)
        y_ = yb
        if y_.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
            y_ = yb[:, None]
        mse = mse_loss(preds, y_)
        rmse = mx.sqrt(mse)

        total_mse += float(mse.item())
        total_rmse += float(rmse.item())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return total_mse / n_batches, total_rmse / n_batches

def evaluate_model(model, x, y, batch_size, task_type = "regression"):
    if task_type == "regression":
        return evaluate_regression(model,x,y,batch_size)
    elif task_type == "classification":
        return evaluate_classification(model, x, y, batch_size)
    else:
        raise ValueError("The task type is not suppored now!")


class MoEInspector:
    """
    记录每个 expert 在训练过程中的使用情况：
      - usage_counts: 被选中的 token 数 / 步数
      - prob_sums: gate 概率总和
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.usage_counts = mx.zeros((num_experts,), dtype=mx.float32)
        self.prob_sums = mx.zeros((num_experts,), dtype=mx.float32)
        self.steps = 0

    def update(self, probs: mx.array, mask: Optional[mx.array] = None):
        """
        probs: (B, E) gate 概率
        mask:  (B, E) 0/1 掩码（top-k 时可以用来统计真正激活的 expert）
        """
        probs = probs.astype(mx.float32)
        B, E = probs.shape

        # gate 概率求和
        prob_sum = mx.sum(probs, axis=0)  # (E,)
        self.prob_sums = self.prob_sums + prob_sum

        # 如果有 mask，就按 mask 统计使用次数；否则按 argmax 统计
        if mask is not None:
            mask = mask.astype(mx.float32)
            used = mx.sum(mask, axis=0)   # (E,)
        else:
            argmax_idx = mx.argmax(probs, axis=-1)  # (B,)
            used = mx.zeros((E,), dtype=mx.float32)
            for i in range(B):
                used[argmax_idx[i]] = used[argmax_idx[i]] + 1.0

        self.usage_counts = self.usage_counts + used
        self.steps += 1

    def summary(self):
        if self.steps == 0:
            return None

        usage = self.usage_counts / mx.sum(self.usage_counts)
        avg_prob = self.prob_sums / mx.sum(self.prob_sums)

        return {
            "usage_counts": self.usage_counts,
            "usage_ratio": usage,      # 每个 expert 被用的比例
            "avg_prob_ratio": avg_prob # gate 概率占比
        }

def flatten_params(obj):
    """把 model.parameters() 或 grads 展平成 list[mx.array]"""
    params = []
    if isinstance(obj, dict):
        for v in obj.values():
            params.extend(flatten_params(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            params.extend(flatten_params(v))
    elif isinstance(obj, mx.array):
        params.append(obj)
    return params

def params_l2_norm(model) -> float:
    flat = flatten_params(model.parameters())
    if not flat:
        return 0.0
    s = mx.array(0.0, dtype=mx.float32)
    for p in flat:
        p = p.astype(mx.float32)
        s = s + mx.sum(p * p)
    return float(mx.sqrt(s).item())

def params_delta_norm(model, before) -> float:
    after = flatten_params(model.parameters())
    assert len(after) == len(before), (len(after), len(before))
    s = mx.array(0.0, dtype=mx.float32)
    for a, b in zip(after, before):
        d = (a - b).astype(mx.float32)
        s = s + mx.sum(d * d)
    return float(mx.sqrt(s).item())

def snapshot_params(model):
     return [mx.array(p) for p in flatten_params(model.parameters())]

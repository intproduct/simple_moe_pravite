import mlx.core as mx
import numpy as np
from SimpleMoEs.losses import cross_entropy, accuracy

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

def evaluate_moe(model, x, y, batch_size=512):
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
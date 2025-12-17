# Gating functions or Routings for Mixture of Experts

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional, Union

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_dir = os.path.join(parent_dir, 'ADQC')

#from ADQC import ADQC, ADQC_LatentGate, VQC

def softmax_with_temperature(logits: mx.array, temperature: float, axis = -1) -> mx.array:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    type_logits = logits.dtype
    logits = (logits / mx.array(temperature)).astype(mx.float32)
    return mx.softmax(logits, axis=axis).astype(type_logits)

def log_softmax(x:mx.array, temperature:float, axis=-1)->mx.array:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    x_type = x.dtype
    x = (x / temperature).astype(mx.float32)  
    x_max = mx.max(x, axis=axis, keepdims=True) 
    x_exp = mx.exp(x - x_max)
    x_exp_sum = mx.sum(x_exp, axis=axis, keepdims=True)
    return (x - x_max - mx.log(x_exp_sum))

def topk_routing(probs:mx.array, k:int)->Union[mx.array, mx.array]:
    probs = probs.astype(mx.float32)
    idx_sorted = mx.argsort(probs, axis=-1)  #Sort index for each element in a row, the biggest element's index at the last
    topk_idx = idx_sorted[:, -k:]     #get the index of the top k elements(the last k elements in the sorted index)
    topk_idx = mx.stop_gradient(topk_idx)  #stop the gradient of mask term for proving the VJP Error.                        

    B, N = probs.shape
    mask = mx.zeros((B, N), dtype=probs.dtype)
    rows = mx.arange(B).reshape(-1, 1)
    mask[rows, topk_idx] = 1.0
    mask = mx.stop_gradient(mask)  #Like the above.

    masked = probs * mask    #Select the top k elements at their positions, other positions are 0
    norm = masked / (mx.sum(masked, axis=-1, keepdims=True) + 1e-9)

    return norm.astype(probs.dtype), mask.astype(probs.dtype)

class SoftGate(nn.Module):
    def __init__(self,data_dim:int, num_experts:int, temperature:float=1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.temperature = temperature

    def __call__(self, x:mx.array)->mx.array:
        logits = self.fc(x)
        probs = softmax_with_temperature(logits, self.temperature, axis=-1)
        return probs, None    #这里返回None是为了和topk gate的返回值保持一致，后者要返回掩码
    
class TopKGate(nn.Module):
    def __init__(self, data_dim:int, num_experts:int, k:int=2, temperature:float=1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.k = k
        self.temperature = temperature

    def __call__(self, x:mx.array)->Union[mx.array, mx.array]:
        logits = self.fc(x)
        probs = softmax_with_temperature(logits, self.temperature, axis=-1)
        topk_probs, mask = topk_routing(probs, self.k)
        return topk_probs, mask

'''#一个量子门控机制
class QuantumGate(VQC):
    def __init__(self, qubits, layers):
        super().__init__()

    def __call__(self):
        return self'''
    

def gate_factory(gate_type: str, data_dim: int, num_experts: int, **kwargs) -> nn.Module:
    gate_type = gate_type.lower() #统一转换为小写
    if gate_type == "topk":
        return TopKGate(data_dim, num_experts, k=kwargs.get("k", 2), temperature=kwargs.get("temperature", 1.0))
    if gate_type == "softmax":
        return SoftGate(data_dim, num_experts, temperature=kwargs.get("temperature", 1.0))
    if gate_type == "quantum":
        return Quantum_layer_Gate(data_dim, num_experts, 2, "tensor", 1, True)
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")


def normalize_state_general(psi: mx.array, num_qubits: int) -> mx.array:
    """
    psi: [..., 2,2,...,2]  (最后 num_qubits 个维度是 Hilbert 空间)
    在 Hilbert 空间上做归一化：对每个 token 的波函数单独归一化
    """
    full_shape = psi.shape
    hilbert_shape = full_shape[-num_qubits:]         # e.g. (2,2,2,2)
    leading_shape = full_shape[:-num_qubits]         # e.g. (B,S)

    T = 1
    for d in leading_shape:
        T *= d
    D = 1
    for d in hilbert_shape:
        D *= d

    psi_flat = mx.reshape(psi, (T, D))               # [T, D]
    norm = mx.linalg.norm(psi_flat, axis=-1, keepdims=True) + 1e-8
    psi_flat = psi_flat / norm                       # 每个 token 归一化
    psi_norm = mx.reshape(psi_flat, full_shape)
    return psi_norm


def measure_sigma_z_probs_general(
    psi: mx.array,
    num_qubits: int,
    eps: float = 1e-8,
) -> mx.array:
    """
    psi: [..., 2,2,...,2]，最后 num_qubits 个 2 是 Hilbert 维度
    物理过程：
      - 对每个 token 的 N 个 qubit 分别测量 σ_z
      - 对第 i 个 qubit 取本征值 -1 的 Born 概率 (即该位为 |1⟩ 的概率)
      - 再在 i=1..N 之间做一次归一化，得到 gate 概率

    返回: [..., num_qubits]
    """
    full_shape = psi.shape
    hilbert_shape = full_shape[-num_qubits:]         # 例如 (2,2,2,2)
    leading_shape = full_shape[:-num_qubits]         # 例如 (B,S)

    assert all(d == 2 for d in hilbert_shape), "目前只实现 dims=2 的 qubit 情况"

    # 1. Hilbert 内归一化
    psi = normalize_state_general(psi, num_qubits)   # [...,2,2,...,2]

    # 2. 把前导维合成一个大 batch T
    T = 1
    for d in leading_shape:
        T *= d

    psi_flat = mx.reshape(psi, (T,) + hilbert_shape) # [T, 2,2,...,2]

    # 3. 在计算基上求概率
    prob_full = mx.abs(psi_flat) ** 2                # [T, 2,2,...,2]

    # 4. 对每个 qubit i，算“该位为 1”的概率 p_i^{(1)} (Born 概率)
    p1_list = []
    for i in range(num_qubits):
        # 构造索引：batch T + N 个 qubit 轴
        idx = [slice(None)] * (num_qubits + 1)
        idx[i + 1] = 1       # 第 i 个 qubit 取值 1，其它两态都保留

        slice_i = prob_full[tuple(idx)]   # [T, 2,2,...] 少一维
        axes_to_sum = tuple(range(1, slice_i.ndim))
        p1_i = mx.sum(slice_i, axis=axes_to_sum)     # [T]
        p1_list.append(p1_i)

    p1 = mx.stack(p1_list, axis=-1)                  # [T, num_qubits]
    Z = mx.sum(p1, axis=-1, keepdims=True) + eps
    probs_flat = p1 / Z                              # [T, num_qubits]

    # 6. reshape 回 [..., num_qubits]
    probs = mx.reshape(probs_flat, leading_shape + (num_qubits,))
    return probs

def pauli_z():
    return mx.array([[1, 0], [0, -1]], mx.float32)

def normalize(x: Union[mx.array, List[mx.array]], form: str):
    if form == "tensor":
        return x / (mx.linalg.norm(x) + 1e-8)

def shape_init(n_qubits: int, dims: Optional[int] = None):
    if dims is not None:
        return list(dims for _ in range(n_qubits))
    else:
        return list(2 for _ in range(n_qubits))

#关于初始化
def mps_init(n_qubits: int, chi: int = 10, init_way: str = "standard", d_in: Optional[int] = None, gamma: Optional[int] = None):
    if init_way == "standard":
       return [mx.random.normal([1,2,chi])] + [mx.random.normal([chi, 2, chi]) for _ in range(n_qubits-2)] + [mx.random.normal([chi, 2, 1])]
    elif init_way == "kaiming":
        return [mx.random.normal([1,2,chi], mx.float32, mx.array([0]), mx.array([2 / d_in]))] + \
               [mx.random.normal([chi, 2, chi], mx.float32, mx.array([0]), mx.array([2 / d_in])) for _ in range(n_qubits-2)] + \
               [mx.random.normal([chi, 2, 1], mx.float32, mx.array([0]), mx.array([2 / d_in]))]
    elif init_way == "gamma":
        return [mx.random.normal([1,2,chi], mx.float32, mx.array([0]), mx.power(mx.array([(1 / d_in)])))] + \
               [mx.random.normal([chi, 2, chi], mx.float32, mx.array([0]), mx.power(mx.array([(1 / d_in)]))) for _ in range(n_qubits-2)] + \
               [mx.random.normal([chi, 2, 1], mx.float32, mx.array([0]), mx.power(mx.array([(1 / d_in)])))]

def mpo_init(n_qubits: int, chi: int = 10, init_way: str = "standard", din: int = 5, gamma: Optional[int] = None):
    if init_way == "strandard":
        #顺时针定义mpo的bond顺序，从横置最左第一个virtual bond开始，din在下，dout在上，下同
        return [mx.random.normal([1, 2, chi, din])] + [mx.random.normal([chi, 2, chi, din]) for _ in range(n_qubits-2)] + [mx.random.normal([chi, 2, 1, din])]
    elif init_way == "kaiming":
        d_in = din ** n_qubits
        return [mx.random.normal([1, 2, chi, din], None, mx.array([0]), mx.array([2 / d_in]))] + \
               [mx.random.normal([chi, 2, chi, din], None, mx.array([0]), mx.array([2 / d_in])) for _ in range(n_qubits-2)] + \
               [mx.random.normal([chi, 2, 1, din], None, mx.array([0]), mx.array([2 / d_in]))]
    elif init_way == "gamma":
        d_in = din ** n_qubits
        return [mx.random.normal([1, 2, chi, din], None, mx.array([0]), mx.power(mx.array([(1 / d_in)])))] + \
               [mx.random.normal([chi, 2, chi, din], None, mx.array([0]), mx.power(mx.array([(1 / d_in)]))) for _ in range(n_qubits-2)] + \
               [mx.random.normal([chi, 2, 1, din], None, mx.array([0]), mx.power(mx.array([(1 / d_in)])))]
    
def get_quantum_layer(din: int, num_exp: int, method: str, init_way: str = "standard", dims: Optional[int] = None, gamma: Optional[int]=None, chi: Optional[int]=None):
    if method == 'tensor':
        if dims is not None:
           indexs = shape_init(dims * num_exp)
        else:
            indexs = shape_init(2 * num_exp)
        if init_way == "standard":
           return mx.random.normal(indexs)
        elif init_way == "kaiming":
            return mx.random.normal(indexs, None, mx.array([0]), mx.array([2/din]))
        elif init_way == "gamma":
            return mx.random.normal(indexs, None, mx.array([0]), mx.power(mx.array([1/din]), gamma))
    if method == "mpo":
        return mpo_init(num_exp, chi, init_way, din, gamma)
    
'''def measure_ope(name:Optional[str]=None, dims: Optional[int] = None):
    if name is None:
        if dims is not None:
            return spin_ops(dims)['Jz'].astype(mx.float32)
        return pauli_ops('z').astype(mx.float32)'''
    
def evolution(tensor: mx.array, evo_ope: mx.array) -> mx.array:
    if isinstance(tensor, List):
        shape_t = tensor.shape
        shape_o = evo_ope.shape

def measurement(tensor: mx.array, ope: mx.array) -> mx.array:
    shape_t = tensor.shape
    shape_o = ope.shape
    if len(shape_t) == len(shape_o) and shape_t[0] == shape_o[0]:
        return None

'''class Quantum_LatentGate(ADQC_LatentGate):
    def __init__(self, dims_in: int, dims_out: int, 
                 num_experts: int, if_tensor: bool,
                 chi: int):
        super().__init__()
        self.params = dims_out
        self.n_qubits = num_experts
        self.down = nn.Linear(dims_in, dims_out)
        if if_tensor:
            self.psi_init = mx.random.normal(shape_init(num_experts), mx.float32)
        else:
            self.psi_init = mps_init(num_experts, chi)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.down(x).reshape(-1, self.params)'''

class Quantum_layer_Gate(nn.Module):
    def __init__(self, d_in: int, num_experts: int, dims: Optional[int] = None, form: str = "tensor", k:int =1, sample: bool = True):
        super().__init__()
        self.form = form
        self.dims = dims
        self.num_experts = num_experts
        self.k = k
        self.sample = sample
        if dims is not None:
            self.project = nn.Linear(d_in, dims ** num_experts) #这里是数据降维map
            self.gate = get_quantum_layer(dims ** num_experts, num_experts, self.form, "standard", dims, None, None)
        else:
            self.project = nn.Linear(d_in, 2 ** num_experts)
            self.gate = get_quantum_layer(2 ** num_experts, num_experts, self.form, "standard", None, None, None)
        
        #self.measure = measure_ope()

    def __call__(self, x: mx.array) -> mx.array:
        psi = self.project(x)  #[B,S,d_in] -> [B,S, 2 ** num_experts]
        if self.form == "tensor":
            '''
            shape_ = psi.shape[-1]
            H = self.gate.reshape(shape_, shape_)
            U,S,Vh = mx.linalg.svd(H)   #⚠️在MLX中，svd操作只能在CPU上使用，不支持CUDA和Apple sclion，因此这里暂时不被使用，仅表示和torch对照作用
            H_ = U @ Vh
            psi = (psi @ H_).reshape(shape_, -1)
            '''
            psi = psi @ self.gate.reshape(psi.shape[-1], -1) #[B,S, 2** num_experts] -> [B,S,2 ** num_experts]
            if self.dims != 2:
                raise NotImplementedError("目前只实现 dims=2 的 Pauli-Z 测量")

            # 把最后一个维度 2**N reshape 成 N 个 2：[..., 2,2,...,2]
            full_shape = psi.shape          # e.g. (B,S,16)
            leading_shape = full_shape[:-1] # (B,S)
            psi_qubits = mx.reshape(
                psi, leading_shape + (2,) * self.num_experts
            )                               # (B,S,2,2,2,2)

            # 在 Hilbert 维度上做 σ_z 测量 + 归一化
            probs = measure_sigma_z_probs_general(
                psi_qubits,
                num_qubits=self.num_experts,
            )                               # (B,S,num_experts)

            if self.sample == False:
               return probs, None
            else:
                B, E = probs.shape
                k = min(self.k, E)

                # Gumbel-Max trick: 采样近似 categorical(probs)
                U = mx.random.uniform(shape=probs.shape)      # (B, E)
                g = -mx.log(-mx.log(U + 1e-8) + 1e-8)         # Gumbel noise
                logits = mx.log(probs + 1e-8) + g             # (B, E)

                idx_sorted = mx.argsort(logits, axis=-1)
                topk_idx = idx_sorted[:, -k:]
                topk_idx = mx.stop_gradient(topk_idx)

                mask = mx.zeros((B, E), dtype=probs.dtype)
                rows = mx.arange(B).reshape(-1, 1)
                mask[rows, topk_idx] = 1.0
                mask = mx.stop_gradient(mask)

                masked = probs * mask
                norm = masked / (mx.sum(masked, axis=-1, keepdims=True) + 1e-9)

                return norm.astype(probs.dtype), mask.astype(probs.dtype)

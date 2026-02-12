# KDA/GDN Context Parallelism (CP) 实现分析

> 文档生成时间: 2026-02-12

## 目录

1. [概述](#概述)
2. [第一性原理：从算法公式到 CP 推导](#第一性原理从算法公式到-cp-推导)
   - [KDA/GDN 的核心递归公式](#kdagdn-的核心递归公式)
   - [Chunk 并行化分解](#chunk-并行化分解)
   - [跨 Rank 状态传递的数学推导](#跨-rank-状态传递的数学推导)
   - [反向传播的 CP 推导](#反向传播的-cp-推导)
   - [通信复杂度分析](#通信复杂度分析)
3. [FLA 仓库中的 CP 实现](#fla-仓库中的-cp-实现)
   - [核心数据结构](#核心数据结构)
   - [通信原语](#通信原语)
   - [前向传播 CP 实现](#前向传播-cp-实现)
   - [后向传播 CP 实现](#后向传播-cp-实现)
4. [KDA 与 GDN 的 CP 差异](#kda-与-gdn-的-cp-差异)
5. [性能优化](#性能优化)
6. [优化方向建议](#优化方向建议)
7. [使用示例](#使用示例)

---

## 概述

Context Parallelism (CP) 是一种针对长序列训练的分布式策略，通过将序列维度切分到多个 GPU 上来突破单卡显存限制。FLA 项目中的 KDA (Kimi Delta Attention) 和 GDN (Gated Delta Net) 算法都实现了完整的 CP 支持。

### 什么是 KDA 和 GDN？

**GDN (Gated Delta Net)** 和 **KDA (Kimi Delta Attention)** 都属于 **Linear Attention with Delta Rule** 家族，其核心思想是：
- 使用 **递归状态矩阵** `S ∈ ℝ^{K×V}` 替代标准 Attention 的 KV Cache
- 通过 **Delta Rule** 进行增量更新，实现 O(1) 的单步推理复杂度
- 使用 **遗忘门 (Gating)** 控制历史信息的衰减

### CP 的核心挑战

```
全局序列: [Token_0, Token_1, ..., Token_{T-1}]
                    ↓ 按序列维度切分到 N 个 Rank
Rank 0: [Token_0, ..., Token_{L-1}]           → 计算局部状态 S_L
Rank 1: [Token_L, ..., Token_{2L-1}]          → 需要 S_L 作为初始状态
...
Rank N-1: [Token_{(N-1)L}, ..., Token_{T-1}]  → 需要 S_{(N-1)L} 作为初始状态

其中 L = T / N 是每个 Rank 负责的 token 数量
```

**关键挑战：**
1. **跨 Rank 状态依赖**: 递归状态 `S` 需要从前一个 Rank 传递到后一个 Rank
2. **通信与计算重叠**: 如何最小化通信开销
3. **反向传播**: 梯度 `dS` 需要从后向前传递
4. **显存优化**: 需要压缩跨 Rank 传递的状态以节省显存

---

## 第一性原理：从算法公式到 CP 推导

### KDA/GDN 的核心递归公式

#### 单步递归形式

KDA 和 GDN 的核心是 **Gated Delta Rule**，其单步递归公式为：

```
输入: q_t, k_t, v_t ∈ ℝ^d,  β_t ∈ ℝ,  g_t ∈ ℝ (或 ℝ^d)
状态: S_t ∈ ℝ^{K×V}

递归更新:
┌─────────────────────────────────────────────────────────────────┐
│  S_t = γ_t ⊙ S_{t-1} + β_t · k_t ⊗ (v_t - S_{t-1}^⊤ k_t)      │  (状态更新)
│  o_t = S_t^⊤ q_t                                                │  (输出计算)
└─────────────────────────────────────────────────────────────────┘

其中:
- γ_t = exp(g_t): 遗忘门 (decay factor)
- ⊗: 外积 (outer product)
- ⊙: 逐元素乘法 (对于 scalar gate) 或矩阵-向量广播乘法 (对于 vector gate)
```

> **KDA vs GDN 的区别**: 
> - KDA 使用 **per-head-dim gate** `g_t ∈ ℝ^K`，即 `γ_t` 是向量
> - GDN 使用 **per-head gate** `g_t ∈ ℝ`，即 `γ_t` 是标量
> - KDA 还有额外的 gate 激活函数：`g = -exp(A_log) · softplus(g + dt_bias)`

#### 展开递归

将递归展开，状态 `S_T` 可以表示为所有历史 token 的加权和：

```
S_T = Σ_{t=1}^{T} [ (Π_{τ=t+1}^{T} γ_τ) · β_t · k_t ⊗ ṽ_t ]

其中 ṽ_t = v_t - S_{t-1}^⊤ k_t 是 "delta" 项
```

这个公式揭示了 CP 的核心难点：**每个位置的输出都依赖于之前所有位置的状态**。

### Chunk 并行化分解

为了实现并行计算，我们将序列分成大小为 `C` 的 **chunks**：

```
序列长度 T，分成 T/C 个 chunks
Chunk i: 包含 tokens [i·C, (i+1)·C)
```

#### Chunk 内的状态转移

对于 Chunk i，定义：
- `S_i^{start}`: chunk 开始时的状态 (即上一个 chunk 的结束状态)
- `S_i^{end}`: chunk 结束时的状态

我们可以将 chunk 内的状态转移表示为 **仿射变换**：

```
┌─────────────────────────────────────────────────────────────────┐
│  S_i^{end} = M_i · S_i^{start} + H_i                            │
└─────────────────────────────────────────────────────────────────┘

其中:
- M_i ∈ ℝ^{K×K}: 状态转移矩阵 (Transition Matrix)
- H_i ∈ ℝ^{K×V}: 累积状态增量 (Accumulated Delta)
```

#### 推导 M_i 和 H_i

**Step 1: 单步转移的矩阵形式**

将单步更新重写为矩阵形式：
```
S_t = γ_t · S_{t-1} + β_t · k_t ⊗ (v_t - S_{t-1}^⊤ k_t)
    = γ_t · S_{t-1} + β_t · k_t ⊗ v_t - β_t · k_t ⊗ (S_{t-1}^⊤ k_t)
    = γ_t · S_{t-1} + β_t · k_t · v_t^⊤ - β_t · k_t · k_t^⊤ · S_{t-1}
    = (γ_t · I - β_t · k_t · k_t^⊤) · S_{t-1} + β_t · k_t · v_t^⊤
```

定义单步转移矩阵和增量：
```
A_t = γ_t · I - β_t · k_t · k_t^⊤  ∈ ℝ^{K×K}  (单步转移矩阵)
u_t = β_t · k_t · v_t^⊤            ∈ ℝ^{K×V}  (单步增量)
```

则：`S_t = A_t · S_{t-1} + u_t`

**Step 2: Chunk 内的累积转移**

对于 chunk i 内的 tokens `{t: t ∈ [i·C, (i+1)·C)}`，依次应用转移：

```
S_{i·C+1} = A_{i·C+1} · S_{i·C} + u_{i·C+1}
S_{i·C+2} = A_{i·C+2} · S_{i·C+1} + u_{i·C+2}
         = A_{i·C+2} · (A_{i·C+1} · S_{i·C} + u_{i·C+1}) + u_{i·C+2}
         = A_{i·C+2} · A_{i·C+1} · S_{i·C} + A_{i·C+2} · u_{i·C+1} + u_{i·C+2}
...
```

归纳得到：
```
┌─────────────────────────────────────────────────────────────────┐
│  M_i = Π_{t=(i+1)·C}^{i·C+1} A_t = A_{(i+1)·C} · ... · A_{i·C+1}│
│                                                                 │
│  H_i = Σ_{t=i·C+1}^{(i+1)·C} (Π_{τ=t+1}^{(i+1)·C} A_τ) · u_t   │
└─────────────────────────────────────────────────────────────────┘
```

**关键观察**: `M_i` 和 `H_i` 只依赖于 chunk i 内部的数据，可以**独立并行计算**！

#### 简化形式 (WY 表示)

在实际实现中，使用 **WY 表示** 来高效计算：

```
定义:
- w_t = A_{kk} @ (k_t * β_t)  ∈ ℝ^K  (预计算的 "key-like" 向量)
- u_t = A_{kk} @ (v_t * β_t)  ∈ ℝ^V  (预计算的 "value-like" 向量)
- k̃_t = k_t * γ_t             ∈ ℝ^K  (衰减后的 key)

其中 A_{kk} 是 chunk 内的下三角矩阵 (intra-chunk attention)
```

则状态更新简化为：
```
S_t = γ_t · S_{t-1} + k̃_t ⊗ (u_t - w_t^⊤ · S_{t-1})
```

### 跨 Rank 状态传递的数学推导

现在考虑 **CP 场景**：将 `T/C` 个 chunks 分配到 `N` 个 Ranks 上。

#### 设定

```
- 每个 Rank 处理 T/(N·C) 个 chunks
- Rank r 处理 chunks: {i: i ∈ [r·T/(N·C), (r+1)·T/(N·C))}
- 定义 Rank r 的本地状态:
  - S_r^{in}: Rank r 的输入状态 (从 Rank r-1 接收)
  - S_r^{out}: Rank r 的输出状态 (发送给 Rank r+1)
```

#### 前向传播：链式状态传递

**朴素方案 (Sequential)**:
```
Rank 0: S_0^{out} = f_0(S_0^{in}=0)      # 初始状态为 0
        ↓ send S_0^{out}
Rank 1: S_1^{out} = f_1(S_1^{in}=S_0^{out})
        ↓ send S_1^{out}
...
Rank N-1: S_{N-1}^{out} = f_{N-1}(S_{N-1}^{in}=S_{N-2}^{out})
```

这种方案的问题：**串行通信，无法并行！**

#### 并行化方案：(M, H) 对的 All-Gather

**核心思想**: 每个 Rank 可以**独立**计算其本地的 `(M_r, H_r)` 对，然后通过 **all-gather** 通信来合并。

**Step 1: 每个 Rank 独立计算本地 (M_r, H_r)**

Rank r 处理的所有 chunks 可以合并为一个大的 chunk：
```
M_r = 本地所有 M_i 的连乘 = M_{last_chunk} · ... · M_{first_chunk}
H_r = 本地所有 H_i 的累积 (考虑后续 M 的衰减)
```

即：
```
S_r^{out} = M_r · S_r^{in} + H_r
```

**Step 2: All-Gather 收集所有 (M, H) 对**

```
all_gather({(M_0, H_0), (M_1, H_1), ..., (M_{N-1}, H_{N-1})})
```

**Step 3: 合并计算初始状态**

对于 Rank r，其初始状态由前 r 个 Rank 的 (M, H) 对决定：

```
S_r^{in} = M_{r-1} · (M_{r-2} · (... · (M_0 · 0 + H_0) + H_1) ...) + H_{r-1}
```

迭代计算：
```python
def compute_initial_state(rank, all_M, all_H):
    S = 0  # S_0^{in} = 0
    for i in range(rank):
        S = all_M[i] @ S + all_H[i]
    return S
```

#### 通信模式总结

```
┌─────────────────────────────────────────────────────────────────┐
│                     Forward Pass CP                             │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: Local Compute (各 Rank 并行)                           │
│   Rank 0: compute (M_0, H_0)                                    │
│   Rank 1: compute (M_1, H_1)                                    │
│   ...                                                           │
│   Rank N-1: skip (last rank 不需要发送状态)                     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: All-Gather (同步点)                                    │
│   all_gather([M_0, H_0], [M_1, H_1], ..., [M_{N-1}, H_{N-1}])   │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Merge (各 Rank 独立计算，但工作量不同)                 │
│   Rank 0: S_0^{in} = 0 (无需计算)                               │
│   Rank 1: S_1^{in} = H_0 (1 次迭代)                             │
│   Rank 2: S_2^{in} = M_1 @ H_0 + H_1 (2 次迭代)                 │
│   ...                                                           │
│   Rank r: 独立迭代 r 次，存在冗余计算                           │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: Local Forward (各 Rank 并行)                           │
│   各 Rank 使用 S_r^{in} 作为初始状态，计算本地输出              │
└─────────────────────────────────────────────────────────────────┘
```

### 反向传播的 CP 推导

#### 梯度传播的依赖关系

前向传播中：`S_r^{in}` 依赖于 `S_0, S_1, ..., S_{r-1}`

反向传播中：`dS_{r-1}^{out}` 依赖于 `dS_r^{in}, dS_{r+1}^{in}, ..., dS_{N-1}^{in}`

**方向相反！**

#### 反向传播公式推导

对于状态转移 `S^{out} = M · S^{in} + H`，反向传播：

```
给定 dL/dS^{out}，计算 dL/dS^{in}

由链式法则:
dL/dS^{in} = (∂S^{out}/∂S^{in})^⊤ · dL/dS^{out}
           = M^⊤ · dL/dS^{out}
```

#### 反向传播的 CP 方案

类似前向，每个 Rank 计算本地的 `(M_r^⊤, dH_r)` 对：

```
dS_r^{in} = M_r^⊤ · dS_r^{out} + dH_r

其中 dH_r 是本地梯度累积
```

**通信模式 (与前向对称)**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Backward Pass CP                            │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: Local Compute (各 Rank 并行)                           │
│   Rank 0: skip (first rank 不需要发送梯度)                      │
│   Rank 1: compute (M_1^⊤, dH_1)                                 │
│   ...                                                           │
│   Rank N-1: compute (M_{N-1}^⊤, dH_{N-1})                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: All-Gather (同步点)                                    │
│   all_gather([(M_0^⊤, dH_0), ..., (M_{N-1}^⊤, dH_{N-1})])       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Merge (各 Rank 独立计算，方向相反)                     │
│   Rank N-1: dS_{N-1}^{out} = 0 (末端无后续梯度)                 │
│   Rank N-2: dS_{N-2}^{out} = dH_{N-1} (1 次迭代)                │
│   ...                                                           │
│   Rank r: 独立迭代 (N-1-r) 次，存在冗余计算                     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: Local Backward (各 Rank 并行)                          │
│   各 Rank 使用 dS_r^{out} 计算本地参数梯度                      │
└─────────────────────────────────────────────────────────────────┘
```

### 通信复杂度分析

#### 通信量

每个 Rank 需要 all-gather 的数据量：
```
M_r: K × K × sizeof(float32) = K² × 4 bytes
H_r: K × V × sizeof(float32) = KV × 4 bytes

总通信量 per Rank: N × (K² + KV) × 4 bytes
```

对于典型配置 (K=128, V=256, N=8):
```
= 8 × (128² + 128×256) × 4
= 8 × (16384 + 32768) × 4
= 8 × 49152 × 4
= 1.57 MB per rank
```

#### 与 Tensor Parallelism 的对比

| 维度 | Context Parallelism | Tensor Parallelism |
|------|--------------------|--------------------|
| 切分维度 | 序列 (T) | 隐藏维度 (H) |
| 通信量 | O(N × K² × H_heads) | O(T × H_hidden) per layer |
| 通信次数 | 2 次 (fwd + bwd) | 2 次 per FFN + 2 次 per Attn |
| 适用场景 | 超长序列 | 超大模型 |

#### 计算/通信比

```
计算量 (per Rank): O(T/N × K × V × H)
通信量 (per Rank): O(N × K² × H)

计算/通信比 = O(T × V / (N² × K))
```

当 `T` 足够大时，计算/通信比很高，CP 效率好。

---

## FLA 仓库中的 CP 实现

基于上述数学推导，我们现在分析 FLA 仓库中的具体实现。

### 架构概览

```
fla/ops/cp/
├── __init__.py          # 导出公共接口
├── context.py           # FLACPContext 数据结构和 build_cp_context
├── comm.py              # 通信原语 (all_gather, send_recv)
└── chunk_delta_h.py     # 核心 CP Triton kernels (1176 行)
    ├── pre_process_fwd_kernel_merged    # 计算 (M, H) 对
    ├── pre_process_bwd_kernel_merged    # 计算 (M^T, dH) 对
    └── merge_fwd_bwd_kernel             # 合并状态
```

---

## 核心数据结构

### FLACPContext

```python
# 文件: fla/ops/cp/context.py

@dataclass
class FLACPContext:
    """FLA Context Parallel Context - Operator-level context management."""
    
    # 进程组
    group: ProcessGroup | None = None
    
    # 当前 Rank 的局部 cu_seqlens (GPU tensor, int32)
    cu_seqlens: torch.Tensor | None = None
    
    # CPU 版本的 cu_seqlens (用于避免 D2H 同步)
    cu_seqlens_cpu: torch.Tensor | None = None
    
    # 是否是序列链的最后一个 Rank
    is_last_rank: bool | None = None
    
    # 当前 Rank 之前有多少个 Rank 处理同一序列
    pre_num_ranks: int | None = None
    
    # 是否是序列链的第一个 Rank
    is_first_rank: bool | None = None
    
    # 当前 Rank 之后有多少个 Rank 处理同一序列
    post_num_ranks: int | None = None
    
    # Conv1d 相关参数
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None
```

### 关键属性说明

| 属性 | 说明 | 用途 |
|------|------|------|
| `is_first_rank` | 当前 Rank 是否是某序列的起始 Rank | 决定是否需要接收前序状态 |
| `is_last_rank` | 当前 Rank 是否是某序列的结束 Rank | 决定是否需要发送后续状态 |
| `pre_num_ranks` | 前序 Rank 数量 | 用于 all_gather 后的状态合并 |
| `post_num_ranks` | 后续 Rank 数量 | 用于反向传播时的梯度合并 |

### build_cp_context 函数

```python
def build_cp_context(
    cu_seqlens: torch.Tensor,      # 全局 cu_seqlens (切分前)
    group: ProcessGroup,            # CP 进程组
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
```

此函数的核心逻辑:

1. **计算当前 Rank 负责的 token 范围**: `[rank_start, rank_end)`
2. **使用 searchsorted 快速定位**: 找到与当前范围重叠的序列
3. **计算局部 cu_seqlens**: 将全局坐标映射到局部坐标
4. **确定 Rank 角色**: 计算 `is_first_rank`, `is_last_rank`, `pre_num_ranks`, `post_num_ranks`

```python
# 核心计算逻辑
part_len = total_tokens // world_size
rank_start = part_len * rank
rank_end = rank_start + part_len

# 使用 searchsorted 快速定位
start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side='right')
end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side='left')

# 计算局部 cu_seqlens
local_cu_seqlens = (subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start)
```

---

## 通信原语

### 文件: `fla/ops/cp/comm.py`

#### 1. all_gather_into_tensor

```python
def all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False
) -> tuple[torch.Tensor, dist.Work | None]:
    """
    将所有 Rank 的 tensor 聚合到一起。
    输出 shape: [world_size, *inp.shape]
    """
```

#### 2. send_recv_fwd / send_recv_bwd

```python
def send_recv_fwd(send_tensor, group, recv_from_prev=True):
    """
    前向传播通信: 发送到下一个 Rank, 从上一个 Rank 接收。
    使用 all_gather 实现以确保所有 Rank 参与。
    """
    gathered, _ = all_gather_into_tensor(send_tensor, group=group)
    if recv_from_prev:
        if rank == 0:
            return torch.zeros_like(send_tensor)  # 第一个 Rank 无前序
        return gathered[rank - 1].clone()

def send_recv_bwd(send_tensor, group, recv_from_next=True):
    """
    后向传播通信: 梯度从后向前传递。
    """
    gathered, _ = all_gather_into_tensor(send_tensor, group=group)
    if recv_from_next:
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)  # 最后一个 Rank 无后续
        return gathered[rank + 1].clone()
```

---

## 前向传播 CP 实现：`chunk_gated_delta_rule_fwd_h_pre_process`

### 核心文件

- `fla/ops/cp/chunk_delta_h.py`: CP 核心 Triton kernels
- `fla/ops/kda/chunk_fwd.py`: KDA 前向调用入口

### 函数作用

`chunk_gated_delta_rule_fwd_h_pre_process` 是 **CP 与非 CP 模式的核心差异点**。它负责：
1. 计算当前 Rank 的本地 `(H_local, M_local)` 对
2. 通过 `all_gather` 收集所有 Rank 的数据
3. 合并前序 Rank 的状态，得到当前 Rank 的 `initial_state`

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              chunk_gated_delta_rule_fwd_h_pre_process 流程                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 分配缓冲区                                                          │
│          hm = zeros([H, K, V+K])   # packed buffer for (H_local, M_local)   │
│          initial_state = zeros([N, H, K, V])                                │
│                                                                             │
│  Step 2: 计算本地 (H_local, M_local)                                         │
│          if not is_last_rank:                                               │
│              pre_process_fwd_kernel_merged(k, w, u, g, hm)                  │
│              # hm[:, :, :V] = H_local (累积状态增量)                         │
│              # hm[:, :, V:] = M_local (状态转移矩阵)                         │
│                                                                             │
│  Step 3: All-Gather 通信                                                     │
│          ag_hm = all_gather(hm)  # shape: [world_size, H, K, V+K]           │
│          ↑ 同步点：所有 Rank 等待通信完成                                    │
│                                                                             │
│  Step 4: 合并计算初始状态 (关键!)                                            │
│          if not is_first_rank:                                              │
│              merge_fwd_bwd_kernel(initial_state[0], ag_hm, pre_num_ranks)   │
│              # 迭代合并前 pre_num_ranks 个 Rank 的 (H, M)                    │
│                                                                             │
│  Return: initial_state                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 2 详解：`pre_process_fwd_kernel_merged`

此 kernel 计算当前 Rank 对全局状态的贡献 `(H_local, M_local)`：

**数学定义：**
```
给定 Rank 本地的 NT 个 chunks，假设初始状态为 0:
S_final = M_local · 0 + H_local = H_local

其中:
- H_local ∈ [K, V]: 累积状态增量 = Σ (decay × k ⊗ v_delta)
- M_local ∈ [K, K]: 状态转移矩阵 = Π (diag(decay) - k^T @ w)
```

**Kernel 伪代码：**
```python
# Part A: 计算 H_local (处理 hm[:, :, :V] 部分)
b_h = zeros([K, V])
for i_t in range(NT):
    v_delta = u[i_t] - w[i_t] @ b_h    # Delta rule
    decay = exp(g[i_t_last])
    b_h = b_h * decay + k[i_t]^T @ v_delta
# 最终 b_h 即为 H_local

# Part B: 计算 M_local (处理 hm[:, :, V:] 部分)
b_m = I  # 单位矩阵
for i_t in range(NT):
    decay = exp(g[i_t_last])
    A_t = diag(decay) - k[i_t]^T @ w[i_t]
    b_m = A_t @ b_m
# 最终 b_m 即为 M_local
```

**关键点：**
- 只有 **非 last_rank** 需要计算，因为 last_rank 不需要向后续 Rank 传递状态
- H 和 M 的计算被融合到同一个 kernel，通过 `i_col` 区分处理哪部分

### Step 4 详解：`merge_fwd_bwd_kernel`

此 kernel 根据 `ag_hm` 中前序 Rank 的数据，计算当前 Rank 的 `initial_state`。

**数学公式：**
```
S_r^{in} = M_{r-1} @ (M_{r-2} @ (... @ (M_0 @ 0 + H_0) ...) + H_{r-1})
```

**迭代形式：**
```python
b_h = 0
for idx in range(pre_num_ranks):
    cur_rank = rank - pre_num_ranks + idx  # 从最远的前序 Rank 开始
    H_cur = ag_hm[cur_rank, :, :, :V]
    M_cur = ag_hm[cur_rank, :, :, V:]
    b_h = M_cur @ b_h + H_cur
# 最终 b_h 即为 initial_state
```

**示例 (rank=3, pre_num_ranks=3)：**
```
idx=0: cur_rank=0, b_h = M_0 @ 0 + H_0 = H_0
idx=1: cur_rank=1, b_h = M_1 @ H_0 + H_1
idx=2: cur_rank=2, b_h = M_2 @ (M_1 @ H_0 + H_1) + H_2
```

### 关于"并行性"的说明

**重要澄清：** Merge 阶段**不是真正的并行计算**，而是**各 Rank 独立计算但工作量不对称**。

```
时间线 (Merge 阶段):

Rank 0: [无需 merge, h=0]
Rank 1: [1 次矩阵乘] ─────────────────────────────┐
Rank 2: [2 次矩阵乘] ──────────────────┐          │
Rank 3: [3 次矩阵乘] ─────────┐        │          │ 各 Rank 独立计算
...                           │        │          │ 但工作量递增
Rank r: [r 次矩阵乘]          ▼        ▼          ▼
```

**特点：**
1. **独立性**：每个 Rank 独立计算，不需要等待其他 Rank
2. **冗余计算**：Rank 2 和 Rank 3 都会计算 `M_1 @ H_0 + H_1`
3. **负载不均衡**：Rank r 需要 O(r) 次矩阵乘法

**这种设计的 tradeoff：**
- ✅ 消除了串行依赖（不需要等待前序 Rank 发送 S）
- ❌ 存在冗余计算（每个 Rank 重复计算前缀）
- ❌ 负载不均衡（后序 Rank 工作量更大）

### 与非 CP 版本的对比

| 方面 | 非 CP 版本 | CP 版本 |
|------|-----------|---------|
| `initial_state` 来源 | 用户提供或默认 0 | 通过 `pre_process` + 通信计算 |
| 额外 kernel | 无 | `pre_process_fwd_kernel_merged`, `merge_fwd_bwd_kernel` |
| 通信 | 无 | 1 次 All-Gather (per head) |
| 计算开销 | 无 | Rank r: O(r × K² × K) 矩阵乘 |
| 内存开销 | 无 | `ag_hm`: world_size × H × K × (V+K) × 4 bytes |

### 时序图

```
时间 →
                     ┌─ intra_chunk ─┐ ┌─ pre_process ─┐ ┌ all_gather ┐ ┌─ merge ─┐ ┌─ fwd_h ─┐

Rank 0: ─────────────[chunk_kda_intra]─[compute (H,M)]──[  ag_hm    ]──[  skip  ]──[fwd_h]───▶
Rank 1: ─────────────[chunk_kda_intra]─[compute (H,M)]──[  ag_hm    ]──[1次 mm ]──[fwd_h]───▶
Rank 2: ─────────────[chunk_kda_intra]─[compute (H,M)]──[  ag_hm    ]──[2次 mm ]──[fwd_h]───▶
Rank 3: ─────────────[chunk_kda_intra]─[compute (H,M)]──[  ag_hm    ]──[3次 mm ]──[fwd_h]───▶
                                                             ↑
                                                        同步点 (barrier)
```

---

## 后向传播 CP 实现

### 核心文件

- `fla/ops/cp/chunk_delta_h.py`: CP 核心 Triton kernels
- `fla/ops/kda/chunk_bwd.py`: KDA 后向调用入口

### 函数作用

`chunk_gated_delta_rule_bwd_dhu_pre_process` 是 **CP 与非 CP 模式的核心差异点**。它负责：
1. 计算当前 Rank 的本地 `(dh, M^T)` 对
2. 通过 `all_gather` 收集所有 Rank 的数据
3. 合并后续 Rank 的梯度，得到当前 Rank 的 `dS_r^{out}`

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              chunk_gated_delta_rule_bwd_dhu_pre_process 流程                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 分配缓冲区                                                          │
│          dhm = zeros([H, K, V+K])   # packed buffer for (dh, M)            │
│          dS_r_out = zeros([N, H, K, V])                                    │
│                                                                             │
│  Step 2: 计算本地 (dh, M^T)                                                  │
│          if not is_first_rank:                                              │
│              pre_process_bwd_kernel_merged(q, k, w, do, dv, dhm)            │
│              # dhm[:, :, :V] = dH (梯度)                                    │
│              # dhm[:, :, V:] = M^T (转置后的状态转移矩阵)                   │
│                                                                             │
│  Step 3: All-Gather 通信                                                     │
│          ag_dhm = all_gather(dhm)  # shape: [world_size, H, K, V+K]         │
│          ↑ 同步点：所有 Rank 等待通信完成                                    │
│                                                                             │
│  Step 4: 合并计算梯度 (关键!)                                                │
│          if not is_last_rank:                                              │
│              merge_fwd_bwd_kernel(dS_r_out, ag_dhm, post_num_ranks, rank)   │
│              # 迭代合并后 post_num_ranks 个 Rank 的 (dH, M)                  │
│                                                                             │
│  Return: dS_r_out                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 2 详解：`pre_process_bwd_kernel_merged`

此 kernel 计算当前 Rank 对全局梯度的贡献 `(dH, M^T)`：

**数学定义：**
```
给定 Rank 本地的 NT 个 chunks，假设后续状态为 0:
dS_r^{out} = M_{r} @ (dS_{r+1}^{in} = 0) + dH_r = dH_r

其中:
- dH_r ∈ [K, V]: 局部梯度 = Σ (decay × k ⊗ v_delta)
- M_r ∈ [K, K]: 状态转移矩阵 = Π (diag(decay) - k^T @ w)
```

**Kernel 伪代码：**
```python
# Part A: 计算 dH (处理 dhm[:, :, :V] 部分)
b_dh = zeros([K, V])
for i_t in range(NT):
    v_delta = dv[i_t] - w[i_t] @ b_dh    # Delta rule
    decay = exp(g[i_t])
    b_dh = b_dh * decay + k[i_t]^T @ v_delta
# 最终 b_dh 即为 dH

# Part B: 计算 M^T (处理 dhm[:, :, V:] 部分)
b_m = I  # 单位矩阵
for i_t in range(NT):
    decay = exp(g[i_t])
    A_t = diag(decay) - k[i_t]^T @ w[i_t]
    b_m = A_t @ b_m
# 最终 b_m 即为 M^T
```

**关键点：**
- 只有 **非 first_rank** 需要计算，因为 first_rank 不需要接收前序 Rank 的梯度
- dH 和 M 的计算被融合到同一个 kernel，通过 `i_col` 区分处理哪部分

### Step 4 详解：`merge_fwd_bwd_kernel`

此 kernel 根据 `ag_dhm` 中后续 Rank 的数据，计算当前 Rank 的 `dS_r^{out}`。

**数学公式：**
```
dS_r^{out} = M_r @ dS_{r+1}^{out} + dH_r
```

**迭代形式：**
```python
b_dS = 0
for idx in range(post_num_ranks):
    cur_rank = rank + idx + 1  # 从最近的后续 Rank 开始
    dH_cur = ag_dhm[cur_rank, :, :, :V]
    M_cur = ag_dhm[cur_rank, :, :, V:]
    b_dS = M_cur @ b_dS + dH_cur
# 最终 b_dS 即为 dS_r^{out}
```

**示例 (rank=1, post_num_ranks=2)：**
```
idx=0: cur_rank=2, b_dS = M_2 @ 0 + dH_2 = dH_2
idx=1: cur_rank=3, b_dS = M_3 @ dH_2 + dH_3
```

### 关于"并行性"的说明

**重要澄清：** Merge 阶段**不是真正的并行计算**，而是**各 Rank 独立计算但工作量不对称**。

```
时间线 (Merge 阶段):

Rank 0: [无需 merge, dS=0]
Rank 1: [1 次矩阵乘] ─────────────────────────────┐
Rank 2: [2 次矩阵乘] ──────────────────┐          │
Rank 3: [3 次矩阵乘] ─────────┐        │          │ 各 Rank 独立计算
...                           │        │          │ 但工作量递增
Rank r: [r 次矩阵乘]          ▼        ▼          ▼
```

**特点：**
1. **独立性**：每个 Rank 独立计算，不需要等待其他 Rank
2. **冗余计算**：Rank 1 和 Rank 2 都会计算 `M_1 @ dH_2 + dH_1`
3. **负载不均衡**：Rank r 需要 O(r) 次矩阵乘法

**这种设计的 tradeoff：**
- ✅ 消除了串行依赖（不需要等待后续 Rank 发送 dS）
- ❌ 存在冗余计算（每个 Rank 重复计算后缀）
- ❌ 负载不均衡（前序 Rank 工作量更大）

### 与非 CP 版本的对比

| 方面 | 非 CP 版本 | CP 版本 |
|------|-----------|---------|
| `dS_r^{out}` 来源 | 用户提供或默认 0 | 通过 `pre_process` + 通信计算 |
| 额外 kernel | 无 | `pre_process_bwd_kernel_merged`, `merge_fwd_bwd_kernel` |
| 通信 | 无 | 1 次 All-Gather (per head) |
| 计算开销 | 无 | Rank r: O(r × K² × K) 矩阵乘 |
| 内存开销 | 无 | `ag_dhm`: world_size × H × K × (V+K) × 4 bytes |

### 时序图

```
时间 →
                     ┌─ intra_chunk ─┐ ┌─ pre_process ─┐ ┌ all_gather ┐ ┌─ merge ─┐ ┌─ bwd_h ─┐

Rank 0: ─────────────[chunk_kda_intra]─[compute (dH,M)]──[  ag_dhm   ]──[  skip  ]──[bwd_h]───▶
Rank 1: ─────────────[chunk_kda_intra]─[compute (dH,M)]──[  ag_dhm   ]──[1次 mm ]──[bwd_h]───▶
Rank 2: ─────────────[chunk_kda_intra]─[compute (dH,M)]──[  ag_dhm   ]──[2次 mm ]──[bwd_h]───▶
Rank 3: ─────────────[chunk_kda_intra]─[compute (dH,M)]──[  ag_dhm   ]──[3次 mm ]──[bwd_h]───▶
                                                             ↑
                                                        同步点 (barrier)
```

---

## KDA 特有的 CP 支持

### 文件: `fla/ops/kda/chunk_fwd.py` 和 `fla/ops/kda/chunk_bwd.py`

KDA 的 CP 实现复用了 `chunk_delta_h.py` 中的基础设施，但有一些特殊处理:

### 前向传播

```python
def chunk_kda_fwd(..., cp_context: FLACPContext | None = None):
    # 1. Gate 处理 (KDA 特有)
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(g_org, A_log, dt_bias, ...)
    else:
        g = chunk_local_cumsum(g, ...)
    
    # 2. Intra-chunk 计算
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(...)
    
    # 3. CP 预处理 (与 GDN 相同)
    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg, w=w, u=u, gk=g,
            cu_seqlens=cu_seqlens,
            context=cp_context,
            use_exp2=True,  # KDA 使用 exp2
        )
    
    # 4. 计算隐状态
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(...)
    
    # 5. 压缩初始状态 (节省显存)
    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)
    
    # 6. 计算输出
    o = chunk_gla_fwd_o_gk(...)
```

### 后向传播

```python
def chunk_kda_bwd(..., cp_context: FLACPContext | None = None):
    # 1. 重计算或加载中间状态
    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(...)
        w, u, qg, kg = recompute_w_u_fwd(...)
        
        # 展开压缩的初始状态
        if cp_context is not None:
            initial_state = expand_h0(initial_state, context=cp_context)
        
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(...)
    
    # 2. 局部梯度计算
    dAqk, dv = chunk_kda_bwd_dAv(...)
    
    # 3. CP 后向预处理
    if cp_context is not None:
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=qg, k=kg, w=w, do=do, dv=dv, gk=g,
            context=cp_context,
            use_exp2=True,
        )
    
    # 4. 计算 dh
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(...)
    
    # 5. 其余梯度计算
    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(...)
    dq, dk, db, dg = chunk_kda_bwd_intra(...)
```

### KDA 的 use_exp2 参数

KDA 使用 `exp2` (以 2 为底的指数) 而非 `exp` (自然指数)，这是因为:
- KDA 的 gate 计算使用 `RCP_LN2` (1/ln(2)) 作为 scale
- `exp2(x * RCP_LN2) = exp(x)`，但 `exp2` 在 GPU 上更高效

---

## GDN 的 CP 支持

### 文件: `fla/ops/gated_delta_rule/chunk.py`

GDN 的 CP 实现与 KDA 类似，但有以下区别:

### 对比表

| 特性 | KDA | GDN |
|------|-----|-----|
| Gate 计算 | `kda_gate_chunk_cumsum` | `chunk_local_cumsum` |
| use_exp2 | `True` | `False` (默认) |
| L2 Norm | 可选 (`use_qk_l2norm_in_kernel`) | 内置 |
| 额外参数 | `A_log`, `dt_bias` | 无 |

### GDN 前向传播

```python
def chunk_gated_delta_rule_fwd(..., cp_context: FLACPContext | None = None):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    
    # WY 表示计算
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, ...)
    A = solve_tril(A=A, ...)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, ...)
    
    # CP 预处理
    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k, w=w, u=u, g=g,
            context=cp_context,
            # 注意: GDN 默认使用 exp 而非 exp2
        )
    
    # 隐状态计算
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(...)
    
    # 压缩初始状态
    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)
    
    o = chunk_fwd_o(...)
```

---

## KDA 与 GDN 的 CP 差异

### 文件位置

- **KDA**: `fla/ops/kda/chunk_fwd.py`, `fla/ops/kda/chunk_bwd.py`
- **GDN**: `fla/ops/gated_delta_rule/chunk.py`

### 对比表

| 特性 | KDA | GDN |
|------|-----|-----|
| Gate 类型 | Per-head-dim (`g ∈ ℝ^{B×T×H×K}`) | Per-head (`g ∈ ℝ^{B×T×H}`) |
| Gate 计算 | `kda_gate_chunk_cumsum` (含激活) | `chunk_local_cumsum` |
| Gate 激活函数 | `-exp(A_log) · softplus(g + dt_bias)` | 无额外激活 |
| use_exp2 | `True` | `False` (默认) |
| L2 Norm | 可选 (`use_qk_l2norm_in_kernel`) | 通常在外部处理 |
| 额外参数 | `A_log`, `dt_bias`, `lower_bound` | 无 |

### KDA 的 exp2 优化

KDA 使用 `exp2` (以 2 为底的指数) 而非 `exp`：
```python
# KDA gate 计算使用 RCP_LN2 = 1/ln(2) 作为 scale
g_scaled = g * RCP_LN2

# 在 kernel 中使用 exp2
# exp2(g * RCP_LN2) = 2^(g/ln(2)) = e^g
decay = tl.exp2(g_scaled)  # 比 tl.exp(g) 在 GPU 上更高效
```

### CP 调用路径

```python
# KDA 前向
def chunk_kda_fwd(..., cp_context):
    g = kda_gate_chunk_cumsum(...)  # KDA 特有的 gate 计算
    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            ..., gk=g, use_exp2=True, context=cp_context  # KDA 使用 exp2
        )

# GDN 前向
def chunk_gated_delta_rule_fwd(..., cp_context):
    g = chunk_local_cumsum(...)  # 普通 cumsum
    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            ..., g=g, use_exp2=False, context=cp_context  # GDN 使用 exp
        )
```

---

## 性能优化

### 1. 初始状态压缩

```python
def compress_h0(h0: torch.Tensor, context: FLACPContext):
    """
    在 CP 模式下，只有第一个序列可能跨 Rank。
    因此只需保存 h0[0] 而非整个 batch。
    """
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    return h0[:1].clone()

def expand_h0(h0: torch.Tensor, context: FLACPContext):
    """在后向传播时展开压缩的初始状态。"""
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    expand_h0 = h0.new_zeros(B, *h0.shape[1:])
    expand_h0[:1] = h0
    return expand_h0
```

### 2. 融合 Kernel

`pre_process_fwd_kernel_merged` 和 `pre_process_bwd_kernel_merged` 将 Stage 1 和 Stage 2 融合为单个 kernel，减少:
- Kernel launch overhead
- 全局内存访问
- 同步开销

### 3. 使用 searchsorted 快速定位

在 `build_cp_context` 中使用 `torch.searchsorted` 进行 O(log N) 的二分查找，而非 O(N) 的线性搜索。

### 4. CPU 预计算

`cu_seqlens` 相关的计算在 CPU 上完成，避免 GPU 同步:
```python
cu_seqlens_cpu = cu_seqlens.cpu()
# ... 所有计算在 CPU 上
local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(device=cu_seqlens.device, non_blocking=True)
```

### 5. int64 防止溢出

对于长序列，使用 `int64` 计算偏移量以防止溢出:
```python
i_tg = i_t.to(tl.int64)
bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), ...
```

---

## 优化方向建议

基于上述第一性原理分析，以下是一些可能的优化方向：

### 1. 通信优化

#### 1.1 Ring All-Reduce 替代 All-Gather

**当前实现**: 使用 `all_gather` 收集所有 `(M, H)` 对到每个 Rank。

**潜在优化**: 使用 **Ring-based 前缀计算**：
```
Rank 0 → Rank 1 → Rank 2 → ... → Rank N-1
         传递累积状态 S，而非 (M, H) 对
```

**优势**:
- 通信量从 `O(N × (K² + KV))` 降到 `O(KV)`
- 每个 Rank 只需接收一次累积状态

**挑战**:
- 增加了通信轮次 (从 1 轮 all-gather 变为 N-1 轮 send-recv)
- 需要仔细设计 overlap

#### 1.2 通信-计算 Overlap

**当前实现**: Phase 分离，通信完成后才开始计算。

**潜在优化**: 
```
时间线:
Rank 0: [计算 (M_0,H_0)] [发送 S_0^{out}] [计算本地输出]
Rank 1:                  [接收 S_1^{in}] [计算本地输出] ...
```

使用 **异步通信** 和 **CUDA Streams**：
```python
# 启动异步 send
handle = dist.isend(S_out, dst=rank+1)
# 同时计算不依赖 S_in 的部分
compute_local_intra_chunk(...)
# 等待接收完成
handle.wait()
```

#### 1.3 减少 M 矩阵的通信

**观察**: M 矩阵 (K×K) 通常比 H 矩阵 (K×V) 小，但在某些配置下可能相当。

**潜在优化**:
- 如果 M 接近单位矩阵 (gate 接近 1)，可以只传递 `M - I` 的稀疏表示
- 使用低秩近似：`M ≈ I + U @ V^T`

### 2. 计算优化

#### 2.1 M 矩阵的累积计算

**当前实现**: 在 `merge_fwd_bwd_kernel` 中迭代计算：
```python
for idx in range(pre_num_ranks):
    S = M[idx] @ S + H[idx]
```

**潜在优化**: 预计算 M 的前缀积：
```
M_prefix[0] = I
M_prefix[1] = M_0
M_prefix[2] = M_1 @ M_0
...
M_prefix[r] = M_{r-1} @ ... @ M_0
```

然后：`S_r^{in} = M_prefix[r] @ 0 + Σ M_prefix[r-i] @ H[i]`

这可以用 **并行前缀和 (Parallel Prefix Sum)** 算法实现。

#### 2.2 融合更多计算到 CP Kernel

**当前实现**: CP 预处理是独立的 kernel。

**潜在优化**: 将 CP 相关计算融合到主 chunk kernel 中：
```python
# 当前: 3 个 kernel
kernel_1: chunk_kda_fwd_intra(...)  # 本地计算
kernel_2: pre_process_fwd_kernel_merged(...)  # CP 预处理
kernel_3: chunk_gated_delta_rule_fwd_h(...)  # 状态更新

# 优化: 融合为 2 个 kernel
kernel_1_fused: chunk_kda_fwd_with_cp_preprocess(...)
kernel_2: chunk_gated_delta_rule_fwd_h(...)
```

#### 2.3 利用稀疏性

对于 **低 beta (学习率)** 场景：
- Delta update `β_t · k_t ⊗ ṽ_t` 较小
- M 矩阵接近单位矩阵
- 可以使用近似计算或跳过某些更新

### 3. 内存优化

#### 3.1 激进的状态压缩

**当前实现**: `compress_h0` 只压缩第一个序列的状态。

**潜在优化**: 使用更激进的压缩：
- 低秩分解：`H = U @ V^T`，只存储 `U, V`
- 量化：将 fp32 状态量化为 fp16/bf16
- 选择性存储：只存储重要的状态分量

#### 3.2 Gradient Checkpointing with CP

**当前实现**: `disable_recompute=False` 时重计算中间状态。

**潜在优化**: 更细粒度的 checkpointing：
- 只 checkpoint CP 边界的状态
- 使用 selective recomputation

### 4. 算法层面优化

#### 4.1 异步 Pipeline CP

借鉴 Pipeline Parallelism 的思想：
```
时间步:  t0    t1    t2    t3
Rank 0: [C0] [C1] [C2] [C3]
Rank 1:      [C0] [C1] [C2]
Rank 2:           [C0] [C1]
Rank 3:                [C0]

C_i: 第 i 个 micro-batch 的 chunk
```

这样可以在等待状态传递时处理下一个 micro-batch。

#### 4.2 近似 CP

对于超长序列，可以考虑**近似方法**：
- 只传递最近 k 个 chunk 的状态影响
- 使用指数衰减加权历史状态
- 截断过小的状态分量

### 5. 系统层面优化

#### 5.1 CUDA Graph

**当前实现**: 每次前向/反向都重新 launch kernels。

**潜在优化**: 
- 将 CP 相关的 kernel sequence 捕获为 CUDA Graph
- 减少 kernel launch overhead
- 注意: 需要处理动态 shape (variable length sequences)

#### 5.2 通信后端优化

- 使用 **NCCL** 的 `ncclGroupStart/End` 批量化通信
- 探索 **MSCCL** 或自定义 collective 算法
- 在 NVLink 环境下使用 **P2P 通信**替代 all-gather

### 优化优先级建议

| 优化项 | 实现难度 | 预期收益 | 优先级 |
|--------|----------|----------|--------|
| 通信-计算 Overlap | 中 | 高 | ⭐⭐⭐⭐⭐ |
| Ring-based 状态传递 | 高 | 高 | ⭐⭐⭐⭐ |
| Kernel 融合 | 中 | 中 | ⭐⭐⭐ |
| CUDA Graph | 低 | 中 | ⭐⭐⭐ |
| 状态压缩/量化 | 中 | 中 | ⭐⭐ |
| 近似 CP | 高 | 视场景 | ⭐⭐ |

---

## 使用示例

### KDA with CP

```python
import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda

# 初始化分布式环境
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()

# 模型参数
B, T_total, H, K, V = 1, 131072, 32, 128, 128  # 128K 序列
T_local = T_total // world_size

# 全局 cu_seqlens (所有序列)
cu_seqlens = torch.tensor([0, T_total], device='cuda', dtype=torch.int32)

# 构建 CP context
cp_context = build_cp_context(cu_seqlens, group=dist.group.WORLD)

# 局部输入 (每个 Rank 的部分)
q = torch.randn(B, T_local, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T_local, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T_local, H, V, device='cuda', dtype=torch.bfloat16)
g = torch.randn(B, T_local, H, K, device='cuda', dtype=torch.bfloat16)
beta = torch.rand(B, T_local, H, device='cuda', dtype=torch.bfloat16).sigmoid()

# Gate 参数
A_log = torch.randn(H, device='cuda', dtype=torch.float32)
dt_bias = torch.randn(H * K, device='cuda', dtype=torch.float32)

# 前向传播
o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log,
    dt_bias=dt_bias,
    use_gate_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    safe_gate=True,
    lower_bound=-5,
    cu_seqlens=cp_context.cu_seqlens,  # 使用局部 cu_seqlens
    cp_context=cp_context,
)

# 后向传播正常工作
loss = o.sum()
loss.backward()
```

### GDN with CP

```python
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# 与 KDA 类似，但不需要 A_log, dt_bias
o, final_state = chunk_gated_delta_rule(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=K ** -0.5,
    cu_seqlens=cp_context.cu_seqlens,
    cp_context=cp_context,
)
```

---

## 测试与验证

### 测试文件

- `tests/context_parallel/test_cp_kda.py`: KDA 的 CP 单元测试
- `benchmarks/cp/test_gdn_with_cp.py`: GDN 的 CP 集成测试和性能基准
- `benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py`: KDA 在不同 CP 配置下的性能对比

### 运行测试

```bash
# KDA CP 测试 (需要 2/4/8 GPU)
pytest tests/context_parallel/test_cp_kda.py -v

# 使用 torchrun 运行 GDN CP 测试
torchrun --nproc_per_node=4 benchmarks/cp/test_gdn_with_cp.py --ops --backward

# KDA CP8 vs CP2TP 性能基准
torchrun --nproc_per_node=8 benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py --config cp8 --seqlen 131072 --backward
```

---

## 总结

FLA 项目中 KDA/GDN 的 Context Parallelism 实现具有以下特点:

1. **统一的基础设施**: `fla/ops/cp/` 提供了通用的 CP 支持，KDA 和 GDN 共享大部分代码
2. **高效的通信**: 使用 all_gather 而非 point-to-point 通信，确保所有 Rank 参与
3. **显存优化**: 通过初始状态压缩减少跨 Rank 传输和存储的数据量
4. **融合 Kernel**: 使用融合的 Triton kernel 减少 kernel launch 和内存访问开销
5. **Variable-Length 支持**: 完整支持跨多个 Rank 的变长序列
6. **数值稳定性**: 使用 int64 防止长序列场景下的整数溢出

这套实现使得 KDA/GDN 能够高效地处理超长序列（128K+），突破单卡显存限制。

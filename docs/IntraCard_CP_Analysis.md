# KDA Intra-Card Context Parallel：算法设计与实现分析

> 文档覆盖范围：`fla/ops/common/intracard_cp.py` 及其入口 `fla/ops/common/backends/intracard.py`

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心数学基础：Delta Rule 状态递推](#2-核心数学基础delta-rule-状态递推)
3. [Intra-Card CP 算法总览](#3-intra-card-cp-算法总览)
4. [五步流水线详解](#4-五步流水线详解)
   - 4.1 序列切分策略（Splitting）
   - 4.2 Pre-Scan：并行扫描子序列端点状态
   - 4.3 Merge：串行传播初始状态
   - 4.4 初始状态展开（Expand）
   - 4.5 主 Kernel 并行运行
5. [关键数据结构与缓存设计](#5-关键数据结构与缓存设计)
6. [SM 饱和度分析](#6-sm-饱和度分析)
7. [与跨卡 CP 的设计对比](#7-与跨卡-cp-的设计对比)
8. [CPU 侧开销优化：Pure Python Loop](#8-cpu-侧开销优化pure-python-loop)
9. [精度风险与控制](#9-精度风险与控制)
10. [可能的优化方向](#10-可能的优化方向)
11. [激活条件与使用场景](#11-激活条件与使用场景)

---

## 1. 背景与动机

### 1.1 问题场景

KDA（Kimi Delta Attention）及 GDN（Gated Delta Network）的 prefill 阶段面临一个根本性的并行瓶颈：

**状态递推本质上是串行的**。`chunk_gated_delta_rule_fwd_h` 的核心循环为：

```
h_{c+1} = decay(g_last_c) * h_c + k_c^T @ (u_c - w_c @ h_c)
```

对于 N=1（单序列）、长度 L 的序列，kernel grid 为 `(num_v_blocks, N*H)`，约等于 `2 * H` 个 SM block。
对于 H=32 的模型，只能占用约 64 个 SM block，**完全无法饱和**现代 GPU 的数百个 SM。

### 1.2 跨卡 CP 的局限

跨卡 Context Parallel（KCP）通过把序列切分到多个 GPU 来增加并行度，但需要：
- 多 GPU 集群（至少 2 卡）
- All-gather 通信开销
- 无法在单卡推理场景下使用

### 1.3 Intra-Card CP 的定位

**Intra-Card CP（卡内并行）**：在单卡上，将一条长序列切分为多个子序列，借助与跨卡 CP 完全相同的数学结构（pre-scan + merge），让主 kernel 可以并行处理所有子序列，从而**以算力换延迟**，大幅提升 SM 利用率。

| 特性 | 跨卡 CP | Intra-Card CP |
|------|---------|---------------|
| 硬件需求 | 多 GPU | 单 GPU |
| 通信 | All-gather（NCCL） | 无（同设备内存） |
| 适用阶段 | 训练 + 推理 | **仅推理（inference_mode）** |
| 激活条件 | `cp_context` 非空 | `varlen` 模式 + `inference_mode` |
| 并行度来源 | 跨卡 | 序列内切分 → 多子序列 |

---

## 2. 核心数学基础：Delta Rule 状态递推

### 2.1 基本递推式

KDA 和 GDN 共享以下 chunk-level 状态递推：

```
h_{c+1} = M_c · h_c + h_ext_c
```

其中：
- **`h_c`**：chunk `c` 输入时的 `[K, V]` 状态矩阵
- **`M_c`**：chunk `c` 的转移矩阵（Transition Matrix），`[K, K]`
- **`h_ext_c`**：chunk `c` 在 `h_c=0` 假设下的外推状态，`[K, V]`

转移矩阵的计算（forward 方向）：

```
M_c = diag(decay(g_last_c)) - k_c^T @ w_c
```

- KDA（per-dim gate）：`decay = exp2(gk_last)` → 对角矩阵
- GDN（标量 gate）：`decay = exp(g_last)` → 标量乘以单位阵

### 2.2 子序列的初始状态推导

设一条序列被切分为 `[ss_0, ss_1, ..., ss_{P-1}]` 共 P 个子序列。

- `ss_0` 的初始状态：直接继承原序列的 `initial_state`（或零）
- `ss_j`（`j>=1`）的初始状态需要通过合并前 `j` 个子序列的 `(h_ext, M)` 来计算：

```
h_init[ss_j] = M_{j-1} · (M_{j-2} · (... · h_init[ss_0] + h_ext[ss_0]) + ...) + h_ext[ss_{j-1}]
```

这正是 Pre-Scan + Merge 步骤的数学含义。

---

## 3. Intra-Card CP 算法总览

```
输入：长序列 (k, w, u, g/gk)，cu_seqlens（描述序列边界）

Step 1：compute_subseq_len()
        ↓  计算目标子序列长度（基于 SM 数量 + head 数）

Step 2：prepare_subseq_cu_seqlens()
        ↓  决定哪些序列需要切分、切成几段
        ↓  输出扩展后的 cu_seqlens_subseq（子序列边界数组）

Step 3：intracard_pre_scan()  ← Triton kernel（完全并行）
        ↓  对所有被切分序列的所有子序列，并行计算 (h_ext, M)
        ↓  输出 hm[S_split, H, K, V+K]

Step 4：intracard_merge()  ← Triton kernel（按序列串行，子序列内并行）
        ↓  对每条被切分序列，串行合并 (h_ext, M) 计算各子序列的初始状态
        ↓  输出 initial_states_merge[num_non_first, H, K, V]

Step 5：_raw_chunk_gated_delta_rule_fwd_h()  ← 主 Triton kernel（完全并行）
        ↓  以 total_subseqs 作为"虚拟 batch"，所有子序列并行运行
        ↓  输出 h（chunk-level 状态）、v_new、final_state_subseq

Step 6：抽取 final_state（取每条原始序列最后一个子序列的 final_state）
        输出：h, v_new, final_state
```

---

## 4. 五步流水线详解

### 4.1 序列切分策略（Splitting）

#### `compute_subseq_len()`

核心思想：**目标切分数量由 SM 数量和 head 数共同决定**。

```python
NUM_V_BLOCKS = 2  # fwd_h kernel 每条序列贡献的 V 方向 block 数
target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))
subseq_chunks = ceil(seq_chunks / target_splits)
subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)  # MIN=128
subseq_len = subseq_chunks * chunk_size
```

**设计要点**：
- `target_splits` 确保即使只有一条长序列，也能让所有 SM 都有工作
- `MIN_SUBSEQ_CHUNKS = 128`（即 `subseq_len >= 8192 tokens`）是一个"切分门槛"的隐含下界：序列长度须超过 `3 × subseq_len`（即 ~24K tokens）才会被切分，避免短序列被无意义切分

#### `prepare_subseq_cu_seqlens()`

**切分触发条件**：`seq_len >= 3 × subseq_len`（防止 `num_subseqs=1` 的无效切分）

切分算法：
```python
num_ss = min(max_splits, ceil(seq_chunks / subseq_chunks))
chunks_per = ceil(seq_chunks / num_ss)   # 尽量均匀分配
actual_ssl = chunks_per * chunk_size      # 对齐到 chunk 边界
```

输出 `SplitSeqInfo`（纯 Python 对象）：
- `split_seq_ids`：哪些序列被切分了
- `start_subseq_idx`：每条切分序列在子序列数组中的起始下标
- `num_subseqs`：每条切分序列的子序列数量

### 4.2 Pre-Scan：并行扫描子序列端点状态

**函数**：`intracard_pre_scan()`  
**Kernel**：`pre_process_fwd_kernel_merged`（来自 `fla/ops/cp/chunk_delta_h.py`）

Grid 设计：
```python
grid = (cdiv(V, BLOCK_SIZE) + cdiv(K, BLOCK_SIZE), H, S_split)
```

关键：第三维 `S_split`（**被切分的子序列总数**），使得所有子序列并行执行。

每个子序列独立计算并输出 `hm[s, H, K, V+K]`，其中：
- `hm[s, h, :K, :V]`：**h_ext** — 该子序列假设初始状态为零时的累积状态
- `hm[s, h, :K, V:]`：**M** — 该子序列的转移矩阵（K×K）

`pre_process_fwd_kernel_merged` 是融合的两阶段 kernel：

- **Stage 1（h_ext 列）**：按 V 方向分块，每块运行完整 chunk 扫描
  ```
  h = 0
  for each chunk c:
      h *= decay(g_last_c)       # inter-chunk 衰减
      v_new = u_c - w_c @ h      # delta rule 减去历史贡献
      h += k_c^T @ v_new         # 外积累加
  ```

- **Stage 2（M 列）**：按 K 列分块，每块运行矩阵连乘
  ```
  M = I
  for each chunk c:
      M_c = diag(decay) - k_c^T @ w_c
      M = M_c @ M
  ```

**重要实现细节**：M 的累乘始终保持 fp32，避免 bf16 精度退化。

### 4.3 Merge：串行传播初始状态

**函数**：`intracard_merge()`  
**Kernel**：`merge_fwd_bwd_kernel`（来自 `fla/ops/cp/chunk_delta_h.py`，`INTRACARD_MODE=True`）

Grid 设计：
```python
grid = (cdiv(V, BV), num_split_seqs, H)
```

- 第二维 `num_split_seqs`：不同的被切分序列之间**并行**
- 每个序列内部：子序列**串行**合并（天然的数据依赖）

Merge 伪代码（INTRACARD_MODE 下）：
```python
for i_seq in range(num_split_seqs):  # 并行
    ss_start = seq_offsets[i_seq]
    ss_end   = seq_offsets[i_seq + 1]
    h = initial_state[orig_seq_id[i_seq]]  # 原始初始状态（若有）
    for idx in range(num_subseqs):         # 串行
        h_ext = hm[ss_start + idx, :, :V]
        M     = hm[ss_start + idx, :, V:]
        h = M @ h + h_ext
        if idx < num_subseqs - 1:
            initial_states_merge[init_base + idx] = h  # 存储中间初始状态
```

输出 `initial_states_merge[num_non_first, H, K, V]`：所有非第一个子序列的初始状态。

### 4.4 初始状态展开（Expand）

将 merge 的结果"展开"到全量子序列的初始状态数组：

```python
initial_state_expanded = zeros(total_subseqs, H, K, V)

# 第一个子序列继承原始初始状态
initial_state_expanded[first_subseq_indices] = initial_state  # 若有

# 非第一个子序列填入 merge 结果
initial_state_expanded[non_first_indices] = initial_states_merge
```

`first_subseq_indices` 和 `non_first_indices` 均为 Python `list[int]`，通过 `_precompute_intracard_indices()` 预计算（并被缓存）。

### 4.5 主 Kernel 并行运行

```python
h, v_new, final_state_subseq = _raw_chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g, gk=gk,
    initial_state=initial_state_expanded,  # [total_subseqs, H, K, V]
    cu_seqlens=cu_seqlens_subseq_gpu,      # 扩展后的子序列边界
    ...
)
```

此时 `cu_seqlens` 包含 `total_subseqs` 条"序列"，kernel grid 变为 `(num_v_blocks, total_subseqs * H)`，SM 利用率相对原来提升 `~total_subseqs` 倍。

最后，抽取每条原始序列最后一个子序列的 final state：
```python
final_state = final_state_subseq[last_subseq_indices]
```

---

## 5. 关键数据结构与缓存设计

### 5.1 `_CacheEntry`（LRU 缓存条目）

`_intracard_cache` 是一个 `OrderedDict`，上限为 32 条（`_INTRACARD_CACHE_MAXSIZE`），以下列元组为 key：

```python
cache_key = (
    id(cu_seqlens),   # 对象身份，非内容哈希
    subseq_len,
    chunk_size,
    max_splits,
    str(device),
)
```

**设计理由**：vLLM 等推理框架在同一个 batch 的多次 forward 中会复用相同的 `cu_seqlens` Python 对象（同地址），因此用 `id()` 可以零哈希开销地命中缓存。

缓存内容分两类：

| 类型 | 字段 | 作用 |
|------|------|------|
| Python 对象 | `cu_seqlens_subseq_values`、`split_info`、`non_first_indices` 等 | 避免重复 Python loop 计算 |
| GPU Tensor | `cu_seqlens_subseq_gpu`、`cu_seqlens_split_flat` | 避免重复 H2D 传输 |

**弱引用守卫**：通过 `weakref.ref(cu_seqlens)` 防止 Python id 复用导致的缓存污染（老 tensor 被 GC，新 tensor 恰好复用同一地址）。

### 5.2 `SplitSeqInfo`（NamedTuple）

纯 Python 列表容器，零 tensor 开销：
```python
class SplitSeqInfo(NamedTuple):
    split_seq_ids: list[int]      # 哪些序列被切分
    start_subseq_idx: list[int]   # 在子序列数组中的起始位置
    num_subseqs: list[int]        # 每条序列的切分数
```

### 5.3 CPU 侧 vs GPU 侧数据流

```
cu_seqlens (GPU)
    ↓ .cpu()
cu_seqlens_cpu (CPU)
    ↓ Python loop
cu_seqlens_subseq_values (Python list)
    ↓ torch.tensor(..., device=device)
cu_seqlens_subseq_gpu (GPU) ← 缓存避免重复 H2D
```

---

## 6. SM 饱和度分析

以 H100（132 SM）、`H=32`（head 数）、`K=V=128` 为例：

### 原始单序列 fwd_h

```
Grid = (cdiv(V, BV), N*H) = (2, 1*32) = 64 blocks
SM 利用率 = 64 / 132 ≈ 48%（且受限于长序列串行扫描时间）
```

### Intra-Card CP 后（以 target_splits=4 为例）

```
target_splits = max(4, 132 // (2 * 32)) = max(4, 2) = 4
total_subseqs = 4（一条序列切成 4 段）
Grid = (2, 4*32) = 256 blocks
SM 利用率 ≈ 256 / 132 → ~1.94x 超载，充分饱和
关键路径从 4×subseq_len 降为 1×subseq_len（理论加速比 ~4x）
```

### Pre-Scan 步骤的额外开销

Pre-scan kernel grid：`(cdiv(V,BS) + cdiv(K,BS), H, S_split)`

对于 V=K=128，`BLOCK_SIZE=64`：`(2+2, 32, 4) = 512 blocks`（并行）
Pre-scan 的扫描长度为 `subseq_len`，计算量为原序列的 `1/num_splits`

因此总额外开销 ≈ `pre_scan(subseq_len) + merge(O(1))` ≪ `fwd_h(seq_len)` 节省量。

---

## 7. 与跨卡 CP 的设计对比

Intra-Card CP 与 KCP（Kimi Context Parallel）共享完全相同的数学框架（pre-scan + merge），核心 kernel 也被复用：

| 组件 | 跨卡 CP（KCP） | Intra-Card CP |
|------|----------------|---------------|
| Pre-Scan Kernel | `pre_process_fwd_kernel_stage1` + `stage2` | `pre_process_fwd_kernel_merged`（融合版） |
| Merge Kernel | `merge_fwd_bwd_kernel`（`INTRACARD_MODE=False`） | `merge_fwd_bwd_kernel`（`INTRACARD_MODE=True`） |
| 子序列/Rank 来源 | 跨 GPU rank | 单卡内切分子序列 |
| 通信 | All-gather（NCCL） | 无 |
| Merge 方向 | 向前（h_r 依赖前 r-1 ranks） | 向前（ss_j 依赖前 j-1 子序列） |
| 支持反向 | 是 | **否（仅推理）** |
| `h0` 处理 | 仅第一条序列有效 | 每条原始序列均可有独立 h0 |
| 编排层 | `FLACPContext` + `dist` | `IntraCardCPBackend` + `BackendRegistry` |

**关键差异**：KCP 的 merge kernel 在 `INTRACARD_MODE=False` 时以 rank 为索引；Intra-Card 在 `INTRACARD_MODE=True` 时以 `(i_seq, i_subseq)` 为索引，并通过 `seq_offsets` / `init_offsets` 两个小数组描述拓扑。

---

## 8. CPU 侧开销优化：Pure Python Loop

代码注释中明确提到："uses pure Python loops instead of torch tensor operations (repeat_interleave, arange, cumsum, etc.)"。

**背景**：在高吞吐推理服务中，每个请求都触发一次 `intracard_fwd_h`，而序列数组（`cu_seqlens`，通常 1~32 条序列）极小。对这种"tiny tensor"使用 `torch.cumsum`、`torch.arange` 等操作会引入：
1. Python GIL 竞争
2. CUDA kernel launch overhead（每个 PyTorch op 约 5-20μs）
3. `cudaStreamSynchronize`（当结果需要回到 CPU 时）

替换为 Python `for` 循环后，这些开销归零——Python 循环处理 32 个元素的速度远快于启动一个 CUDA kernel。

对比示例（`_precompute_intracard_indices` 内的 `cumsum`）：

```python
# 原来（torch 方式，有 CUDA kernel launch）
merge_seq_offsets = torch.cumsum(torch.tensor([0] + num_ss), dim=0).tolist()

# 优化后（pure Python）
merge_seq_offsets: list[int] = [0]
for n in num_ss:
    merge_seq_offsets.append(merge_seq_offsets[-1] + n)
```

---

## 9. 精度风险与控制

### 9.1 转移矩阵链乘的精度退化

Merge 步骤的核心运算是：

```
h = M_{P-1} @ M_{P-2} @ ... @ M_0 @ h_0
```

如果 M 的累乘以 bf16 进行，每次矩阵乘法都会损失约 3 bits 的精度，P 次累乘后误差指数级增长。

**当前控制**：`merge_fwd_bwd_kernel` 内部保持 fp32 精度：
```python
b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)
```

### 9.2 `MAX_SUBSEQS` 的作用

```python
MAX_SUBSEQS = int(os.environ.get('FLA_INTRACARD_MAX_SPLITS', 32))
```

这是"合并链深度"的上限，其设计原则：
- 过小（如 4）：长序列无法充分切分，SM 利用率不足
- 过大（如 128）：合并链过深，即使 fp32 也可能累积误差，且 merge kernel 延迟增加

32 的默认值提供了足够深的切分（32 段对应约 1M tokens 的序列可以有效切分），同时将误差链长度控制在可接受范围。

---

## 10. 可能的优化方向

### 10.1 树形并行 Merge（最高优先级）

**当前瓶颈**：`intracard_merge` 在每条序列内是线性扫描（深度 = P-1），merge kernel 延迟 = O(P)。

**优化方向**：采用树形归约（parallel scan/prefix sum）：

```
当前（线性）：ss_0 → ss_1 → ss_2 → ss_3   深度 3
树形归约：
  层0：(ss_0, ss_1)   (ss_2, ss_3)           深度 1
  层1：(ss_01, ss_23)                         深度 1
                                              总深度 2 = log2(4)
```

对 P=32 的切分，树形归约将 merge 深度从 31 降到 5，**大幅降低 merge 延迟**，且树形中间结果可以完全并行计算。

实现思路：
1. 奇偶层交替的归约 kernel，每层对应一个 kernel launch
2. 使用 workspace 存储中间层的 `(h_ext, M)` 对
3. 总 kernel launch 数从 1 增加到 log2(P)，但总延迟从 O(P) 降到 O(log P)

### 10.2 自适应阈值调整

**当前问题**：`MIN_SUBSEQ_CHUNKS = 128`（subseq_len >= 8192）是硬编码常量，切分门槛固定为 `3 × subseq_len`。

**优化方向**：
- 基于实测的 pre-scan + merge overhead vs fwd_h 加速量的 profile 数据，学习最优 `MIN_SUBSEQ_CHUNKS`
- 区分不同 GPU 型号（A100/H100/H20/B100）使用不同默认值
- 将切分决策建模为：`gain(num_splits, seq_len) > overhead(pre_scan + merge)`，用实测拟合得到最优分界线

### 10.3 CUDA Graph 兼容

**当前限制**：Intra-Card CP 仅在 `torch.inference_mode()` 下工作，但对 CUDA Graph 的兼容性未显式处理。

CUDA Graph 要求所有 kernel launch 在 capture 阶段确定，而动态切分逻辑（`compute_subseq_len`、`prepare_subseq_cu_seqlens`）依赖序列长度在运行时的实际值。

**优化方向**：
- 为固定形状的请求（如服务端 continuous batching 中已知 max_seq_len 的场景）预生成 CUDA Graph，与缓存机制联动
- 对 `total_subseqs` 进行 padding 到固定值，避免动态 grid 导致的 graph 不兼容

### 10.4 支持反向传播（训练模式扩展）

**当前限制**：`chunk_gated_delta_rule_fwd_h_verifier` 明确限制为 `torch.is_inference_mode_enabled()`。

**分析**：反向传播的阻碍在于中间 chunk-level 状态 `h[NT, H, K, V]` 需要被 `save_for_backward`，而 Intra-Card 切分使 NT 动态变化，且 `initial_state_expanded` 是 detached 的（通过 merge kernel 非 autograd 生成）。

**可行路径**：
1. 仅支持 `no_grad()` 下的训练评估（eval 模式）
2. 为 Intra-Card 实现独立的 `chunk_gated_delta_rule_bwd_dhu_intracard`，复用 merge backward（`merge_fwd_bwd_kernel(FORWARD=False)`）

### 10.5 混合批次的负载均衡

**当前问题**：在混合长度的 varlen 批次中（如 `[1K, 512K, 2K, 256K]`），只有超过门槛的序列被切分，导致切分后的 `total_subseqs * H` 块数依然可能不均匀。

**优化方向**：
- 全局 SM 视角的切分：对所有序列联合优化 `num_subseqs_i`，使 `sum(num_subseqs_i * H * NUM_V_BLOCKS)` 恰好等于 `num_sms` 的整数倍
- 短序列也参与切分（降低 `MIN_SUBSEQ_CHUNKS`），通过全局负载均衡而非局部门槛决策

### 10.6 Intra-Card pre_scan 与 KCP pre_process 的 kernel 统一

**当前状态**：KCP 使用 `pre_process_fwd_kernel_stage1 + stage2`（两个 kernel），Intra-Card 使用 `pre_process_fwd_kernel_merged`（一个 kernel）。

**优化方向**：
- 彻底将 KCP 也迁移到 `pre_process_fwd_kernel_merged`，合并两套 kernel 的维护负担
- `MULTI_SEQS=False` 对应单序列（KCP 本地序列），`MULTI_SEQS=True` 对应多子序列（Intra-Card）

### 10.7 子序列边界对齐

**当前实现**：`actual_ssl = ceil(seq_chunks / num_ss) * chunk_size` 确保每段对齐到 `chunk_size`（默认 64）。

**优化方向**：进一步对齐到 `warp_tile_size`（如 128 tokens），减少 boundary check 的分支，同时改善 L2 cache 利用率。

---

## 11. 激活条件与使用场景

### 11.1 后端注册机制

```python
# fla/ops/common/backends/__init__.py
common_registry = BackendRegistry("common")
common_registry.register(IntraCardCPBackend())
```

`chunk_gated_delta_rule_fwd_h` 在调用时通过 `dispatch` 函数遍历注册的后端，调用 `chunk_gated_delta_rule_fwd_h_verifier` 检查是否应由该后端处理：

```python
# IntraCardCPBackend.chunk_gated_delta_rule_fwd_h_verifier
if not torch.is_inference_mode_enabled():
    return False, "Not in inference mode"
if cu_seqlens is None:
    return False, "cu_seqlens is None"
return True, None
```

### 11.2 适用场景

| 场景 | 是否激活 | 说明 |
|------|---------|------|
| KDA prefill，长序列，单卡 | ✅ | 主要目标场景 |
| GDN prefill，长序列，单卡 | ✅ | 同样适用 |
| KDA 训练（backward）| ❌ | `inference_mode` 不满足 |
| KDA + KCP（多卡推理）| 条件激活 | 若 `cu_seqlens != None` 则激活，但与 KCP 的 `initial_state` 交互需验证 |
| 短序列（< 3 × subseq_len）| 跳过切分 | `early_return` 或 `not split_info` 走原始 kernel |
| `cu_seqlens = None`（padded batch）| ❌ | varlen 模式才有意义 |

### 11.3 环境变量控制

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `FLA_INTRACARD_CP` | `1`（若后端启用） | 完全禁用 Intra-Card CP |
| `FLA_INTRACARD_MAX_SPLITS` | `32` | 控制最大切分数（精度/性能权衡） |

---

## 总结

KDA Intra-Card CP 是一个**单卡内的序列并行加速**技术，通过将长序列切分为多个子序列，借助与跨卡 CP 完全相同的数学框架（pre-scan + merge），将主 kernel 的有效 batch size 放大，从而充分利用 GPU SM。

其最核心的工程创新有三点：
1. **数学等价性**：复用 KCP 的 `(h_ext, M)` 框架，无需为单卡推理设计新算法
2. **CPU 侧零开销**：所有索引计算用 pure Python loop 完成，结合 LRU 缓存（含 GPU tensor 缓存），使每次 forward 的 CPU overhead 近乎为零
3. **渐进式退化**：短序列自动绕过，确保不会因切分开销大于收益而变慢

未来最有价值的优化方向是**树形并行 Merge**，它可以将当前 O(P) 的 merge 延迟降至 O(log P)，特别对于大切分数（P=16~32）的场景有显著收益。

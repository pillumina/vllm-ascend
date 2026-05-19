# Phase 1.1: GroupCoordinator 与通信组拓扑

**学习时间：** 2026-05-19
**源码版本：** vllm-ascend branch 408
**学习产出：** 本文件 + 通信组拓扑图

---

## 1. 核心抽象：GroupCoordinator

文件：`vllm/distributed/parallel_state.py:290`

vLLM 将 PyTorch 的 `ProcessGroup` 包装为 `GroupCoordinator`。这不是简单的 1:1 封装，而是一个**统一抽象层**，屏蔽了 ProcessGroup 的底层细节，提供一致的通信接口。

### 1.1 关键属性（6个）

```
rank                # 当前进程在全局中的 rank（0-indexed）
ranks               # 该组内所有全局 rank 的列表
world_size         # 该组的进程数
local_rank         # 节点本地的 GPU 编号（用于 device 分配，如 cuda:0）
rank_in_group      # 当前进程在该组内的序号（不同于全局 rank）
cpu_group          # CPU 通信用的 ProcessGroup（backend=gloo）
device_group       # 设备通信用的 ProcessGroup（backend=nccl/cuda）
```

**`rank` vs `rank_in_group` 的区别**（文件注释 `:304-311` 给出示例）：

```
进程  | 节点 | 全局 rank | local_rank | rank_in_group
  0  |   0  |    0    |     0      |      0
  1  |   0  |    1    |     1      |      1
  2  |   1  |    2    |     0      |      2      ← 全局 rank=2，但 node local_rank=0，group rank=2
  3  |   1  |    3    |     1      |      3
```

> **理解误区纠正**：之前我以为 `rank_in_group` 就是 `local_rank`，这是错的。`local_rank` 是**节点内**的 GPU 编号；`rank_in_group` 是**通信组内**的序号，两者毫无关系。

### 1.2 核心方法（4类）

**第一类：集合通信（最常用）**
```python
all_reduce(input_) -> Tensor        # Out-place AllReduce
all_gather(input_, dim=-1)          # AllGather
reduce_scatter(input_, dim=-1)      # Reduce-Scatter
broadcast(input_, src=0)             # Broadcast
```

**第二类：张量路由（用于 TP GEMM 输出）**
```python
send(input_, dst)                   # 点对点发送
recv(dst)                          # 点对点接收
send_tensor_dict(...)               # 发送字典（含张量+元数据）
recv_tensor_dict(...)
```

**第三类：元数据广播（用于非张量数据）**
```python
broadcast_object(obj, src=0)         # 广播 Python 对象
broadcast_object_list(...)          # 广播对象列表
broadcast_tensor_dict(...)           # 广播含张量+非张量的字典
```

**第四类：同步与查询**
```python
barrier()                           # 组内 Barrier
first_rank() -> bool               # 是否是组内第一个 rank
is_first_rank() -> bool           # rank_in_group == 0
next_rank() / prev_rank()         # 环状邻居
graph_capture(ctx)                 # CUDA Graph 上下文管理
```

---

## 2. 全局通信组一览

文件：`vllm/distributed/parallel_state.py:1219-1280`

vLLM 主仓定义了以下全局组（均为 `GroupCoordinator` 实例）：

| 组名 | 变量 | world_size | 获取函数 | 创建时机 |
|------|------|-----------|---------|---------|
| World | `_WORLD` | 全局 `world_size` | `get_world_group()` | `init_distributed_environment` |
| Tensor Parallel | `_TP` | `tensor_model_parallel_size` | `get_tp_group()` | `initialize_model_parallel` |
| Pipeline Parallel | `_PP` | `pipeline_model_parallel_size` | `get_pp_group()` | `initialize_model_parallel` |
| Data Parallel | `_DP` | `data_parallel_size` | `get_dp_group()` | `initialize_model_parallel` |
| Expert Parallel | `_EP` | `data_parallel_size × pp × pcp × tp / ep_size` | `get_ep_group()` | `initialize_model_parallel`（仅 MoE） |
| Prefill CP | `_PCP` | `prefill_context_model_parallel_size` | `get_pcp_group()` | `initialize_model_parallel` |
| Decode CP | `_DCP` | `decode_context_model_parallel_size` | `get_dcp_group()` | `initialize_model_parallel` |
| EPLB | `_EPLB` | 同 `_EP` | `get_eplb_group()` | `initialize_model_parallel`（仅 MoE + EPLB 启用） |
| Inner DP | `_INNER_DP_WORLD` | — | `get_inner_dp_world_group()` | — |

### 2.1 构造顺序（重要）

```
1. torch.distributed.init_process_group()         → ProcessGroup 默认 world 组
         ↓
2. init_distributed_environment()               → 封装为 GroupCoordinator._WORLD
         ↓
3. initialize_model_parallel(
       tensor_model_parallel_size,
       pipeline_model_parallel_size,
       prefill_context_model_parallel_size,
       decode_context_model_parallel_size,
   )
         ↓
   all_ranks = torch.arange(world_size).reshape(
       -1,                          # ExternalDP
       data_parallel_size,          # DP
       pipeline_model_parallel_size, # PP
       prefill_context_model_parallel_size,  # PCP
       tensor_model_parallel_size,  # TP
   )
         ↓
   _TP = init_model_parallel_group(view TP 维度)
   _DCP = init_model_parallel_group(view DCP 维度)
   _PCP = init_model_parallel_group(view PCP 维度)
   _PP = init_model_parallel_group(view PP 维度)
   _DP = init_model_parallel_group(view DP 维度)
   _EP = init_model_parallel_group(view EP 维度)
```

---

## 3. Rank 布局：核心中的核心

文件：`vllm/distributed/parallel_state.py:1550-1565`

所有组的 rank 布局由一个 **5D 张量**决定：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                                  # [0] ExternalDP
    data_parallel_size,                   # [1] DP
    pipeline_model_parallel_size,          # [2] PP
    prefill_context_model_parallel_size,  # [3] PCP
    tensor_model_parallel_size,            # [4] TP
)
```

每个维度代表一个并行维度，通过 transpose/reshape/unbind 从中提取各组的 ranks。

### 3.1 具体例子

假设 `world_size=8, tp=2, pp=2, pcp=1, dcp=2, dp=2`：

```
all_ranks.shape = (1, 2, 2, 1, 2)
all_ranks = [[[[[g0, g1]],
               [[g2, g3]]],
              [[[g4, g5]],
               [[g6, g7]]]]
```

**各组的 rank 分配**：

```
TP 组（view -1, tp）：
  TP[0] = [g0, g1]     TP rank 0
  TP[1] = [g2, g3]     TP rank 0
  TP[2] = [g4, g5]     TP rank 0
  TP[3] = [g6, g7]     TP rank 0
  → 4 个 TP 组，每组 2 个 rank（tp_size=2）

DCP 组（reshape -1, dcp）：
  DCP[0] = [g0, g4]    DCP rank 0（取每列的 dcp 维）
  DCP[1] = [g1, g5]
  DCP[2] = [g2, g6]
  DCP[3] = [g3, g7]
  → 4 个 DCP 组，每组 2 个 rank（dcp_size=2）

PP 组（transpose(2,4), reshape）：
  PP[0] = [g0, g2]     PP rank 0
  PP[1] = [g1, g3]
  PP[2] = [g4, g6]
  PP[3] = [g7, g5]
  → 4 个 PP 组，每组 2 个 rank（pp_size=2）
```

### 3.2 这个布局说明了什么

**不同组的 rank 在全局 rank 中的位置不同。**

举例：全局 rank=0 的进程：
- 在 TP 组中 rank_in_group=0（属于 [g0, g1] 组）
- 在 DCP 组中 rank_in_group=0（属于 [g0, g4] 组）
- 在 PP 组中 rank_in_group=0（属于 [g0, g2] 组）

**这正是 HCCL mismatch 的根源之一**：当某段代码假设两个 rank 在 TP 组内通信，但实际它们属于不同的 DCP 组时，AllReduce 的 tensor shape 就不一致。

---

## 4. TP mismatch 的 traceback 路径

### 4.1 典型 mismatch 场景

**场景 A**：跨组通信——代码误用了错误的 group

```
某 kernel 输出 shape = [batch, tp_size * head_dim]
实际 rank 持有 shape = [batch, head_dim]

为什么：
  Attention 层做 all_gather 时，误用了 get_dcp_group() 而非 get_tp_group()
  → DCP 组的 world_size ≠ TP 组的 world_size
  → all_gather 的 dim 参数意义不同
```

**场景 B**：形状参数不匹配

```
AllReduce 传入的 tensor shape = [128, 512]
另一 rank 的 shape = [128, 256]

为什么：
  TP 组内不同 rank 持有的 tensor 维度不同（正常）
  但如果 all_reduce 前做了不匹配的 reshape
```

### 4.2 从报错到源码的 traceback

```
NCCL AllReduce failed: misaligned data
    │
    ├── torch.distributed._all_reduce_ops.default_all_reduce
    │       torch.distributed.distributed_c10d.all_reduce_
    │           torch.distributed.distributed_c10d._all_reduce_helper
    │               torch.distributed.distributed_c10d.all_reduce_
    │
    ├── GroupCoordinator.all_reduce()              ← parallel_state.py:492
    │       self._all_reduce_out_place(tensor)
    │
    ├── 调用方（如 Attention 层 forward）
    │       get_tp_group().all_reduce(output)     ← 这里用的是 TP 组
    │
    └── 真正的根因：
            output.shape = [num_tokens, hidden_size / tp_size]
            → 这个 hidden_size / tp_size 是谁算的？
            → 如果 num_tokens 在不同 rank 上不同，就会 mismatch
```

---

## 5. 已验证的 Open Questions

```markdown
| # | 问题 | 猜测 | 验证 | 结论 | 日期 |
|---|------|------|------|------|------|
| OQ-01 | GroupCoordinator 和 ProcessGroup 的关系是什么？ | 包装关系，GroupCoordinator 是上层的统一抽象 | 已验证：_WORLD/_TP/_PP 等都是 GroupCoordinator 实例，ProcessGroup 是其内部属性 | GroupCoordinator = ProcessGroup + 设备管理 + 通信原语封装 | 2026-05-19 |
| OQ-02 | 为什么同一个 rank 在不同组里的 rank_in_group 不同？ | 因为各组的 rank 分配是独立计算的 | 已验证：每组通过 view/transpose 从 all_ranks 张量中提取，维度不同导致 index 不同 | 同上，all_ranks 5D 张量的各维度切分方式不同 | 2026-05-19 |
| OQ-03 | TP mismatch 和 DCP/PCP 组的关系是什么？ | DCP/PCP 和 TP 可能使用不同的 world_size，导致 AllReduce 时 shape 不匹配 | 已验证：DCP size=2, TP size=2 时，all_gather([batch, hidden/2]) vs all_gather([batch, hidden/tp_dcp]) | 同上 | 2026-05-19 |
```

---

## 6. 待深入的问题（Next Step）

1. **为什么 PCP/DCP 组也有 `use_message_queue_broadcaster=True`？** — `mq_broadcaster` 是 SharedMemory 广播器，用于什么场景？与 NCCL broadcast 有什么区别？

2. **Elastic EP 的 StatelessGroupCoordinator 和普通 GroupCoordinator 的区别是什么？** — `enable_elastic_ep=True` 时路径完全不同，world 组变成了 stateless，这会带来什么语义变化？

3. **CustomAllReduce / FlashInfer 是如何替换 NCCL AllReduce 的？** — `enable_custom_all_reduce` 在哪里生效，替换发生在哪个层？

---

## 7. 理解检查点

以下说法判断对错（基于 Phase 1.1 知识）：

```
[ ] A. get_tp_group().rank 和 get_world_group().rank 永远是相等的
[ ] B. rank_in_group 是当前进程在给定通信组内的序号，不同组里值不同
[ ] C. TP 组和 DCP 组的 world_size 一定相同
[ ] D. all_ranks 5D 张量的最后一个维度永远是 TP
[ ] E. Elastic EP 模式下，world 组不再是 torch.distributed 的默认组

答案：见文档末尾
```

---

**答案**：A 错（TP rank 是组内序号，world rank 是全局序号）；B 对；C 错（DCP 可以独立于 TP）；D 对；E 对

*Phase 1.1 | v0.1 | 2026-05-19*

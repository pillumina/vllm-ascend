# Distributed 包学习路线图

**创建时间：** 2026-05-18
**目标：** 系统掌握 vLLM distributed 通信体系（主仓 + vllm-ascend）
**学习方式：** 自顶向下、以问题为锚、双轨并行（实现 + 异常模式）
**最后更新：** 2026-05-18

---

## 整体架构图

```
Layer 0 ── 通信原语抽象 ─────────────────────────────────────────────
              base_device_communicator.py
              cuda_communicator.py / npu_communicator.py
              custom_all_reduce.py / flashinfer_all_reduce.py
                         ↑ 使用 Layer 1 的 GroupCoordinator
                         │
Layer 1 ── 并行状态管理 ────────────────────────────────────────────
              parallel_state.py (vLLM 主仓 2122 行)
              └── GroupCoordinator: TP / PP / CP / DP / world_group
                         ↑ 被 Layer 2 消费
                         │
Layer 2 ── 高层通信方案 ────────────────────────────────────────────
              ├── KV Transfer
              │     ├── 主仓框架：KVConnectorBase_V1
              │     └── Ascend 实现：MooncakeConnector / AscendStoreConnector
              ├── FlashComm V1/V2        (→ parallel_state 组)
              ├── MC2 (MoE Communication)
              └── EPLB (Expert Load Balancing)
```

---

## 学习阶段

### Phase 1：Layer 1 — parallel_state 全局拓扑（目标：建立通信组地图）

**目标：** 搞懂 vLLM 的通信组体系，包括各组的 rank 布局、world_size、构造顺序、相互引用关系。

**主线任务：**

- [x] 1.1 读 `vllm/distributed/parallel_state.py` 骨架 ✅ (2026-05-19)
      - ✅ GroupCoordinator 抽象类 6 个关键属性 + 4 类方法
      - ✅ 6 个全局组一览（TP/PP/DP/EP/PCP/DCP/EPLB）
      - ✅ 构造顺序：world → TP → DCP → PCP → PP → DP → EP → EPLB
      - ✅ all_ranks 5D 张量布局（ExternalDP × DP × PP × PCP × TP）
      - ✅ TP mismatch 的 traceback 路径（从 NCCL → GroupCoordinator → 调用方）
      - ✅ rank vs rank_in_group vs local_rank 区分
      - 产出：`vllm-ascend/.phase1-notes/phase1-1-groupcoordinator.md`

- [ ] 1.2 读 vllm-ascend 的 `parallel_state.py` 扩展
      - 识别新增了哪些组（MC2 / FLASHCOMM2 / SHARD_WEIGHT / EPLB）
      - 与主仓组的关系：继承、替换、还是新增？

- [ ] 1.3 绘制完整通信组拓扑图
      - 每张图节点：组名、world_size、rank_in_group 计算方式
      - 每张图连接：谁使用谁（被谁引用）
      - 区分"主仓标准组"和"ascend 扩展组"

**问题锚（从已知问题反推）：**
> 我们分析的 Mooncake transfer 失败 bug，调用链经过哪些组？`_SHARD_WEIGHT` 组和 `get_tp_group()` 在哪里分叉？

**Open Questions（学习中产生的问题）：**

```markdown
| # | 问题 | 猜测 | 验证 | 结论 | 日期 |
|---|------|------|------|------|------|
| OQ-01 | GroupCoordinator 和 ProcessGroup 的关系 | 包装关系 | ✅ 已验证 | GroupCoordinator = ProcessGroup + 设备管理 + 通信原语封装 | 2026-05-19 |
| OQ-02 | rank vs rank_in_group vs local_rank 的区别 | 用途不同 | ✅ 已验证 | rank=全局rank；rank_in_group=组内序号；local_rank=节点内GPU编号 | 2026-05-19 |
| OQ-03 | 为什么 PCP/DCP 组有 use_message_queue_broadcaster=True？ | SharedMemory 广播器 vs NCCL | 待验证 | mq_broadcaster 用途待查 | 2026-05-19 |
| OQ-04 | Elastic EP 的 StatelessGroupCoordinator 和普通 GroupCoordinator 的区别 | world 组变 stateless，跨节点协调方式不同 | 待验证 | 需读 stateless_coordinator.py | 2026-05-19 |
| OQ-05 | CustomAllReduce / FlashInfer 如何替换 NCCL AllReduce？ | 在 device_communicator 层做条件分支 | 待验证 | 需读 cuda_communicator.py | 2026-05-19 |
```

**Case Study 锚点：**
- **CS-1**：MooncakeConnector 的 `_handle_request` 为什么在 `finally` 中发 DONE？→ 追溯到 `KVCacheTaskTracker` 与 Scheduler 的交互 → 发现 parallel_state 中 `kv_role` 的初始化时机

**交付物：**
- [ ] `distributed-groups-map.md` — 通信组拓扑图（ASCII/Mermaid）
- [ ] Phase 1 总结（3-5 句话）

---

### Phase 2：Layer 2 上层方案 — KV Transfer + Ascend 特有扩展

**目标：** 理解各 Connector 如何消费 Layer 1 的组，以及 Ascend 扩展的实现差异。

**KV Transfer 主仓框架：**

- [ ] 2.1 `KVConnectorBase_V1` 抽象接口（`vllm/distributed/kv_transfer/kv_connector/v1/base.py`）
      - `start_load_kv` / `wait_for_save` / `get_finished` / `post_forward` 的契约
      - `update_connector_output` 与 Scheduler 的交互

- [ ] 2.2 Scheduler 侧集成（`vllm/v1/core/sched/scheduler.py` 中 `_update_from_kv_xfer_finished`）
      - `finished_recving_kv_req_ids` / `failed_recving_kv_req_ids` 的流转
      - → **复盘我们已发现的 bug**：为什么 failed 请求没进入 `failed_recving_kv_req_ids`

**Ascend 实现：**

- [ ] 2.3 `mooncake_connector.py` 全链路（已完成文档）
      - 补充：`_SHARD_WEIGHT` 组在哪里被引用？

- [ ] 2.4 `mooncake_layerwise_connector.py` 与 `mooncake_connector.py` 的区别
      - Layerwise 版本多出了什么？

- [ ] 2.5 `ascend_store_connector.py` vs `mooncake_connector.py`
      - 池化与直传的架构差异
      - 谁使用哪个 Group？

**Open Questions：**
```markdown
| # | 问题 | 猜测 | 验证 | 结论 | 日期 |
|---|------|------|------|------|------|
```

**Case Study 锚点：**
- **CS-2**：PD 分离文档中 ACK 计数机制 → `port_send_num` 的更新在 P 侧哪个线程中执行？该线程与 Scheduler 线程是否是同一个？
- **CS-3**：AscendStoreConnector 的 Layerwise 模式 → Pool 的 `get_num_new_matched_tokens` 命中后，请求何时进入 `WAITING_FOR_REMOTE_KVS`？与 MooncakeConnector 的时序差异是什么？

**交付物：**
- [ ] KV Connector 族谱图（主仓抽象 → Ascend 各实现 → 消费关系）
- [ ] Phase 2 总结

---

### Phase 3：Layer 0 — 通信原语

**目标：** 理解 broadcast / all_reduce / send / recv 等原语在不同硬件上的实现差异，以及 Ascend NPU 的 HCCL vs 主仓 NCCL 的关系。

**主仓路径：**

- [ ] 3.1 `base_device_communicator.py` — 统一接口抽象
      - 各方法签名（broadcast / all_reduce / send / recv / gather / scatter）
      - `DeviceConfig` 如何决定使用哪个后端

- [ ] 3.2 `cuda_communicator.py` — NCCL 实现
      - 关键路径：`all_reduce` 如何调用 NCCL
      - CustomAllReduce / FlashInfer 的 fallback 机制

**Ascend NPU 路径：**

- [ ] 3.3 `npu_communicator.py` — HCCL 实现
      - 与 `cuda_communicator.py` 的接口是否完全一致？
      - HCCL 的集合通信（AllReduce/AllGather）如何映射到 NPU 硬件拓扑

- [ ] 3.4 `pyhccl_wrapper.py` / `pyhccl.py` — HCCL Python 绑定
      - HCCL vs NCCL 在 API 层面的差异

- [ ] 3.5 FlashComm V1/V2 通信内核
      - `npu_mm_reduce_scatter_base` / `npu_all_gather_base` 在哪被调用
      - FlashComm2 的 OTP/ODP 组如何在 `npu_communicator` 中使用

**Open Questions：**
```markdown
| # | 问题 | 猜测 | 验证 | 结论 | 日期 |
|---|------|------|------|------|------|
```

**Case Study 锚点：**
- **CS-4**：FlashComm V1 的 AllReduce 为什么能 overlap GEMM？追踪 `enable_sp()` 为 True 时，`npu_mm_reduce_scatter_base` 的调用栈 → 发现它是在哪个 stream 上执行的
- **CS-5**：layer_sharding 的 broadcast 使用了哪个 communicator？验证 `get_shard_weight_group()` 返回的 GroupCoordinator 用的是 `npu_communicator` 还是 `cuda_communicator`

**交付物：**
- [ ] 通信原语族谱（主仓 NCCL / Ascend HCCL / CustomAllReduce / FlashInfer 各路径对比）
- [ ] Phase 3 总结

---

## 异常模式分析（贯穿全阶段）

每个 Phase 中发现的 failure pattern 统一记录：

```markdown
## 异常模式沉淀

### Pattern-[编号]：[简述]

**触发场景：**
**根因：**
**影响：**
**定位方法：**
**修复方向：**
**源码位置：**
```

**已沉淀：**

| 编号 | 模式名称 | Phase | 触发场景 | 影响 |
|------|---------|-------|---------|------|
| P-01 | Mooncake Transfer 失败静默腐化 | Phase 2 | RDMA ret<0 | Attention 用未初始化 KV → 垃圾输出 |

---

## 对齐节奏

- **频率**：每周一次（或每完成一个 Phase）
- **形式**：
  1. 我汇报本阶段进展 + 新产生的问题
  2. 你 review：理解是否正确、是否有遗漏
  3. 共同确认下一阶段的具体任务
  4. 我更新 tracker 文件

---

## 当前状态

**当前 Phase：** Phase 1.1 已完成 ✅ → 进入 Phase 1.2
**下一步任务：** 读 `vllm-ascend/distributed/parallel_state.py`，对比主仓新增了哪些组
**Open Questions：** 5 个（OQ-01~OQ-05），已解答 2 个
**已沉淀 Case Study：** CS-1（MooncakeConnector finally 块与 Scheduler 交互）→ 已在 PD 文章 §4.7 中沉淀

---

*Roadmap 版本：v0.2 | 最后同步：2026-05-19*

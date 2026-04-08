# vLLM-Ascend MoE 量化与昇腾特性协同 — 设计文档

**Author:** Claude Sonnet 4.6
**Date:** 2026-04-08
**Status:** Approved

---

## 1. 背景与动机

### 1.1 MoE 模型在 NPU 上的核心挑战

MoE（Mixture of Experts）模型的核心特征是**稀疏激活**：每个 token 只会被路由到少数几个 expert 处理。这带来了两个相互制约的挑战：

- **计算密集**：每个 token 的实际计算量虽小，但访存密集
- **通信密集**：在 Expert Parallel（EP）部署下，token 需要先路由、再分发、最后合并，涉及跨设备通信

在 Ascend NPU 上，这两者的矛盾被进一步放大：
- NPU 的矩阵计算单元对数据排布有严格要求（FRACTAL_NZ 私有格式）
- 跨 device 的通信带宽有限，MC2（Memory-Centric Communication）是关键优化点
- **量化**作为通用优化手段，必须和以上两个特性协同设计，才能真正发挥作用

### 1.2 为什么量化不能单独讲

如果把量化方案（W8A8_DYNAMIC、MXFP8 等）单独抽出来讲，你会得到一套和上游 vLLM 差异不大的 scheme 列表。**真正的差异点在于**：

1. 量化权重能否走 FRACTAL_NZ 路径（决定 NPU 算子效率）
2. 量化 matmul 能否和 MC2 token dispatch fuse 到一起（决定 MoE 通信效率）
3. 量化的 scale 能否在 FUSED_MC2 模式下被 fold 进 kernel（减少内存带宽）

因此，本文档以 **MC2 通信策略选择为骨架**，把量化方案作为骨架上的执行者来展开。

---

## 2. 架构全景

### 2.1 整体数据流

```
Token 输入
    ↓
┌──────────────────────────────────────────────┐
│  MC2 通信层（AscendForwardContext）          │
│  ├─ Token 路由（routing）                    │
│  ├─ AllGather / MC2 / FUSED_MC2 / ALLTOALL  │
│  └─ 取决于 quant_type + device + EP 配置     │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│  量化 Matmul 层（AscendLinearScheme）        │
│  ├─ 权重：INT8/FP8/INT4 → FRACTAL_NZ        │
│  ├─ 激活：动态量化（per-token scales）       │
│  └─ 算子：torch_npu.npu_quant_matmul        │
└──────────────────────────────────────────────┘
    ↓
Expert 计算结果合并
```

### 2.2 关键设计约束

| 约束维度 | 具体限制 |
|---------|---------|
| FUSED_MC2 通信 | 仅支持 W8A8_DYNAMIC（mode=1/2 均要求 w1/w2 scale 为 per-token） |
| FRACTAL_NZ 格式 | W8A8、W4A8、W8A16 支持；MXFP 系列、W4A4 系列不支持 |
| 动态 EPLB | 仅在 W8A8_DYNAMIC / W4A8_DYNAMIC / MXFP8 / MXFP4 的 MoE 路径上可用 |
| 两条配置路径 | ModelSlim（华为工具链闭环）vs LLM-Compressor（上游生态），不可混用 |

---

## 3. MC2 通信策略选择机制

### 3.1 四种通信方法

| 方法 | 类 | 通信量 | 适用场景 |
|------|-----|--------|---------|
| ALLGATHER | AllGatherCommImpl | O(EP × M) | EP=1，或 MC2 容量不足 |
| ALLTOALL | AlltoAllCommImpl | O(M) | 高 EP（EP≥32），token 均匀分布 |
| MC2 | MC2CommImpl | O(M × k/num_experts) | A2/A3 设备，token 分布稀疏，w8a8_dynamic |
| FUSED_MC2 | FusedMC2CommImpl | 同 MC2 | A3 设备，EP≤32，token≤512，量化 matmul fuse |

### 3.2 MC2 选择决策树

**复用 CLAUDE.md 中已有的决策树**，重点关注量化类型在判断逻辑中的位置：

```
is MoE? ──否──→ None（不涉及 MoE 通信）
  │
  是
  ↓
is EP enabled? ──否 / EP=1──→ ALLGATHER
  │
  是
  ↓
is Device A2?
  │
  ├── 是：is (expert≤24 && EP≥16 && token≤512)?
  │         │
  │         是 → MC2 ✅
  │         否 → ALLGATHER
  │
  └── 否：is Device A3?
            │
            ├── 是：is token≤512?
            │         │
            │         是：is (FUSED_MC2=1 && EP≤32 && 非MTP)?
            │         │     │
            │         是 → FUSED_MC2 ✅
            │         否：is (FUSED_MC2=2 && w8a8_dynamic)?
            │             │
            │             是 → FUSED_MC2 ✅
            │             否 → MC2
            │
            否：is Device A5?
                  │
                  ├── 是：is (token≤512 && 跨DP)?
                  │     │
                  │     是 → MC2
                  │     否 → ALLTOALL
                  │
                  否：is Device 310P?
                        │
                        是 → ALLGATHER
```

**量化类型（quant_type）在判断中的位置**：
- 在 A3 设备 FUSED_MC2 mode=2 的判断中，`quant_type == "w8a8_dynamic"` 是前置条件
- 在 FUSED_MC2 mode=1 中，虽然不显式检查 quant_type，但只有 W8A8_DYNAMIC 准备了 `fused_w1_scale` / `fused_w2_scale`（见 w8a8_dynamic.py line 299-301）

---

## 4. 量化方案体系

### 4.1 两套配置路径

```
量化模型来源
    │
    ├── ModelSlim 工具链生成 ──→ quant_model_description.json
    │   配置类：AscendModelSlimConfig
    │   注册：@register_quantization_config("ascend")
    │   特点：华为工具链闭环，scheme 由 quant_model_description.json 字段决定
    │
    └── LLM-Compressor 生成 ──→ config.json（quantization_config.quant_method="compressed-tensors"）
        配置类：AscendCompressedTensorsConfig
        注册：替换上游 CompressedTensorsConfig
        特点：上游生态对接，_detect_quant_type() 映射到 ascend scheme
```

**为什么需要两套？**
- ModelSlim 是华为自研的模型压缩工具链，输出格式由华为定义，scheme 和权重布局与 Ascend NPU 算子强绑定
- LLM-Compressor 是上游 vLLM 生态（neuralmagic/LLMCompressor），vllm-ascend 通过替换注册的方式接入
- 两者在 API 层面等效，但适用场景不同（华为工具链用户 vs 上游生态用户）

### 4.2 Scheme 注册机制

```
@register_scheme("w8a8_dynamic", "linear")  ──┐
@register_scheme("w8a8_dynamic", "moe")     ──┤──→ 存于 registry 字典
@register_scheme("w8a8_mxfp8", "linear")   ──┤
...                                        ──┘

模型加载时：
  config.get_quant_method(layer) → AscendLinearMethod / AscendFusedMoEMethod
      ↓
  method.create_weights(...) → AscendLinearScheme.apply()
      ↓
  scheme 实例通过 registry[quant_type, layer_type] 查到
```

### 4.3 Linear 层量化方案

| Scheme | 权重格式 | 激活格式 | FRACTAL_NZ | 适用场景 |
|--------|---------|---------|------------|---------|
| W8A8_DYNAMIC | INT8 | INT8 (per-token) | ✅ | 通用动态量化，首选 |
| W8A8 (static) | INT8 | INT8 (per-tensor) | ✅ | 延迟敏感场景 |
| W8A8_MXFP8 | FP8 E4M3 | FP8 E4M3 (per-token) | ❌ | 精度敏感场景 |
| W8A8_MIX | INT8 | 混合 | ✅ | Prefill-Decode 混合部署 |
| W8A16 | INT8 | FP16/BF16 | ✅ | 权重压缩场景 |
| W4A8_DYNAMIC | INT4 (packed) | INT8 (per-token) | ✅ | 更极致压缩 |
| W4A4_MXFP4 | FP4 E2M1 | FP4 E2M1 (per-token) | ❌ | 极限压缩 |
| W4A4_FLATQUANT_DYNAMIC | INT4 | INT4 (per-token) | ❌ | FlatQuant 分布平滑 |
| W4A4_DYNAMIC | INT4 | INT4 (per-token) | ❌ | LAOS 定制 |

**关键实现差异**：
- W8A8、W4A8、W8A16 在 `process_weights_after_loading()` 中调用 `maybe_trans_nz()` 将权重转为 FRACTAL_NZ
- MXFP 系列使用 `torch_npu.npu_quant_matmul` 配合 `scale_dtype=FLOAT8_E8M0FNU_DTYPE`，不走 FRACTAL_NZ
- W4A4_FLATQUANT_DYNAMIC 使用 `torch_npu.npu_kronecker_quant()` 进行分布平滑，不支持 NZ

### 4.4 MoE 层量化与 MC2/FUSED_MC2 协同

**MoE 量化的特殊性**：
- 每个 token 被路由到 top-k 个 expert，需要 dispatch + combine 通信
- 权重分 w13（gate+up）和 w2（down）两部分
- 量化参数更多：per-token scale、per-group scale、offset 等

**FUSED_MC2 的因果链**：

```
Q: 为什么只有 W8A8_DYNAMIC 支持 FUSED_MC2？

A: 需要满足三个条件：

条件1：per-token 激活量化
  └─ 只有 W8A8_DYNAMIC 的激活是 per-token INT8
     MXFP 系列激活是 per-token FP8，但 NPU 的 fused dispatch kernel
     只支持 INT8 路径

条件2：权重为 INT8 FRACTAL_NZ
  └─ dispatch_ffn_combine / dispatch_gmm_combine_decode 的 weight 参数
     必须是 FRACTAL_NZ 格式
     W8A8_DYNAMIC 满足（maybe_trans_nz）
     W4A8_DYNAMIC 不满足（W4 走不同 kernel）

条件3：scale 可 fold 进 kernel
  └─ FUSED_MC2 mode=1 中，w1/w2 的 float scale 被转为 int64 bit-pattern
     (scale_from_float_to_int64)，作为 fused kernel 的参数传入
     这个转换是 W8A8_DYNAMIC scheme 专属的
```

**Mode 1 vs Mode 2**：

| | Mode 1 | Mode 2 |
|-|--------|--------|
| 环境变量 | VLLM_ASCEND_ENABLE_FUSED_MC2=1 | =2 |
| Kernel | dispatch_ffn_combine | dispatch_gmm_combine_decode |
| 适用设备 | A2/A3 | A3 |
| Scale 来源 | fused_w1_scale / fused_w2_scale | w8a8_dynamic per-token scale |
| 量化要求 | W8A8_DYNAMIC | **仅限 w8a8_dynamic**（严格校验） |

### 4.5 Attention 层量化（点到为止）

| Scheme | 用途 | 与 MC2 关系 |
|--------|-----|------------|
| FAKQuant | DeepSeek MLA 架构的 Q/K/V per-channel 量化 | 无关，独立作用于 attention 层 |
| INT8_DYNAMIC | Attention KV 动态量化 | 无关 |
| C8KVCache | INT8 KV 缓存压缩 | 无关 |

这些 scheme 在 `kv_c8.py` 中注册，通过 `AscendKVCacheMethod` 适配器接入，不参与 MC2 通信路径。

---

## 5. 量化与 FRACTAL_NZ 格式的耦合

### 5.1 maybe_trans_nz() 决策逻辑

```python
def maybe_trans_nz(weight):
    if is_310p():
        return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
    if VLLM_ASCEND_ENABLE_NZ == 2 or (VLLM_ASCEND_ENABLE_NZ == 1 and is_quant_weight(weight)):
        return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
    return weight
```

| 环境变量 | 含义 |
|---------|------|
| 0 | 不转换，保持原始格式 |
| 1 | 仅量化权重转换（默认） |
| 2 | 全部权重转换（含非量化） |

### 5.2 为什么 MXFP 系列不走 NZ

MXFP（Microscaling Floating Point）使用 `torch_npu.npu_quant_matmul`，该算子内部已经对 scale 做了融合处理，不需要外部以 FRACTAL_NZ 格式提供权重。它的优化路径是：

```
激活（FP16/BF16）
    ↓
npu_dynamic_mx_quant → FP8/FP4 per-token
    ↓
npu_quant_matmul(scale_dtype=FP8_E8M0FNU)
    ↓
结果
```

而 NZ 路径是：

```
权重（INT8）→ FRACTAL_NZ
    ↓
激活（INT8）→ npu_dynamic_quant
    ↓
npu_quant_matmul / npu_weight_quant_batchmatmul
    ↓
结果
```

两条路径服务于不同的精度/性能权衡，不需要合并。

---

## 6. 横向对比与实用决策

### 6.1 量化 scheme 对比表

| Scheme | 权重位宽 | 激活位宽 | 内存节省 | 精度损失 | MC2 兼容性 | FRACTAL_NZ | 适用场景 |
|--------|---------|---------|---------|---------|-----------|------------|---------|
| FP16/BF16 | 16bit | 16bit | - | - | 全部 | N/A | 基线 |
| W8A8_DYNAMIC | 8bit | 8bit | ~50% | 低 | MC2+FUSED | ✅ | **首选，MoE 推荐** |
| W8A8_MXFP8 | 8bit(FP8) | 8bit(FP8) | ~50% | 更低 | MC2 | ❌ | 精度敏感 |
| W8A16 | 8bit | 16bit | ~50% | 中 | 全部 | ✅ | 权重受限 |
| W4A8_DYNAMIC | 4bit | 8bit | ~75% | 中 | MC2 | ✅ | 极致压缩 |
| W4A4_MXFP4 | 4bit(FP4) | 4bit(FP4) | ~87.5% | 较高 | MC2 | ❌ | 极限压缩 |
| W8A8_MIX | 8bit | 8bit | ~50% | 低 | 全部 | ✅ | Prefill-Decode 混部 |

### 6.2 场景决策表

| 场景 | 推荐配置 |
|------|---------|
| 通用 MoE 推理（Qwen3-MoE 等） | W8A8_DYNAMIC + FUSED_MC2=1 |
| A3 设备极致性能 | W8A8_DYNAMIC + FUSED_MC2=2 |
| 精度优先 | W8A8_MXFP8 |
| 内存极度受限 | W4A8_DYNAMIC |
| DeepSeek V3/R1 类超大规模 MoE | W8A8_DYNAMIC + FUSED_MC2=2 + 动态 EPLB |
| Prefill-Decode 混部 | W8A8_MIX |

### 6.3 环境变量速查

| 环境变量 | 默认值 | 选项 | 作用 |
|---------|-------|------|------|
| VLLM_ASCEND_ENABLE_NZ | 1 | 0/1/2 | 权重 FRACTAL_NZ 转换策略 |
| VLLM_ASCEND_ENABLE_FUSED_MC2 | 0 | 0/1/2 | FUSED_MC2 通信模式 |

---

## 7. 源码索引

### 7.1 关键文件清单

**配置与注册**
- `vllm_ascend/quantization/quant_type.py` — QuantType 枚举定义
- `vllm_ascend/quantization/modelslim_config.py` — ModelSlim 配置，packed_modules_model_mapping（行 52-245）
- `vllm_ascend/quantization/compressed_tensors_config.py` — LLM-Compressor 配置，_detect_quant_type（行 330）
- `vllm_ascend/quantization/method_adapters.py` — AscendLinearMethod / AscendFusedMoEMethod 适配器
- `vllm_ascend/quantization/methods/registry.py` — @register_scheme 装饰器

**基础 Scheme 类**
- `vllm_ascend/quantization/methods/base.py` — AscendLinearScheme / AscendMoEScheme / AscendAttentionScheme ABC

**Linear 量化方案**
- `vllm_ascend/quantization/methods/w8a8_dynamic.py` — W8A8_DYNAMIC，含 FUSED_MC2 scale fusion（行 299-301）
- `vllm_ascend/quantization/methods/w8a8_static.py` — W8A8 静态量化
- `vllm_ascend/quantization/methods/w8a8_mxfp8.py` — W8A8_MXFP8，RL weight restore（行 134-175）
- `vllm_ascend/quantization/methods/w8a8_pdmix.py` — W8A8_MIX，Prefill-Decode 混合
- `vllm_ascend/quantization/methods/w8a16.py` — W8A16
- `vllm_ascend/quantization/methods/w4a8.py` — W4A8_DYNAMIC，两级量化（行 96-127）
- `vllm_ascend/quantization/methods/w4a4_mxfp4.py` — W4A4_MXFP4
- `vllm_ascend/quantization/methods/w4a4_flatquant.py` — W4A4_FLATQUANT_DYNAMIC，Kronecker quant
- `vllm_ascend/quantization/methods/w4a4_laos_dynamic.py` — W4A4_DYNAMIC（LAOS）

**MoE 量化与 MC2**
- `vllm_ascend/ops/fused_moe/moe_mlp.py` — quant_apply_mlp / unquant_apply_mlp，动态 EPLB gmm_swiglu 路径（行 38）
- `vllm_ascend/ops/fused_moe/moe_comm_method.py` — MC2/FUSED_MC2/ALLGATHER/ALLTOALL 实现，dispatch_ffn_combine（行 283-295），dispatch_gmm_combine_decode（行 297-313）
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py` — build_fused_experts_input，类型契约
- `vllm_ascend/ascend_forward_context.py` — select_moe_comm_type，MC2 选择逻辑（行 200-278）

**Attention 量化**
- `vllm_ascend/quantization/methods/kv_c8.py` — FAKQuant / INT8_DYNAMIC / C8KVCache

**工具函数**
- `vllm_ascend/quantization/utils.py` — auto-detection（行 77），maybe_trans_nz（行 166），enable_fa_quant（行 202）
- `vllm_ascend/utils.py` — ACL_FORMAT_FRACTAL_NZ=29，maybe_trans_nz
- `vllm_ascend/envs.py` — 环境变量定义
- `vllm_ascend/device/mxfp_compat.py` — MXFP dtype 兼容性 shim

**310P 特定**
- `vllm_ascend/_310p/quantization/modelslim_config.py` — AscendModelSlimConfig310，310P scheme 注册覆盖

---

## 8. 文档交叉引用

- **MC2 通信策略**：见 inference-tech-notes 仓库 `vllm-ascend/vLLM-Ascend MC2通信策略详解.md`
- **FRACTAL_NZ 格式**：见 inference-tech-notes 仓库 `vllm-ascend/vLLM-Ascend NZ私有格式详解.md`
- **vLLM-Ascend 版本配套**：见 inference-tech-notes 仓库 `vllm-ascend/vLLM-Ascend版本配套关系详解.md`

---

## 9. 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v0.1 | 2026-04-08 | 初稿，基于代码调研生成 |

---

*文档版本：v0.1 | 写作时间：2026-04-08 | 源码版本：vllm-ascend branch 408*

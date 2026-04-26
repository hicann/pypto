---
name: pypto-fused-op-integration
description: PyPTO算子整网集成工作流。打点采集真实tensor→Golden验证→算子开发→模型集成→端到端验证。触发词：算子融合、整网集成、replace small ops、模型算子替换（GLM、LLaMA、Qwen、MoE、Attention等）。
---

# PyPTO 融合算子整网集成 Skill

将 PyPTO 融合算子替换到整网中，替代原始算子实现的完整工作流程。

---

## 工作流程概览

**阶段一：前置准备** → 环境验证 + 网络基线 ★ + 智能推荐
**阶段二：理解验证** → 打点采集 ★ + Golden编写（场景区分） + 场景验证 ⚠️
**阶段三：算子开发** → 设计方案 + 实现 + 单算子验证 ★
**阶段四：模型集成** → 目录结构 + 适配层 + 调用逻辑 + 缓存处理 ★
**阶段五：验证与提交** → 端到端验证（必须） + 性能调优 [可选] + 提交

> **★ 标注：** 关键步骤，必须完成  
> **⚠️ 标注：** 需特别注意  
> **[可选]：** 后续迭代进行

---

## 详细步骤指南

### 阶段一：前置准备

#### 步骤 1：前置验证（环境+网络基线）★

**目标：** 确认环境和网络可运行，作为后续工作基础。

**环境验证：**
- 推荐 Skill：`pypto-environment-setup`
- 检查 NPU 状态：`npu-smi info`

**网络基线验证：**
- 检查模型路径和文件完整性
- 执行完整推理，确认输出正常（无乱码/NaN/Inf）
- **关键成功标志：生成对 Prompt 的通顺回答**

> **为什么需要？** 网络必须可运行，否则后续工作基础不稳定。

**找不到运行方式时，必须询问用户：** 入口脚本路径、环境变量、模型路径。

**验证检查点：**
- ✅ NPU 驱动正常
- ✅ 模型可正常加载和推理
- ✅ 输出为自然语言（非乱码）

---

#### 步骤 2：需求分析（智能推荐）

**目标：** 分析网络可融合部分，推荐给用户确认。

**操作：**
1. **分析网络结构**：阅读模型代码，识别算子组合模式（Attention、MoE、FFN等）
2. **匹配 pypto 现有算子**：搜索 `models/` 目录下的实现案例
3. **推荐融合点**：
   
   向用户推荐候选融合点（示例格式）：
   ```
   发现可融合算子组合：
   | 位置 | 原始实现 | pypto算子 | 收益 |
   | Attention层 | Q/K/V投影+Softmax+Output | pypto.flash_attention | 减少3次matmul |
   | FFN层 | Gate+Up+SwiGLU+Down | pypto.swiglu_ffn | 减少中间存储 |
   
   请确认目标算子。
   ```
4. **无合适推荐**：直接询问用户融合位置和目标

**推荐 Skill：** `pypto-api-explore`（探索 pypto API）

---

### 阶段二：理解验证 ⚠️

> **为什么需要？** 推测的计算逻辑需 Golden 验证确认。

#### 步骤 3：打点采集真实 Tensor 信息 ★

**目标：** 从原始网络采集真实 shape/dtype，构造必须 pass 的测试用例。

**操作：**
1. 定位打点位置 → 插入打印代码：
   ```python
   print(f"[DEBUG] input: shape={x.shape}, dtype={x.dtype}")
   ```
2. 运行原始网络采集数据
3. 创建 `test_cases.json` 记录结果

**采集流程参考：** `references/test-cases-template.md`  
**测试格式参考：** `pypto-op-develop/templates/test-template.py`

**输出物：** test_cases.json、test_{op}.py

**关键原则：** 真实用例是必须 pass 的基准，覆盖所有调用场景。

---

#### 步骤 4：编写 Golden 脚本（场景区分）

**目标：** 编写 PyTorch 参考实现，精度对比基准。

**场景判断：** 检查**被替换逻辑**是否使用 torch_npu 融合算子（只看逻辑本身，不看文件import）。

- **场景A**：只使用基础算子（torch.matmul、torch.softmax等）→ 直接复制原始代码
- **场景B**：使用融合算子（torch_npu.contrib.flash_attention等）⚠️ 需用torch重写并验证

---

**场景A：未引用 torch_npu**

**策略：** 直接复制原始代码作为 Golden（无需理解，无偏差风险）

```python
# 直接复制原始实现到 xxx_golden.py
def xxx_golden(...):
    # 从网络代码复制原始逻辑
    return original_impl(...)
```

---

**场景B：引用 torch_npu**

**策略：** 用理解编写纯 torch 等价实现

```python
def xxx_golden(query, key, value, ...):
    # 理解 torch_npu 算子语义，用 torch 实现等价计算
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    ...
    return output
```

---

**通用要点：**
- 纯 PyTorch，禁止引入 pypto/torch_npu
- 导出 `{op}_golden()` 函数
- 独立 `{op}_golden.py` 文件

**推荐 Skill：** `pypto-golden-generate`

---

#### 步骤 5：验证理解正确性（分场景验证）

**目标：** 验证 Golden 与原始实现等价。

---

**场景A：未引用 torch_npu**

**无需验证**：Golden = 原始代码，直接进入步骤7。

---

**场景B：引用 torch_npu ★ 必须验证**

验证流程：
1. **构造真实测试用例**（来自步骤3采集的 shape/dtype）
2. **对比 torch Golden 与 torch_npu 原始实现**：
   ```python
   output_npu = torch_npu.flash_attention(query, key, value)
   output_golden = xxx_golden(query, key, value)
   diff = torch.abs(output_npu - output_golden).max().item()
   assert diff < 1e-3
   ```
3. **一致后替换 Golden 到整网**，验证输出正确
4. **全部通过** → 说明理解正确，进入步骤7

---

**验证检查点：**
- ✅ Golden 与原始实现一致（场景B必查）
- ✅ 整网替换后输出正常
- ✅ 无 NaN/Inf

**详细步骤参考：** `references/golden-verification.md`

---

#### 步骤 6：决策与迭代

- **通过 ✅**：
  - 场景A：无需验证 → 进入步骤7
  - 场景B：torch Golden 与 torch_npu 一致 + 整网验证通过 → 进入步骤7
- **失败 ❌**：
  - 场景A：不存在（Golden = 原始代码）
  - 场景B：重新理解融合算子语义，修正 Golden

---

### 阶段三：算子开发

#### 步骤 7：设计方案

**目标：** 设计 PyPTO 实现方案（API 映射、Tiling 策略）。

**推荐 Skill：** `pypto-op-design`

---

#### 步骤 8：算子实现

**目标：** 编写 PyPTO 算子代码。

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 9：单算子验证

**目标：** 验证 PyPTO 实现正确性。

**关键说明：**
- 步骤 3 采集的真实用例是必须 pass 的基准
- 输出无 NaN/Inf，与 Golden 对齐（diff < 2e-3）

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 10：调整目录结构

**目标：** 创建 PyPTO 算子库目录结构。

**典型结构：**
```
xxx_pto_kernels/
├── __init__.py      # 开关定义
├── xxx_impl.py      # PyPTO kernel
└── utils/
    └── xxx_golden.py
```

---

#### 步骤 11：配置适配层

**目标：** 封装 PyPTO 算子调用。

**关键要点：**
- 使用开关变量切换 PyPTO 和原始实现
- 适配层只负责参数转换和桥接

---

#### 步骤 12：修改模型调用逻辑 + 处理缓存 ★

**目标：** 替换原始算子调用，处理 transformers 缓存。

**修改调用逻辑：**
```python
if xxx_pto_kernels.USE_PTO:
    output = xxx_pto_kernels.xxx_wrapper(...)
else:
    output = self.original_op(...)  # fallback
```

**处理 transformers 缓存：**

适用场景：transformers 内置模型 + `trust_remote_code=True`

快速诊断：
```python
print(f"[Debug] __file__ = {__file__}")  # 缓存路径则有问题
```

**完整解决方案参考：** `references/cache-sync-template.md`

**验证检查点：**
- ✅ 本地修改自动生效
- ✅ 缓存目录存在算子库

---

### 阶段五：验证与提交

#### 步骤 13：验证与排查

**目标：** 端到端精度验证 + 问题排查。

**精度验证：**
- 运行整网推理，确认输出正常
- 与原始实现对比

**问题排查：**
- 参考 Skill：`pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debug`

---

#### 步骤 14：性能调优 [可选]

**目标：** 采集性能数据，分析瓶颈。

> **优先级：** 精度验证（必须） > 性能优化（可选）

**操作：**
- 采集泳道图、timeline
- 分析瓶颈（KV组装、MatMul、循环开销）
- 优化 Tile 配置、合图策略

**推荐 Skill：** `pypto-op-perf-tune`

---

#### 步骤 15：提交与文档

**目标：** 创建 Issue 和 PR。

**操作：**
1. 创建 Issue 跟踪变更
2. 提交 PR（含修改说明）

**推荐 Skill：** `pypto-issue-creator`、`pypto-pr-creator`

---

## 相关资源

### 参考模板
- **测试集模板**：`references/test-cases-template.md`
- **缓存处理**：`references/cache-sync-template.md`
- **Golden验证**：`references/golden-verification.md`

### 相关 Skill
- `pypto-environment-setup`：环境安装
- `pypto-intent-understand`：需求理解
- `pypto-golden-generate`：Golden 生成
- `pypto-op-design`：设计方案
- `pypto-op-develop`：算子实现
- `pypto-precision-compare`：精度对比
- `pypto-precision-debug`：精度调试
- `pypto-op-perf-tune`：性能调优
- `pypto-aicore-error-locator`：aicore 错误定位
- `pypto-host-stacktrace-analyzer`：堆栈分析
- `pypto-issue-creator`：创建 Issue
- `pypto-pr-creator`：创建 PR

---

**Skill 版本：** v2.0
**最后更新：** 2026-04-25
**维护者：** PyPTO Team
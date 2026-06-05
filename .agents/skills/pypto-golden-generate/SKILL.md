---
name: pypto-golden-generate
description: 当需要生成 golden 参考实现时使用此 skill。基于算子规格信息，生成 torch + torch_npu NPU 参考实现 `{op}_golden.py`，导出 `{op}_golden()` 函数，作为精度验证基准。计算在 NPU 上执行；torch_npu 未安装时直接报错引导安装，仅无 NPU 硬件时回退 CPU。触发词：生成 golden、生成参考实现、写 golden 函数、golden script、golden reference、reference implementation、generate golden、torch 参考、验证基准、baseline implementation、写验证代码、'帮我写 golden'、golden.py、参考代码。
---

# PyPTO Golden 参考实现生成（NPU）

基于算子规格信息，自动生成 torch + torch_npu NPU golden 参考实现及完整验证代码。计算的 golden 脚本用于开发阶段快速验证算子实现的正确性，可以作为独立模块被 `test_{op}.py` 等其他脚本导入调用。基于固定模板 [templates/golden-template.py](templates/golden-template.py) 生成（在§9 生成文件结构阶段读取该模板）。使用 torch 标准操作在 NPU 上执行；torch_npu 未安装时直接报错引导安装，仅无 NPU 硬件（`device_count() == 0`）时回退 CPU。禁止引入 pypto。性能采集不写入 golden 文件，统一由通用脚本 [scripts/profile_golden.py](scripts/profile_golden.py) 执行。

1. 从用户输入提取算子名称、公式、输入输出规格等必要信息
2. 如果信息不足，向用户逐步提问补充
3. 按工作流执行 golden 函数生成和验证
4. 输出 `{op}_golden.py` 到当前目录或用户指定位置

## 1. 所需信息

| 项目 | 说明 |
|------|------|
| **输入** | 算子规格信息（如结构化规格内容、自然语言描述等） |
| **输出** | `{op}_golden.py`，导出 `{op}_golden()` 函数，路径由调用者决定 |

---

## 2. 算子信息获取

从输入中提取算子名称，或由调用者指定。如果信息不足，向用户逐步提问补充。

---

## 3. 规格字段检查

读取算子规格信息后，按以下分类检查字段完整性：

### 必须字段（缺失则报错退出）

| 字段 | 位置 | 用途 |
|------|------|------|
| 算子名称 | §1 基础信息 | 文件名、函数名 |
| 数学公式 | §1 基础信息 | 生成 PyTorch 实现 |
| 输入规格 | §4 数据规格 | 函数参数、验证 shape |
| 输出规格 | §4 数据规格 | 返回类型、验证 shape |

### 建议字段（缺失时引导补充）

| 字段 | 位置 | 用途 | 缺失时处理 |
|------|------|------|------------|
| 典型配置 | §11 应用场景 | 典型 case 验证 | 引导用户补充到规格信息中 |
| 动态轴范围 | §7 动态轴说明 | 泛化 case 采样 | 使用默认范围 (1-1024) |

### 典型配置缺失时的处理

```
检查规格信息中是否包含典型配置
    │
    ├── 有 → 直接使用
    │
    └── 无 → 引导用户提供
            │
            ├── 用户提供 → 补充到规格信息中，继续生成
            │
            └── 用户跳过 → 根据动态轴范围生成默认配置
                          │
                          ├── 有动态轴范围 → 按范围推荐
                          │   └── 补充到规格信息中，继续生成
                          └── 无动态轴范围 → 使用通用默认值
                              └── 补充到规格信息中，继续生成
```

典型配置采用 7 列格式：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|

---

## 4. Golden 函数生成规范

### 实现方式

使用 **PyTorch + torch_npu** 实现 golden 函数，计算在 NPU 上执行。优先使用 PyTorch 内置 API，在没有直接对应 API 时由 LLM 基于公式生成实现。输入自动转移到 NPU device，返回 NPU tensor。torch_npu 未安装时直接报错引导安装；仅无 NPU 硬件（`device_count() == 0`）时回退 CPU。

**设备卡号（device ID）**：如果用户指定了 NPU 卡号（如"在卡 3 上运行"），必须严格使用用户指定的卡号，通过 `torch.device("npu:<id>")` 指定。模板中 `_get_device()` 已支持通过环境变量 `ASCEND_DEVICE_ID` 或函数参数覆盖默认卡号，生成 golden 时应保留该机制，不要硬编码为 `torch.device("npu")`（等价于卡 0）。

### 函数签名

```python
# 单输入
def silu_golden(x: torch.Tensor) -> torch.Tensor:

# 多输入
def swiglu_golden(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:

# 带可选参数
def layer_norm_golden(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
```

**规则**：
- 函数名：`{算子名}_golden`
- 参数顺序：必要张量参数在前，可选参数在后
- 所有 spec 中定义的参数都要实现，包括可选参数
- dtype 不作为参数，计算精度跟随输入张量

### 边界条件映射

根据规格信息中的边界条件定义生成对应代码逻辑：

```python
# spec 定义：零值返回 1
def safe_div_golden(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x / y
    result[y == 0] = 1.0  # 映射边界条件
    return result
```

---

## 5. 置信度系统

根据实现方式评估置信度，影响最终输出的标注和提示强度：

```
1. 是否有等效 PyTorch API？
   ├─ 有 → 直接调用 → ⭐⭐⭐⭐⭐
   │
   └─ 无 → 2. 是否为已知论文算子？
            ├─ 是 → 搜索第三方实现 → ⭐⭐⭐⭐
            │
            └─ 否 → 3. LLM 智能转换
                     ├─ 简单公式转换 → ⭐⭐⭐⭐
                     └─ 复杂公式转换 → ⭐⭐⭐
```

**低置信度加强提示**（⭐⭐⭐ 及以下）：

在输出文件的 docstring 中添加醒目警告，提醒人工审查代码逻辑、与论文对比验证、使用多组数据验证边界情况。

置信度和验证是独立机制——置信度只影响标注和提示强度，验证标准对所有算子一致。自动修复后根据最终代码更新置信度。

---

## 6. 验证机制

### 验证执行方式

生成文件后，**必须通过直接执行脚本完成验证**（在 NPU 上运行）：

```bash
python3 {op}_golden.py
```

脚本内含 `if __name__ == "__main__": _validate()` 入口，会自动运行全部检查项（典型 case、泛化 case、值域、数值稳定性等）并输出验证报告。验证在 NPU 上执行；torch_npu 未安装时报错引导安装，仅无 NPU 硬件时回退 CPU。

**禁止**使用以下方式替代直接执行：
- `exec(open(...).read())` — 绕过脚本的独立执行环境
- 手动构造测试数据单独调用 golden 函数 — 与脚本内置验证逻辑重复且不完整

**门禁判定依据**：以 `python3 {op}_golden.py` 的 exit code（0 = 通过）和验证报告输出为准。

**device 约定**：golden 函数内部对输入执行 `.to(device)` 做设备迁移。为避免跨设备 mismatch，验证代码创建张量时必须直接指定 `device=device`（如 `torch.randn(..., device=device)`），不要依赖 golden 内部的隐式 `.to()`。

**device ID 指定**：若用户明确要求在特定 NPU 卡号上运行验证，必须通过环境变量 `ASCEND_DEVICE_ID=<id>` 传入，或在验证代码中直接使用 `torch.device("npu:<id>")`。禁止忽略用户指定的卡号而始终使用默认卡 0。

### 验证 shape 来源

**两层来源**：

```
├── 典型 case（来自算子规格中的典型配置）
│   ├── 性能类型：按优先级验证，P0 必须通过
│   └── 功能类型：按优先级验证，P0 必须通过
│
└── 泛化 case（来自算子规格中的动态轴取值范围）
    └── 每个动态轴采样：最小值、中间值、最大值
        如 b: 1-128 → [1, 64, 128]
           s: 64-2048 → [64, 1024, 2048]
        组合生成 shape
```

### 验证顺序（按重要性）

1. 性能类型 P0 配置（最高优先级，必须通过）
2. 性能类型 P1 配置
3. 功能类型 P0 配置（必须通过）
4. 功能类型 P1 配置
5. 泛化 case（边界+中间采样）
6. 其他低优先级配置

### 验证检查项

| 检查项 | 说明 | 级别 |
|--------|------|------|
| 语法检查 | 代码能正常 import | 🔴 严重 |
| 形状一致性 | 输出 shape 与 spec 定义一致 | 🔴 严重 |
| 函数签名 | 参数与 spec 定义匹配 | 🔴 严重 |
| 值域检查 | 从公式推导值域约束 | 🟡 警告 |
| 特殊点验证 | 边界值、零值等 | 🟡 警告 |
| 数值稳定性 | 无 NaN/Inf | 🟡 警告 |
| 数学属性 | 单调性、对称性、守恒量等 | 🟡 警告 |
| API 对比 | 与 PyTorch API 对比（如适用） | 🟢 信息 |

### 从公式推导的验证属性

**值域约束**：根据公式特性推导输出值域。例如 sigmoid 的输出在 (0, 1)、ReLU 的输出 >= 0、softmax 的输出在 (0, 1) 且沿归约轴和为 1。

**数学属性**：检查奇偶性（tanh 是奇函数）、单调性（sigmoid 单调增）、守恒量（softmax 和为 1）等。

**特殊点验证**：验证关键输入值的输出，如 sigmoid(0) = 0.5、tanh(0) = 0、relu(0) = 0。

---

## 7. 自动修复策略

验证失败时，按优先级自动尝试修复：

1. **使用 PyTorch 稳定 API** — 将手写实现替换为 PyTorch 内置 API
   - `x / (1 + torch.exp(-x))` → `torch.sigmoid(x)`
   - `torch.exp(x) / torch.exp(x).sum()` → `torch.softmax(x, dim=-1)`

2. **添加数值稳定性保护** — 防止除零、log 负数等
   - `a / b` → `a / (b + 1e-8)`
   - `torch.log(x)` → `torch.log(torch.clamp(x, min=1e-8))`

3. **使用更稳定的等价形式** — 数学等价但数值更稳定的替代

**限制**：最多 3 次修复尝试。修复后根据最终代码更新置信度。

---

## 8. 错误分级与输出控制

| 级别 | 含义 | 修复失败后处理 |
|------|------|----------------|
| 🔴 严重 | 语法错误、shape 不匹配、签名错误 | **阻止输出**，报告错误原因 |
| 🟡 警告 | 值域检查失败、数值不稳定 | 输出文件 + 在 docstring 中标注警告 |
| 🟢 通过 | 所有检查通过 | 正常输出 |

---

## 9. 生成文件结构

生成的文件遵循固定模板 [`templates/golden-template.py`](templates/golden-template.py)，在生成时读取模板并将占位符替换为实际算子内容。模板包含以下结构：

- `import torch` + `import torch_npu` + NPU 设备初始化
- 文件级 docstring（算子名、公式、置信度）
- `{op}_golden()` 函数：torch + torch_npu 参考实现，计算在 NPU 上执行（含示例注释）
- `_validate()` 函数：自动验证（典型 case、泛化 case、值域检查、数值稳定性、API 对比）
- `if __name__ == "__main__": _validate()` 入口

性能采集相关代码不进入 `{op}_golden.py`，统一使用 [scripts/profile_golden.py](scripts/profile_golden.py)，保证 golden 文件只包含可在 NPU 上运行的 PyTorch 参考实现和验证逻辑。

---

## 10. 用户交互

全自动生成与验证，仅在以下场景需要用户参与：

| 场景 | 交互方式 |
|------|----------|
| 生成代码、验证、修复 | 全自动，无需用户参与 |
| 典型配置缺失 | 引导用户提供或确认推荐配置 |
| 覆盖已有文件 | 通过 `AskUserQuestion` 询问确认 |

---

## 11. 异常处理

| 场景 | 处理方式 |
|------|----------|
| 缺少算子规格信息 | 报错退出，提示先补齐需求信息 |
| 规格信息解析失败 | 报错退出，提示输入格式不正确 |
| 规格信息缺少必须字段 | 列出缺失字段，引导用户补充 |
| 多个算子未指定 | 列出所有可用算子，要求用户指定 |
| golden 文件已存在 | 通过 `AskUserQuestion` 询问是否覆盖 |

---

## 12. 验证报告（必须执行）

文件生成完成后，向用户展示验证结果，示例：

```
✅ Golden 参考实现已生成:
  • {name}_golden.py

============================================================
attention_golden 验证报告
============================================================

[典型 case 验证]
  性能_P0: b=2,h=8,s=512,d=64,w=128 ... ✓ PASS
  功能_P0: b=1,h=4,s=256,d=64,w=128 ... ✓ PASS

[泛化 case 验证]
  b=1,h=4,s=128,d=32,w=64 ... ✓ PASS
  b=64,h=8,s=1024,d=64,w=128 ... ✓ PASS

[值域检查]
  检查 softmax 归一化 ... ✓ PASS

[数值稳定性检查]
  大值输入 (x=100) ... ✓ PASS
  小窗口 (w=4) ... ✓ PASS

[功能正确性检查]
  验证窗口边界 ... ✓ PASS

============================================================
✅ 所有验证通过
============================================================
```

---

## 13. 既有参考的 PyPTO 友好化（Reference Normalization Path）

> **适用场景**：用户提供了已有的 PyTorch / NumPy 参考实现，需要将其规范化为
> PyPTO 友好的 golden。当用户没有提供参考实现而是直接通过规格信息生成 golden
> 时，走 §1-§12 的从规格生成路径即可，本节不适用。

### 13.1 选择最强参考

参考实现选择优先级：

1. PyTorch forward/backward 参考
2. NumPy 参考
3. 仅当无任何已有参考时，自行编写数学参考

在编写新参考前，优先搜索现有实现：

```bash
grep -rn "<operator name>" examples/ custom/ models/ docs/zh/api/operation/
```

### 13.2 审计参考实现的 PyPTO 不友好模式

主动扫描以下不友好模式：

- 隐式多轴广播
- 不透明的库操作
- 复杂的复合调用
- 隐藏的 layout 变化
- 必须显式化的控制流
- 4D/5D 操作（PyPTO 中可能脆弱）
- 不能干净映射到 tile_fwk IR 的 host-side 便利

对每个可疑模式，直接读取相应 op 文档：

```bash
cat docs/zh/api/operation/pypto-<op>.md
```

### 13.3 规范化 golden

将参考改写为 PyPTO 友好的 golden。规范化强度取决于 Stage 3（DESIGN.md §0.3）
选择的分解路径：

> 路径在该步骤之后由 architect 通过 `count_golden_lines.py` 决定。但可以基于算法类型预判：
> - 纯逐元素 / softmax / layernorm / 标准 attention / FlashAttention forward → 可能 **L0 路径**（轻量规范化）
> - 多状态递归（gated_delta_rule / mamba）/ 复杂算子 backward → 可能 **L1 路径**（完整规范化）
>
> 不确定时，默认 **L1（完整规范化）**——它是严格超集，architect 在 Stage 3 落到 L0 时可以忽略边界标记。

**通用规则（L0 与 L1 均适用）**：

- 保留语义，不保留源码语法
- 显式化所有 shape
- 显式化 dtype 转换
- 隐式 broadcast 链改写为单轴形式
- 窄向量 / 奇怪 layout 改写为对齐友好的表示
- **禁用 `.T` / `.t()`**：使用 `torch.transpose(t, dim0, dim1)`。对 matmul `a @ b.T`，
  写为 `torch.matmul(a, b.transpose(-2, -1))` 并注释 `# a @ b^T → pypto: b_trans=True`
- 在每个中间 tensor 上加 shape 注释 `# [B, H, T, K]`

**L1 路径专用（`module_count ≥ 2`）**：

- 在每个未来 module 边界处给中间 tensor 起有意义的命名
- 标记语义 module 边界（用 `# --- Module M1: <role> ---` 注释）——这些会成为 DESIGN.md §0.5 中的 breakpoint

**L0 路径专用（`module_count == 1`）**：

- 中间命名 + `# --- Module M1 ---` 标记**不要求**（kernel 是一个块）
- golden 可以保留单个高层调用（例：`out = torch.softmax(x, dim=-1)`），**除非** Golden function inventory（§13.5）因 shape 变换追踪需要而要求展开

### 13.4 Full vs Tiled 实现策略

规范化 golden 可采用两种等价策略：

**策略 1: Full computation（默认）**

一次性处理整个 input tensor。最简单直接。

```python
def attention_golden(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

**策略 2: Tiled computation（可选，复杂 kernel 推荐）**

将输入切成小 tile，每个 tile 独立计算，然后拼接 / 累积。该模式：

- 映射 PyPTO kernel 的实际执行方式（tile-by-tile）
- 允许早期验证边界处理、padding、accumulator 逻辑
- 在 PyPTO 实现前暴露 tile-size 对数值精度的影响
- 对有 tiling 结构的 kernel（window attention、blockwise matmul、FlashAttention）是必要的

例（按 batch tiled）：

```python
def attention_golden_tiled(q, k, v, window_size=None):
    """Tiled attention golden (matches PyPTO kernel tile-by-tile execution)."""
    outputs = []
    for b in range(q.shape[0]):
        q_tile = q[b:b+1, ...]
        k_tile = k[b:b+1, ...]
        v_tile = v[b:b+1, ...]
        scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        out_tile = torch.matmul(probs, v_tile)
        outputs.append(out_tile)
    return torch.cat(outputs, dim=0)
```

**何时选 Tiled**：

- Kernel spec 明确描述 tiling 或 loop-based 计算
- 算法涉及 split / partial results / state accumulation
- 需要在完整 PyPTO 实现前验证 tile 边界的 edge case

**两种策略必须产生相同的数值结果**（在浮点容差范围内）。若两者都实现，在
`{op}_golden.py` 中包含两者并在验证套件中验证等价性。

### 13.5 构建 Golden function inventory（强制）

规范化 golden 写完后，在 `custom/<op>/MEMORY.md` → **Golden function inventory**
中列出每个数学操作：

```
| # | Golden operation          | Shape transformation              | PyPTO implementation | Line | Status |
|---|---------------------------|-----------------------------------|----------------------|------|--------|
| 1 | matmul(q, k^T)            | [B,H,T,K]@[B,H,K,T]->[B,H,T,T]    | pypto.matmul(...)    | L.42 | ✅     |
| 2 | softmax(scores, dim=-1)   | [B,H,T,T]->[B,H,T,T]              |                      |      | ❌     |
```

**门禁**：inventory 不存在不得进入 module 设计阶段。

### 13.6 用原始 golden 验证规范化 golden

总是用以下条件验证：

- 同一 random seed
- 小 shape
- 代表性 shape
- 边界 / edge shape
- dtype-aware 比较
- NaN / Inf 检查
- `assert_allclose` 使用要求的 tolerance policy

若规范化 golden 不匹配，**停止并修复**。不要开始 PyPTO 实现。

### 13.7 Freeze 规范化 golden

规范化 golden 匹配后：

- 在 memory 中标记 frozen
- 作为后续单一参考
- 除非有证据表明规范化本身错误，否则**不**改动

---

## 14. 升级路径（仅在 pre-check 触发约束时只读查阅）

当 golden 在 pre-check 阶段触发归约（reduction）对齐或 matmul 相关约束时，只读查阅
`pypto-general-debug` 中 `references/DEBUG_GUIDEBOOK.md` 的归约对齐与 matmul 章节即可，
**不要** fork 到 debug 子流程。本 skill 只生成纯 torch 的 golden，调试流程归 debugger 拥有。

---

## 15. NPU 性能 Profiling（验证通过后执行）

### 说明

验证通过后，使用通用脚本 `scripts/profile_golden.py` 调用 `{op}_golden.py`，并通过 `torch_npu.profiler` 采集 golden 算子在 NPU 上的性能数据。golden 文件本身不包含 profiling 代码。

### 执行方式

```bash
python3 .agents/skills/pypto-golden-generate/scripts/profile_golden.py \
  custom/{op}/{op}_golden.py \
  --function {op}_golden \
  --input x:8x1024x4096:float32 \
  --device <DEVICE_ID>   # 可选，指定 NPU 卡号；用户有要求时必须填入
```

参数说明：

- `--function`：golden 函数名；缺省时脚本会优先使用文件名同名函数，或自动寻找唯一的 `*_golden` 函数
- `--input NAME:SHAPE[:DTYPE]`：tensor 输入规格，可重复；多输入算子必须为每个 tensor 参数传入一条
- `--arg NAME=JSON`：非 tensor 参数，可重复，如 `--arg eps=1e-5`、`--arg normalized_shape=[4096]`
- `--device DEVICE_ID`：NPU 卡号（整数），默认为 0；若用户指定了卡号，必须传入对应值，如 `--device 3`
- `--warmup` / `--iters`：默认分别为 5 / 5

如果未提供 `--input`，脚本会按历史默认值为第一个必需参数生成一个 `(8, 1024, 4096)`、`torch.float32` 的 NPU tensor。

### Profiling 流程

1. **由 `profile_golden.py` 预分配输入 tensor**（避免 `randn` 开销混入 kernel 计时）
2. **Warmup 5 轮 + 实测 5 轮**（profiler 模式），排除 JIT 编译和首次调度开销
3. 用 `torch_npu.profiler` 记录 NPU + CPU 事件；Profiler 配置与 PyPTO kernel profiling 保持一致
4. 输出两种数据：
   - CANN 分析数据（`operator_memory.csv`、`memory_record.csv`、设备端原始数据）→ `prof/{op}_golden/`，由 `on_trace_ready` 回调生成
   - Chrome Trace JSON（`trace_view.json`）→ `prof/{op}_golden/`，由 `tensorboard_trace_handler` 导出
5. 解析 Chrome Trace：
   - 按 `SynchronizeDevice` 边界将 AICore kernel 事件分组，每轮 golden 调用对应一组
   - 过滤无效分组（总耗时过小或 kernel 数过少）和离群点
   - 计算 `aicore_e2e`（平均 kernel 耗时）、`aicore_e2e_jitter`（后期抖动）、`aicpukernel_gap`
6. 写入 GOLDEN_PERF_REPORT.md：op 级耗时表 + Golden Kernel Performance 章节

### Profiling 配置

| 项目 | 配置 |
|------|------|
| Profiler 等级 | `ProfilerLevel.Level1` |
| AiC Metrics | `AiCMetrics.PipeUtilization` |
| Activities | NPU + CPU（缺一不可，否则 CANN 分析数据为空） |
| Warmup 轮数 | 5 |
| 实测轮数 | 5 |
| 输入 shape | 取自 spec 典型配置；若无则使用默认值 |
| 输出目录 | `custom/{op}/prof/{op}_golden/` + `custom/{op}/GOLDEN_PERF_REPORT.md` |
| 执行模型 | 直接调用 `{op}_golden()`；不使用 `NPUGraph` replay，不注入与 golden 无关的额外 kernel |

### 产物

| 产物 | 位置 | 说明 |
|------|------|------|
| `GOLDEN_PERF_REPORT.md` | `custom/{op}/` | 可直接阅读的性能报告（op 级耗时、device、shape、时间戳） |
| `trace_view.json` | `custom/{op}/prof/{op}_golden/` | Chrome Trace 格式，可导入 chrome://tracing |
| `operator_memory.csv` | 同上 | 每次调用的内存分配量+分配时长 |
| `memory_record.csv` | 同上 | 内存分配/释放事件记录 |
| 设备端原始数据 | 同上 | ffts/hbm/llc/stars_soc 等硬件计数器 |

### GOLDEN_PERF_REPORT.md 示例

```markdown
# silu Golden NPU Performance Report

- **Device**: npu
- **Input Shape**: (8, 1024, 4096)
- **dtype**: torch.float32
- **Timestamp**: 2026-05-19T19:53:36
- **Profiling Data**: `prof/silu_golden/`

## Op Performance

| op | warmup_avg | stable_avg | stable_min | stable_max |
|----|-----------|-----------|-----------|-----------|
| aten::mul | 504.0us | 12.1us | 9.2us | 29.8us |
| aten::sigmoid | 16066.3us | 13.9us | 12.2us | 24.1us |
| aclnnSigmoid | 6.6us | 4.3us | 3.7us | 5.4us |
| ...

## Golden Kernel Performance (from profiler trace)

Per-call total AICore kernel time, grouped by `SynchronizeDevice`
boundaries in `trace_view.json`.

| metric | warmup_avg | stable_avg | stable_min | stable_max |
|--------|-----------|-----------|-----------|-----------|
| silu_golden (kernel E2E) | 234.5us | 45.2us | 42.1us | 51.3us |

## Notes
- `warmup_avg`: first 5 iterations (includes JIT/compile overhead)
- `stable_avg/min/max`: subsequent iterations (steady-state kernel execution)
- Kernel E2E extracted from profiler trace, not host-side wall-clock
- Full trace: `prof/silu_golden/trace_view.json` (chrome://tracing)
- CANN analysis: `prof/silu_golden/` (operator_memory.csv, device counters)
```

### 注意事项

- 仅 NPU 环境下生效；无 NPU 硬件时 `profile_golden.py` 跳过采集
- 若 torch_npu 未安装，profiling 阶段返回 `npu_import_error`，引导调用 `pypto-environment-setup`
- golden 文件只负责 `{op}_golden()` 和 `_validate()`，不包含 profiling 逻辑
- `profile_golden.py` 会按 `--input` 生成 floating/complex/int/bool tensor；需要特殊数值分布的算子必须在命令中补齐必要 `--arg`，或在 golden 自验证中覆盖该分布
- Chrome Trace JSON 末尾可能被 CANN profiler 截断（缺闭合 `]`），`profile_golden.py` 内置修复逻辑
- **Kernel E2E** 从 profiler trace 中提取：按 `SynchronizeDevice` 边界将 AICore kernel 事件分组，逐轮计算总耗时，过滤无效分组和离群点后取平均，作为 golden 算子在 AICore 上的 kernel 执行时间基准

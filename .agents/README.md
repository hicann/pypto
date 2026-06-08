# PyPTO Agent Team

预配置的 9 智能体团队，覆盖昇腾 NPU 算子从零到一的端到端开发，由显式的 Stage 1–7 状态机驱动，并由自动化 lint 门禁守护。

```
Stage 1 规划 → 2 算法 → 3 架构 → 4 设计
        → 5 构造 → 6 验证 → 7 调优
```

本仓库提供：

- **`AGENTS.md`** — 项目入口，stage、agent、规约的权威说明
- **`.opencode/agents/`** — 9 个 agent 定义（1 个 primary 编排者 + 8 个专职 sub-agent）
- **`.opencode/plugins/`** — 状态机插件与 OL lint 守卫（TypeScript）
- **`.agents/skills/`** — 43 个专家技能（编排、规划、golden、设计、构造、验证、调优、调试、Pass 分析、PR 流程、环境）
- **`.agents/hooks/pypto-op-lint/`** — Python lint 引擎：43 条规则、5 个维度，在每次文件写入与每次测试执行后自动运行

同时支持 [opencode](https://opencode.ai) 与 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)。

> **使用 Claude Code 的用户请先参考 [Claude Code 配置](#claude-code-配置)** 完成一次性迁移。opencode 用户无需任何额外配置。

---

## 快速上手 — 实际怎么跑起来

> **最关键的一步**：会话开始时，切换到 `pypto-op-orchestrator` agent。如果跳过这一步，默认 agent 不会调度 9 智能体团队，会自己包揽所有工作。

| 工具 | 切换到编排者的方法 |
|:---|:---|
| **opencode** | 仓库根目录启动 `opencode`，按 **`Tab`**，选 `pypto-op-orchestrator`。无需任何安装 — agent / skill / plugin / lint 全部自动发现。 |
| **Claude Code** | 需要一次性配置（见下方 [Claude Code 配置](#claude-code-配置)）。然后在仓库根目录运行 `claude --agent pypto-op-orchestrator`。 |

接着用自然语言描述算子（数学公式、规格文档或论文链接）。编排者会读取 `AGENTS.md`，加载操作手册（[`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md)），按 Stage 1–7 推进 — 在每个 stage 调度对应专职 agent，并在每次 Stage Stop 时跑 OL lint 引擎。

如果你跳过这一步，对默认 agent 直接说「帮我做一个 PyPTO 算子」，得到的是单 agent 答复，会绕过门禁、lint 引擎和状态机。**永远先切换到编排者。**

---

## 快速开始

切换到编排者之后，用以下任一方式描述目标。

**方式 1 — 数学公式**

```
我要开发一个名为 sinh 的算子。公式：(e^x - e^(-x)) / 2。
输入：shape 为 [b, s, n, d] 的 float32 tensor。输出 shape 相同。
精度：atol=2.5e-5, rtol=5e-3。
```

**方式 2 — 算子规格文档**

```
请根据 ./docs/my_operator_spec.md 中的方案文档开发对应的 PyPTO 算子。
```

**方式 3 — 算子论文**

```
请将 https://arxiv.org/abs/2205.14135 (Flash Attention) 中描述的算子实现为 PyPTO kernel。
```

编排者将自动按 Stage 1–7 推进。

---

## 9 智能体团队

| Agent | Mode | Stage | 职责 |
|:---|:---:|:---:|:---|
| [`pypto-op-orchestrator`](../.opencode/agents/pypto-op-orchestrator.md) | primary | 1–8 | 入口。推进 stage、强制门禁、调度 sub-agent。本身不直接做领域工作。 |
| [`pypto-op-planner`](../.opencode/agents/pypto-op-planner.md) | subagent | 1 | 将用户需求翻译为 `SPEC.md` + `API_REPORT.md`；初始化 `MEMORY.md`。 |
| [`pypto-op-mathematician`](../.opencode/agents/pypto-op-mathematician.md) | subagent | 2 | 产出 PyPTO 友好的 `<op>_golden.py` 参考实现与 Golden 函数清单。 |
| [`pypto-op-architect`](../.opencode/agents/pypto-op-architect.md) | subagent | 3 | 产出 `DESIGN.md`：tiling 策略、loop 结构、性能目标表。 |
| [`pypto-op-designer`](../.opencode/agents/pypto-op-designer.md) | subagent | 4 | 将 kernel 拆分为语义模块，定义 `module_interfaces.yaml`。 |
| [`pypto-op-coder`](../.opencode/agents/pypto-op-coder.md) | subagent | 5 | 每次调度只写一个 impl 文件。先 per-module 累计构建 (`modules/<op>_module<k>_impl.py`)，最后一个模块通过 verify 后做 cleanup 把累计 impl 整理成 `<op>_impl.py` 并写 `README.md`。从不写测试，从不调试。 |
| [`pypto-op-verifier`](../.opencode/agents/pypto-op-verifier.md) | subagent | 4–7 | 仅评判。运行 `detailed_tensor_compare`、布局检查、prefix-eval、回归检查。分类失败原因。从不调查、从不修复。 |
| [`pypto-op-debugger`](../.opencode/agents/pypto-op-debugger.md) | subagent | 5（按需） | 一次加载一个调试子技能，定位根因，给出补丁建议。补丁由 coder 应用。 |
| [`pypto-op-optimizer`](../.opencode/agents/pypto-op-optimizer.md) | subagent | 7 | 在精度冻结后做三阶段性能调优（frontend → swimlane → incore）。 |

Agent 之间通过两个产物交换信息：
- `custom/<op>/MEMORY.md` — 共享叙事（所有 agent 读写）
- `custom/<op>/.orchestrator_state.json` — 机器可读状态（仅编排者写入；由 lint 插件强制保护）

更多细节（职责边界、完成判据、信息隔离）见 [`AGENTS.md`](../AGENTS.md) 与操作手册 [skill `pypto-orchestration-manual` (SKILL.md auto-loads)](skills/pypto-orchestration-manual/SKILL.md)。

---

## Stage 1–7 工作流

| Stage | 名称 | Agent | 输入 | 产出 |
|:---:|:---|:---|:---|:---|
| 1 | Planning | planner | 用户需求 | `SPEC.md`, `API_REPORT.md` |
| 2 | Algorithm | mathematician | `SPEC.md` | `<op>_golden.py` |
| 3 | Architecture | architect | `<op>_golden.py` | `DESIGN.md` |
| 4 | Design | designer（+ verifier 搭脚手架） | `DESIGN.md` | `module_interfaces.yaml`、scaffolding |
| 5 | Construction | coder ↔ verifier ↔ debugger | `module_interfaces.yaml` | 每次 Phase M_k 产出一个 `_module<k>_impl.py`；最后 cleanup 阶段产出 `<op>_impl.py`、`test_<op>.py`、`README.md` |
| 6 | Verification | verifier | `<op>_impl.py` | 布局 / 结构 / 端到端 PASS/FAIL 裁决 |
| 7 | Optimization | optimizer + verifier | Stage 6 已通过 | 优化后的 impl |

### Stage 5 内部循环（per-module M_k）

```
coder 写模块 M_k
        │
        ▼
verifier 评判（detailed_tensor_compare + prefix-eval --up-to-module k + 布局检查）
        │
        ├── PASS ──► 下一个模块 M_{k+1}
        │
        └── FAIL ──► debugger 调查
                          │
                          ▼
                     在 MEMORY.md 中给出补丁建议
                          │
                          ▼
                     coder 应用补丁 ──► 回到 verifier
```

M_{k+1} 不能在 M_k 通过之前启动。

---

## Agent 跑出来的产物

每个算子在 `custom/<op>/` 目录下生成。下方是 Stage 1–7 完整跑通后的文件结构，已标注每个文件由哪个 stage 产出。

```
custom/<op>/
├─ MEMORY.md                              ← 共享叙事；所有 agent 读写（编排者在 S1 初始化）
├─ .orchestrator_state.json               ← 机器可读状态机（仅编排者写入）
│
├─ SPEC.md                                ← S1：结构化算子规格（公式、shape、dtype、容差）
├─ API_REPORT.md                          ← S1：PyPTO API 映射、不支持算子、规避方案
│
├─ <op>_golden.py                         ← S2：纯 PyTorch fp32 参考实现 + Golden 函数清单
│
├─ DESIGN.md                              ← S3：tiling 策略、loop 结构、数值稳定性档案、
│                                              Layers A–L、性能目标表
│
├─ module_interfaces.yaml                 ← S4：语义模块分解 + 每模块契约
│
├─ modules/
│  ├─ <op>_module<k>_golden.py            ← S4：每模块 torch golden（verifier 搭脚手架时建立）
│  ├─ <op>_module<k>_impl.py              ← S5：每模块 PyPTO impl（每次 Phase M_k 调度产一个）
│  └─ test_<op>_module<k>.py              ← S4：每模块测试（adversarial harness，verifier 编写）
│
├─ <op>_impl.py                           ← S5（cleanup）：集成 PyPTO impl（最终 kernel）
├─ test_<op>.py                           ← S5（cleanup）：端到端测试
├─ README.md                              ← S5（cleanup）：算子级 README（用法、配置、已知约束）
│
└─ eval/
   ├─ module_interfaces.yaml              ← S4：机器可读契约（lint OL49 交叉验证）
   ├─ adversarial_cases.json              ← S4：verifier 生成的边界用例
   ├─ evaluation_report.json              ← S5/S6：verifier 裁决（PASS/FAIL + failure_category）
   └─ prefix_eval_results.json            ← S5：prefix 评测（<op>_impl.py 至 module k）
```

**Stage 7 不会新增文件** — 它原地修改 `<op>_impl.py`。优化报告追加到 `MEMORY.md`；中间产物（swimlane / leafhash dump）按需生成在同级的 `<op>_perf/` 目录下。

| Stage | 新增文件 | 一句话说明 |
|:---:|:---|:---|
| 1 | `MEMORY.md`, `SPEC.md`, `API_REPORT.md` | 框定问题与 API 表面 |
| 2 | `<op>_golden.py` | kernel 必须匹配的位级精度参考 |
| 3 | `DESIGN.md` | 实现方案（tiling、loop、稳定性） |
| 4 | `module_interfaces.yaml`、`modules/<op>_module<k>_golden.py`、`modules/test_<op>_module<k>.py`、`eval/adversarial_cases.json` | 契约 + 脚手架，使每个模块能独立构造与评判 |
| 5 | `modules/<op>_module<k>_impl.py`、`eval/prefix_eval_results.json`；最后一个 M_k 通过后 cleanup：`<op>_impl.py`, `test_<op>.py`, `README.md` | 每模块 impl 累计构建；最后整理出最终 kernel + 端到端测试 |
| 6 | （无 — 仅做门禁） | 布局、结构性、端到端精度门禁 |
| 7 | （修改 `<op>_impl.py`） | 调优后的 kernel；性能报告写入 `MEMORY.md` |

Lint 引擎会把 `module_interfaces.yaml` 中的契约与 impl 做交叉验证（OL49）、强制 `_golden.py` / `_impl.py` / `test_*.py` 三文件分离（D3 规则）、并对每个 impl 强制 Layer A–L 结构（D1 规则）。任一不通过，对应的 Write/Edit 在工具边界即被拦截 — 详见下方 [Lint 与状态机](#lint-与状态机--自动门禁)。

---

## Skills（共 43 项）

每个 skill 是 `.agents/skills/<name>/` 下的一个目录，包含入口 `SKILL.md`，可选子目录 `references/`、`scripts/`、`templates/`。点击 skill 名跳转到对应 `SKILL.md`。

### 编排与共享状态

| Skill | 用途 |
|:---|:---|
| [`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md) | 编排者启动手册：principles / stage 计划 / 团队名册 / 强制规则 / 路由目录 |
| [`pypto-memory-template`](skills/pypto-memory-template/SKILL.md) | `custom/<op>/MEMORY.md` 的必填结构（章节、机器可读字段、更新节奏） |

### Stage 1–4 — 规划、算法、架构、设计

| Skill | 用途 |
|:---|:---|
| [`pypto-intent-understand`](skills/pypto-intent-understand/SKILL.md) | Stage 1：将自然语言算子需求转为结构化规格 |
| [`pypto-api-explore`](skills/pypto-api-explore/SKILL.md) | Stage 1：PyPTO API 映射、约束检查、tiling 可行性 |
| [`pypto-op-plan`](skills/pypto-op-plan/SKILL.md) | Stage 1：triage、API 可用性检查、规划文件初始化 |
| [`pypto-golden-generate`](skills/pypto-golden-generate/SKILL.md) | Stage 2：产出纯 PyTorch `<op>_golden.py` 与必备函数清单；§13 涵盖既有参考的规范化、`.T` 禁用、shape 注释、Golden function inventory、freeze |
| [`pypto-op-design`](skills/pypto-op-design/SKILL.md) | Stage 3：产出 `DESIGN.md`（API 映射、精度路由、tiling 推导、loop 排布） |

### Stage 5–6 — 构造、集成、验证

| Skill | 用途 |
|:---|:---|
| [`pypto-op-construct`](skills/pypto-op-construct/SKILL.md) | Stage 4 模块分解 + Stage 5 单模块构造循环 |
| [`pypto-op-develop`](skills/pypto-op-develop/SKILL.md) | coder 编码手册：实现模式、错误码表、空闲 chip 选择 |
| [`pypto-op-verify`](skills/pypto-op-verify/SKILL.md) | 验证 runner 规格、`detailed_tensor_compare` 用法、成功判据、交付物 |
| [`pypto-op-review`](skills/pypto-op-review/SKILL.md) | `custom/<op>/` 布局检查（CI、pre-commit、agent 可运行） |
| [`pypto-fused-op-integration`](skills/pypto-fused-op-integration/SKILL.md) | 整网集成：tensor 打点 → golden 验证 → 算子替换 → 端到端验证 |

### Stage 7 — 性能调优

| Skill | 用途 |
|:---|:---|
| [`pypto-op-perf-tune`](skills/pypto-op-perf-tune/SKILL.md) | Stage 7 入口（只在精度冻结后启动）：用例执行 + 性能采集 + 分步配置级调优 + 算法级优化 + 报告 |
| [`tune-frontend`](skills/pypto-op-perf-tune/tune-frontend/SKILL.md) | 开箱调优：loop 写法、TileShape、数据搬运 |

| [`tune-swimlane`](skills/pypto-op-perf-tune/tune-swimlane/SKILL.md) | 泳道图深度调优：Stitch、TileShape 深度调优、合图、调度策略；含自动调优脚本（泳道图提取、AIV 依赖链、leafhash → code 映射） |

| [`tune-incore`](skills/pypto-op-perf-tune/tune-incore/SKILL.md) | 核内调优：指令级、核内流水、特殊 shape 处理 |
| [`perf-analyzer`](skills/pypto-op-perf-tune/perf-analyzer/SKILL.md) | 性能数据提取、评级、瓶颈分析、优化建议 |

### 调试与精度

| Skill | 用途 |
|:---|:---|
| [`pypto-general-debug`](skills/pypto-general-debug/SKILL.md) | 调试路由：失败历史、策略切换、逐算子协议、按主题的参考索引 |
| [`pypto-precision-compare`](skills/pypto-precision-compare/SKILL.md) | 两种精度对比：文件保存（`pypto.pass_verify_save` + `torch.save`）与检查点 tensor 二分对比 |
| [`pypto-precision-debug`](skills/pypto-precision-debug/SKILL.md) | 用户代码层面精度排查：语法/逻辑检查、规避方法尝试 |
| [`pypto-aicore-error-locator`](skills/pypto-aicore-error-locator/SKILL.md) | aicore error 时定位 CCE 文件与源代码行 |
| [`pypto-host-stacktrace-analyzer`](skills/pypto-host-stacktrace-analyzer/SKILL.md) | host 端 Python/C++ 堆栈的地址—源码映射与符号解析 |
| [`pypto-memory-overlap-detector`](skills/pypto-memory-overlap-detector/SKILL.md) | MACHINE workspace 内存重叠与管理问题的检测与修复 |
| [`pypto-machine-workspace`](skills/pypto-machine-workspace/SKILL.md) | workspace 内存异常偏大的诊断；逐层拆解内存预算 |
| [`pypto-fracture-point-detector`](skills/pypto-fracture-point-detector/SKILL.md) | 识别当前会话的框架/文档断裂点，产出可转化为 Issue 的报告 |

### Pass 模块（编译期）

| Skill | 用途 |
|:---|:---|
| [`pypto-pass-error-locator`](skills/pypto-pass-error-locator/SKILL.md) | Pass 模块错误诊断：定位、根因分析、修复建议 |
| [`pypto-pass-module-analyzer`](skills/pypto-pass-module-analyzer/SKILL.md) | Pass 模块代码分析；产出结构化 Pass 模块文档 |
| [`pypto-pass-workflow-analyzer`](skills/pypto-pass-workflow-analyzer/SKILL.md) | Pass 业务流分析：职责、执行顺序、数据依赖 |
| [`pypto-pass-perf-optimizer`](skills/pypto-pass-perf-optimizer/SKILL.md) | Pass 编译期性能优化 |
| [`pypto-pass-ut-generate`](skills/pypto-pass-ut-generate/SKILL.md) | 根据业务描述生成 Pass UT 用例 |

### PR / Issue / 代码质量

| Skill | 用途 |
|:---|:---|
| [`pypto-pr-creator`](skills/pypto-pr-creator/SKILL.md) | 创建/更新 `cann/pypto` PR：fork 校验、Git 认证、upstream 同步、commit 检查、CLA |
| [`pypto-pr-fixer`](skills/pypto-pr-fixer/SKILL.md) | 修复 PyPTO PR 的 CodeCheck CI 失败与 review 评论 |
| [`pypto-issue-creator`](skills/pypto-issue-creator/SKILL.md) | 创建结构化 GitCode Issue：Bug、Feature、Doc、Question 等 |
| [`pypto-skill-reviewer`](skills/pypto-skill-reviewer/SKILL.md) | 审计 skill 目录的质量与最佳实践合规性，并打分 |
| [`pypto-skill-validation-prompt`](skills/pypto-skill-validation-prompt/SKILL.md) | 为任意 skill 生成校验提示词，验证产物是否匹配自身声明 |

### 环境与安装

| Skill | 用途 |
|:---|:---|
| [`pypto-environment-setup`](skills/pypto-environment-setup/SKILL.md) | PyPTO 环境安装与修复：CANN、torch_npu、工具链、第三方依赖 |
| [`gitcode-mcp-install`](skills/gitcode-mcp-install/SKILL.md) | 安装与配置 GitCode MCP Server，使 AI 客户端可与 GitCode 平台交互 |
| [`migrate-huggingface-to-npu`](skills/migrate-huggingface-to-npu/SKILL.md) | 将 Hugging Face 大模型迁移到昇腾 NPU；解决 torch / torch-npu 版本问题 |

---

##　Skills 技能详解

技能是定义在 `.agents/skills/` 目录下的可复用行为模块。每个 skill 包含一个 `SKILL.md` 文件，描述完整的执行流程。

**调用方式**：

**自动匹配** — 描述目标，OpenCode 自动选择：

```
我需要开发一个 PyPTO 算子，请帮我完成环境检查和开发流程。
```

**斜杠命令** — 明确指定技能：

```
/pypto-op-workflow
```

**自然语言点名** — 在对话中提及：

```
请使用 pypto-op-workflow 技能帮我开发一个算子。
```

> 进一步了解：[OpenCode Skills 文档](https://opencode.ai/docs/zh-cn/skills/)

按场景快速定位：[算子开发](#算子开发与编排) · [精度调试](#精度验证与调试) · [性能分析](#性能分析-1) · [环境配置](#环境与工具) · [PR提交](#pr-与代码质量)

### 算子开发与编排

#### `pypto-op-workflow` — 算子开发工作流程

**适用场景**：接到算子开发任务，确保开发过程规范、高效、符合最佳实践

**工作流程**：`需求理解 → 环境准备 → Golden → 设计 → 算子实现 → 精度调试 → 性能分析 → 性能调优`

**关键串联**：调用 `pypto-intent-understand`、`pypto-api-explore`、`pypto-golden-generate`、`pypto-op-design`、`pypto-op-develop`、`pypto-precision-debug`、`pypto-op-perf-tune`

#### `pypto-intent-understand` — 需求意图理解

**适用场景**：将用户的自然语言算子描述转化为结构化需求文档（SPEC.md）

**你需要提供**：算子名称、数学公式、输入输出规格

**你会得到**：结构化的 SPEC.md，包含 ASCII 数据流图、规格确认清单、典型配置

#### `pypto-api-explore` — API 探索

**适用场景**：查找 PyPTO 是否支持某个操作、验证 API 约束、分析算子可行性

**你会得到**：API_REPORT.md，包含公式分解、PyPTO API 映射表、约束分析、Tiling 需求

#### `pypto-golden-generate` — Golden 参考实现生成

**适用场景**：生成用于精度对比的 torch + torch_npu NPU golden 参考实现

**你会得到**：`{op}_golden.py`，导出 `{op}_golden()` 函数，含自动验证代码；验证通过后通过通用 profiler 生成 `GOLDEN_PERF_REPORT.md`

#### `pypto-op-design` — 设计方案生成

**适用场景**：设计 PyPTO 算子实现方案（Tiling 策略、Loop 结构）

**你会得到**：DESIGN.md，包含 API 映射设计、数据规格设计、Tiling 策略、Loop 结构、验证方案

#### `pypto-op-develop` — 代码实现

**适用场景**：编写 PyPTO 算子实现、测试和文档

**你会得到**：`{op}_impl.py`（Kernel 实现）、`test_{op}.py`（测试入口）、`README.md`（算子文档）

---

### 精度验证与调试

#### `pypto-precision-debug` — 精度问题排查

**适用场景**：算子精度验证失败，需要系统化定位问题根因

**排查流程**：基础检查 → 内存排查（workspace/内存重叠）→ 特性排除（unroll/合轴/submit_before_loop）→ 二分定位

**常见问题**：workspace 不足、循环展开问题、合轴问题、并行执行问题、valid_shape 错误

#### `pypto-precision-compare` — 精度对比与定位

**适用场景**：调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比

**核心原理**：提供两种方法 - 文件保存方法（使用 `pass_verify_save` 和 `torch.save`）和二分对比方法（使用检查点 tensor）

#### `pypto-aicore-error-locator` — AICore 错误定位

**适用场景**：测试案例出现 AICore error，需要定位问题 CCE 文件和代码行

**工作流程**：启用追踪日志 → 重新编译 → 分析 trace 日志 → 二分查找定位问题代码行

---

### 性能分析

#### `pypto-op-perf-tune` — 性能分析及调优

**适用场景**：分析已生成的性能数据，评估算子性能表现，基于实测性能数据迭代调优，并验证精度与性能收益

**核心指标**：核心利用率、气泡率、AicoreTime、等待时间

**评级标准**：⭐⭐⭐⭐⭐（利用率>90%，气泡<2%）到 ⭐（利用率<50%，气泡>20%）

**调优手段**：Stitch 调优、loop_unroll、Tilesize 调整、L2 亲和调度、CubeNBuffer 合并

---

### 环境与工具

#### `pypto-environment-setup` — 环境诊断与修复

**适用场景**：环境安装失败、import 报错、NPU 设备检测不到、依赖冲突

**你会得到**：诊断报告 + 修复步骤 + 验证命令

#### `gitcode-mcp-install` — GitCode MCP Server 安装

**适用场景**：安装和配置 GitCode MCP Server，使 AI 代理能与 GitCode 平台交互

**你会得到**：安装命令 + 配置模板 + 验证步骤

---

### PR 与代码质量

#### `pypto-pr-creator` — 创建 PR

**适用场景**：将开发完成的算子提交到 cann/pypto 仓库

**你会得到**：fork 验证 → 用户确认 → 分支创建 → PR 创建链接 + 结构化报告

#### `pypto-pr-fixer` — 修复 PR 问题

**适用场景**：PR 收到 review 评论或 CodeCheck CI 失败

**你会得到**：评论解析 → 修复方案 → 自动应用 → 同步更新

#### `pypto-skill-reviewer` — Skill 质量评审

**适用场景**：审计某个 Skill、检查是否遵循规范、发布前评估

**你会得到**：48 条规则评分报告，包含 9 维度评分、问题列表、修复建议

#### `pypto-skill-validation-prompt` — Skill 校验提示词生成

**适用场景**：为任意 Skill 生成校验提示词，验证实际执行效果是否符合规范

**你会得到**：≤80 行的校验提示词文档，可交给 AI 代理执行，产出数据驱动的优化报告

#### `pypto-issue-creator` — 创建 GitCode Issue

**适用场景**：基于会话上下文智能创建 GitCode Issue

**支持类型**：Bug Report、Feature Request、Documentation、Question、Task

#### `pypto-fracture-point-detector` — 断裂点识别

**适用场景**：识别 PyPTO 框架或文档不完善导致的断裂点，产出可转化为 Issue 的报告

**断裂点类型**：文档类（D1-D6）、API/框架类（A1-A5）、错误信息类（E1-E4）、行为模式类（C1-C6）

---

### Pass 分析与优化

#### `pypto-pass-error-locator` — Pass 模块错误诊断

**适用场景**：Pass 模块抛出错误，需要从问题定位到给出修复建议的完整排查流程

**工作流程**：错误定位 → 原因分析 → 修复建议

#### `pypto-pass-module-analyzer` — Pass 模块代码分析

**适用场景**：需要理解 PyPTO Pass 中某个模块的代码、功能和设计

**你会得到**：Pass 模块分析文档，包含接口描述、功能说明与特殊场景分析

#### `pypto-pass-perf-optimizer` — Pass 编译性能优化

**适用场景**：Pass 编译耗时过长，需要分析和优化编译性能

**你会得到**：性能分析报告 + 优化方案 + 验证步骤

#### `pypto-pass-ut-generate` — Pass 单元测试生成

**适用场景**：需要根据 Pass 业务描述生成对应的单元测试用例（UT）

**你会得到**：基于 GTest 框架的 UT 用例，含环境配置、图构建、Pass 执行与结果校验

#### `pypto-pass-workflow-analyzer` — Pass 业务流分析

**适用场景**：需要理解某个业务场景中 Pass 各模块的执行顺序、依赖关系与数据流转

**你会得到**：业务流分析文档，包含模块职责、执行顺序、数据流转说明

---

## Lint 与状态机 — 自动门禁

两个 opencode 插件在每次 tool 调用时自动运行。**任何 agent 都不需要手动调用它们。**

### `pypto-op-lint.ts`

在每次相关的 tool 事件后运行 OL lint 引擎（`.agents/hooks/pypto-op-lint/`）：

| 事件 | 触发条件 | 执行内容 |
|:---|:---|:---|
| `tool.execute.after`（Write/Edit） | 文件名匹配 `*_impl.py`、`*_golden.py` 或 `test_*.py` | `post-edit` hook — 用 43 条 OL 规则校验；命中 S0/S1 规则时**拦截**本次 tool 调用 |
| `tool.execute.after`（Bash） | 命令匹配 `python test_*.py` | `post-bash` hook — 解析 stdout/stderr/exit code 并产出裁决 |
| `tool.execute.before`（Bash） | 命令尝试写入 `.orchestrator_state.json` | **拦截** — 该文件仅允许编排者通过状态机插件修改 |

规则定义在 [`.agents/hooks/pypto-op-lint/rules.json`](hooks/pypto-op-lint/rules.json)（v2.0，43 条）。五个维度：

| 维度 | 覆盖范围 |
|:---|:---|
| **D1** | 框架约束合规 — 装饰器、签名 shape、JIT 要求 |
| **D2** | 工件完整性 — 每个 stage 的必备文件 |
| **D3** | 三文件分离 — golden / impl / test 的边界（impl 中无 torch、golden 中无 pypto） |
| **D4** | 测试规范 — adversarial 覆盖、tolerance 模式（`atol/rtol` 或 `mare/mere/rmse` 矩阵） |
| **D5** | 跨文件一致性 — 模块契约与 impl 一致 |

严重级别：S0（致命）→ S1（必修）→ S2（警告）→ S3（信息）。

### `pypto-state-transition.ts`

守护 stage 间转移，发出 Phase M_k 循环事件。支持 `rollback_to_stage`，让单个失控模块不会污染其余流水线。核心逻辑放在 `lib/state-transition-core.ts`，与 opencode 解耦，可独立单测。

两个插件都在 `.opencode/plugins/__tests__/` 下提供单元测试。

---

## Claude Code 配置

opencode 自动发现 `.opencode/agents/`、`.agents/skills/`、`.opencode/plugins/`。Claude Code 使用不同的目录结构（`.claude/agents/`、`.claude/skills/`、`CLAUDE.md`、`.claude/settings.json`），需要一次性迁移配置。lint 插件（`.opencode/plugins/pypto-op-lint.ts`）属于 opencode 专用，不会自动迁移 — 但同一份 Python lint 引擎可以通过 Claude Code 的 hooks API 接入（见步骤 3）。

> **一次性配置**，在仓库根目录运行。仅在新增 agent/skill 或刷新 hook 配置时需要重跑。

### 1. 创建 `.claude/` 目录结构

Claude Code 使用不同的目录结构，需要先迁移项目配置：

```bash
# 1. 创建 Claude Code 目录结构
mkdir -p .claude/skills .claude/agents

# 2. 复制项目指令文件
cp AGENTS.md CLAUDE.md

# 3. 复制 Skills 到 Claude Code 目录
cp -r .agents/skills/* .claude/skills/

# 4. 复制 Agents 到 Claude Code 目录
cp -r .opencode/agents/* .claude/agents/

# 5. 复制 Hook 配置（权限、lint hooks 等）
cp .agents/settings.json .claude/settings.json
```

### 2. 指定 Agent 启动 Claude Code

使用 `--agent` 参数让 Claude Code 启动时直接进入编排者：

```bash
claude --agent pypto-op-orchestrator
```

启动后，在对话中描述算子开发任务即可。

### 3. 验证配置

启动后，确认编排者已激活：
- 它的开场白会提到 Stage 1–7 与 9 智能体团队
- 输入 `/agents`，确认列表里能看到全部 9 个 `pypto-op-*` agent
- 编辑任意 `*_impl.py` 写一些违规内容；OL lint hook 应触发并（如果是 S0/S1 违规）拦截本次编辑

如果上述任一项失败，最常见原因：
- 工作目录不在仓库根（Claude Code 从 `cwd` 向上检索 `.claude/`）
- agent 的 `mode:` frontmatter 行让 Claude Code 报警告（这是 opencode 专用字段）。如果遇到，按下行去掉 `mode:`：`for f in .opencode/agents/*.md; do sed '/^mode:/d' "$f" > ".claude/agents/$(basename "$f")"; done`

---

## 单技能模式（不走编排者）

如果只想用某一项能力 — 比如对已有 kernel 跑一次 `pypto-precision-compare`，或让 `pypto-pr-creator` 把本地修改提成 PR — 可以不通过编排者，直接调用单个 skill。它读取的是同一份 `SKILL.md`，但不会推进 Stage 1–7，也不会触发 lint 门禁。

这种模式适合诊断、orchestrated run 之后的补丁修复，以及一次性的 PR/Issue 工作。

---

## 仓库布局

```
pypto/
├─ AGENTS.md                          ← 项目入口（权威规约）
├─ .opencode/
│  ├─ agents/                         ← 9 个 agent 定义（markdown frontmatter + body）
│  └─ plugins/                        ← 状态机 + OL lint 插件（TypeScript + 单测）
└─ .agents/
   ├─ README.md                       ← 当前文件
   ├─ skills/                         ← 43 个专家技能，每个含 SKILL.md
   ├─ hooks/pypto-op-lint/            ← Python lint 引擎（43 规则、11 单测、JSON 事件日志）
   ├─ settings.json                   ← agent 运行时设置
   └─ user_in.md                      ← 用户提示词模板
```

---

## 常见问题

<details>
<summary><b>AGENTS.md、Skills 和 Agents 有什么区别？</b></summary>

| 维度 | AGENTS.md | Skills | Agents |
|:---|:---|:---|:---|
| 作用 | 项目级自定义规范 | 特定任务的执行流程 | 编排和隔离执行复杂任务 |
| 加载方式 | 自动加载，对所有对话生效 | 按需加载，调用时才生效 | Orchestrator 主导，Subagent 被调度 |
| 内容 | 通用开发规范和原则 | 具体任务的步骤、工具、验证标准 | 状态机、工件契约、重试策略 |
| 执行模式 | 规则约束 | 直接执行 | Primary 编排 + Subagent 隔离执行 |

三者配合使用：AGENTS.md 定义"怎么做才对"，Skills 定义"怎么一步步做完"，Agents 定义"怎么编排和隔离执行"。

</details>

<details>
<summary><b>什么时候用 Orchestrator，什么时候直接用 Skill？</b></summary>

- **完整算子开发**：使用 `pypto-op-orchestrator` agent（或触发 `pypto-op-workflow` skill）
- **单步任务**：直接调用对应 Skill，如只需生成 Golden 就调用 `pypto-golden-generate`
- **调试修复**：直接调用调试类 Skill，如 `pypto-precision-debug`、`pypto-aicore-error-locator`

</details>

<details>
<summary><b>其他 AI 工具兼容性</b></summary>

本项目支持多种 AI 编程工具，包括 [OpenCode](https://opencode.ai)、[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)、Cursor、Codex 等。

**Claude Code 目录结构映射**：

| 组件 | OpenCode | Claude Code |
|:---|:---|:---|
| 项目指令 | `AGENTS.md` | `CLAUDE.md` |
| Skills | `.agents/skills/` | `.claude/skills/` |
| Agents | `.opencode/agents/` | `.claude/agents/` |
| Hook/Plugin | `.opencode/plugins/` | `.claude/settings.json` |

**格式兼容性**：
- **SKILL.md**：YAML frontmatter + Markdown，两种工具完全兼容
- **Agents**：YAML frontmatter + Markdown，`mode: primary` 为 OpenCode 特有字段，Claude Code 会忽略

</details>

<details>
<summary><b>Claude Code 启动后 settings.json 报错怎么办？</b></summary>

`.claude/settings.json` JSON 格式错误时，可用 `jq` 校验语法：

```bash
jq . .claude/settings.json
```

如果输出报错信息（如 `parse error`），按提示定位行号并修正。校验通过则会回显完整 JSON。

</details>

<details>
<summary><b>OpenCode 和 Claude Code 的 Hook/Lint 机制有什么区别？</b></summary>

两种工具共享同一份 Python lint 脚本（`.agents/hooks/pypto-op-lint/pypto_op_lint.py`），但触发方式不同：

- **OpenCode**：通过 `.opencode/plugins/` 下的 TypeScript 插件自动加载，无需手动配置
  - `pypto-op-lint.ts` — 编辑/测试时自动 lint，阻断 S1 级违规
  - `pypto-state-transition.ts` — 提供 `state_transition` 工具，封装阶段状态迁移与门禁检查
- **Claude Code**：通过 `.claude/settings.json` 中的 hooks 配置触发（从 `.agents/settings.json` 拷贝）
  - `PostToolUse[Write|Edit|MultiEdit]` → post-edit lint
  - `PostToolUse[Bash]` → 测试输出三态判定
  - `PreToolUse[Write|Edit|MultiEdit]` → Stage 6 自动备份
  - `Stop` → 交付门禁

</details>

---

## 进一步阅读

- [`AGENTS.md`](../AGENTS.md) — 完整的项目规约、Stage 1–7 详解、lint/状态机规格、核心算子开发原则
- [skill `pypto-orchestration-manual` (SKILL.md auto-loads)](skills/pypto-orchestration-manual/SKILL.md) — 编排者启动手册（调试编排逻辑时先读它）
- [`hooks/pypto-op-lint/rules.json`](hooks/pypto-op-lint/rules.json) — 权威 OL 规则集（v2.0，43 条）

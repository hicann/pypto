# PyPTO OpenCode 工作流

通过 AI 代理与专家技能，自动完成昇腾 NPU 算子开发全流程：

```
需求分析 → API 探索 → Golden 生成 → 设计方案 → 代码实现 → 精度验证 → 性能调优 → PR 提交
```

本仓库为 [OpenCode](https://opencode.ai) 预配置了项目规范（AGENTS.md）、专家技能（Skills）和协作代理（Agents），开箱即用。

---

## 快速开始

在本仓库目录下启动 OpenCode，通过以下任一方式描述你的目标：

### 方式一：数学公式描述

```
我需要开发一个名为 sinh 的算子，数学公式是 (e^x - e^(-x)) / 2。
输入是 shape 为 [b, s, n, d] 的 float32 tensor，输出 shape 相同。
精度要求：atol=0.000025, rtol=0.005。
```

### 方式二：提供算子方案文档

```
请根据 ./docs/my_operator_spec.md 中的方案文档，开发对应的 PyPTO 算子。
```

### 方式三：提供算子论文链接

```
请根据 https://arxiv.org/abs/2205.14135 这篇 Flash Attention 论文，实现对应的 PyPTO 算子。
```

OpenCode 会自动加载项目规范，选择合适的技能，按标准流程执行开发任务。无需手动配置。

---

## 使用方式

本节介绍如何在 OpenCode 和 Claude Code 中使用本项目的 Skills 和 Agents 进行算子开发。

### OpenCode

在 PyPTO 仓库主目录启动 OpenCode，通过以下方式开始算子开发：

#### 方式一：直接调用 Skill

在对话中描述开发任务，自动触发 `pypto-op-workflow`：

```
开发一个 sinh 算子，数学公式是 (e^x - e^(-x)) / 2
```

或使用斜杠命令明确指定：

```
/pypto-op-workflow 开发一个 sinh 算子
```

#### 方式二：切换到 Orchestrator Agent

按 `Tab` 键切换到 `pypto-op-orchestrator` 代理，然后输入开发任务：

```
开发一个 sinh 算子，数学公式是 (e^x - e^(-x)) / 2
```

> **提示**：Orchestrator 会自动编排 7 阶段状态机：需求理解 → API 探索 → Golden 生成 → 设计方案 → 代码实现 → 精度修复 → 性能调优

---

### Claude Code

#### 前置准备

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
```

#### 方式一：直接调用 Skill

启动 Claude Code 后，直接在对话中调用 skill：

```bash
# 启动 Claude Code
claude
```

然后在对话中使用斜杠命令：

```
/pypto-op-workflow 开发一个 sinh 算子
```

或自然语言描述：

```
请使用 pypto-op-workflow 技能开发一个 sinh 算子
```

#### 方式二：指定 Agent 启动

使用 `--agent` 参数直接指定代理启动 Claude Code：

```bash
claude --agent pypto-op-orchestrator
```

启动后，在对话中输入算子开发任务即可。

---

### 使用建议

| 场景 | 推荐方式 |
|:---|:---|
| 完整算子开发流程 | 方式二（Orchestrator Agent） |
| 单步任务（如只需生成 Golden） | 方式一（直接调用对应 Skill） |
| 调试修复类任务 | 方式一（直接调用 `pypto-precision-debug` 等） |

---

## 核心架构

### AGENTS.md — 项目规范

AGENTS.md 是 OpenCode 的项目级自定义指令文件。当你在本仓库中使用 OpenCode 时，它会自动加载并生效——无需手动操作。

该文件定义了：

- **Skills 索引**：按场景分类的技能清单（算子开发、精度调试、性能分析、环境工具、PR 质量）
- **通用原则**：如实报告、先验证再下结论、区分事实与推断、最小必要改动
- **核心算子开发原则**：理解原理后实现、遇问题先定位、工件职责分离、以官方资料为准、优先 NPU 真机验证

> 进一步了解：[OpenCode 自定义规则文档](https://opencode.ai/docs/zh-cn/rules/)

---

### Agents — 协作代理

代理是定义在 `.opencode/agents/` 目录下的协作实体，负责编排和隔离执行复杂任务。

| 代理 | 模式 | 职责 |
|:---|:---|:---|
| `pypto-op-orchestrator` | Primary | 算子端到端开发编排，管理 7 阶段状态机 |
| `pypto-op-analyst` | Subagent | Golden 生成与 Design 设计分析（上下文隔离） |
| `pypto-op-developer` | Subagent | 代码实现与精度修复（上下文隔离） |
| `pypto-op-perf-tuner` | Subagent | 性能分析与调优（上下文隔离） |
| `pypto-code-merge-agent` | Subagent | 代码变更到 PR 提交的自动化流程 |

**Orchestrator 状态机**：

```
Stage 1: 需求理解 (pypto-intent-understand)
    ↓
Stage 2: API 探索 (pypto-api-explore)
    ↓
Stage 3: Golden 生成 → Analyst Subagent
    ↓
Stage 4: Design 设计 → Analyst Subagent
    ↓
Stage 5: 代码实现 → Developer Subagent
    ↓
Stage 6: 精度修复 → Developer Subagent (可选)
    ↓
Stage 7: 性能调优 → PerfTuner Subagent
```

---

### Skills（专家技能）

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

---

## 技能详解

按场景快速定位：[算子开发](#算子开发与编排) · [精度调试](#精度验证与调试) · [性能分析](#性能分析) · [环境配置](#环境与工具) · [PR提交](#pr-与代码质量)

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

**适用场景**：生成用于精度对比的 PyTorch golden 参考实现

**你会得到**：`{op}_golden.py`，导出 `{op}_golden()` 函数，含自动验证代码

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

#### `pypto-issue-creator` — 创建 GitCode Issue

**适用场景**：基于会话上下文智能创建 GitCode Issue

**支持类型**：Bug Report、Feature Request、Documentation、Question、Task

#### `pypto-fracture-point-detector` — 断裂点识别

**适用场景**：识别 PyPTO 框架或文档不完善导致的断裂点，产出可转化为 Issue 的报告

**断裂点类型**：文档类（D1-D6）、API/框架类（A1-A5）、错误信息类（E1-E4）、行为模式类（C1-C6）

---

### Pass 分析与优化

#### `pypto-pass-error-fixer` — Pass 模块错误诊断与修复

**适用场景**：Pass 模块抛出错误，需要从问题定位到修复验证的完整排查流程

**工作流程**：错误定位 → 原因分析 → 问题修复 → 验证确认

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

**格式兼容性**：
- **SKILL.md**：YAML frontmatter + Markdown，两种工具完全兼容
- **Agents**：YAML frontmatter + Markdown，`mode: primary` 为 OpenCode 特有字段，Claude Code 会忽略

</details>

# PyPTO OpenCode 工作流

通过 AI 代理与专家技能，自动完成昇腾 NPU 算子开发全流程：

```
需求分析 → 环境准备 → 编码实现 → 精度验证 → 性能调优 → PR 提交
```

本仓库为 [OpenCode](https://opencode.ai) 预配置了项目规范（AGENTS.md）和专家技能（Skills），开箱即用。技能持续增加中，以 `.opencode/skills/` 目录下的实际内容为准。

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

## 我想做…→ 用这个

根据你的目标快速定位合适的技能：

| 我想…                     | 推荐技能                             | 一句话说明               |
| :------------------------ | :----------------------------------- | :----------------------- |
| 开发一个新算子            | `pypto-operator-develop-workflow`     | 全流程引导，从需求到交付 |
| 修复环境报错              | `pypto-environment-setup`            | 诊断 + 修复 + 验证      |
| 分析编译 pass 校验结果    | `pypto-verify-pass`                  | 定位失败的 pass 及原因   |
| 排查精度不一致            | `pypto-verify-binary-search`         | 二分法定位首个误差点     |
| 分析算子性能瓶颈          | `pypto-operator-perf-autotune`       | 泳道图分析 + 优化建议    |
| 迭代调优到目标性能        | `pypto-perf-tuning-loop`             | 参数扫描 + 对比报告      |
| 提交 PR 到 cann/pypto     | `pypto-pr-creator`                   | 自动创建规范 PR          |
| 修复 PR review 意见       | `pypto-pr-fixer`                     | 解析评论 + 自动修复      |
| 安装 GitCode MCP Server   | `gitcode-mcp-install`                | 安装 + 配置 + 验证       |

> **不确定用哪个？** 直接描述目标，OpenCode 会自动匹配。

---

## 技能详解

### 算子开发

#### `pypto-operator-develop-workflow` — 算子开发全流程

**适用场景**：从零开发一个昇腾 NPU 自定义算子

**你需要提供**：算子名称、数学公式、输入输出规格、数据类型、精度要求

**你会得到**：完整的分阶段执行流程——需求检查 → 环境准备 → 方案设计 → 编码实现 → 构建测试 → 高阶参数使能

#### `pypto-environment-setup` — 环境诊断与修复

**适用场景**：环境安装失败、import 报错、NPU 设备检测不到、依赖冲突

**你需要提供**：问题描述（如错误信息、Python 版本、已安装的包）

**你会得到**：诊断报告 + 修复步骤 + 验证命令

#### `pypto-verify-pass` — Pass 校验分析

> **注意**：目录名为 `pypto-verify-pass`，frontmatter name 为 `pypto-verify`，两种方式均可调用。

**适用场景**：编译后需要确认各 pass 是否通过

**你需要提供**：编译输出日志或 pass 验证结果

**你会得到**：逐 pass 成功/失败状态 + 失败原因定位

#### `pypto-verify-binary-search` — 精度二分定位

**适用场景**：算子输出与 golden 不一致，需要定位误差来源

**你需要提供**：精度不匹配的算子代码、golden 实现

**你会得到**：第一个出现误差的 op 位置

---

### 性能调优

#### `pypto-operator-perf-autotune` — 性能分析与调优建议

**适用场景**：算子开发完成后，分析性能瓶颈并获取优化方向

**你需要提供**：算子代码、输出目录路径

**你会得到**：泳道图分析 + 性能瓶颈定位 + 优化建议

#### `pypto-perf-tuning-loop` — 迭代式性能调优

**适用场景**：需要系统性地搜索最优参数组合

**你需要提供**：算子代码、性能目标、可调参数范围

**你会得到**：基准性能 → 参数扫描 → 调优后性能对比报告 + 最优参数组合

---

### 代码贡献

#### `pypto-pr-creator` — 创建 PR

**适用场景**：将开发完成的算子提交到 cann/pypto 仓库

**你需要提供**：分支名、commit 信息、PR 标题和描述

**你会得到**：fork 验证 → 用户确认 → 分支创建 → PR 创建链接 + 结构化报告

#### `pypto-pr-fixer` — 修复 PR 问题

**适用场景**：PR 收到 review 评论或 CodeCheck CI 失败

**你需要提供**：PR URL 或 `owner/repo/pull_number`、需要修复的评论

**你会得到**：评论解析 → 修复方案 → 自动应用 → 同步更新

---

### 工具集成

#### `gitcode-mcp-install` — GitCode MCP Server

**适用场景**：首次安装或重新配置 GitCode MCP Server，使 AI 代理能与 GitCode 平台交互

**你需要提供**：安装方式偏好（Go 二进制 / Python 源码）

**你会得到**：安装命令 + 配置模板 + 验证步骤

---

## 工作原理

### AGENTS.md — 项目规范，自动生效

AGENTS.md 是 OpenCode 的项目级自定义指令文件。当你在本仓库中使用 OpenCode 时，它会自动加载并生效——无需手动操作。

该文件定义了：

- **核心原则**：遇问题优先定位修复、基于官方文档实现、优先保证方案可用
- **环境配置**：默认版本（CANN 8.5.0 / PyTorch 2.6.0 / torch_npu 2.6.0.post3）
- **开发规范**：目录结构、分阶段流程、错误处理策略
- **默认值**：输入输出规格、数据类型、精度要求的合理缺省

> 📖 进一步了解：[OpenCode 自定义规则文档](https://opencode.ai/docs/zh-cn/rules/)

### Skills — 专家技能，按需加载

Skills 是定义在 `.opencode/skills/` 目录下的可复用行为模块。每个 skill 包含一个 `SKILL.md` 文件，描述完整的执行流程。OpenCode 会在需要时自动发现并加载。

**调用方式**（三选一，效果等价）：

**自动匹配** — 描述目标，OpenCode 自动选择：

```
我需要开发一个 PyPTO 算子，请帮我完成环境检查和开发流程。
```

**斜杠命令** — 明确指定技能：

```
/pypto-operator-develop-workflow
```

**自然语言点名** — 在对话中提及：

```
请使用 pypto-operator-develop-workflow 技能帮我开发一个算子。
```

> 📖 进一步了解：[OpenCode Skills 文档](https://opencode.ai/docs/zh-cn/skills/)

---

## 常见问题

<details>
<summary><b>AGENTS.md 和 Skills 有什么区别？</b></summary>

| 维度     | AGENTS.md                | Skills                         |
| :------- | :----------------------- | :----------------------------- |
| 作用     | 项目级自定义规范         | 特定任务的执行流程             |
| 加载方式 | 自动加载，对所有对话生效 | 按需加载，调用时才生效         |
| 内容     | 通用开发规范和原则       | 具体任务的步骤、工具、验证标准 |

两者配合使用：AGENTS.md 定义"怎么做才对"，Skills 定义"怎么一步步做完"。

</details>

# AGENTS.md

## 项目概述

本项目用于开发华为昇腾 AI 处理器（CANN PyPTO）自定义算子，支持完整的开发、测试及性能调优流程。

### 核心功能

- 使用 PyPTO 编程语言开发昇腾 AI 处理器自定义算子
- 提供完整的开发、构建、测试及性能调优工作流支持
- 遵循官方开发规范和性能优化最佳实践

---

## 9-Agent 多智能体团队（入口）

本仓库通过 **9 个职责分明的子智能体** 协同完成算子开发。所有调度由 `pypto-op-orchestrator` 统一负责，子智能体之间通过 `custom/<op>/MEMORY.md`（共享叙事）和 `custom/<op>/.orchestrator_state.json`（机器可读状态）交换信息。

| 智能体 | 职责 | 阶段 |
|---|---|---|
| `pypto-op-orchestrator` | 总编排、阶段门禁、调度子智能体 | 全阶段 |
| `pypto-op-planner` | 需求规格 (`SPEC.md`) + API 报告 (`API_REPORT.md`) | Stage 1 |
| `pypto-op-mathematician` | golden 参考实现 (`<op>_golden.py`) | Stage 2 |
| `pypto-op-architect` | 设计方案 (`DESIGN.md`)、tiling、性能目标 | Stage 3 |
| `pypto-op-designer` | 模块分解 (`module_interfaces.yaml`) | Stage 4 |
| `pypto-op-coder` | 算子实现（per-module impl + 集成 impl + README） | Stage 5 |
| `pypto-op-verifier` | 评判 / 测试生成 / 精度校验 | Stage 4 scaffolding + 5 + 6 |
| `pypto-op-debugger` | Stage 5 失败时的根因调查与补丁建议 | Stage 5 (按需) |
| `pypto-op-optimizer` | 三阶段性能调优（frontend / swimlane / incore） | Stage 7 |

> **完整操作手册**：参见 [skill `pypto-orchestration-manual` (SKILL.md auto-loads)](.agents/skills/pypto-orchestration-manual/SKILL.md) 及其 `references/` 子目录（principles / agents / rules / catalog）。

### Stage 1–7 工作流概览

| Stage | 内容 | 拥有者 | 主要产出 |
|---|---|---|---|
| **1** Planning | 需求规格 + API 调研 | planner | `SPEC.md`, `API_REPORT.md` |
| **2** Algorithm | golden 参考（PyPTO 友好） | mathematician | `<op>_golden.py` |
| **3** Architecture | 计算图、tiling、数值稳定性、性能目标 | architect | `DESIGN.md` |
| **4** Design | 模块分解、契约、scaffolding (per-module golden + test + adversarial harness) | designer + verifier | `module_interfaces.yaml`, `modules/<op>_module<k>_golden.py`, `modules/test_<op>_module<k>.py` |
| **5** Construction | per-Phase M_k 实现循环（coder → verifier → debugger 按需 → coder → verifier）。最后一个 M_k 通过后：cleanup 把 `modules/<op>_module1...N_impl.py` 拷贝到 `<op>_impl.py`（保持算法、整理函数名 / imports / 调试码），verifier 写 `test_<op>.py`，coder 写 `README.md` | coder + verifier | `modules/<op>_module<k>_impl.py`, `<op>_impl.py`, `test_<op>.py`, `README.md` |
| **6** Verification | 布局检查、结构性规则验证 | verifier | layout 通过 |
| **7** Optimization | 性能调优（≤ 2 倍加速等目标达成） | optimizer | 优化后的 `<op>_impl.py` |

### Stage 5 内部 Phase 循环

Stage 5 不是单次调度，而是 per-module 串行循环：

```
for M_k in [M1, M12, M123, …, M123…N]:
    coder    → modules/<op>_module<k>_impl.py
    verifier → 跑 modules/test_<op>_module<k>.py + prefix-eval（PASS / FAIL + failure_category）
    若 FAIL：
        debugger → 提出补丁建议（写入 MEMORY.md）
        coder    → 应用补丁
        verifier → 重判
    cycles ≥ 10 时进入 blocked 状态，orchestrator 决定升级用户或 rollback_to_stage
```

---

## Skills 索引（43 项）

> 命名规则：本项目原创 skill 统一以 `pypto-` 前缀，少数 upstream 派生 skill（`perf-analyzer`、`tune-frontend/incore/swimlane`、`gitcode-mcp-install`、`migrate-huggingface-to-npu`）保持原名。

### 编排与共享状态

- `pypto-orchestration-manual`：orchestrator 入口；包含 principles / agents / rules / catalog 四份控制文档（progressive disclosure）
- `pypto-memory-template`：`custom/<op>/MEMORY.md` 模板（共享叙事日志）

### Stage 1–4：规划与设计

- `pypto-intent-understand`：将自然语言算子需求转化为结构化规格
- `pypto-api-explore`：探索 PyPTO API 映射、约束条件与可行性
- `pypto-op-plan`：Stage 1 规划工作流
- `pypto-golden-generate`：Stage 2 生成 PyPTO 友好的 torch golden
- `pypto-op-design`：Stage 3 架构设计 / DESIGN.md 生成
- `pypto-op-construct`：Stage 4–5 模块分解 + per-Phase 构建工作流

### Stage 5–7：实现与验证

- `pypto-op-develop`：Stage 5 单文件实现指引
- `pypto-op-verify`：`detailed_tensor_compare` runner、success criteria、deliverables
- `pypto-op-review`：`extract_pypto_calls.py`（op-by-op 调试用）；layout / 结构检查已由 pypto-op-lint 钩子自动执行

### Stage 7：性能调优

- `pypto-op-perf-tune`：Stage 7 调优工作流总入口；三阶段配置级调优（frontend → swimlane → incore）+ 收敛后可选算法级优化
- `perf-analyzer`：性能数据采集与瓶颈定位
- `tune-frontend`：前端 tile / dtype 调优
- `tune-swimlane`：swimlane 并行调优 / 自动化搜索 / leafhash → code 映射
- `tune-incore`：incore loop unroll / 寄存器复用

### 调试与精度定位

- `pypto-general-debug`：debug router；包含 `DEBUG_GUIDEBOOK.md` §1–§9 故障排查 playbook；tile shape / L0/L1 / alignment / `set_cube_tile_shapes` 问题由 `DEBUG_GUIDEBOOK.md` 路由到 `references/tile-shapes.md`
- `pypto-precision-debug`：精度问题定位与修复
- `pypto-precision-compare`：精度对比与二分定位
- `pypto-aicore-error-locator`：AICore 错误源定位
- `pypto-host-stacktrace-analyzer`：Host 段错误 / stack trace 分析
- `pypto-memory-overlap-detector`：workspace overlap 检测
- `pypto-machine-workspace`：OOM / `rtMalloc failed` 处理

### 环境与安装

- `pypto-environment-setup`：PyPTO 环境安装与问题修复
- `gitcode-mcp-install`：GitCode MCP Server 安装与配置
- `migrate-huggingface-to-npu`：HuggingFace 模型迁移到 NPU

### Pass 模块分析（编译期）

- `pypto-pass-error-locator`：Pass 模块错误诊断
- `pypto-pass-module-analyzer`：Pass 模块代码分析
- `pypto-pass-perf-optimizer`：Pass 编译性能优化
- `pypto-pass-ut-generate`：Pass 单元测试生成
- `pypto-pass-workflow-analyzer`：Pass 业务流分析

### PR / Issue / 代码质量

- `pypto-pr-creator`：准备并创建符合规范的 PR
- `pypto-pr-fixer`：修复 PR 的 CI 失败与 review 意见
- `pypto-issue-creator`：基于上下文创建 GitCode Issue
- `pypto-fracture-point-detector`：识别 PyPTO 框架或文档断裂点
- `pypto-skill-reviewer`：评审 skill 目录质量
- `pypto-skill-validation-prompt`：为任意 skill 生成校验提示词
- `pypto-fused-op-integration`：算子融合与模型集成

---

## Lint 与状态机（自动化门禁）

每次 `state_transition(action="complete_stage", stage=N)` 调用都会触发 lint 引擎（`.agents/hooks/pypto-op-lint/`）。Stage N 关联的所有规则同步评估，FAIL 直接阻断状态迁移；状态文件不会被错误地更新。

- 规则总数：**43 条**（`OL01–OL47`，跨 5 个维度 D1–D5）
- 严重程度：`S0` 致命（直接阻断）/ `S1` 必修 / `S2` 警告 / `S3` 信息提示
- 状态机 schema v2.0：支持 Stage 1–7、Stage 5 内部 per-Phase M_k 循环、10-cycle blocker、`rollback_to_stage` 审计追踪
- 详细规则清单：参见 `.agents/hooks/pypto-op-lint/rules.json`

---

## 通用原则

> **严格遵循以下原则**

1. **如实报告，禁止伪完成**
   - 未验证的结果，不得表述为"已完成"或"已通过"
   - 未实际执行的命令、测试、构建、提交或发布，不得声称已执行
   - 遇到失败、阻塞、权限不足或信息缺失时，必须明确说明，不得伪造过程或结果
2. **先验证，再下结论**
   - 能通过代码、文件、日志、测试或工具直接确认的事项，优先基于证据判断，不以猜测代替验证
   - 若当前环境无法完成验证，必须明确说明验证缺口、已知范围与剩余风险
3. **区分事实、推断与建议**
   - 结论应明确区分"已确认事实""基于上下文的推断""建议采取的动作"
   - 禁止编造不存在的文件、输出、报错、性能收益、验证状态或用户意图
4. **遵循最小必要改动原则**
   - 优先复用现有实现、既有模式和项目约定，避免无依据的重写、扩面或过度设计
   - 只解决当前任务要求的问题，不擅自引入额外功能、依赖或流程复杂度

---

## 核心算子开发原则

> **严格遵循以下原则**

1. **Layer A–L 结构强制**
   - 每个 staged file 与集成 impl 必须遵循 `pypto-op-develop/references/pypto-kernel-design-format.md` 的 Layer A–L 分层
   - 三个起始模板分工：
     - `impl_template.py`（coder，Layer G–K）
     - `golden_template.py`（mathematician/verifier，Layer A–F）
     - `test_template.py`（verifier，Layer L）
2. **职责分离不可越界**
   - golden / impl / test 三文件分离（D3 维度规则强制）
   - coder 不写测试，verifier 不写实现
   - state machine 仅 orchestrator 可写入
3. **遇问题先定位，不简化代码**
   - 第一步：直接搜索 `docs/` API 文档
   - 第二步：查阅官方示例 `examples/`
   - 第三步：定位问题点后修复，**禁止简化代码或推翻重写**
4. **PyPTO 场景以官方资料和仓内样例为准**
   - 在 API 映射、约束、算子行为、编译、精度和性能判断等场景中，优先依据 `docs/`、`examples/`、现有实现和官方文档
   - 当文档、样例与经验推断冲突时，应先指出冲突并回到可核实依据，不凭经验强行定论
   - `examples/` 仅作 **API 用法参考**，不是 production 实现标准；当 example 的写法与 lint / 门禁冲突时以 **lint 为准**，不得以「example 这样写」为由判定 lint 误报或绕过门禁
5. **验证模式优先使用真实 NPU 环境**
   - 若 `npu-smi info` 检测到可用 NPU 环境，且用户未明确要求使用 sim 模式，则禁止使用 sim 模式进行验证
6. **严禁绕过或规避门禁，必须正向解决问题**
   - 算子开发过程中，禁止绕过、规避门禁，或针对门禁报错采用取巧手段临时"过检"
   - 遇到门禁报错时，必须优先正向分析根本原因，并基于规范、实现和验证结果采取正确修复方案
   - 禁止以关闭检查、放宽约束、修改验证条件、伪造结果或其他规避性方式替代真实修复
   - lint 规则与 NPU 运行结果同为硬性客户要求；NPU 运行通过不能替代 lint 通过，lint 失败时必须继续修到门禁允许
7. **失败可回退而非硬撑**
   - Stage 5 中同一 Phase M_k 失败 3 次进入 `blocked` 状态时，应通过 `state_transition(action=rollback_to_stage, target_stage=N, reason=...)` 回到上游阶段（如 architect 或 designer）重新设计
8. **代码仓探索必须使用 subagent**
   - 禁止在 primary agent 中大规模探索代码仓

---

## 入口路径速查

| 你想做什么 | 第一站 |
|---|---|
| 开始一个新算子 | `pypto-op-orchestrator`（自动从 Stage 1 引导） |
| 了解整体编排逻辑 | skill `pypto-orchestration-manual` (SKILL.md auto-loads) |
| 查看某个 Stage 的产出/门禁/交接 | skill `pypto-orchestration-manual`'s `references/agents.md` |
| 查看 lint 规则 | `.agents/hooks/pypto-op-lint/rules.json` |
| 查看 kernel 编码规范 | skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md` |
| 查看 MEMORY.md / state.json 字段 | skill `pypto-memory-template`'s `templates/MEMORY.template.md` 与 `.opencode/plugins/lib/state-transition-core.ts` |
| 查看某个 op 的开发历史 | `custom/<op>/MEMORY.md` + `custom/<op>/.orchestrator_state.json` |

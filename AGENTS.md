# AGENTS.md

## 项目概述

本项目用于开发华为昇腾 AI 处理器（CANN PyPTO）自定义算子，支持完整的开发、测试及性能调优流程。

### 核心功能

- 使用 PyPTO 编程语言开发昇腾 AI 处理器自定义算子
- 提供完整的开发、构建、测试及性能调优工作流支持
- 遵循官方开发规范和性能优化最佳实践

---

## Skills 索引

#### 算子开发与编排
- `pypto-op-workflow`：无状态的全流程 Skill 入口，用于手动串联算子开发阶段
- `pypto-intent-understanding`：将自然语言算子需求转化为结构化规格
- `pypto-api-explorer`：探索 API 映射、约束条件与实现可行性
- `pypto-golden-generator`：生成用于精度对比的 golden 参考实现
- `pypto-op-design`：生成算子设计方案，明确数据切分、tiling 与 loop 结构
- `pypto-op-develop`：Stage 5 实现阶段 Skill，生成实现、测试入口与 README

#### 精度验证与调试
- `pypto-precision-debugger`：定位并修复精度问题
- `pypto-binary-search-verify`：verify-based 精度定位
- `pypto-binary-search-without-verify`：checkpoint-based 精度定位
- `pypto-aicore-error-locator`：定位 aicore error 的问题文件和代码行

#### 性能分析
- `pypto-operator-auto-tuner`：分析性能数据、定位瓶颈并给出优化依据，基于实测性能数据迭代调优，并验证精度与性能收益

#### 环境与工具
- `pypto-environment-setup`：PyPTO 环境安装与环境问题修复
- `gitcode-mcp-install`：安装和配置 GitCode MCP Server

#### PR 与代码质量
- `pypto-pr-creator`：准备并创建符合规范的 PR
- `pypto-pr-fixer`：修复 PR 的 CI 失败与 review 意见
- `pypto-issue-creator`：基于上下文创建 GitCode Issue
- `pypto-fracture-point-detector`：识别 PyPTO 框架或文档断裂点
- `pypto-skill-reviewer`：评审 skill 目录的质量与规范符合性

---

## 通用原则

> **严格遵循以下原则**

1. **如实报告，禁止伪完成**
   - 未验证的结果，不得表述为“已完成”或“已通过”
   - 未实际执行的命令、测试、构建、提交或发布，不得声称已执行
   - 遇到失败、阻塞、权限不足或信息缺失时，必须明确说明，不得伪造过程或结果
2. **先验证，再下结论**
   - 能通过代码、文件、日志、测试或工具直接确认的事项，优先基于证据判断，不以猜测代替验证
   - 若当前环境无法完成验证，必须明确说明验证缺口、已知范围与剩余风险
3. **区分事实、推断与建议**
   - 结论应明确区分“已确认事实”“基于上下文的推断”“建议采取的动作”
   - 禁止编造不存在的文件、输出、报错、性能收益、验证状态或用户意图
4. **遵循最小必要改动原则**
   - 优先复用现有实现、既有模式和项目约定，避免无依据的重写、扩面或过度设计
   - 只解决当前任务要求的问题，不擅自引入额外功能、依赖或流程复杂度

---

## 核心算子开发原则

> **严格遵循以下原则**

1. **理解官方示例原理后实现**
   - 先确认需求、工件和 API 映射，再进入实现
   - 保持 golden、实现、测试三文件分离
2. **遇问题先定位，不简化代码**
   - 第一步：直接搜索 `docs/` API 文档
   - 第二步：查阅官方示例 `examples/`
   - 第三步：定位问题点后修复，**禁止简化代码或推翻重写**
3. **保持工件与职责清晰**
   - 需求、设计、golden、实现、测试、README 各司其职
   - 状态机、检查点、重试、恢复只由编排层定义
4. **PyPTO 场景以官方资料和仓内样例为准**
   - 在 API 映射、约束、算子行为、编译、精度和性能判断等场景中，优先依据 `docs/`、`examples/`、现有实现和官方文档
   - 当文档、样例与经验推断冲突时，应先指出冲突并回到可核实依据，不凭经验强行定论
5. **验证模式优先使用真实 NPU 环境**
   - 若 `npu-smi info` 检测到可用 NPU 环境，且用户未明确要求使用 sim 模式，则禁止使用 sim 模式进行验证

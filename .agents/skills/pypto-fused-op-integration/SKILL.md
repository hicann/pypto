---
name: pypto-fused-op-integration
description: 将 PyPTO 融合算子集成到整网中，替代多个小算子组合的完整工作流程。包含理解验证、算子开发、模型集成、精度验证与性能调优。Whenever users mention "fused op"、"model integration"、"算子融合"、"整网集成"、"replace small ops" or any operator optimization in neural networks (GLM, LLaMA, MoE, Attention), use this skill even if they don't explicitly ask for 'fused operator' or 'integration'.
---

# PyPTO 融合算子整网集成 Skill

将 PyPTO 融合算子替换到整网中，替代多个小算子组合的完整工作流程。

**本文档基于 GLM-4.5 模型实践总结，完整代码示例请参考：**
- 原始文档：`models/glm_v4_5/intergrated_example.md`
- 示例代码：`models/glm_v4_5/` 目录下的算子实现文件

---

## 工作流程概览

**阶段一：前置准备** → 环境验证、需求分析  
**阶段二：理解验证** → 分析原始算子、编写 Golden、验证理解 ⚠️  
**阶段三：算子开发** → 设计方案、实现、验证（如已有可跳过）  
**阶段四：模型集成** → 目录结构、适配层、调用逻辑  
**阶段五：验证调优** → 端到端精度、性能分析、问题排查  
**阶段六：提交文档** → Issue、PR

---

## 详细步骤指南

### 阶段一：前置准备

#### 步骤 1：环境与工程验证

验证 PyPTO 环境和目标工程就绪。

**推荐 Skill：** `pypto-environment-setup`

---

#### 步骤 2：算子需求分析

定位目标算子，分析输入输出规格。

**推荐 Skill：** `pypto-intent-understanding`

---

### 阶段二：理解验证（关键环节）⚠️

> **为什么需要这个阶段？**
> 
> 在分析整网中的小算子组合时，我们往往无法直接确定其精确实现细节。只能根据算子名称、前后依赖关系，人为推测计算逻辑。这个推测是否正确，需要通过 Golden 验证来确认。

#### 步骤 3.1：分析原始小算子组合

**目标：** 深入理解原始实现的计算逻辑

**操作：**

1. **阅读原始代码**
   
   参考 `models/glm_v4_5/intergrated_example.md` 中的原始代码示例。

2. **梳理数据流**
   
   绘制数据流图，记录所有中间变量的形状和 dtype。

**输出物：**
- 原始代码片段注释版
- 数据流图

---

#### 步骤 3.2：编写 Golden 脚本

**目标：** 基于理解编写 PyTorch 参考实现

**操作：**

1. **创建 Golden 文件**
   ```bash
   mkdir -p models/<model_name>/utils/golden
   touch models/<model_name>/utils/golden/<op>_golden.py
   ```

2. **编写 Golden 实现**
   
   参考 `models/glm_v4_5/utils/golden/attn_golden.py` 中的 `attention_pre_golden()` 函数，编写 Golden 参考实现。
   
   或使用 `pypto-golden-generator` skill 自动生成，参考 `.agents/skills/pypto-golden-generate/templates/golden-template.py` 模板。
   
   关键要点：
   - Golden 实现必须是纯 PyTorch 实现，禁止引入 pypto
   - Golden 实现应完整覆盖原始小算子组合的计算逻辑
   - 导出 `{op}_golden()` 函数供测试脚本调用

**输出物：**
- Golden 实现文件（`*_golden.py`）

**推荐 Skill：** `pypto-golden-generator`

**参考示例：**
- `models/glm_v4_5/utils/golden/attn_golden.py:322` (attention_pre_golden 函数)

---

#### 步骤 3.3：验证理解正确性

**目标：** 将 Golden 脚本集成到整网中，验证理解是否正确

**操作：**

1. **创建适配层临时版本**
   
   参考 `models/glm_v4_5/intergrated_example.md` 中的"配置算子适配层接口"章节，创建临时适配层用于验证。

2. **修改模型调用逻辑**
   
   在目标文件中替换原始算子调用为 Golden 实现，参考 intergrated_example.md 的"修改算子调用逻辑"章节。

3. **运行整网验证**
   ```bash
   python examples/run_glm4.py --input "测试输入"
   ```

4. **端到端精度对比**
   
   对比 Golden 替换后的输出与原始输出，检查 max_diff < 1e-3。

**验证检查点：**
- ✅ 模型能正常启动和运行
- ✅ 中间激活值形状、dtype 正确
- ✅ 无 NaN/Inf 等异常值
- ✅ 端到端输出与原始实现对齐（diff < 1e-3）

---

#### 步骤 3.4：决策与迭代

**目标：** 根据验证结果决定下一步行动

- **验证通过 ✅**：Golden 脚本可作为 PyPTO 算子的精度标杆，进入阶段三
- **验证失败 ❌**：重新阅读原始代码，修正理解或参数传递，重新运行验证

---

### 阶段三：PyPTO 算子开发（如已有可跳过）

#### 步骤 4：设计方案

**目标：** 设计 PyPTO 算子的实现方案

**操作：**

参考 `models/glm_v4_5/` 目录下的现有算子设计文档，完成 API 映射分析和 Tiling 策略设计。

**推荐 Skill：** `pypto-op-design`

---

#### 步骤 5：算子实现

**目标：** 编写 PyPTO 算子实现代码

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 6：单算子验证

**目标：** 验证 PyPTO 算子实现正确性

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 7：调整目录结构

**目标：** 创建 PyPTO 算子库目录结构

**操作：**

参考 `models/glm_v4_5/intergrated_example.md` 中的"调整目录结构"章节。

---

#### 步骤 8：配置适配层

**目标：** 在适配层中封装 PyPTO 算子调用

参考 `models/glm_v4_5/intergrated_example.md` 中的"配置算子适配层接口"章节。

关键要点：
- 使用开关变量灵活切换 PyPTO 和原始实现
- 适配层只负责参数转换和调用桥接

---

#### 步骤 9：修改模型调用逻辑

**目标：** 在模型代码中替换原始算子调用

参考 `models/glm_v4_5/intergrated_example.md` 中的"修改算子调用逻辑"章节。

注意事项：
- 严格匹配目标文件和目标函数
- 使用开关变量保证可回滚
   ```

**输出物：**
- 修改后的模型代码
- Git commit 记录

---

### 阶段五：整网验证与调优

#### 步骤 10：端到端精度验证

运行整网推理，验证融合算子精度正确性。

**推荐 Skill：** `pypto-precision-compare`、`pypto-precision-debugger`

---

#### 步骤 11：性能分析与调优

采集性能数据，分析瓶颈并优化。

**推荐 Skill：** `pypto-operator-auto-tuner`、`tune-frontend`、`tune-incore`、`tune-swimlane`

---

#### 步骤 12：问题排查与修复

排查集成过程中的问题。

**推荐 Skill：** `pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debugger`

---

### 阶段六：提交与文档

#### 步骤 13：创建 Issue 跟踪

**推荐 Skill：** `pypto-issue-creator`

---

#### 步骤 14：提交 PR

**推荐 Skill：** `pypto-pr-creator`

---

## 相关资源

### 参考文档
- **完整示例**：`models/glm_v4_5/intergrated_example.md`
- **算子实现**：`models/glm_v4_5/` 目录下的算子文件
- **PyPTO 官方文档**：`docs/`

### 相关 Skill
- `pypto-golden-generator`：生成 Golden 参考实现
- `pypto-op-design`：算子设计方案
- `pypto-op-develop`：算子实现
- `pypto-precision-compare`：精度对比
- `pypto-precision-debugger`：精度调试
- `pypto-operator-auto-tuner`：性能调优
- `pypto-aicore-error-locator`：定位 aicore error
- `pypto-host-stacktrace-analyzer`：分析堆栈信息
- `pypto-issue-creator`：创建 Issue
- `pypto-pr-creator`：创建 PR

---

**Skill 版本：** v1.1
**最后更新：** 2026-04-11
**维护者：** PyPTO Team

## 相关文件
- models/glm_v4_5/glm_attention_pre_quant.py
- models/glm_v4_5/glm_attention_pre_quant_golden.py
- glm_pto_kernels/__init__.py
```

**推荐 Skill：** `pypto-issue-creator`

---

#### 步骤 14：提交 PR

**目标：** 将修改提交到 GitCode 仓库

**操作：**

1. **检查修改**
   ```bash
   git status
   git diff
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/glm-attention-pre-quant
   ```

3. **提交修改**
   ```bash
   git add .
   git commit -m "feat(glm): integrate PyPTO fused attention_pre_quant operator

- Add PyPTO implementation of attention_pre_quant fused operator
- Add golden reference for verification
- Add adapter layer for model integration
- Update Glm4MoeDecoderLayer to use fused operator
- Verified precision and performance

Performance: XX% improvement
Precision: max_diff < 1e-3
"
   ```

4. **推送到远程**
   ```bash
   git push origin feature/glm-attention-pre-quant
   ```

5. **创建 PR**
   ```bash
   # 使用 pypto-pr-creator skill
   # 自动创建符合规范的 PR
   ```

**推荐 Skill：** `pypto-pr-creator`

---

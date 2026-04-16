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

**阶段一：前置准备** → 环境验证、网络基线验证 ★、需求分析  
**阶段二：理解验证** → 分析原始算子、打点采集 ★、编写 Golden、验证理解 ⚠️  
**阶段三：算子开发** → 设计方案、实现、真实用例验证 ★  
**阶段四：模型集成** → 目录结构、适配层、调用逻辑  
**阶段五：验证调优** → 端到端精度（必须）、性能分析（可选）  
**阶段六：提交文档** → Issue、PR

> **★ 标注：** 关键/新增步骤，必须完成  
> **⚠️ 标注：** 需要特别注意的环节  
> **[可选]：** 优先级不高，可在后续迭代中进行

---

## 详细步骤指南

### 阶段一：前置准备

#### 步骤 1：环境与工程验证

验证 PyPTO 环境和目标工程就绪。

**推荐 Skill：** `pypto-environment-setup`

---

#### 步骤 2：网络基线验证 ★ 关键新增

**目标：** 确认目标网络在当前环境下能正常运行，作为所有后续工作的基础。

> **为什么需要这个步骤？**
> 
> 在进行任何算子开发或替换之前，必须确保目标网络本身是可运行的。如果网络本身就无法启动或推理异常，后续的所有工作都建立在不稳定的基础上，会导致问题定位困难、返工浪费。

**操作：**

1. **检查模型资源**

2. **检查 NPU 状态**
   ```bash
   npu-smi info                  # 确认 NPU 驱动正常，进程可见
   ```

3. **查找入口脚本**
   ```bash
   ls -la run*.sh                # 查找运行脚本
   ls -la vllm_infer.py          # 查找推理入口
   ls -la examples/*.py          # 查找示例脚本
   cat README.md                 # 查看项目文档
   ```

4. **如果无法确定运行方式，询问用户**
   
   当尝试以上方式后仍无法确定如何运行整网时，**必须询问用户**：
   
   ```
   我尝试查找整网入口脚本但未找到明确的运行方式。请告知：
   1. 整网入口脚本路径是什么？（如 run_tp8_30B.sh）
   2. 需要哪些环境变量或前置配置？
   3. 模型路径是什么？
   
   请提供运行命令，我将先验证整网能够正常运行，再继续后续算子开发工作。
   ```

5. **执行完整推理**
   - 使用用户提供的或找到的入口脚本
   - 执行一次完整推理
   - 观察启动日志无 Error/Exception

6. **验证推理输出**
   
   **关键成功判断：确认生成对 Prompt 的通顺回答**
   - 输出应当是自然语言，而非乱码或空字符串
   - 例如 Prompt="你好"，输出应为合理的回复
   - 如果输出异常（乱码/截断/重复循环），需排查问题

7. **记录 Baseline**
   - 保存一次推理输出结果
   - 记录原始性能数据

**验证检查点：**
- ✅ 模型路径存在，文件完整
- ✅ NPU 驱动正常，进程可见
- ✅ 已确定整网运行方式（用户告知或自行找到）
- ✅ 网络能正常启动和加载模型
- ✅ 推理完整执行，输出无 NaN/Inf
- ✅ **生成对 Prompt 的通顺回答（关键成功标志）**

**失败处理：**
| 问题类型 | 处理方法 |
|---------|---------|
| 模型路径不存在 | 确认正确路径或更换模型 |
| NPU 异常 | 检查 CANN 安装、权限 |
| 无法确定运行方式 | **询问用户** |
| 网络无法启动 | 先修复依赖/配置 |
| 推理输出异常 | 定位具体错误，修复后继续 |

**Tips for Running Network:**
1. 优先查找 `run*.sh`、`vllm_infer.py`、`examples/*.py`
2. 查看项目 `README.md` 通常包含运行说明
3. 很多脚本需要 `source set_env.sh` 配置环境变量

> **⚠️ 关键原则：**
> 
> 1. 网络基线验证失败时，**必须先修复问题**
> 2. 找不到运行方式时，**必须询问用户**，不能假设或猜测
> 3. 确保整网能够运行后，才继续算子开发工作

---

#### 步骤 3：算子需求分析

定位目标算子，分析输入输出规格。

**推荐 Skill：** `pypto-intent-understand`

---

### 阶段二：理解验证（关键环节）⚠️

> **为什么需要这个阶段？**
> 
> 在分析整网中的小算子组合时，我们往往无法直接确定其精确实现细节。只能根据算子名称、前后依赖关系，人为推测计算逻辑。这个推测是否正确，需要通过 Golden 验证来确认。

#### 步骤 4.1：分析原始小算子组合

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

#### 步骤 4.2：打点采集真实 Tensor 信息 ★ 关键新增

**目标：** 从原始网络中采集待替换算子涉及的 tensor 的真实 shape、dtype，用于构造必须 pass 的测试用例。

> **为什么需要这个步骤？**
> 
> 随机生成的测试用例（如 torch.randn）可能与真实网络场景脱节。只有基于真实网络数据构造的测试用例，才能保证算子在真实场景中正确工作。**这些用例是必须 pass 的基准用例**。

**操作：**

1. **定位打点位置**
   
   在原始网络代码中，找到待替换算子的输入输出位置。

2. **插入打点代码**
   ```python
   # 方式一：打印 tensor 信息
   print(f"[DEBUG] input_query: shape={query.shape}, dtype={query.dtype}")
   print(f"[DEBUG] key_cache: shape={key_cache.shape}, dtype={key_cache.dtype}")
   
   # 方式二：保存 tensor 到文件（可选）
   torch.save(query, "debug_query.pt")
   torch.save(key_cache, "debug_key_cache.pt")
   torch.save(output, "debug_output.pt")
   ```

3. **运行原始网络采集数据**
   ```bash
   python vllm_infer.py --model=/path/to/model
   # 观察控制台输出的 shape、dtype 信息
   ```

4. **记录采集结果**
   
   创建记录文件 `tensor_info.md`：
   ```markdown
   ## 真实 Tensor 信息
   
   | Tensor | Shape | Dtype | 说明 |
   |--------|-------|-------|------|
   | query | [8, 12, 128] | bfloat16 | decode 阶段输入 |
   | key_cache | [1024, 128, 1, 128] | bfloat16 | KV cache |
   | output | [8, 12, 128] | bfloat16 | 输出 |
   ```

5. **基于真实数据构造测试用例**
   ```python
   # 使用真实 shape、dtype
   batch_size = 8
   num_heads = 12
   head_size = 128
   query = torch.randn(batch_size, num_heads, head_size, dtype=torch.bfloat16, device='npu')
   ```

**输出物：**
- tensor 信息记录文件（`tensor_info.md`）
- 基于真实数据构造的测试用例代码

**关键原则：**
- 基于真实网络采集的数据构造的测试用例是**必须 pass 的基准用例**
- 如果这些用例失败，说明算子实现有问题，不能进入集成阶段
- 可以额外增加随机用例进行更广泛测试，但真实用例优先级最高

---

#### 步骤 4.3：编写 Golden 脚本

**目标：** 基于理解编写 PyTorch 参考实现

**操作：**

1. **创建 Golden 文件**
   ```bash
   mkdir -p models/<model_name>/utils/golden
   touch models/<model_name>/utils/golden/<op>_golden.py
   ```

2. **编写 Golden 实现**
   
   参考 `models/glm_v4_5/utils/golden/attn_golden.py` 中的 `attention_pre_golden()` 函数，编写 Golden 参考实现。
   
   或使用 `pypto-golden-generate` skill 自动生成，参考 `.agents/skills/pypto-golden-generate/templates/golden-template.py` 模板。
   
   关键要点：
   - Golden 实现必须是纯 PyTorch 实现，禁止引入 pypto
   - Golden 实现应完整覆盖原始小算子组合的计算逻辑
   - 导出 `{op}_golden()` 函数供测试脚本调用
   - 每个融合算子必须有独立 `{op}_golden.py` 文件，禁止将 golden 逻辑内联在测试或实现文件中

**输出物：**
- Golden 实现文件（`*_golden.py`）

**推荐 Skill：** `pypto-golden-generate`

**参考示例：**
- `models/glm_v4_5/utils/golden/attn_golden.py:322` (attention_pre_golden 函数)

---

#### 步骤 4.4：验证理解正确性

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

#### 步骤 4.5：决策与迭代

**目标：** 根据验证结果决定下一步行动

- **验证通过 ✅**：Golden 脚本可作为 PyPTO 算子的精度标杆，进入阶段三
- **验证失败 ❌**：重新阅读原始代码，修正理解或参数传递，重新运行验证

---

### 阶段三：PyPTO 算子开发（如已有可跳过）

#### 步骤 5：设计方案

**目标：** 设计 PyPTO 算子的实现方案

**操作：**

参考 `models/glm_v4_5/` 目录下的现有算子设计文档，完成 API 映射分析和 Tiling 策略设计。

**推荐 Skill：** `pypto-op-design`

---

#### 步骤 6：算子实现

**目标：** 编写 PyPTO 算子实现代码

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 7：单算子验证

**目标：** 验证 PyPTO 算子实现正确性

**关键说明：**
- 基于步骤 4.2 采集的真实数据构造的测试用例是**必须 pass 的基准用例**
- 如果真实用例失败，说明算子实现与实际场景不匹配，需要修复
- 可以额外增加随机 shape 测试覆盖更多场景，但不能替代真实用例

**验证检查点：**
- ✅ 真实用例（基于采集数据）全部 pass
- ✅ 输出无 NaN/Inf
- ✅ 与 Golden 参考精度对齐（max_diff < 2e-3）

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 8：调整目录结构

**目标：** 创建 PyPTO 算子库目录结构

**操作：**

参考 `models/glm_v4_5/intergrated_example.md` 中的"调整目录结构"章节。

---

#### 步骤 9：配置适配层

**目标：** 在适配层中封装 PyPTO 算子调用

参考 `models/glm_v4_5/intergrated_example.md` 中的"配置算子适配层接口"章节。

关键要点：
- 使用开关变量灵活切换 PyPTO 和原始实现
- 适配层只负责参数转换和调用桥接

---

#### 步骤 10：修改模型调用逻辑

**目标：** 在模型代码中替换原始算子调用

参考 `models/glm_v4_5/intergrated_example.md` 中的"修改算子调用逻辑"章节。

注意事项：
- 严格匹配目标文件和目标函数
- 使用开关变量保证可回滚

**输出物：**
- 修改后的模型代码
- Git commit 记录

---

### 阶段五：整网验证与调优

#### 步骤 11：端到端精度验证

运行整网推理，验证融合算子精度正确性。

**推荐 Skill：** `pypto-precision-compare`、`pypto-precision-debug`

---

#### 步骤 12：性能分析与调优 [可选]

> **⚠️ 优先级说明：**
> 
> **精度验证是必须完成的**，性能调优优先级不高，可在后续迭代中进行。
> 
> 优先级排序：
> 1. 精度正确性（必须）
> 2. 整网集成成功（必须）
> 3. 性能优化（可选，后续迭代）

**目标：** 采集性能数据，分析瓶颈并优化（可选步骤）。

**操作：**

当精度验证通过且需要提升性能时：
- 采集性能数据（泳道图、timeline）
- 分析瓶颈（KV 组装、MatMul、循环开销）
- 优化 Tile 配置、L1 reuse、合图策略

**推荐 Skill：** `pypto-op-perf-tune`、`tune-frontend`、`tune-incore`、`tune-swimlane`

---

#### 步骤 13：问题排查与修复

排查集成过程中的问题。

**推荐 Skill：** `pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debug`

---

### 阶段六：提交与文档

#### 步骤 14：创建 Issue 跟踪

**推荐 Skill：** `pypto-issue-creator`

---

#### 步骤 15：提交 PR

**推荐 Skill：** `pypto-pr-creator`

---

## 相关资源

### 参考文档
- **完整示例**：`models/glm_v4_5/intergrated_example.md`
- **算子实现**：`models/` 目录下的算子文件
- **PyPTO 官方文档**：`docs/`

### 相关 Skill
- `pypto-golden-generate`：生成 Golden 参考实现
- `pypto-op-design`：算子设计方案
- `pypto-op-develop`：算子实现
- `pypto-precision-compare`：精度对比
- `pypto-precision-debug`：精度调试
- `pypto-op-perf-tune`：性能调优
- `pypto-aicore-error-locator`：定位 aicore error
- `pypto-host-stacktrace-analyzer`：分析堆栈信息
- `pypto-issue-creator`：创建 Issue
- `pypto-pr-creator`：创建 PR

---

**Skill 版本：** v1.3
**最后更新：** 2026-04-14
**维护者：** PyPTO Team
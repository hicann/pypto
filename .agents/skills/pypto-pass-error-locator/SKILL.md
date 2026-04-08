---
name: pypto-pass-error-locator
description: PyPTO Pass 模块错误诊断技能。包含错误定位、原因分析和提供问题修复建议，提供从问题定位到修复建议的完整工作流程。当遇到 PyPTO Pass 模块抛出错误时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass 错误诊断与修复技能

本技能提供 PyPTO Pass 模块错误的完整诊断和修复建议能力，从错误原因定位到提供修复建议的端到端解决方案。

## 使用场景

当某业务场景执行失败，需要分析定位某pass抛出异常原因并获取修复建议时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志，提取错误信息
- 识别错误相关的 Pass 模块
- 分析错误产生的根本原因
- 定位问题代码位置
- 提供修复建议

## 前置条件

使用此技能前需要满足以下条件：

1. **环境要求**
   - PyPTO 开发环境已正确配置
   - 可访问 PyPTO 源代码目录
   - 具有日志文件读取权限

2. **输入要求**
   - 用户必须提供复现问题的执行命令
   - 可选：提供相关的测试用例或脚本路径

3. **依赖技能**
   - `pypto-environment-setup`：用于检查环境状态

## 触发机制

当用户输入包含以下错误日志或关键字时，自动触发此技能：

- **定位 pass 错误**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **分析 pass 异常**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 报错**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 失败**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 异常**：定位 Pass 抛出异常的具体原因，并提供修复方案

**触发示例**
- 执行 python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4 异常，pass 报错
- 执行 python3 test.py 异常，分析 pass 失败原因

## 工作流程

### 步骤 1：问题复现

1. 如果用户执行的是python相关的脚本，开启图编译阶段调试模式开关：

   ```python
    @pypto.frontend.jit(
        debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
    )
    ```

2. 设置日志相关环境变量：
   - 设置日志输出目录环境变量：`export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs/$(date +%Y%m%d%H%M%S) `
   - 设置日志输出级别环境变量：`export ASCEND_GLOBAL_LOG_LEVEL=0 `

3. 根据用户提供的执行命令，复现用户问题

**重要**: 上述步骤中的 `设置日志相关环境变量` 与 `根据用户提供的执行命令，复现用户问题`，必须同一个会话中完成，否则环境变量的设置无法生效。

**验证检查点**：
- [ ] 问题成功复现
- [ ] 获得可复现的执行命令
- [ ] 记录复现时的环境信息
- [ ] 记录用户代码文件路径

### 步骤 2：获取日志内容

**日志查找策略：**

1. 从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录下获取 `pypto-*.log`
2. 优先查找pass模块相关的[ERROR]、[WARN]级别的日志

**日志文件验证：**
- 确认日志文件包含 [PASS] 标记
- 确认日志文件包含文件路径和行号信息

**验证检查点**：
- [ ] 成功定位日志文件
- [ ] 日志内容可读取
- [ ] 日志内容包含错误或警告信息
- [ ] 日志内容包含 Pass 模块信息

### 步骤 3：解析日志关键信息

**日志格式示例**：

以如下 pass 模块打印的日志格式为例，分段解析：
```
[ERROR] PYPTO(638465):2026-03-16 10:02:24.711 [n_buffer_merge.cpp:530][PASS]:[NBufferMerge.Config]:The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

**解析字段**：

1. 日志级别：[ERROR]
2. 模块名及进程：PYPTO(638465)
3. 日志打印时间：2026-03-16 10:02:24.711
4. 代码文件及行号：[n_buffer_merge.cpp:530] 此处 `n_buffer_merge.cpp` 为文件名，`530` 为打印该日志代码所在行号
5. 所属模块类型：[PASS]
6. 具体模块信息：[NBufferMerge.Config]，此处 `NBufferMerge` 可能是pass模块名，也可能是其他工具模块名，如果不是pass模块名称要根据日志上下文分析是哪个pass模块调用的该方法
7. 日志内容：The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

**多行错误处理**：
- 检查错误信息是否跨越多行
- 合并相关联的错误上下文
- 提取完整的错误堆栈信息

**日志级别处理**：
- ERROR：必须处理的关键错误
- WARNING：可能导致问题的警告信息
- INFO：辅助调试的信息日志

**验证检查点**：
- [ ] 日志级别正确识别
- [ ] 代码位置准确提取
- [ ] Pass 模块名称正确解析
- [ ] 错误内容完整提取
- [ ] 时间戳格式正确解析

### 步骤 4：异常分类

#### 4.1 常见异常类型

| 异常类型 | 日志特征 | 关键信息提取 |
|---------|---------|-------------|
| **参数配置异常** | `vec_nbuffer_setting`, `cube_l1_reuse_setting`, `cube_nbuffer_setting`, `sg_set_scope` | 异常参数名、参数配置值 |
| **算子约束违反** | `constraint violated`, `invalid parameter` | opcode类型、op_magic信息 |
| **段错误** | `segment fault` | 调用堆栈信息 |
| **属性缺失** | `attribute not found`, `missing attribute`, `required attribute`, `get attribute failed` | 属性名称、算子名称、期望属性类型、缺失的属性列表 |
| **计算图成环** | `cycle detected`, `graph cycle`, `topological sort failed`, `circular dependency` | 成环节点、循环路径、依赖关系、拓扑排序失败节点 |
| **索引越界** | `index out of range`, `invalid index` | 索引值、范围、张量形状 |
| **合轴异常** | `AxisCombine process failed`, `CombineAxis failed` | 轴信息、形状信息、期望维度、实际维度、期望形状和实际形状 |


#### 4.2 将异常进行分类
- 分析用户提供的测试代码，了解用户业务场景
- 根据日志中获取的异常代码位置，分析异常代码上下文(通常需要向上取 20 行，向下取 10 行), 理解异常代码业务逻辑
- 使用 `pypto-pass-module-analyzer` 技能分析理解对应pass模块整体业务逻辑
- 根据日志内容结合代码逻辑，推断异常类型，异常类型及特征如常见异常类型表格中所示

**验证检查点**：
- [ ] 异常类型正确分类
- [ ] 业务场景理解准确
- [ ] 代码上下文分析完整

### 步骤 5：异常分析

#### 5.1 按类型进行异常分析（必须执行）
- 读取 `references/pass-error-analysis-guide.md` 文件
- 根据异常类型定位到对应的分析指导章节
- 严格按照章节定义的步骤顺序执行，记录每步结果
- 不得跳过或合并步骤

#### 5.2 汇总分析结果给出修复建议
- 生成根因链：现象(日志) -> 触发位置(源码切片) -> 状态异常(计算图/IR变化) -> 违反规则(约束文档)
- 汇总异常定位过程中的关键信息及分析结果，形成分析报告，格式参考 `PyPTO Pass 错误分析报告`
- 给出可行的修复建议（如：增加判空、修改算子映射逻辑、调整内存分配策略）

**验证检查点**：
- [ ] 分析指导获取准确
- [ ] 错误原因分析准确
- [ ] 代码位置精确定位
- [ ] 修复建议合理可行

## 输出格式

### 错误分析报告格式

```markdown
## PyPTO Pass 错误分析报告

### 一、基本信息
- **错误级别**: ERROR
- **发生时间**: 2026-03-16 10:02:24.711
- **进程ID**: 638465
- **复现命令**: python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4

### 二、错误位置
- **文件**: n_buffer_merge.cpp
- **行号**: 530
- **Pass模块**: NBufferMerge
- **Element类型**: Config

### 三、错误信息
```
The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

### 四、错误原因分析
1. **主要原因**: 参数配置错误
2. **详细分析**: VEC_NBUFFER_SETTING 参数值 -3 超出有效范围
3. **影响范围**: 仅影响当前Pass模块
4. **相关代码片段**:
   ```cpp
   // n_buffer_merge.cpp:530
   if (key < -1 || key > max_hash_order) {
       GELOGE(INTERNAL_ERROR, "The VEC_NBUFFER_SETTING key %d is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder %d.", key, max_hash_order);
       return INTERNAL_ERROR;
   }
   ```

### 五、修复建议
1. **修复方案**: 调整 VEC_NBUFFER_SETTING 参数值
2. **具体步骤**:
   - 检查参数配置文件
   - 将参数值调整为有效范围（-1 到 0）
   - 重新执行测试用例
3. **风险提示**:
   - 修改参数可能影响性能
   - 建议先在测试环境验证

### 六、附录
- **完整日志**: [日志文件路径]
- **相关文档**: [文档链接]
- **参考案例**: [案例链接]

## 输出成功标准

一个有效的错误分析报告必须满足：
- 报告包含基本信息、错误位置、错误信息、原因分析、修复建议
- 代码位置精确到文件和行号
- 修复建议具体可执行
- 错误原因分析基于日志和源码证据
- 根因链完整（现象 -> 触发位置 -> 状态异常 -> 违反规则）

## 版本兼容性

本技能支持以下 PyPTO 版本：
- PyPTO 8.5.0+
- 建议使用最新版本以获得最佳支持

不同版本可能存在以下差异：
- 日志格式可能略有不同
- Pass 模块名称可能变化
- 错误信息内容可能更新

## 性能优化建议

处理大规模日志文件时：
1. 使用流式读取，避免一次性加载整个文件
2. 优先搜索 ERROR 级别日志，减少处理范围
3. 使用多线程并行处理多个日志文件
4. 缓存已解析的日志信息，避免重复解析

## 使用示例

### 示例 1：参数配置错误

**用户输入：**
```
执行 python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4 异常，pass报错
```

**执行流程：**
1. 设置调试选项和环境变量
2. 执行命令复现问题
3. 从日志中提取提取错误信息
4. 定位到 NBufferMerge.Config Pass
5. 分析参数配置错误
6. 提供修复建议

**输出：**
```markdown
## 错误分析报告

### 错误位置
- 文件: n_buffer_merge.cpp
- 行号: 530
- Pass模块: NBufferMerge
- Element类型: Config

### 错误信息
The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

### 修复建议
将 VEC_NBUFFER_SETTING 参数值从 -3 改为 -1
```

## 参考文档

### 核心文档
- [查看计算图](docs/tools/computation_graph/查看计算图.md)
- [PyPTO IR分析指导](references/ir-analysis-guide.md)
- [计算图JSON解析指导](references/computation-graph-parse.md)
- [pass异常分类分析指导](references/pass-error-analysis-guide.md)

### 相关技能
- [pypto-environment-setup](../pypto-environment-setup/SKILL.md)
- [pypto-pass-module-analyzer](../pypto-pass-module-analyzer/SKILL.md)
- [pypto-pass-workflow-analyzer](../pypto-pass-workflow-analyzer/SKILL.md)

### API文档
- [Pass配置API](docs/api/config/pypto-set_pass_options.md)

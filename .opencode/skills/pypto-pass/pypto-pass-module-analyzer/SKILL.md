---
name: pypto-pass-module-analyzer
description: PyPTO Pass 模块分析技能。用于分析 PyPTO pass 文档中的模块介绍部分，帮助理解各个模块的功能、职责和接口。当需要理解 PyPTO pass 中某个模块的功能和设计时使用此技能。
---

# PyPTO Pass Module Analyzer Skill

## 概述

本技能用于分析 PyPTO pass 文档中的模块介绍部分，结合源代码分析，帮助理解各个模块的功能、职责和接口。

## 功能

- 解析 pass 文档中的模块介绍章节
- 在 framework/src/passes 中查找对应源代码
- 结合文档和代码生成综合分析
（按照 Pass_Analysis_Template.md 模板格式输出）
- 提取模块的关键信息：
  - Pass 概述（名称、类型、阶段、位置）
  - 业务功能（主要功能、处理流程、关键函数逻辑）
  - 在编译流程中的位置（Pass 顺序、依赖关系）
  - 业务价值（适用场景）
  - 架构特定行为
  - 注意事项（包括 OPCode 特判分析）
  - 典型应用场景
  - - 总结
  - 相关文件

## 触发机制

当用户输入包含以下关键字时，自动触发此技能：

- **分析Pass模块XXX**：分析指定 Pass 模块的功能和设计
- **分析Pass代码XXX**：分析指定 Pass 的代码实现
- **介绍Pass XXX**：介绍指定 Pass 的功能和特点
- **Pass XXX的功能是什么**：查询指定 Pass 的功能说明
- **分析XXX Pass**：分析指定 Pass 的详细信息
- **XXX Pass的作用**：查询指定 Pass 的作用
- **XXX Pass做什么**：查询指定 Pass 的功能

**触发示例**：
- "分析Pass模块AutoCast"
- "分析Pass代码SubgraphToFunction"
- "介绍Pass AutoCast"
- "Pass AutoCast的功能是什么"
- "分析AutoCast Pass"
- "AutoCast Pass的作用"
- "AutoCast Pass做什么"

## 使用场景

当需要理解 PyPTO pass 中某个模块的功能和设计时使用此技能。

## 工作流程

### 场景1：指定输入文档时

1. 确认文档描述的是哪个pass模块
2. 总结文档内容
3. 在项目中 framework/src/passes 中查找对应代码
4. 结合代码分析模块实现
5. **特别关注 OPCode 特判分析**：
   - 搜索代码中所有 `GetOpcode()`、`GetOpCode()` 相关的判断
   - 搜索 `Opcode::OP_` 相关的常量使用
   - 搜索 `if (op->GetOpcode() == Opcode::OP_XXX)` 或类似的特判逻辑
   - **重点搜索 OP_VIEW、OP_ASSEMBLE、OP_RESHAPE 的特判**
   - 记录每个特判的场景和条件
   - **详细记录视图类 OPCode 的特殊处理逻辑**
   - 将这些特判场景整理到输出文档的"注意事项"模块中
6. 按照 Pass_Analysis_Template.md 格式生成综合文档

### 场景2：未指定文档时

1. 询问用户是否要查找全部pass
2. 查找特定pass名称：
    - 搜索 pass_manager.cpp 中 PassName 保存的所有pass，询问用户想要分析哪个pass，不要分批展示
    - 根据pass_manager.cpp中的注册信息，在 framework/src/passes 中查找用户选择的pass的代码
    - 分析代码并生成文档
    - **特别关注 OPCode 特判分析**：
      - 搜索代码中所有 `GetOpcode()`、`GetOpCode()` 相关的判断
      - 搜索 `Opcode::OP_` 相关的常量使用
      - 搜索 `if (op->GetOpcode() == Opcode::OP_XXX)` 或类似的特判逻辑
      - **重点搜索 OP_VIEW、OP_ASSEMBLE、OP_RESHAPE 的特判**
      - 记录每个特判的场景和条件
      - **详细记录视图类 OPCode 的特殊处理逻辑**
      - 将这些特判场景整理到输出文档的"注意事项"模块中
    - 按照 Pass_Analysis_Template.md 格式生成综合文档
3. 查找全部：
    - 搜索 pass_manager.cpp 中 PassName 保存的所有pass
    - 根据pass_manager.cpp中的注册信息，依次遍历每个pass的代码
    - 对每个pass分析代码并生成总结文档
    - **特别关注 OPCode 特判分析**：
      - 搜索代码中所有 `GetOpcode()`、`GetOpCode()` 相关的判断
      - 搜索 `Opcode::OP_` 相关的常量使用
      - 搜索 `if (op->GetOpcode() == Opcode::OP_XXX)` 或类似的特判逻辑
      - **重点搜索 OP_VIEW、OP_ASSEMBLE、OP_RESHAPE 的特判**
      - 记录每个特判的场景和条件
      - **详细记录视图类 OPCode 的特殊处理逻辑**
      - 将这些特判场景整理到输出文档的"注意事项"模块中
    - 按照 Pass_Analysis_Template.md 格式生成综合文档

## OPCode 特判分析要点

在分析代码时，需要特别关注以下内容并记录到"注意事项"模块中：

**重点关注的视图类 OPCode**：
- **OP_VIEW**：视图操作，用于创建视图
- **OP_ASSEMBLE**：组装操作，用于组装张量
- **OP_RESHAPE**：重塑操作，用于改变张量形状

这三个视图类 OPCode 在 Pass 中经常需要特殊处理，**必须**在分析时重点记录它们的特判场景。

### 1. 特判类型识别

- **直接特判**：`if (op->GetOpcode() == Opcode::OP_XXX)`
- **集合特判**：`if (xxxOps.count(op->GetOpcode()) > 0)`
- **否定特判**：`if (op->GetOpcode() != Opcode::OP_XXX)`
- **多重特判**：`if (op->GetOpcode() == Opcode::OP_XXX || op->GetOpcode() == Opcode::OP_YYY)`

### 2. 特判场景记录

对于每个 OPCode 特判，记录以下信息：
- **特判的 OPCode**：具体是哪个或哪些操作码
- **特判条件**：判断的具体条件
- **特判位置**：在哪个函数或代码块中
- **处理逻辑**：特判后执行的操作
- **业务含义**：这个特判的业务目的

**特别强调**：
- 如果特判涉及 **OP_VIEW**、**OP_ASSEMBLE**、**OP_RESHAPE**，必须详细记录
- 标注这些视图类 OPCode 的特殊处理逻辑
- 说明为什么这些视图类 OPCode 需要特殊处理

### 3. 特判分类

- **跳过处理**：某些 OPCode 被跳过不处理
- **特殊处理**：某些 OPCode 需要特殊逻辑处理
- **禁止操作**：某些 OPCode 不允许出现
- **兼容处理**：某些 OPCode 需要兼容性处理

### 4. 注意事项模块格式

在输出文档的"注意事项"模块中，OPCode 特判分析应按以下格式组织：

```
## 6. 注意事项

### OP Code 特判分析

**[场景1名称]**：
- 特判 OPCode：[OPCode 列表]
- 特判条件：[条件描述]
- 处理逻辑：[处理说明]
- 业务含义：[业务目的]

**[场景2名称]**：
- 特判 OPCode：[OPCode 列表]
- 特判条件：[条件描述]
- 处理逻辑：[处理说明]
- 业务含义：[业务目的]

### 其他注意事项

- [其他注意事项1]
- [其他注意事项2]
```

## 输出格式

按照 Pass_Analysis_Template.md 模板格式输出，包含以下章节：

1. Pass 概述
2. 业务功能
   - 主要功能
   - 处理流程
     - PreCheck 阶段
     - RunOnFunction 阶段
     - PostCheck 阶段
     - 关键函数的核心逻辑
     - Pass 业务核心逻辑图
3. 在编译流程中的位置
   - Pass 顺序
   - 依赖关系
4. 业务价值
   - 适用场景
5. 架构特定行为
6. 注意事项（包含 OPCode 特判分析）
7. 典型应用场景
8. 总结
9. 相关文件
10. 附录

# PyPTO Pro概述与学习路径

## 什么是PyPTO Pro

PyPTO Pro是一种面向Ascend 950PR/Ascend 950DT、以Python为前端的DSL（Domain-Specific Language，领域特定语言）。它采用SPMD（Single Program Multiple Data，单程序多数据）执行模型，并以二维Tile作为Tile API中Cube、Vector计算和数据搬运的主要载体；对于需要寄存器级编程的场景，还提供基于RegTensor和MaskReg的Reg API。PyPTO Pro通过Tile等抽象，在保留硬件控制能力的同时简化算子开发，并可在经过合理的Tiling切分和流水设计后获得较好的性能。

## 适用场景

PyPTO Pro适用于开发高性能深度学习算子，可以较为快速地实现各种神经网络算子，尤其是Cube和Vector都涉及的融合算子，并达到较为理想的性能。

## 核心特性

- **SPMD执行模型**：参与执行的AI Core运行同一份Kernel代码，并通过核索引划分各自处理的数据。
- **基于Tile的编程模型**：Tile API使用Tile（硬件感知的数据块）描述片上数据及其计算，通过Tile简化坐标偏移计算和指令参数配置。
- **基于TileGroup的核内流水表达**：使用TileGroup将同一流水线中可轮转复用的多块Tile封装为一组，用户通过`next()`和`current()`等接口表达多缓冲流水。使用`@pl.jit(auto_mutex=True)`开启自动同步后，框架根据每块Tile绑定的`mutex_id`自动插入核内同步。
- **融合场景的自动核间流水编排**：在支持的Cube、Vector融合场景中，用户通过stage机制和相应标签表达计算阶段，框架可自动插入核间同步，并进行Preload流水编排。
- **Python前端API**：提供更友好的Python API前端，贴近算法开发者的思维模式。
- **IR与编译优化**：Python前端代码会被转换为PyPTO IR，在IR层完成通用代码优化，再由CCE CodeGen生成Device侧代码并编译为可在NPU上运行的二进制文件。

## 产品支持情况

PyPTO Pro当前支持以下产品型号：

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 学习路径

建议按照以下路径学习PyPTO Pro：

1. **环境准备**：参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。
2. **快速入门**：从[HelloWorld](quick_start/SIMD_programming/HelloWorld.md)开始，了解Kernel函数定义、JIT编译和运行的基本流程；再通过[Add算子快速入门](quick_start/SIMD_programming/Add_operator_quick_start.md)学习Tile配置、数据搬运和计算。
3. **编程模型**：阅读[编程模型概述](../programming_guide/programming_model/programming_model_overview.md)，理解SPMD编程、Tile抽象和流水机制。
4. **编程指南**：深入学习[基于Tile的Python编程](../programming_guide/programming_model/AI_Core_SIMD_programming/tile_based_python_programming/Python_programming_overview.md)，掌握Tensor、Tile、TileGroup等核心数据结构的使用。
5. **API参考**：查阅[SIMD API](../../api/SIMD-API/index.md)和[Utils API](../../api/Utils-API/index.md)，了解各接口的详细用法。

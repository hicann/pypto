# 编程模型概述

如[异构系统](heterogeneous_system.md)章节所述，基于昇腾处理器的应用程序通常分为两部分：**Host代码**与**Device代码**。其中，Host代码运行在CPU上，负责设备资源管理、数据准备及任务调度等；Device代码运行在NPU（神经网络处理器）上，专门执行实际的计算任务。本编程指南重点讲解如何基于**PyPTO Pro编程语言**编写Device代码，以及如何通过Host代码完成Device代码的调度与执行。

## SIMD并行执行模型

为编写高性能的Device端代码，首先需要理解底层的并行计算原理。SIMD定义了指令驱动多计算单元协同工作的核心机制，是提升程序数据吞吐量、优化计算性能的关键技术，也是学习PyPTO Pro编程的核心内容。

### 单指令多数据流

**核心概念**：SIMD（Single Instruction Multiple Data）是一种 **数据并行** 模型，核心逻辑是：一条指令在同一个时钟周期内，对多个数据元素执行完全相同的操作，实现数据的批量并行处理。

**核心特征**：

- 单指令驱动：所有并行计算单元同步执行同一条指令，操作完全一致；
- 数据同构：要求参与计算的数据类型统一、长度相同，确保指令可批量处理；
- 同步执行：所有数据的操作在同一个指令周期内完成，无独立调度逻辑，执行节奏完全统一。

**适用场景**：主要适配数据密集、操作规整、无分支或分支极少的计算任务，典型场景包括：

- 图像像素处理（如灰度化、滤波、像素缩放）；
- 音频信号分析（如降噪、信号预处理）；
- 矩阵乘法、卷积等深度学习核心运算；
- 逐元素数学函数（如向量加减、乘除、指数/对数运算）。

**SIMD[核函数Kernel](AI_Core_SIMD_programming/kernel_function.md)编程四步法**：SIMD编程遵循SPMD模型（Single Program Multiple Data,单程序多数据），即每个AI Core运行同一份核函数，但负责处理不同的数据块，具体步骤如下：

1. **Tiling（分块）设计**：对全局超大数据进行均匀切分，为各AI Core分配大小均衡的独立数据分片，精准适配SPMD多核并行架构，规避单核算力瓶颈，实现全域负载均衡。
2. **数据搬入**：调用`pl.load`/`pl.load_tile`等搬运接口，将计算所需的数据从Device Memory搬运到AI Core的片上Buffer，减少全局内存访问延迟。
3. **数据计算**：调用Tile级别的计算API（如`pl.add`/`pl.matmul`等），一次处理多个同构数据；通过TileGroup + `auto_mutex=True`机制，框架自动插入核内同步，确保计算时数据已就绪。
4. **数据搬出**：调用`pl.store`/`pl.store_tile`等搬运接口，将片上Buffer中完成计算的结果搬运回Device Memory，供后续任务使用。

## AI Core硬件基础

上述抽象的并行执行模型，在昇腾AI处理器中有着清晰的物理硬件对应。Device端的核心计算单元为**AI Core**，是昇腾AI处理器的核心算力载体。单枚昇腾NPU芯片通常集成多个AI Core，各核心可并行协作，大幅提升设备整体计算吞吐量。每个AI Core内部架构模块化分工明确，核心组成组件如下：

- **标量处理单元**：负责处理控制流（如分支、循环）和地址计算，功能类似传统CPU核心，是AI Core的"控制中枢"。
- **向量处理单元**：承担核心向量运算任务，是SIMD并行执行模型的主要硬件载体。
- **矩阵运算单元**：针对矩阵乘加运算做深度硬件优化，是深度学习卷积、全连接层等核心算子的极速加速单元。
- **本地存储**：AI Core内置的高速存储资源（包含L1 Buffer、L0A/L0B/L0C Buffer、UB等），用于缓存实时计算所需数据，可有效规避全局Device Memory的高延迟访问问题，显著提升整体计算效率。

PyPTO Pro将AI Core的片上Buffer抽象为内存空间（`pl.MemorySpace`），开发者通过Tile和内存空间来描述数据在片上的位置，而无需直接操作底层硬件寄存器。详细的内存层次与数据流路径请参考[抽象硬件架构](AI_Core_SIMD_programming/abstract_hardware_architecture.md)。

## AI Core编程模型

PyPTO Pro采用SIMD编程模型，通过向量计算、矩阵计算以及两者的融合计算覆盖深度学习中的核心计算场景。

### SIMD编程

- **能力范围**：支持向量计算、矩阵计算，以及向量与矩阵的融合计算，覆盖深度学习大部分核心场景；
- **适用场景**：规整的、高密度的数据并行任务，如卷积、矩阵乘、逐元素变换等，是昇腾NPU开发的主流选择；
- **优势**：能效比高，指令执行开销低，可充分发挥硬件性能，接近AI Core的峰值计算能力；
- **学习路径**：详见[AI Core SIMD编程](AI_Core_SIMD_programming/overview.md)；算子开发流程参见[基于Tile的Python编程](AI_Core_SIMD_programming/tile_based_python_programming/Python_programming_overview.md)。

## PyPTO Pro的Tile编程模型

在SIMD编程框架内，PyPTO Pro采用**Tile编程模型**，核心思想是使用Tile作为NPU核内计算的载体，通过一系列对于Tile的基本运算和操作来描述完整的SPMD计算流程。所有的计算都以Tensor作为输入，通过搬运操作把数据从Tensor搬运到Tile，经过Tile级别的核内计算后，再将数据从Tile搬运到Tensor。相比于传统SIMD的算子编程，PyPTO Pro保留了几乎所有的硬件控制力度，同时通过Tile的封装，简化了指令参数填写、核内流水、核内同步、核间同步等代码的编写，提升了易用性。

### 核内流水简化表达与自动核内同步

对于Cube、Vector核内流水，PyPTO Pro提供了TileGroup机制，简化了N-Buffer流水的表达。通过`pl.make_tile_group`将同一pipeline的多块tile封装为一组，用户通过`tile_group.next()`获取当前应使用的tile。同时使用该机制，免除了用户插入核内不同pipeline之间同步的烦恼——组中每个tile都打了一个mutex_id标记，当kernel用`auto_mutex=True`编译时，框架会在每次使用轮转tile的前后自动发出`mutex_lock`/`mutex_unlock`。

### 自动Preload流水与自动核间同步

对于融合算子，可以通过stage机制来描述Cube和Vector的计算代码段，框架识别后自动插入核间同步，还可以自动完成经典的Preload核间流水优化。用户只需要表达Cube和Vector的计算逻辑，并打上对应的标签，由框架自动完成preload流水的编排以及核间同步的插入。

### 设计理念

传统的NPU SIMD算子编程面临四大难题：

1. **多核Tiling切分**——算子间差异极大，不存在通用算法
2. **Buffer复用**——算子间差异极大，不存在通用算法
3. **offset计算、参数填写**——代码量较多，容易出错
4. **核内/核间流水排布**（包含同步插入）——复杂度高

由于多核Tiling切分和Buffer复用算子间差异极大，不存在一套通用的切分、复用算法让所有算子都达成较好的理论性能。PyPTO Pro编程将这两点仍然交给算子开发程序员，并通过以下设计理念简化了后两个难题：

1. **通过Tile简化offset计算与参数填写**：全局使用2维的Tile来描述核内buffer，通过Tile将底层的指令进行了封装和简化。用户只需要表达当前Tile在GM Tensor上的坐标即可，不需要将坐标转换成一维offset。同时Tile的表达简化了硬件指令中的参数填写。

2. **通过TileGroup简化核内流水表达**：引入Tile管理机制，通过`make_tile_group`描述一组相同功能的tile，用户通过`next()`/`current()`等接口获取当前应使用的tile。通过`mutex_ids`指定每一块buffer的id，用户无需手动插入同步，框架自动插入对应id的同步。

3. **自动核间流水编排**：对于融合算子，框架自动完成preload流水的编排以及核间同步的插入，用户只需表达计算逻辑并打上标签。

关于Tile的详细信息请参考[基于Tile的Python编程](AI_Core_SIMD_programming/tile_based_python_programming/Python_programming_overview.md)。

## 多核切分

在PyPTO / Ascend术语里，一个**block**就是一个**AI Core**。启动一个kernel时，**同一份kernel代码**会在`block_dim`个核上各跑一遍（SPMD模型）。每个核通过`pl.get_block_idx()`和`pl.get_block_num()`获取自身核编号和总核数，认领自己的数据分片。

最常见、最均衡的多核切分方式是**跨步（strided）**分配：第`core_id`个核处理第`core_id, core_id+num_cores, core_id+2*num_cores, ...`个tile，天然让各核tile数最多相差1，无需手写首/尾核分支代码。

多核切分的详细实践（跨步循环、TilingData、tiling_key、负载均衡方法论）请参考[多核切分与Tiling](AI_Core_SIMD_programming/tile_based_python_programming/multi_core_partitioning_and_Tiling.md)。

## AI Core编程小结

PyPTO Pro采用SIMD并行设计，保留了SIMD高能效、高吞吐量的核心优势。在此基础上，通过Tile编程模型和TileGroup机制，在保留硬件控制力的同时大幅简化了指令参数填写、核内流水、核内同步、核间同步等代码的编写，提升了易用性。开发者可根据算法的**访存模式**（连续规整/离散随机）和**分支密度**（低分支/高分支），基于Tile API完成各类算子开发；对性能要求极高的场景，还可使用Reg API直接操作向量寄存器。Host侧通过PyTorch张量完成数据准备与同步，与Device侧代码协同工作，共同实现异构计算的高效运行。

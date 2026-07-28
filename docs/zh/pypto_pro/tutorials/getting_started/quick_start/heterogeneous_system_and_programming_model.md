# 异构系统与编程模型

> 本文作为算子编程快速入门章节，将从昇腾异构系统基础入手，逐步拆解编程模型核心要素，帮助您快速建立PyPTO Pro算子编程的整体认知，为后续实操开发奠定基础。

---

## 什么是异构系统？

基于昇腾处理器的异构系统，核心是**通过两类功能互补的处理器协同工作**，兼顾通用计算与高密度并行计算，实现整体性能最优。

| 角色 | 组成 | 职责 |
|------|------|------|
| **Host（主机）** | CPU + Host Memory | 通用计算、资源管理、任务调度、统筹协调 |
| **Device（设备）** | 昇腾NPU + Device Memory | 高密度并行计算（深度学习推理/训练、图像处理等） |

> **通俗理解**：Host如同项目经理，负责规划、分配、协调；Device如同专业工程师团队，专注于高效完成高强度、重复性的计算工作。两者分工协作，提升整体效率。

---

## PyPTO Pro应用程序的组成

基于昇腾异构系统的PyPTO Pro应用程序必然包含两部分代码，分工明确、协同运行：

| 代码类型 | 运行位置 | 编程语言 | 核心职责 |
|----------|----------|----------|----------|
| **Host代码** | CPU（Host侧） | **Python**（PyTorch） | 通过PyTorch张量准备数据、调用Kernel函数、同步等待结果 |
| **Device代码** | NPU（Device侧） | **PyPTO Pro** | 执行具体的并行计算任务，称为[核函数](../../programming_guide/programming_model/AI_Core_SIMD_programming/kernel_function.md) |

> **便捷提示**：PyPTO Pro的Host代码与Device代码可编写在同一个`.py`文件中，由`@pl.jit()`装饰器标记Kernel函数，框架自动完成JIT编译与任务下发，简化开发流程。

---

## Host与Device的协作流程

Host侧通过调用**PyTorch NPU接口**完成与Device的协同工作。典型流程如下：

1. **准备数据**：通过PyTorch张量在Device Memory上分配输入/输出空间（张量直接创建在NPU上，无需显式拷贝）。
2. **调用Kernel函数**：Host端调用通过`@pl.jit()`装饰的Kernel函数，框架首次调用时触发JIT编译并下发NPU任务，后续调用直接执行缓存的二进制。
3. **NPU并行计算**：Device侧每个AI Core执行Kernel代码，通过`load`/`load_tile`搬入数据、完成Tile级别计算、通过`store`/`store_tile`搬出结果。
4. **同步等待**：Host端通过`torch.npu.synchronize()`等待NPU执行完成，确保数据计算完整（避免未完成就读取结果）。
5. **访问结果**：计算结果已写入输出张量，同步后即可在Host端访问。

> **关键点**：Host与NPU是异步执行的，第4步的同步必不可少，否则可能读取到未完成的数据。

---

## AI Core：NPU的核心计算单元

NPU是Device侧的计算核心，而 **AI Core**则是NPU内部的"最小计算单元"。一枚NPU芯片通常集成多个AI Core，可并行处理不同的数据块，大幅提升整体计算吞吐量。

每个AI Core内部结构高度优化，专门适配并行计算需求，核心组件包括：

- **标量处理单元**：处理控制流（如分支、循环）和地址计算，类似传统CPU核心，是AI Core的"控制核心"。
- **向量处理单元**：执行向量运算，是SIMD并行模型的主要载体。
- **矩阵运算单元**：专门优化矩阵乘加运算，是深度学习卷积、全连接层等算子的"性能加速核心"。
- **本地存储**：用于缓存计算所需的数据，有效降低对全局Device Memory的访问延迟，提升计算效率。

在PyPTO / Ascend术语里，一个**block**就是一个**AI Core**。每个核通过以下接口获取自身的核编号和总核数：

- `pl.get_block_num()` → 本次一共有几个核（== launch时传的`block_dim`）
- `pl.get_block_idx()` → 当前核的编号，取值`0 .. block_num - 1`
- `pl.get_subblock_idx()` → 核内子块编号（Vector双子核，一般只在需要区分子核时才用）

---

## SIMD并行模型

**SIMD**（Single Instruction Multiple Data，单指令多数据）是PyPTO Pro采用的核心并行执行模型。一条指令同时操作多个同构数据，适用于矩阵乘、卷积、逐元素运算等规整、高密度计算场景。

### 核函数编程基本步骤（遵循SPMD模型）

> **SPMD（Single Program, Multiple Data）**：将每个AI Core抽象成一个Block，通过`pl.get_block_idx()`作为Block索引。每个Block执行同一份算子Kernel代码，基于block_idx划分每个Block的数据处理范围，实现多核负载均衡与并行调度。

1. **Tiling（分块）**：将数据划分为均匀的块，每个AI Core负责一块，实现负载均衡。PyPTO Pro在Kernel内部完成多核Tiling切分，通过`pl.range(core_id, total, num_cores)`实现跨步分配，无需独立Host侧Tiling函数。
2. **数据搬入**：需要**显式调用数据搬运API**（`pl.load`/`pl.load_tile`）将数据从Device Memory搬到片上Buffer。
3. **数据计算**：调用**Tile级别API**（`pl.add`/`pl.matmul`等）完成计算。通过TileGroup + `auto_mutex=True`机制，框架自动插入核内同步。
4. **数据搬出**：需要**显式调用数据搬运API**（`pl.store`/`pl.store_tile`）将片上Buffer中的结果写回Device Memory。

> **提示**：Host端通过`kernel[stream, block_dim](...)`语法调用并运行核函数，其中`block_dim`指定启动的AI Core数量。

---

## 编程模型说明

| 编程模型 | 支持范围 | 适用场景 |
|----------|----------|----------|
| **SIMD** | 向量、矩阵、融合计算 | 规整、高密度任务（卷积、矩阵乘、逐元素变换） |

---

## 小结

昇腾异构计算的核心逻辑可概括为：**Host（CPU）调度 + Device（NPU）执行**

- **Host**侧：统筹资源、调度任务。
- **Device**侧：核心是**AI Core**，采用SIMD并行编程模型。

**算子编程的关键**：合理划分数据、组织访存与计算流程，充分发挥AI Core的硬件性能，即可高效开发出高性能的PyPTO Pro算子。

理解以上概念后，建议通过以下示例上手实践：

- [HelloWorld](SIMD_programming/HelloWorld.md)：体验Kernel定义、JIT编译与运行的完整流程
- [Add算子快速入门](SIMD_programming/Add_operator_quick_start.md)：学习Tile配置、数据搬运与计算
- [Matmul算子快速入门](SIMD_programming/Matmul_operator_quick_start.md)：学习Cube矩阵计算与多核切分

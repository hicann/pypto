# 编程范式

## 引言

SIMT（Single Instruction Multiple Thread，单指令多线程）以Thread为基本执行单元，同一份函数由多个Thread并行执行，每个Thread通过自身索引处理不同的数据。SIMT适合离散数据访问、复杂控制逻辑和并发更新等场景。PyPTO Pro支持在Ascend 950PR/Ascend 950DT的AIV上使用SIMT。

## 异构并行计算核心模型：SIMT编程范式

PyPTO Pro使用Grid、Thread Block和Thread组织SIMT任务：

- **Grid**：由多个Thread Block组成。当前Grid使用X维，`grid_dim().x`表示Vector执行域中的逻辑Block数量，`block_idx().x`表示当前Thread Block的索引。
- **Thread Block**：由一至三维Thread组成。外层JIT Kernel在`pl.section_vector()`中执行一次`pl.simt.launch`，会在当前Vector逻辑Block上启动一个Thread Block。
- **Thread**：执行SIMT函数中的逐线程逻辑，通过`thread_idx()`或`linear_thread_idx()`确定块内位置。

一个SIMT启动包含的Thread总数为Grid中的Thread Block数量与单个Thread Block中Thread数量的乘积。各Thread执行相同的函数代码，但根据Block索引和Thread索引访问不同的数据。

硬件将Thread Block中的Thread按线性顺序划分为Warp，每个Warp包含32个Thread。同一Warp中的Thread执行相同指令，但拥有各自的索引和执行状态；不同Thread进入不同分支时会产生分支发散。详细说明参见[线程架构](../../development/vector_computation/simt_computation.md#线程架构)。

## 硬件底层支撑：SIMT单元架构与内存层级

SIMT函数运行在AIV上，主要使用Warp调度、Vector侧计算资源、线程私有执行资源以及Global Memory和Unified Buffer（UB）。各硬件资源及访存路径参见[抽象硬件架构](abstract_hardware_architecture.md)。

### 核心硬件资源

- **Warp执行资源**：以Warp为单位调度和执行SIMT指令。
- **Vector侧计算资源**：执行各Thread的标量计算和控制逻辑。
- **线程私有执行资源**：保存函数参数、局部变量和中间结果。
- **共享存储资源**：Global Memory可由Grid中的Thread Block访问，UB可由当前Thread Block内的Thread共享。

### SIMT编程内存层级

| 数据对象 | 可见范围 | 数据位置 | 管理方式 |
|---|---|---|---|
| Tensor | Grid | Global Memory | 由Host准备并通过外层Kernel传入 |
| 二维ND Vec Tile | Thread Block | Vector侧UB | 由外层Kernel分配和搬运 |
| Scalar参数和局部标量 | Thread | 线程私有执行资源 | 参数按值传递，局部数据由各Thread独立维护 |

Tensor由Host准备并通过外层Kernel传入；Vec Tile由外层Kernel分配和搬运，再作为完整Tile传给SIMT函数。

## SIMT编程接口体系

### SIMT编程接口特点

PyPTO Pro通过Python DSL提供SIMT编程能力：

- 使用`@pl.simt.function(max_threads=...)`定义可启动的SIMT入口函数，使用`pl.simt.launch`配置Thread Block并启动函数。
- 使用`thread_idx()`、`block_idx()`、`block_dim()`和`grid_dim()`等接口获取线程执行位置。
- 在线程函数中访问Tensor元素、Vec Tile元素和Scalar。
- 使用同步、内存栅栏和原子操作完成多Thread协作。
- 使用`pl.simt`命名空间下的标量计算接口表达逐Thread计算。

接口原型、支持的数据类型和约束请参见[SIMT API](../../../../../api/pro_api/SIMT-API/index.md)。

## 小结

PyPTO Pro SIMT编程通过Grid、Thread Block和Thread组织线程级并行计算，并使用Tensor、Vec Tile和Scalar表达不同作用范围的数据。后续章节分别介绍[抽象硬件架构](abstract_hardware_architecture.md)，以及算子开发中的[线程架构](../../development/vector_computation/simt_computation.md#线程架构)、[SIMT函数](../../development/vector_computation/simt_computation.md#simt函数)、[同步机制](../../development/vector_computation/simt_computation.md#同步机制)、[原子操作](../../development/vector_computation/simt_computation.md#原子操作)和[编程示例](../../development/vector_computation/simt_computation.md#编程示例)。

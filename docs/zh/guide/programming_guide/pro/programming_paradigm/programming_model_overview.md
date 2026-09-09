# 编程模型概述

PyPTO Pro采用Host与Device协同的异构编程方式，并在Device侧提供SIMD和SIMT两种并行编程方式。本节从整体上介绍程序的组成、并行方式、硬件基础和开发流程；SIMD、SIMT各自的具体编程方法在后续章节展开。

## 并行编程模型：SIMD与SIMT

PyPTO Pro面向AI Core提供SIMD和SIMT两种并行编程方式。SIMD以数据块为主要编程对象，适合规整、高吞吐的批量计算；SIMT以线程为基本编程单元，适合不规则访存和复杂控制逻辑。

### SIMD（单指令多数据）

SIMD（Single Instruction Multiple Data）是一种数据并行模型。一条指令同时对多个同构数据元素执行相同操作，从而提高单位指令的数据处理量。

PyPTO Pro使用Tensor描述Global Memory中的数据，使用Tile描述片上存储中的数据块，并提供以下计算方式：

- Memory矢量计算：以UB中的Tile为计算对象，适合通用矢量计算。
- Reg矢量计算：通过VF函数操作Vector Register中的数据，适合寄存器级控制和精细性能优化。
- Cube矩阵计算：使用L1、L0等片上存储中的矩阵Tile完成矩阵乘加等高密度计算。

SIMD适合连续、规整的数据访问，以及对大量数据执行相同操作的场景，例如逐元素计算、归约、数据变换、矩阵乘和Cube/Vector融合计算。

典型的SIMD Kernel按照以下过程组织：

1. **Tiling设计**：将全局数据划分为适合多核和片上存储处理的数据块。
2. **数据搬入**：使用`load`、`move`等接口将数据从Global Memory搬入相应的片上存储。
3. **数据计算**：在Vector或Cube执行域中调用Tile API或Reg API完成计算，并根据数据依赖组织同步。
4. **数据搬出**：使用`store`等接口将结果写回Global Memory。

详细内容参见[SIMD编程范式](SIMD/programming_paradigm.md)。

### SIMT（单指令多线程）

SIMT（Single Instruction Multiple Threads）是一种线程并行模型。同一份SIMT函数由多个线程执行，每个线程通过自身索引处理不同的数据元素，并可独立进行地址计算和条件分支。

PyPTO Pro通过`@pl.simt.function(max_threads=...)`定义SIMT入口函数，并在外层JIT Kernel的Vector执行域中使用`pl.simt.launch`启动Thread Block。线程由硬件按Warp调度，通过`thread_idx()`、`block_idx()`等接口获取执行位置。

SIMT具有以下编程特征：

- 每个线程拥有独立的标量计算结果和控制流状态，可表达逐线程分支与循环。
- 线程可访问Tensor元素和二维ND Vec Tile元素，并通过线程索引确定访问位置。
- Thread Block内的线程可使用同步、内存栅栏和原子操作完成协作。
- 同一Warp中的线程执行不同分支时会产生分支发散，开发时需要关注控制流差异。

SIMT适合离散索引、不规则数据访问、复杂条件分支和并发原子更新等场景。

典型的SIMT计算按照以下过程组织：

1. 使用`@pl.simt.function`定义SIMT入口函数。
2. 在线程函数中通过线程索引确定各线程负责的数据。
3. 使用标量计算、同步或原子接口完成逐线程处理。
4. 在外层JIT Kernel中通过`pl.simt.launch`启动线程块。

PyPTO Pro支持在Ascend 950PR/Ascend 950DT的AIV上使用SIMT。详细内容参见[SIMT编程范式](SIMT/programming_paradigm.md)。

## AI Core硬件基础

昇腾NPU包含多个AI Core，多个AI Core可以并行处理不同的数据分片。AI Core内部包含Scalar、Vector、Cube、片上存储和数据搬运等单元：

| 硬件组成 | 主要职责 | PyPTO Pro中的对应表达 |
|---|---|---|
| Scalar单元 | 控制流、地址计算和指令调度 | Python控制流和标量表达式 |
| Vector单元 | 矢量计算和SIMT线程计算 | `section_vector()`中的Tile API、VF函数和SIMT函数 |
| Cube单元 | 矩阵乘加等矩阵计算 | `section_cube()`中的矩阵计算接口 |
| 片上存储 | 保存计算输入、输出和中间数据 | Vec、Mat、Left、Right、Acc等MemorySpace中的Tile |
| 数据搬运单元 | 在Global Memory与片上存储、不同片上存储之间搬运数据 | `load`、`move`和`store`等接口 |

Ascend 950PR/Ascend 950DT采用AIC与AIV分离架构，AIC主要执行Cube计算，AIV主要执行Vector和SIMT计算。计算与搬运任务在不同流水上执行，存在数据依赖时需要通过同步机制约束执行顺序。

硬件单元、存储层级及其接口映射参见[SIMD抽象硬件架构](SIMD/abstract_hardware_architecture.md)和[SIMT抽象硬件架构](SIMT/abstract_hardware_architecture.md)。

## 开发流程与学习路径

PyPTO Pro算子的典型开发与运行流程如下：

1. 在Host侧使用PyTorch准备输入、输出和工作空间Tensor。
2. 使用`@pl.jit`定义Kernel，并声明Tensor、Scalar或TilingData等参数。
3. 根据数据规模设计多核切分和Tiling，规划Tile及片上存储。
4. 在Kernel中使用SIMD接口组织数据搬运和批量计算，或通过SIMT函数表达逐线程逻辑。
5. 在Host侧通过`kernel[stream, block_dim](...)`启动Kernel；首次调用触发JIT编译。
6. Kernel异步下发后，在Host读取结果前等待Device任务完成。

建议继续阅读以下内容：

- [SIMD编程](SIMD/index.md)：多核切分、Tensor与Tile、矢量计算、矩阵计算和Kernel定义。
- [SIMT编程](SIMT/index.md)：线程架构、内存层级、SIMT函数、同步和原子操作。
- [编译与运行](../development/compilation_and_execution/index.md)：JIT编译和离线二进制编译。
- [PyPTO Pro快速入门](../../../quick_start/pro/index.md)：完整的SIMD和SIMT算子示例。

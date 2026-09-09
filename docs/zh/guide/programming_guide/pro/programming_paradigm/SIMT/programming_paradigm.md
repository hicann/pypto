# 编程范式

SIMT（Single Instruction Multiple Threads，单指令多线程）是一种线程并行模型，以Thread为基本执行单元。同一份程序由多个Thread并行执行，每个Thread根据自身索引处理不同的数据。SIMT允许每个Thread独立寻址，并根据数据进入不同的条件分支或循环。与面向规则数据块、批量执行相同操作的SIMD相比，SIMT更适合表达不规则数据访问和逐Thread控制逻辑。

PyPTO Pro在AIV上提供SIMT编程能力，具体的实现流程参见[SIMT计算](../../development/vector_computation/simt_computation.md)。

## 线程架构

### 线程层次结构

SIMT采用Grid、Thread Block和Thread三级线程层次，从顶层到底层逐级划分并行任务。Grid和Thread Block的规模，以及Thread Block和Thread的坐标，均为包含X、Y、Z三个分量的dim3三维结构。

**Thread（线程）**

Thread是SIMT结构中的最小执行单元。每个Thread独立完成计算任务，拥有独立的寄存器、栈空间和执行状态，并通过自身在Thread Block内的三维坐标处理不同数据。

**Thread Block（线程块）**

Thread Block是Grid的组成单元，由若干Thread组成。block_dim = (block_x, block_y, block_z)表示一个Thread Block在各维度上的Thread数量，三个分量的乘积为Block内Thread总数，当前不超过2048。

Thread Block具有以下特点：

- 同一Thread Block内的Thread执行相同的SIMT入口函数，并具有相同的Thread Block尺寸；
- 块内Thread可以访问共享的Tile，并通过同步机制进行数据交换和协作；
- 定义SIMT入口函数时，可以声明单个Thread Block允许启动的最大Thread数量。

**Grid（线程块网格）**

Grid是SIMT线程层次结构的最顶层，由多个Thread Block组成。grid_dim = (grid_x, grid_y, grid_z)表示Grid在各维度上的Thread Block数量，各Thread Block通过自身坐标标识。

在PyPTO Pro中，Grid具有以下特点：

- 每个执行到SIMT启动点的Vector逻辑Block启动一个Thread Block，Grid规模由外层Kernel实际启动的Vector逻辑Block数量决定，Kernel执行期间不会改变；
- Grid中的所有Thread Block具有相同的尺寸和维度配置；
- 不同Thread Block彼此独立，不能依赖固定的执行顺序；
- 当前Grid仅使用X维，即grid_dim = (grid_x, 1, 1)，对应的Thread Block坐标中Y、Z均为0。

一次SIMT启动的Thread总数为：

$$
\text{total\_threads} = grid_x \times grid_y \times grid_z
\times block_x \times block_y \times block_z
$$

### 线程索引

每个Thread都有对应的三维坐标。开发者通过线程层级查询接口获取Grid、Thread Block和Thread的信息，从而确定当前Thread负责处理的数据。

| PyPTO Pro接口 | 说明 | 返回形式 | 约束 |
|---|---|---|---|
| pypto_pro.language.simt.grid_dim() | Grid在各维度上的Thread Block数量。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 当前仅使用X维，Y、Z维均为1；X维由外层Kernel实际启动的Vector逻辑Block数量决定。 |
| pypto_pro.language.simt.block_dim() | Thread Block在各维度上的Thread数量。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 三个分量的乘积不能超过入口函数的max_threads，且不能超过2048。 |
| pypto_pro.language.simt.block_idx() | 当前Thread Block在Grid中的三维坐标。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | X坐标范围由Grid的X维大小决定，当前Y、Z坐标均为0。 |
| pypto_pro.language.simt.thread_idx() | 当前Thread在Thread Block内的三维坐标。 | dim3形式的三维对象，各分量为DT_UINT32 Scalar。 | 各维坐标范围由Thread Block对应维度的大小决定。 |

块内线性索引和线程索引的具体使用方法参见[SIMT计算](../../development/vector_computation/simt_computation.md)。

## SIMT函数

PyPTO Pro支持SIMT入口函数和SIMT辅助函数。入口函数描述Thread Block中每个Thread执行的计算逻辑；辅助函数用于复用逐Thread逻辑，在调用它的Thread中执行。

| 函数类型 | 定义方式 | 作用 | 调用方式 |
|---|---|---|---|
| SIMT入口函数 | @pypto_pro.language.simt.function(max_threads=N) | 定义每个Thread执行的完整计算，结果写入传入的Tile或Tensor。 | 由外层JIT Kernel通过pypto_pro.language.simt.launch启动。 |
| SIMT辅助函数 | @pypto_pro.language.simt.function | 封装可复用的逐Thread计算，可以不返回值或返回一个Scalar。 | 由SIMT入口函数或其他辅助函数调用。 |

SIMT函数不能由Host直接启动。Host先启动外层JIT Kernel，再由外层Kernel在Vector执行域中启动SIMT入口函数；入口函数可以继续调用辅助函数。具体定义、参数配置和调用代码参见[SIMT计算](../../development/vector_computation/simt_computation.md)。

## 内存层级和操作对象

SIMT函数可以操作Scalar、Tile和Tensor。三类对象对应不同的内存层级和共享范围：

| 操作对象 | 内存层级 | 作用范围 | 主要用途 |
|---|---|---|---|
| Scalar | 寄存器 | Thread私有 | 保存参数、索引、局部变量和标量计算的中间结果。 |
| Tile | Unified Buffer（UB） | Thread Block内共享 | 保存块内数据，并在Thread之间交换中间结果。 |
| Tensor | Global Memory | Grid范围内可访问 | 保存输入、输出和跨Thread Block访问的数据，支持基于运行时索引进行不规则访存。 |

Scalar、Tile和Tensor与寄存器、UB及Global Memory的映射参见[抽象硬件架构](abstract_hardware_architecture.md)。如何基于这些对象完成计算，并使用同步、内存栅栏和原子操作处理数据依赖，参见[SIMT计算](../../development/vector_computation/simt_computation.md)。

# SIMT计算

## 线程架构

SIMT采用Grid（线程块网格）、Thread Block（线程块）和Thread（线程）组成的分层线程结构。开发者通过线程索引将不同的数据分配给不同Thread，硬件再将Thread Block划分为Warp执行。

### 线程层次结构

**Thread（线程）**

Thread是SIMT编程的基本执行单元。每个Thread执行同一份SIMT函数代码，并通过自身索引处理不同的数据。函数中的Scalar参数按值传递，局部标量由各Thread独立计算。

**Thread Block（线程块）**

Thread Block由一至三维Thread组成，同一Thread Block内的Thread具有相同的`block_idx()`和`block_dim()`，可以访问同一个Vec Tile并使用同步接口协作。

`pl.simt.launch`通过`threads`参数配置Thread Block尺寸，未指定的维度补为1：

| `threads`写法 | Thread Block尺寸 |
|---|---|
| `threads=x`或`threads=(x,)` | `(x, 1, 1)` |
| `threads=(x, y)` | `(x, y, 1)` |
| `threads=(x, y, z)` | `(x, y, z)` |

各维乘积不能超过2048，也不能超过SIMT入口函数声明的`max_threads`。同一Thread Block内的Thread可以通过Vec Tile交换数据，并使用`syncthreads()`完成块内阶段同步。

**Grid（线程块网格）**

Grid由多个Thread Block组成。在PyPTO Pro中，每个执行到`pl.simt.launch`的Vector逻辑Block启动一个Thread Block。当前Grid使用X维，`grid_dim().x`等于Vector执行域中的逻辑Block数量，`grid_dim().y`和`grid_dim().z`为1。

同一Grid中的Thread Block具有相同的Thread Block尺寸，彼此独立，不能依赖固定的执行顺序。一个SIMT启动的Thread总数为：

$$
\text{total\_threads} = \text{grid\_dim().x} \times
\text{block\_dim().x} \times \text{block\_dim().y} \times \text{block\_dim().z}
$$

常用线程索引接口如下：

| 接口 | 说明 |
|---|---|
| `pl.simt.grid_dim()` | Grid在各维度上的Thread Block数量 |
| `pl.simt.block_dim()` | 当前Thread Block在各维度上的Thread数量 |
| `pl.simt.block_idx()` | 当前Thread Block在Grid中的索引 |
| `pl.simt.thread_idx()` | 当前Thread在Thread Block中的索引 |
| `pl.simt.linear_thread_idx()` | 当前Thread在Thread Block内按X维优先展开的一维索引 |
| `pl.simt.warp_size()` | 一个Warp包含的Thread数量 |

三维Thread Block按照X维优先展开，块内线性索引为：

$$
\text{local\_idx} = \text{thread\_idx().x}
+ \text{thread\_idx().y} \times \text{block\_dim().x}
+ \text{thread\_idx().z} \times \text{block\_dim().x} \times \text{block\_dim().y}
$$

`linear_thread_idx()`直接返回该线性索引。对于一维Grid和Thread Block，当前Thread对应的全局索引可以表示为：

```python
index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
```

### Warp执行机制

硬件按照块内线性顺序将每32个Thread划分为一个Warp，Warp是SIMT调度和执行的基本单位。同一Warp中的Thread执行相同指令，并保留各自的索引和执行状态。

当同一Warp中的Thread进入不同控制流分支时，硬件分别执行各分支并屏蔽当前分支之外的Thread，这称为分支发散。各分支完成后，Thread重新汇合并继续执行后续代码。

例如，条件`thread_idx().x < 8`会使一个Warp中的前8个Thread与其余Thread进入不同路径。硬件依次执行两条路径，执行其中一条路径时屏蔽另一条路径上的Thread。

### 配置最大线程数

`@pl.simt.function(max_threads=N)`使用编译期整数`N`声明SIMT入口函数允许启动的最大Thread数量，取值范围为[1, 2048]。`pl.simt.launch`配置的`threads`各维乘积不能超过该值。

`max_threads`是启动上限，不是实际启动数量；实际Thread Block尺寸由调用处的`threads`参数确定。PyPTO Pro会将该值生成到底层SIMT函数的启动边界中，编译器据此规划寄存器等线程执行资源。

## SIMT函数

PyPTO Pro的SIMT代码由外层JIT Kernel和SIMT函数组成。外层Kernel由Host启动，负责组织执行域并调用`pl.simt.launch`；SIMT入口函数定义Thread Block中每个Thread执行的逻辑，不由Host直接调用。

```text
Host
└── 启动 @pl.jit 外层Kernel
    └── 进入 pl.section_vector()
        └── pl.simt.launch(...)
            └── 启动 @pl.simt.function(max_threads=...) 入口函数
                └── 可调用 @pl.simt.function 辅助函数
```

### SIMT函数的定义

PyPTO Pro提供两类SIMT函数：

| 函数类型 | 定义方式 | 调用方式 | 返回值 |
|---|---|---|---|
| SIMT入口函数 | `@pl.simt.function(max_threads=N)` | 由外层Kernel通过`pl.simt.launch`启动 | 无返回值 |
| SIMT辅助函数 | `@pl.simt.function` | 由入口函数或其他辅助函数调用 | 无返回值或一个Scalar |

`max_threads`表示入口函数允许启动的最大Thread数量，实际Thread Block尺寸由调用处决定。

```python
import pypto_pro.language as pl


@pl.simt.function(max_threads=256)
def add_one(dst, src, count: pl.DT_UINT32):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if index < count:
        dst[0, index] = src[0, index] + 1.0
```

SIMT辅助函数在调用它的Thread中执行，用于复用逐Thread计算逻辑；调用辅助函数不会创建新的Thread Block，也不会改变当前Thread的索引。

### SIMT函数的调用

外层JIT Kernel在`pl.section_vector()`中通过`pl.simt.launch`启动SIMT入口函数：

```python
@pl.jit()
def add_one_kernel(
    src: pl.Tensor[[1, 1024], pl.DT_FP32],
    dst: pl.Tensor[[1, 1024], pl.DT_FP32],
    count: pl.DT_UINT32,
):
    with pl.section_vector():
        pl.simt.launch(
            add_one,
            threads=256,
            args=(dst, src, count),
        )
```

`threads`配置一个Thread Block的线程尺寸，支持一至三维编译期整数；`args`按顺序传递Scalar、Tensor或Vec Tile。每个执行到`pl.simt.launch`的Vector逻辑Block启动一个Thread Block，因此Grid中的Thread Block数量由外层Kernel的Vector逻辑Block数量决定。`pl.simt.launch`的参数约束参见[launch](../../../../../api/pro_api/SIMT-API/execution/launch.md)。

对于纯Vector Kernel，Host侧通过`kernel[stream, block_dim](...)`启动外层Kernel时，`block_dim`对应Vector逻辑Block数量。对于同时包含Cube和Vector执行域的Kernel，应以SIMT函数中的`grid_dim().x`表示实际Thread Block数量。

### Grid与Thread索引内置接口

SIMT函数通过以下接口获取执行配置和索引：

| 接口 | 说明 |
|---|---|
| `pl.simt.thread_idx()` | 当前Thread在Thread Block内的三维索引 |
| `pl.simt.block_dim()` | 当前Thread Block各维度的Thread数量 |
| `pl.simt.block_idx()` | 当前Thread Block在Grid中的索引 |
| `pl.simt.grid_dim()` | Grid各维度的Thread Block数量 |
| `pl.simt.linear_thread_idx()` | 当前Thread在Thread Block内按X维优先展开的一维索引 |
| `pl.simt.warp_size()` | 一个Warp包含的Thread数量 |

开发者可组合Block索引与Thread索引，计算当前Thread对应的数据位置。线程层次及索引关系参见[线程架构](#线程架构)。

## 同步机制

同一个Thread Block内的多个Thread并行执行，不同Thread的执行进度和内存访问顺序可能不同。在线程之间存在数据依赖时，需要使用同步屏障或内存栅栏约束执行顺序和内存可见性。

- **同步屏障（Barrier）**：等待当前Thread Block内的所有Thread到达指定位置后再继续执行。
- **内存栅栏（Memory Fence）**：约束调用Thread在栅栏前后的内存访问顺序，使栅栏前的内存操作按指定范围可见；内存栅栏不会等待其他Thread。

### 接口概览

| 类型 | PyPTO Pro接口 | 作用范围 |
|---|---|---|
| 同步屏障 | `pl.simt.syncthreads()` | 当前Thread Block |
| 内存栅栏 | `pl.simt.threadfence_block()` | 当前Thread Block可见范围 |
| 内存栅栏 | `pl.simt.threadfence()` | Device可见范围 |

`syncthreads()`将Thread Block内的计算划分为前后两个阶段。该接口需要由当前Thread Block内的所有Thread在统一控制路径上执行，只有全部Thread到达后才能继续执行屏障后的代码。

`threadfence_block()`保证调用Thread在栅栏前的内存访问先于栅栏后的访问，并按Thread Block范围建立可见性；`threadfence()`提供Device范围的内存顺序约束。二者都只作用于调用Thread，不等待其他Thread到达。

不同Thread Block之间没有Grid级同步屏障。跨Block共享状态时，需要使用Global Memory、Device范围内存栅栏和原子操作构建同步关系。同步屏障和内存栅栏均不能将普通的读—改—写序列变为原子操作。

## 原子操作

SIMT编程中，不同Thread可以并发访问Global Memory中的Tensor或Vector侧UB中的Tile。当多个Thread同时对同一地址执行读—改—写操作时，普通访存可能产生数据竞争，导致部分更新丢失。

原子操作用于保证同一地址上的一次读—改—写过程不可被其他Thread打断，典型场景包括计数、状态更新、直方图统计和并行归约。

### 基本语义

原子操作保证的是单次内存更新的原子性。例如，多个Thread同时对同一个计数器执行`pl.simt.atomic_add`时，每次加法都以原子方式完成，不会因多个Thread读取同一个旧值后相互覆盖而丢失更新。

原子操作不保证多个Thread到达该操作的先后顺序，也不是线程同步屏障。多个Thread、Warp或Thread Block同时访问同一地址时，硬件会串行处理冲突的原子请求，但程序不能依赖其处理顺序。

PyPTO Pro提供加减、交换、比较交换、极值和按位更新等原子操作。不同接口支持的数据类型、访问空间和返回值约束不同，详细说明请参见[原子操作API](../../../../../api/pro_api/SIMT-API/atomic/index.md)。

## 编程示例

考虑从形状为100000 × 128的二维Tensor中读取`indices`指定的12288行数据：

```text
output[row, col] = input_tensor[indices[row], col]
```

每个Thread负责一个输出行。Thread先根据Block索引和块内索引计算输出行号，再读取对应的输入行并遍历列：

- `block_idx().x * block_dim().x + thread_idx().x`计算当前Thread负责的输出行号。
- `indices[0, row]`读取该输出行对应的输入行号。
- 当前Thread遍历128列，将指定输入行复制到输出Tensor。
- 边界检查确保多余Thread不会访问超出`row_count`的行。

```python
import pypto_pro.language as pl


INPUT_ROWS = 100000
WIDTH = 128
OUTPUT_ROWS = 12288
THREADS = 256


@pl.simt.function(max_threads=THREADS)
def gather_rows(
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    row_count: pl.DT_UINT32,
):
    row = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if row < row_count:
        input_row = indices[0, row]
        for col in pl.range(0, WIDTH, 1):
            output[row, col] = input_tensor[input_row, col]


@pl.jit()
def gather_kernel(
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    row_count: pl.DT_UINT32,
):
    with pl.section_vector():
        pl.simt.launch(
            gather_rows,
            threads=THREADS,
            args=(output, input_tensor, indices, row_count),
        )
```

处理12288行数据时，需要的Thread Block数量为：

$$
\text{blocks} = \left\lceil\frac{12288}{256}\right\rceil = 48
$$

外层Kernel使用48个Vector逻辑Block，每个逻辑Block启动一个包含256个Thread的Thread Block。Host侧通过`gather_kernel[stream, 48](...)`启动Kernel，共形成12288个Thread。完整的SIMT入门流程参见[Add算子（SIMT）](../../../../quick_start/pro/add_simt.md)。

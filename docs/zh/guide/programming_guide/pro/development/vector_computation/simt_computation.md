# SIMT计算

SIMT计算以Thread为基本执行单元。同一份SIMT函数由多个Thread并行执行，每个Thread根据自身索引处理不同的数据，适合表达离散索引、不规则访存和逐Thread条件判断等计算。

SIMT函数不能由Host直接启动。开发者需要先定义SIMT函数，再由外层JIT Kernel在Vector执行域中启动。每个执行到启动点的Vector逻辑Block启动一个Thread Block，Thread Block的尺寸由启动参数决定。

SIMT的线程架构、函数关系以及Scalar、Tile、Tensor操作对象参见[SIMT编程范式](../../programming_paradigm/SIMT/programming_paradigm.md)。本文介绍SIMT计算流程。

## 支持的功能

PyPTO Pro当前提供以下SIMT编程能力：

| 功能 | 主要接口或数据对象 | 说明 |
|---|---|---|
| 函数定义与启动 | @pypto_pro.language.simt.function、pypto_pro.language.simt.launch | 定义SIMT入口函数或辅助函数，并从外层JIT Kernel启动Thread Block。 |
| 线程位置获取 | pypto_pro.language.simt.thread_idx()、pypto_pro.language.simt.block_dim()、pypto_pro.language.simt.block_idx()、pypto_pro.language.simt.grid_dim()、pypto_pro.language.simt.linear_thread_idx() | 获取当前Thread、Thread Block和Grid的索引或尺寸。 |
| 数据访问 | Scalar、Tensor、Tile | 使用Scalar保存逐Thread数据，访问Global Memory中的Tensor或UB中的Tile。 |
| 标量计算 | 类型转换、数学函数、舍入、判断和融合乘加等接口 | 对每个Thread中的Scalar执行计算。 |
| 线程协作 | pypto_pro.language.simt.syncthreads()、pypto_pro.language.simt.threadfence_block()、pypto_pro.language.simt.threadfence() | 完成Thread Block内同步，或约束Block、Device范围内的内存访问顺序。 |
| 原子操作 | 原子加减、交换、比较交换、极值和按位操作等接口 | 对Tensor或Tile中的共享地址执行原子更新。 |

不同接口支持的数据类型、存储空间和返回值存在差异，具体约束参见[SIMT API](../../../../../api/pro_api/SIMT-API/index.md)。

### 同步

当同一Thread Block内的Thread需要交换数据时，可以根据数据依赖选择以下接口：

- pypto_pro.language.simt.syncthreads()等待块内所有Thread到达同步点，适用于共享Tile的分阶段读写。
- pypto_pro.language.simt.threadfence_block()和pypto_pro.language.simt.threadfence()分别约束Block和Device范围内的内存访问顺序，但不会等待其他Thread。

同步屏障需要由同一Thread Block内的所有Thread在一致的控制路径上执行。内存栅栏不能替代同步屏障，也不能保证普通读—改—写操作的原子性。

### 原子操作

多个Thread并发更新同一Tensor或Tile地址时，应使用原子操作避免更新丢失。PyPTO Pro提供原子加减、交换、比较交换、极值、自增自减和按位更新等接口，具体的数据类型和存储空间约束参见[原子操作API](../../../../../api/pro_api/SIMT-API/atomic/index.md)。

## 编写约束

- 使用@pypto_pro.language.simt.function(max_threads=N)定义可由pypto_pro.language.simt.launch启动的SIMT入口函数；不带max_threads时定义SIMT辅助函数。
- 启动接口只能在Vector执行域中调用，不能在SIMT函数中嵌套启动。
- threads支持一至三维编译期整数，各维乘积不能超过入口函数声明的max_threads，总Thread数量不能超过2048。
- args可以传递Scalar、ND Tensor或UB中的静态二维ND Tile。Tensor和Tile需要以完整变量传入，不支持传入元素、Slice或Tile subview。
- 同一Thread Block内的Thread可以共享Tile并执行块内同步；不同Thread Block彼此独立，不能依赖固定的执行顺序。

## SIMT计算流程

当前版本的SIMT计算流程包括定义SIMT函数、配置启动参数、定义外层JIT Kernel和启动Device侧Kernel。

### 1. 定义SIMT函数

#### 入口函数

使用@pypto_pro.language.simt.function(max_threads=N)定义入口函数。入口函数描述每个Thread执行的计算，max_threads声明单个Thread Block允许启动的最大Thread数量；实际尺寸由调用处的threads决定。

入口函数没有返回值，计算结果需要写入传入的Tensor或Tile。

#### 辅助函数

使用不带max_threads的@pypto_pro.language.simt.function定义辅助函数。辅助函数在当前Thread中执行，不会启动新的Thread；可以不返回值，也可以返回一个Scalar。

#### 编写逐Thread处理逻辑

入口函数描述单个Thread执行的处理逻辑，包括根据线程索引确定数据位置，以及完成数据读取、计算和写回。需要复用的逐Thread计算可以封装为辅助函数，并在入口函数中直接调用。以下示例使用辅助函数affine完成每个Thread的数据计算：

```python
import pypto_pro.language as pl


@pl.simt.function
def affine(value: pl.DT_FP32, scale: pl.DT_FP32, bias: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + bias


@pl.simt.function(max_threads=256)
def transform(
    output: pl.Tensor[[1, 1024], pl.DT_FP32],
    input_tensor: pl.Tensor[[1, 1024], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if index < count:
        output[0, index] = affine(input_tensor[0, index], scale, bias)
```

示例中的边界判断用于避免多出的Thread越界访问。涉及共享数据依赖时，参见[同步](#同步)和[原子操作](#原子操作)。

### 2. 配置pypto_pro.language.simt.launch

pypto_pro.language.simt.launch通过callee指定SIMT入口函数，通过threads设置Thread Block尺寸，通过args传递函数参数。callee必须是配置了max_threads的入口函数；threads的各维乘积不能超过该值；args的数量、顺序和类型需要与入口函数的形参一致。

### 3. 定义外层JIT Kernel

当前版本需要在外层JIT Kernel的Vector执行域中调用启动接口。以下代码启动步骤1定义的transform入口函数：

```python
@pl.jit()
def transform_kernel(
    input_tensor: pl.Tensor[[1, 1024], pl.DT_FP32],
    output: pl.Tensor[[1, 1024], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        pl.simt.launch(
            transform,
            threads=256,
            args=(output, input_tensor, count, scale, bias),
        )
```

Tensor和Tile需要以完整变量传入。启动接口的完整约束参见[launch](../../../../../api/pro_api/SIMT-API/execution/launch.md)。

### 4. 启动Device侧Kernel

Host侧通过Bracket Launch语法将外层JIT Kernel启动到Device执行。如果一次SIMT计算需要多个Thread Block，需要设置相应数量的block_dim启动外层Kernel。Host侧block_dim的设置参见[多核Tiling切分](../tiling/multi_core_tiling.md#在启动时设置逻辑block数block_dim)：

```python
THREADS = 256
blocks = (task_count + THREADS - 1) // THREADS
transform_kernel[stream, blocks](input_tensor, output, task_count, scale, bias)
```

每个Vector逻辑Block启动一个Thread Block，因此一维配置下本次SIMT计算最多覆盖blocks * THREADS个任务。

## 示例：使用SIMT实现Gather

以下示例从形状为100000 × 128的输入Tensor中读取indices指定的12288行数据。每个Thread负责一个输出行，使用动态索引读取对应的输入行：

```text
output[row, col] = input_tensor[indices[row], col]
```

```python
import pypto_pro.language as pl


INPUT_ROWS = 100000
WIDTH = 128
OUTPUT_ROWS = 12288
THREADS = 256
BLOCKS = (OUTPUT_ROWS + THREADS - 1) // THREADS


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

Host侧使用BLOCKS个逻辑Block启动外层Kernel：

```python
gather_kernel[stream, BLOCKS](input_tensor, indices, output, OUTPUT_ROWS)
```

本例中，BLOCKS为48，每个Thread Block包含256个Thread，共启动12288个Thread。每个Thread通过Block索引和块内Thread索引计算row，边界判断用于处理数据量不能被Thread Block尺寸整除的通用情况。

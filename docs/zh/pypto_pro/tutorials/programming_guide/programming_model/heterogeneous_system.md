# 异构系统

一个基于昇腾处理器的异构系统通常包含CPU与昇腾NPU。其中，CPU及其内存称为Host与Host Memory；NPU及其内存称为Device与Device Memory。

基于昇腾的PyPTO Pro应用程序通常包含两部分：一部分运行在Host CPU上，使用Python（PyTorch）编程；另一部分运行在NPU上，使用PyPTO Pro编程语言编写。运行在NPU上的代码称为[核函数Kernel](./AI_Core_SIMD_programming/kernel_function.md)，需由Host代码调用执行。Host端通过PyTorch张量在Device Memory上准备输入输出数据，调用Kernel函数触发JIT编译并下发NPU任务，通过`torch.npu.synchronize()`等待核函数执行完成。Host代码与核函数可编写在同一个`.py`文件中，由PyPTO Pro框架完成异构编译。

为提升计算效率，NPU上通常有多个计算核并发执行，每个核一般处理不同的数据。每个NPU的计算核称为AI Core（AI处理器的计算核心）。AI Core遵循SIMD（Single Instruction Multiple Data，单指令多数据流）模型，通过一条指令同时操作多个数据实现并行计算。

下图演示了运行在Host CPU上的代码通过调用PyPTO Pro API将核函数下发到AI Core的流程。由于Host Memory与Device Memory拥有独立的内存空间，需要通过PyTorch张量在NPU上的分配与访问完成数据传输。典型流程为：Host代码在Device Memory上准备输入张量，供核函数计算使用；等待核函数执行完成；最后访问输出张量的结果。

**图1**  Kernel调度示意图

![Kernel调度示意图](../../figures/kernel_scheduling_diagram.png)

## Host与Device的分工

| 角色 | 职责 |
|:---|:---|
| **Host（CPU）** | 通过PyTorch张量准备输入输出数据；调用Kernel函数触发JIT编译和NPU任务下发；通过`torch.npu.synchronize()`同步等待结果 |
| **Device（NPU）** | 执行编译后的Kernel二进制；在AI Core上完成Tile级别的数据搬运与计算；将结果写回Global Memory |

## AI Core与Block

在PyPTO / Ascend术语里，一个**block**就是一个**AI Core**。启动一个kernel时，同一份kernel代码会在`block_dim`个核上各跑一遍（SPMD模型）。每个核通过以下接口获取自身的核编号和总核数：

- `pl.get_block_num()` → 本次一共有几个核（== launch时传的`block_dim`）
- `pl.get_block_idx()` → 当前核的编号，取值`0 .. block_num - 1`
- `pl.get_subblock_idx()` → 核内子块编号（Vector双子核，一般只在需要区分子核时才用）

三个API的层级关系：

```text
block_dim 个 block（AI Core）        <- launch 时决定，get_block_num() 读到
   └── 每个 block 内 2 个 subblock   <- Vector 双子核，get_subblock_idx() 读到
```

## 数据流

PyPTO Pro Kernel的典型数据流如下：

1. Host端准备输入数据（PyTorch张量在NPU上）
2. Host端调用Kernel函数，框架完成JIT编译并下发任务
3. Device端每个AI Core执行Kernel代码：
   - 通过`load`/`load_tile`将数据从Global Memory搬运到片上Buffer
   - 在片上完成Tile级别的计算
   - 通过`store`/`store_tile`将结果从片上Buffer写回Global Memory
4. Host端通过`torch.npu.synchronize()`同步后访问结果

## 执行模型

PyPTO Pro的Kernel基于SPMD（Single Program Multiple Data）执行模型，同一份kernel代码会在`block_dim`个核上各跑一遍，每个核通过`pl.get_block_idx()`认领自己的数据分片。框架自动完成Kernel的编译、加载与任务下发。

关于执行模型的详细配置请参考[异步执行](../compilation_and_execution/asynchronous_execution.md)。

## 多核切分数据流

PyPTO Pro在Kernel内部完成多核Tiling切分（无独立Host侧Tiling函数）。每个AI Core通过`pl.get_block_idx()`认领自己的数据分片，通过`pl.range(core_id, total, num_cores)`实现跨步分配。以`[M, N]`张量按`[TILE_M, TILE_N]`切块、跨步分配到`num_cores`个核为例：输入张量被切分为`m_tile_num × n_tile_num`个tile，第`core_id`个核认领tile序号为`core_id, core_id+num_cores, core_id+2*num_cores, ...`的块，各核独立完成`load → 计算 → store`后写回输出张量。

跨步分配天然让各核tile数最多相差1，即负载均衡，无需手写首/尾核分支代码。多核切分的详细实践请参考[多核切分与Tiling](AI_Core_SIMD_programming/tile_based_python_programming/multi_core_partitioning_and_Tiling.md)。

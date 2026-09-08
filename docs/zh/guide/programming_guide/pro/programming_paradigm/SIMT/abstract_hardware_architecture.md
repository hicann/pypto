# 抽象硬件架构

PyPTO Pro的SIMT函数运行在支持SIMT模式的AIV上，目前支持Ascend 950PR/Ascend 950DT。SIMT允许同一条指令中的不同Thread独立寻址和计算，适合离散数据访问和复杂控制逻辑等场景。

昇腾NPU包含多个AIV。每个AIV内部包含Warp调度与计算资源、寄存器和Unified Buffer（UB）；AIV外部的Global Memory由不同AIV共享，L2 Cache位于Global Memory与各AIV之间。

SIMT多线程计算涉及的主要硬件资源如下：

- **Warp调度与计算资源**：硬件将一个Thread Block划分为多个Warp，每个Warp包含32个Thread。同一Warp中的Thread执行相同指令，Vector侧计算资源完成各Thread的运算。
- **寄存器**：每个Thread拥有独立的寄存器，用于保存参数、局部变量和中间结果。AIV上的寄存器总量有限，入口函数配置的最大Thread数量会影响单个Thread可使用的寄存器资源。
- **Unified Buffer（UB）**：UB中的一部分空间作为Thread Block内共享内存，供块内Thread交换数据；另一部分作为Data Cache，用于缓存SIMT线程访问的Global Memory数据。PyPTO Pro使用`MemorySpace.Vec`中的二维ND Tile表示可由Thread Block共享的数据。
- **L2 Cache**：L2 Cache由所有AIV共享，位于Global Memory与各AIV的Data Cache之间，用于缓存Global Memory数据并降低访问延迟，由硬件自动管理。
- **Global Memory**：Global Memory保存输入、输出和跨Thread Block共享的数据，PyPTO Pro使用Tensor表示其中的数据。SIMT线程读取Global Memory时，数据经过L2 Cache和AIV上的Data Cache后进入Thread私有寄存器。

在PyPTO Pro中，外层JIT Kernel负责分配Vec Tile并组织Global Memory与UB之间的数据搬运，再在`pl.section_vector()`中通过`pl.simt.launch`启动SIMT入口函数。`pl.simt.launch`配置Thread Block尺寸并传递Tensor、Vec Tile或Scalar，不单独配置Data Cache空间。

线程与Warp的组织方式参见[线程架构](../../development/vector_computation/simt_computation.md#线程架构)。

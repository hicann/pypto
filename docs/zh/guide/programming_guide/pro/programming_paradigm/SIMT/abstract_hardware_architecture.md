# 抽象硬件架构

SIMT编程允许不同Thread独立寻址和计算，适合表达离散数据访问、复杂控制逻辑和线程协作。PyPTO Pro的SIMT函数运行在支持SIMT模式的AIV上。

## 硬件组成

昇腾NPU包含多个AIV。每个AIV包含SIMT计算资源、寄存器和Unified Buffer（UB），AIV外部的L2 Cache和Global Memory由多个AIV共享。SIMT多线程计算主要涉及以下硬件资源。

### SIMT计算资源

SIMT计算资源负责组织并执行Thread任务。一个Thread Block包含多个Thread，各Thread执行同一份SIMT函数，并根据各自的线程索引访问和处理数据。不同Thread可以具有独立的局部变量，也可以根据运行时条件执行不同的控制分支。

### 寄存器

每个Thread拥有独立的寄存器，用于保存函数参数、线程索引、局部变量和中间结果。AIV上的寄存器总量有限，Thread Block内的Thread数量以及单个Thread的计算复杂度都会影响寄存器资源的使用。

### Unified Buffer

UB中的部分空间用于保存Thread Block内共享的数据，支持块内Thread交换和复用中间结果；部分空间作为Data Cache，缓存SIMT线程访问的Global Memory数据。

### L2 Cache和Global Memory

Global Memory用于保存输入、输出以及不同Thread Block访问的数据。L2 Cache位于AIV与Global Memory之间，由多个AIV共享并由硬件管理。SIMT线程访问Global Memory时，数据经过L2 Cache和AIV内的Data Cache，最终进入Thread私有寄存器。

## PyPTO Pro中的编程映射

PyPTO Pro中的数据对象与上述硬件存储资源对应如下：

- Thread私有的Scalar通常保存在寄存器中，用于表示线程索引、局部变量和中间结果。
- Tile位于UB中，可由同一Thread Block内的Thread共享。
- Tensor位于Global Memory中，用于保存输入、输出和全局数据。

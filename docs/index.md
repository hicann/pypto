# PyPTO文档中心

欢迎使用PyPTO文档。本文档由华为技术有限公司主导编写，并包含来自CANN社区参与者的贡献。保留所有权利。

PyPTO（发音:pai p-t-o）是CANN推出的一款面向AI加速器的高效编程框架，旨在简化算子开发流程，同时保持高性能计算能力。该框架采用创新的PTO（Parallel Tensor/Tile Operation）编程范式，以基于Tile的编程模型为核心设计理念，通过多层次的计算图表达，将用户通过API构建的AI模型从高层次的Tensor计算图逐步编译成硬件指令，最终生成可在目标平台上高效执行的代码，并由设备侧以MPMD（Multiple Program Multiple Data）方式调度执行。

```{toctree}
:maxdepth: 2
:caption: 目录

install/index
tutorials/index
api/index
tools/index
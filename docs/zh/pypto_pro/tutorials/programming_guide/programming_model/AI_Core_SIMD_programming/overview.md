# 概述

## 引言

SIMD并行机制是AI Core的核心算力支撑，承担了整机90%以上的计算吞吐量。因此，本章将以**SIMD编程模型**为核心主线，遵循「宏观架构—硬件调度—代码落地」的递进逻辑，完整覆盖SIMD算子开发全链路：阐述多AI Core集群的并行任务分发机制，拆解单AI Core内部异构计算单元的任务调度逻辑，并逐层讲解如何基于PyPTO Pro的Python编程接口，完成[算子Kernel](kernel_function.md)计算逻辑的开发。

PyPTO Pro的Tile和VF接口建立在AI Core通用SIMD存储层级之上：Cube侧使用L1、L0A、L0B、L0C Buffer，Vector侧使用Unified Buffer（UB）和Register File，输入输出Tensor位于Global Memory。

**图1**  SIMD-Reg向量计算内存层级

![SIMD-Reg向量计算内存层级](../../../figures/simd_reg_vector_memory_hierarchy.jpg)

## 异构并行计算核心模型：SPMD嵌套SIMD编程范式

昇腾NPU异构并行体系采用**外层多核SPMD（Single Program Multiple Data）并行 + 内层单核SIMD细粒度并行**的双层融合架构，是AI Core算子开发的核心理论根基。所有SIMD算子的并行逻辑、数据拆分策略、任务调度机制，均基于该编程范式实现。

**多核集群层面遵循SPMD编程模型**：将每个AI Core抽象成一个Block，通过`pl.get_block_idx()`作为Block索引。每个Block执行同一份算子Kernel代码，基于`pl.get_block_idx()`划分每个Block的数据处理范围，实现多核负载均衡与并行调度。

**单核计算层面依托SIMD并行机制**：单个AI Core内部基于SIMD机制实现细粒度并行，依靠硬件专用计算单元，单条指令批量完成多组同质数据的并行运算，充分挖掘单AI Core的极致算力潜力。

## 硬件底层支撑：AI Core计算单元架构与算子分类

AI Core是昇腾NPU的基础核心计算单元，采用「控制单元+异构计算单元+分级本地存储」的经典架构，各组件分工明确、协同联动。硬件的能力边界直接决定算子的计算形态与开发方式，下文将详细介绍AI Core核心硬件组件，以及基于硬件能力划分的标准化算子类型。

### AI Core核心硬件组件

- **标量处理单元**：作为AI Core的控制中枢，主要负责地址偏移计算、指令调度与发射，统筹管控其他运算单元的指令执行流程，支撑分支、循环等各类控制流逻辑的正常运行。
- **向量处理单元Vector**：遵循标准SIMD并行计算逻辑，专职执行各类向量指令，支持单指令多数据并行运算，适配元素级计算、逻辑运算、数据重组等灵活性要求高的计算场景。
- **矩阵运算单元Cube**：高密度张量专用算力单元，面向矩阵乘加、高维张量卷积等算力密集型场景深度优化。硬件原生支持批量矩阵运算，典型能力为单次完成一组float16类型16×16矩阵乘法，是AI模型训练与推理场景的核心算力支撑。
- **本地存储**：AI Core片内高速存储体系，用于缓存计算中间数据，规避频繁访问低速全局内存的性能损耗，降低访存延迟。其中Cube计算单元配套L1 Buffer、L0C Buffer等；Vector计算单元配套统一缓存UB（Unified Buffer）；Ascend 950PR/Ascend 950DT的AI Core向量单元新增可编程向量寄存器（Register），单寄存器大小为256B。

PyPTO Pro将AI Core的片上buffer抽象为内存空间（`pl.MemorySpace`），开发者通过Tile和内存空间来描述数据在片上的位置，而无需直接操作底层硬件寄存器。详细的内存层次与数据流路径请参考[抽象硬件架构](abstract_hardware_architecture.md)。

### 基于硬件单元的三类标准算子

根据算子核心计算逻辑对硬件单元的依赖差异，AI Core算子可划分为三大标准类型：

- **矢量类算子**：核心计算逻辑完全由Vector计算单元承载，无矩阵密集运算，以元素级、逻辑类、数据重组类计算为主，适配访存密集、灵活度要求高的场景。在PyPTO Pro中通过`pl.section_vector()`标记执行域。
- **矩阵类算子**：核心计算逻辑完全由Cube计算单元承载，以大维度矩阵乘、张量卷积等规整算力密集计算为主，追求极致硬件吞吐。在PyPTO Pro中通过`pl.section_cube()`标记执行域。
- **融合类算子**：工业界主流复杂算子形态，可联动调度Cube计算单元与Vector计算单元协同计算，兼顾矩阵高密度算力运算与向量灵活逻辑处理。在PyPTO Pro中通过stage机制描述Cube和Vector的计算代码段，框架识别后自动插入核间同步并完成Preload核间流水编排。

## 基于AI Core的SIMD算子开发通用步骤

基于SPMD+SIMD双层编程模型，所有AI Core核运行同一份Kernel核函数，通过`pl.get_block_idx()`区分数据分片，实现多核并行计算。因此，算子开发的首要步骤为**Tiling（分块）设计**：对全局超大张量数据做均匀切分，为每个AI Core分配独立数据块，保障多核负载均衡。

此外，与传统CPU串行编程不同，AI Core SIMD编程的显著特征是**显式分层访存**：开发者需手动管控数据流转，将数据从Global Memory（全局内存，又称Device Memory，简称GM）搬运至AI Core片内Local Memory（本地内存，典型包含L1 Buffer、UB）完成计算，最终将运算结果写回全局内存。

### 算子Kernel通用开发四步法

基于SIMD的AI Core算子开发包含四个主要步骤，适配绝大多数算子开发场景：

1. **Tiling（分块）设计**：对全局超大张量数据进行均匀切分，为各AI Core分配大小均衡的独立数据分片，精准适配SPMD多核并行架构，规避单核算力瓶颈，实现全域负载均衡。
2. **数据搬入**：调用`pl.load`/`pl.load_tile`等数据搬运接口，将全局内存中的待计算数据，批量搬运至L1 Buffer、UB等AI Core片内高速本地内存。
3. **数据计算**：根据算子类型在`pl.section_vector()`或`pl.section_cube()`上下文中调度Cube计算单元或Vector计算单元，依托片内高速缓存数据，完成矩阵、向量或混合逻辑计算。
4. **数据搬出**：计算完成后，通过`pl.store`/`pl.store_tile`将片内存储中的结果数据写回全局内存，完成单次算子计算流程。

> [!NOTE]说明
> AI Core片内本地存储空间有限，无法一次性加载超大尺寸张量。实际开发中普遍采用「迭代分块搬运、分批计算、结果累加」的策略完成全域数据计算，同时搭配TileGroup的N-Buffer流水线技术屏蔽数据搬运耗时，提升整体运算效率。

### 新架构双模式矢量计算结构

Ascend 950PR/Ascend 950DT新一代架构在传统UB缓存体系的基础上，开放寄存器（Register）可编程能力，构建出「Global Memory → UB → Register」的三级内存层级，衍生出两套适配不同性能诉求的矢量计算模式，实现通用场景与极致性能场景的全覆盖。

基于全新三级内存架构，矢量计算分为通用的**Memory矢量计算**与高性能的**Reg矢量计算**：

- **Memory矢量计算（传统通用模式）**：具备流程简洁稳定、通用性强的特点，全程基于UB完成数据缓存与运算。对应PyPTO Pro的Tile级别接口（如`pl.add`、`pl.sub`、`pl.relu`等）。
  - 数据搬入：Global Memory → UB
  - 矢量计算：基于UB完成矢量计算
  - 数据搬出：UB → Global Memory

- **Reg矢量计算（高性能模式）**：依托寄存器低延迟、高带宽的硬件优势，专为极致性能优化场景设计。对应PyPTO Pro的VF级别接口（如`vf.add`、`vf.reduce_sum`、`vf.histograms`等），通过`@pl.vector_function`装饰器定义VF函数。
  - 数据搬入：全局内存 → UB → 寄存器（Reg）
  - 矢量计算：基于Reg完成矢量计算
  - 数据搬出：寄存器（Reg） → UB → 全局内存

向量寄存器位于Vector计算单元的最内层，拥有最低访存延迟与最高带宽，开放寄存器可编程能力，有助于开发者充分释放硬件峰值计算性能。VF级别接口是进阶用法，建议先掌握Tile级别接口后再学习，详细说明请参考[Reg矢量计算](tile_based_python_programming/Reg_vector_computation.md)。

## 多级编程接口体系：分层抽象与能力差异

### 算子多级编程接口发展回顾

早期算子开发以C语言裸指针编程为核心，依托C语言底层内存操控能力，可直接通过指针寻址、操作设备内存，同时配套与硬件底层指令精准映射的原生接口，支持向量计算、矩阵乘、数据搬运等硬件原语调用。该模式可实现硬件缓存、寄存器等资源的精细化管控，是底层极致性能优化的核心方案，广泛应用于离散类、归约类、矩阵类等核心算子的底层实现。

随着深度学习持续迭代演进，AI计算核心载体升级为4D/5D高维张量（如NLP模型注意力张量、CV模型特征图张量）。传统裸指针编程需手动完成内存偏移计算、维度拆分、边界校验等重复性工作，不仅开发效率低下，还易引发内存越界、索引错误等稳定性问题。

在此背景下，基于面向对象特性的Tensor抽象应运而生。Tensor内置内存布局（Layout）、数据类型等张量元信息，其中Layout包含维度（Shape）、步长（Stride）等，可简化内存布局的管理，降低高维张量开发成本。业界由此逐步形成主流的Tensor/Tile编程模型：核心思想是将全局张量（Tensor）切分为规则且固定尺寸的数据分块（Tile），以Tile作为计算、存储与硬件调度的最小单元。其中，Tensor是多维数组的标准化数学与计算抽象，是AI场景下向量、矩阵、高维特征图等各类数据的统一表示形式，仅描述全局逻辑结构，不绑定硬件执行细节；Tile则是AI加速芯片张量核心、矢量核心的最小调度、计算与数据搬运粒度，作为编程模型的物理层载体，承担着衔接全局数据与底层硬件的作用。通过将全局Tensor切分为适配硬件存储与算力规格的Tile，结合数据搬运、流水线调度、片上内存复用等优化手段，可充分挖掘并释放硬件极限算力。

当前Tensor/Tile编程范式主要包含两类实现路径：手工Tensor/Tile编程一般基于C/C++实现，由开发者全权管控内存布局、分块策略、数据搬运与流水线调度全流程；自动化Tensor/Tile编程一般基于Python DSL（领域专用语言）实现，由AI编译器自动完成内存分配、资源复用、任务调度与依赖同步，降低开发门槛。PyPTO Pro即属于后者，在Tensor/Tile编程模型的基础上，通过Python前端提供更友好的编程体验，同时保留了精细化的硬件控制力。

### PyPTO Pro多级编程接口

PyPTO Pro秉持「Python语法、最小化扩展」的核心设计原则，打造轻量化、高性能的编程基座。PyPTO Pro基于Tile抽象提供两类核心编程接口，全层级均支持完整AI Core算力调度，各层级能力递进、场景适配清晰：

| API层级 | 编程模式 | 特点 | 主要用途 |
|----------|----------|------|----------|
| **Tile API** | 基于Tile编程 | 通过`make_tile`/`make_tile_group`分配片上buffer，`auto_mutex=True`自动管理核内同步与N-Buffer流水。 | 使用TileGroup自动编排数据搬运和计算，提升编程易用性与开发效率，适配绝大多数算子开发场景。 |
| **Reg API（VF计算）** | 基于寄存器编程 | 通过`@pl.vector_function`定义VF函数，使用`vf.*`接口直接操作向量寄存器，由开发者自主管理寄存器数据加载与存储。 | 自主管理寄存器，开放Vector计算单元最内层硬件能力，支撑精细化调优与极致性能实现。标注为ISASI类别，不保证跨硬件版本兼容。 |

此外，PyPTO Pro提供**Utils API**（公共辅助函数），涵盖Python语法糖（`const`/`min`/`max`等标量操作）和调测接口（`printf`/`pto_assert`/`dump_data`/`trap`等调试工具），支持开发者高效实现算子开发与调试。

开发者可结合自身性能优化诉求与开发效率需求，灵活选用对应层级接口完成算子开发。详细的接口说明请参考[SIMD API](../../../../api/SIMD-API/index.md)和[Utils API](../../../../api/Utils-API/index.md)。

## 控制流

### 循环

支持Python原生语法表达，`range`在kernel内采用`pl.range(start, end, step)`表达：

```python
num_cores = pl.get_block_num()
core_id = pl.get_block_idx()

for i in pl.range(core_id, m_tile_num, num_cores):
    for j in pl.range(0, n_tile_num, 1):
        ...
```

### 条件分支

支持Python原生`if`、`else`、`elif`用法：

```python
if tiling.opkind[4] == 0:
    pl.add(tile_c, tile_a, tile_b)
elif tiling.opkind[4] == 1:
    pl.sub(tile_c, tile_a, tile_b)
else:
    pl.mul(tile_c, tile_a, tile_b)
```

## 编程流程

使用PyPTO Pro开发算子的典型流程：

1. **定义Kernel函数**：使用`@pl.jit()`装饰器标记，通过`pl.Tensor`声明输入输出
2. **定义Tile与TileGroup**：通过`pl.TileType`描述Tile的shape/dtype/内存空间，通过`pl.make_tile`或`pl.make_tile_group`分配片上buffer
3. **编写计算逻辑**：在`pl.section_vector()`或`pl.section_cube()`上下文中，使用`load`/`store`/计算接口完成数据搬运和计算
4. **Host端调用**：通过bracket-launch语法`kernel[stream, block_dim](*args)`启动Kernel

详细的编程方法请参考：

- [核函数](kernel_function.md)
- [基于Tile的Python编程](tile_based_python_programming/Python_programming_overview.md)

## 小结

本章系统梳理了AI Core算子从任务分发、硬件调度到代码落地的全链路技术体系：计算任务依托SPMD+SIMD异构并行范式分发至多AI Core并行执行，再根据算子计算特性精准调度至Cube矩阵单元或Vector向量单元完成运算；开发者可结合自身业务场景，灵活选用Tile API或Reg API等Python编程接口，完成算子Kernel的完整逻辑开发。

后续章节将逐层深入、循序渐进地详解硬件架构原理、Kernel核函数定义、各级编程接口规范、内存管理机制、数据同步逻辑、实战开发流程与性能调优技巧，帮助开发者快速理解核心技术，熟练掌握高性能、高可用的AI Core算子开发能力。

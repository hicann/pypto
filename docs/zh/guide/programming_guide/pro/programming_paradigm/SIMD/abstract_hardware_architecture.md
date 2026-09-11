# 抽象硬件架构

SIMD计算分别使用AIV上的Vector计算资源和AIC上的Cube计算资源，并通过片上存储、数据搬运单元和同步机制组成完整的数据通路。SIMT的线程执行资源请参考[SIMT抽象硬件架构](../SIMT/abstract_hardware_architecture.md)。

## SIMD硬件组成

AI Core中的SIMD硬件分为AIC和AIV：

- **AIV**主要承担Tile矢量计算和Reg矢量计算。其SIMD相关资源包括Scalar、Unified Buffer（UB）、Vector Register File、Vector计算单元以及MTE2/MTE3搬运流水。
- **AIC**主要承担Cube矩阵计算。其资源包括Scalar、L1 Buffer、L0A/L0B/L0C Buffer、Cube计算单元以及MTE2/MTE1/M/FIX等流水。
- **GM和L2 Cache**位于AI Core之外，为多个AIC和AIV提供全局数据。

**图1 AI Core硬件架构**

![AIC、AIV、存储、计算和搬运单元的关系](../../../../figures/pro/hardware_architecture_950.png)

图中黑色实线表示主要数据流，橙色虚线表示指令流。AIV中的SIMD VF指令队列、Vector Register File和Vector单元对应SIMD矢量计算；AIC中的Cube指令队列、矩阵Buffer和Cube单元对应SIMD矩阵计算。图中同时绘制了SIMT资源，用于说明共享的AIV硬件位置，不属于本章展开范围。

PyPTO Pro提供三种SIMD计算方式：

| 计算方式 | 执行位置 | 主要数据载体 | 主要执行单元 |
|:---|:---|:---|:---|
| Tile矢量计算（Membase） | AIV | UB中的Vec Tile | Vector计算单元 |
| Reg矢量计算（Regbase） | AIV | UB Tile和Vector Register中的RegTensor | Reg向量执行单元、Aux Scalar和DMA单元 |
| Cube矩阵计算 | AIC | L1、L0A、L0B和L0C中的矩阵Tile | Cube计算单元 |

## AIV矢量计算架构

AIV负责SIMD矢量指令的控制、数据搬运和执行。Kernel中的Python控制流和标量表达式由Scalar侧处理；Scalar将搬运和矢量指令发射到对应的指令队列，MTE和Vector相关单元异步执行这些任务。

### Tile矢量计算

Tile矢量计算以UB中的Vec Tile作为输入、输出和中间数据。其基本硬件数据路径为：

```text
GM ──MTE2──> UB
               │
               V
          Vector计算
               │
GM <──MTE3── UB
```

PyPTO Pro中的接口与硬件路径对应如下：

| 硬件行为 | Pipe | PyPTO Pro表达 |
|:---|:---|:---|
| GM数据搬入UB | MTE2 | `pypto_pro.language.load`、`load_tile` |
| UB上的批量矢量计算 | V | `pypto_pro.language.add`、`sub`、`sum`等Tile API |
| UB数据写回GM | MTE3 | `pypto_pro.language.store`、`store_tile` |

Tile使用`pypto_pro.language.MemorySpace.Vec`映射到UB。`TileType`描述Tile的shape、dtype和layout，`make_tile`或`make_tile_group`将Tile绑定到UB中的具体地址。详细创建方式请参考[Tile创建和操作](../../development/tile_creation_and_operations.md)。

### Reg矢量计算

Reg矢量计算在Tile矢量数据路径上增加Vector Register File。GM中的数据必须先搬入UB，再由VF搬运接口加载到Vector Register；计算完成后按相反方向写回：

```text
GM → UB → Vector Register File
              ↓
          Reg矢量计算
              ↓
GM ← UB ← Vector Register File
```

**图2 Reg矢量计算内存层级**

![GM、UB和Vector Register File的层级关系](../../../../figures/pro/register_memory_hierarchy.jpg)

Vector侧参与Reg矢量计算的主要硬件资源如下：

- **Vector Register File**：保存VF加载的数据、计算中间结果和待写回结果。
- **Reg向量执行单元**：从Vector Register File读取操作数，执行`vf.*`矢量指令并写回寄存器。
- **Aux Scalar**：处理VF函数中的地址、循环等辅助标量计算。
- **DMA单元**：在UB与Vector Register File之间搬运数据。

**图3 Reg矢量执行单元**

![Aux Scalar、Reg向量执行单元、DMA、Register File和UB的关系](../../../../figures/pro/register_execution_unit.jpg)

PyPTO Pro使用`@pypto_pro.language.vector_function`定义VF函数，使用RegTensor和MaskReg保存寄存器数据，并通过`vf.load*`、`vf.store*`在UB与Vector Register之间搬运。RegTensor的数据类型和寄存器限制请参考[vf.reg_tensor](../../../../../api/pro_api/SIMD-API/reg_computation/reg_tensor.md)，完整编程方法请参考[Reg计算](../../development/vector_computation/reg_computation.md)。

## AIC矩阵计算架构

AIC使用Cube单元执行矩阵乘加。矩阵数据从GM进入AIC后，依次经过L1和L0级片上存储；Cube从L0A和L0B读取矩阵块，将累加结果写入L0C。

```text
                            ┌──MTE1──> L0A──┐
GM ──MTE2──> L1 ┤                   ├──M──> L0C──FIX──> GM
                            └──MTE1──> L0B──┘
```

主要存储空间和对应数据如下：

| `pypto_pro.language.MemorySpace` | 物理存储 | 典型作用 |
|:---|:---|:---|
| `Mat` | L1 Buffer | GM与L0A/L0B之间的矩阵暂存 |
| `Left` | L0A Buffer | Cube左矩阵操作数 |
| `Right` | L0B Buffer | Cube右矩阵操作数 |
| `Acc` | L0C Buffer | 矩阵累加值和计算结果 |
| `Bias` | Bias Buffer | 矩阵计算的融合偏置 |
| `Scaling` | Fixpipe Buffer | 量化或反量化参数 |
| `ScaleLeft` | L0A_MX Buffer | MX矩阵计算的左量化系数矩阵 |
| `ScaleRight` | L0B_MX Buffer | MX矩阵计算的右量化系数矩阵 |

AIC各Pipe及其典型接口如下：

| Pipe | 硬件行为 | PyPTO Pro表达 |
|:---|:---|:---|
| MTE2 | GM搬入L1 | `load`、`load_tile` |
| MTE1 | L1搬入L0A/L0B及MX Buffer | `move` |
| M | Cube矩阵计算 | `matmul`、`matmul_acc`、`matmul_mx`、`matmul_mx_acc` |
| FIX | L0C结果搬出 | `store`、`store_tile` |

矩阵Tile通常使用NZ、ZN等分形布局。不同MemorySpace具有相应的默认layout和fractal约束，数据搬运接口可以在支持的路径上完成格式转换。矩阵存储布局、L1地址规划和Cube计算流程请参考[Cube计算](../../development/cube_computation.md)。

## 存储层级与数据对象

SIMD编程涉及GM、片上Buffer和Vector Register三个层级：

| 存储层级 | 可见范围 | PyPTO Pro数据对象 | 特点 |
|:---|:---|:---|:---|
| GM | Device上的多个AI Core | Tensor、Ptr | 容量大，是Kernel输入、输出和Workspace所在位置 |
| AIV片上存储 | 当前AIV | `MemorySpace.Vec`中的Tile | 延迟和带宽优于GM，容量有限，需要显式规划地址和复用 |
| AIC片上存储 | 当前AIC | Mat、Left、Right、Acc等Tile | 按矩阵数据通路分层，layout和分形约束更强 |
| Vector Register File | 当前VF函数 | RegTensor、MaskReg | 用于Reg矢量计算，生命周期局限于VF函数 |

Tensor不能直接作为片上计算单元的操作数。数据需要通过搬运接口进入对应Tile；Reg矢量计算还需要继续将UB Tile加载到Vector Register。各层数据对象只描述所在存储和数据视图，不隐式申请其他层级的内存，也不会自动完成数据搬运。

## 指令流、数据流与同步流

理解SIMD硬件执行时，需要区分三类关系：

- **指令流**：Scalar解析控制逻辑并把搬运、计算和同步任务发射到不同指令队列。
- **数据流**：MTE、DMA、Vector和Cube单元按照各自的数据路径读取或写入GM、片上Buffer和寄存器。
- **同步流**：当一条Pipe生产的数据将被另一条Pipe消费时，通过事件或mutex约束执行顺序。同步信号只表达依赖，不搬运数据。

各Pipe异步执行，因此源码中的先后顺序不等同于硬件上的完成顺序。例如，Tile矢量计算通常包含“MTE2搬入完成后V才能读取”和“V计算完成后MTE3才能搬出”两条依赖；Cube计算则包含MTE2、MTE1、M和FIX之间的依赖链。

PyPTO Pro提供两种同步方式：

- 使用`make_tile_group`配置mutex元数据，并通过`@pypto_pro.language.jit(auto_mutex=True)`让框架根据Tile读写关系插入同步。
- 使用`make_tile`时，通过`pypto_pro.language.system.sync_src`和`sync_dst`显式表达生产Pipe和消费Pipe的依赖。

自动同步、TileGroup轮转和手动同步的使用边界请参考[Tile创建和操作](../../development/tile_creation_and_operations.md)和[Tile计算](../../development/vector_computation/tile_computation.md)。

## AIC与AIV的并行关系

多个AIC或AIV可以并行执行同一份Kernel，并通过全局逻辑核索引处理不同数据分片。纯Vector Kernel使用AIV资源，纯Cube Kernel使用AIC资源；Cube/Vector混合Kernel同时使用两类资源。

混合Kernel采用AIC:AIV为1:2的映射。每个逻辑Block包含一个AIC和两个AIV，两个AIV通过subblock索引区分。硬件映射在PyPTO Pro中由`get_block_num()`、`get_block_idx()`、`get_subblock_num()`和`get_subblock_idx()`等接口呈现，详细语义请参考[SIMD编程范式](programming_paradigm.md#多核spmd与核内simd)和[多核Tiling切分](../../development/tiling/multi_core_tiling.md)。

## 小结

SIMD硬件包含AIV上的Tile/Reg矢量数据路径和AIC上的Cube矩阵数据路径。Tensor、Tile和RegTensor分别表示GM、片上Buffer和Vector Register中的数据；MTE、Vector、Cube和FIX等Pipe异步执行，通过自动mutex或显式事件同步保证数据依赖。开发者据此选择计算方式、规划存储层级并组织搬运与计算流水。

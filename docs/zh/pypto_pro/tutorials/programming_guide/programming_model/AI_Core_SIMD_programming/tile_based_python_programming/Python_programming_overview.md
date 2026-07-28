# Python语言编程概述

PyPTO Pro基于Python前端，提供了一套完整的DSL用于描述NPU上的Kernel计算。本章介绍PyPTO Pro编程中的核心数据结构与参数化机制。

## 核心数据结构

PyPTO Pro中有三种核心数据抽象：

| 抽象 | 所在位置 | 创建方式 | 用途 |
|:---|:---|:---|:---|
| **Tensor** | 全局内存（GM） | `pl.Tensor[...]` / `pl.make_tensor` | 带shape + stride的片外（GM）数据视图 |
| **Tile** | 片上buffer | `pl.make_tile` | 固定的片上buffer（Vec / Mat / Left / Right / Acc / Scaling） |
| **TileGroup** | 片上buffer | `pl.make_tile_group` | 一组轮转的tile，用于双缓冲 / N缓冲 |

### Tensor

Tensor是对全局内存（GM）的带类型视图，是Kernel的输入/输出。你从中`load`数据到片上tile，并把结果`store`回去。

**在Kernel签名中声明**（最常见形式）：

```python
@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输入 GM tensor
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输入 GM tensor
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输出 GM tensor
):
    ...
```

- `pl.Tensor[[shape...], dtype]` —— 第一个元素是shape列表，第二个是元素dtype
- `pl.DYNAMIC`：尺寸在launch时传入，尺寸变化不会产生新的编译变体
- `pl.STATIC`：尺寸在launch时读取并固化到IR，尺寸变化会选择新的编译变体
- 正整数：固定尺寸，launch时必须精确匹配
- kernel内通过`x.shape[axis]`读取维度，支持负索引

**由裸指针构造视图**（动态rank kernel）：

```python
@pl.jit(auto_mutex=True)
def fa_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    v: pl.Ptr[pl.DT_FP16],
    o: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,
):
    # 由裸指针构造带类型的二维视图
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.d], [tiling.d, 1])
    ...
```

`pl.make_tensor(ptr, shape, stride, dtype=None)`由一个裸指针结合显式shape与stride构造tensor视图，是动态rank kernel的基础。

### Tile与TileType

Tile是一块固定的片上buffer。`pl.TileType`描述一个tile的shape、dtype以及片上摆放，它本身不分配任何东西，而是传给`make_tile`/`make_tile_group`。

```python
@dataclass
class TileType:
    shape: Sequence[int]                 # 例如 [128, 128]
    dtype: DataType                      # pl.DT_FP16, pl.DT_FP32, ...
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None  # 逻辑有效区域（< shape）
    layout: Optional[TensorLayout] = None        # Tile的分型（ND/DN/NZ/ZN/NN/ZZ）
    fractal: Optional[int] = None                # fractal大小
    pad: Optional[int] = None                    # TilePad.null/zero/max/min
    compact: Optional[int] = None                # 紧凑摆放模式
```

**内存空间**（`pl.MemorySpace`）：

| 空间 | 典型角色 |
|:---|:---|
| `Vec` | 向量单元工作buffer（UB）—— 逐元素计算 |
| `Mat` | L1 / 矩阵暂存buffer（GM → L1加载） |
| `Left` | L0A —— matmul左操作数 |
| `Right` | L0B —— matmul右操作数 |
| `Acc` | L0C —— matmul累加器（fp32/int32） |
| `Scaling` | 缩放/量化参数buffer |

> [!NOTE]说明
> 在同一款硬件上，**除了UB之外**，如果target_memory确定了，layout和fractal是确定的，可以不填。

### TileGroup

TileGroup是用`pl.make_tile_group`声明的一组轮转的tile，用于实现双缓冲乃至更广义的N缓冲。当一块buffer正在被消费时，下一块可以同时被生产，从而让多条pipe重叠以提升吞吐。

```python
g = pl.make_tile_group(type=<TileType>, addrs=<base|list>, mutex_ids=[...])
```

- `type` —— 描述组中每一个tile的`pl.TileType`
- `mutex_ids` —— 非空的、互不相同的整数列表，取值范围`[0, 31]`，其长度即buffer数量（2→双缓冲，N→N缓冲）
- `addrs` —— 单个基地址（tile连续排布）或地址列表（每个tile一个显式地址）

> [!NOTE]说明
> `mutex_ids`在整个kernel内必须唯一。两个组共享同一个id会导致同步互相混叠。

关于Tile和TileGroup的详细使用方法请参考[Tile矢量计算](Tile_vector_computation.md)和[Cube矩阵计算](Cube_matrix_computation.md)。

## 编程范式

PyPTO Pro的算子开发遵循「**搬入→计算→搬出**」三段式流水线范式，与AI Core硬件的多级异步流水特性完全贴合：

1. **搬入（CopyIn）**：通过`pl.load`/`pl.load_tile`将数据从Global Memory搬运至片上Buffer（UB/L1等）
2. **计算（Compute）**：在片上Buffer上完成Tile级别的计算，根据算子类型在`pl.section_vector()`或`pl.section_cube()`上下文中调用对应的计算接口
3. **搬出（CopyOut）**：通过`pl.store`/`pl.store_tile`将结果从片上Buffer写回Global Memory

AI Core内部的搬运单元（MTE2/MTE1/MTE3等）与计算单元（V/M等）天然支持异步并行。通过TileGroup的N-Buffer机制，可以让搬入下一块数据与当前块计算重叠执行，实现流水线吞吐叠加。

### 两档内存管理策略

PyPTO Pro提供两种Tile分配方式，对应不同的内存管理与同步复杂度：

| 策略 | 分配方式 | 同步管理 | 适用场景 |
|:---|:---|:---|:---|
| **自动同步（推荐）** | `pl.make_tile_group` + `auto_mutex=True` | 框架自动插入`mutex_lock`/`mutex_unlock`，无需手动同步 | 大多数kernel；流水化/重叠的循环 |
| **手动同步** | `pl.make_tile` | 开发者手动插入`sync_src`/`sync_dst`，精确控制同步时序 | 需要精确、手工放置flag的紧凑流水线 |

新kernel优先使用`make_tile_group` + `auto_mutex=True`，它在构造上即正确，远不易出错。仅在你需要精确、手工放置flag时才使用`make_tile` + 显式同步。两者可在同一kernel中混用。

三类标准编程范式的详细实践请参考：

- [Tile矢量计算](Tile_vector_computation.md)：矢量类算子（Vector单元）
- [Cube矩阵计算](Cube_matrix_computation.md)：矩阵类算子（Cube单元）
- 融合类算子：通过stage机制联动Cube与Vector，框架自动完成Preload核间流水编排

## 同步机制概述

AI Core内部存在多条异步并行流水，当一条流水生产的数据被另一条流水消费时，必须插入同步事件确保数据依赖正确。PyPTO Pro提供两种同步模式：

### 自动同步（auto_mutex）

通过`@pl.jit(auto_mutex=True)`启用。框架根据TileGroup中每个tile的`mutex_id`，在每次使用轮转tile前后自动发出`mutex_lock`/`mutex_unlock`，无需开发者感知流水类型与事件ID。这是最常用的同步方式，覆盖绝大多数场景。

### 手动同步（sync_src / sync_dst）

由`make_tile`创建的裸buffer不附带任何同步，开发者需手动插入`pl.system.sync_src`/`pl.system.sync_dst`对：

- `sync_src(set_pipe, wait_pipe, event_id)` —— 生产方SET flag
- `sync_dst(set_pipe, wait_pipe, event_id)` —— 消费方WAIT flag

PyPTO Pro的流水类型（`pl.PipeType`）与硬件指令流水对应关系：

| PipeType | 含义 | 典型操作 |
|:---|:---|:---|
| `MTE2` | GM→L1/UB搬运 | `pl.load`/`pl.load_tile` |
| `MTE1` | L1→L0A/L0B搬运 | `pl.move` |
| `M` | 矩阵计算 | `pl.matmul` |
| `V` | 向量计算 | `pl.add`/`pl.sub`/... |
| `MTE3` | UB→GM搬运 | `pl.store`/`pl.store_tile` |
| `FIX` | L0C→GM搬运 | `pl.store`（Acc→GM） |

手动同步的典型模式为：搬入后插入`MTE2→V`同步确保数据就绪再计算，计算后插入`V→MTE3`同步确保计算完成再搬出。在循环场景下还需考虑反向同步（循环间依赖），防止当前迭代覆盖上一迭代未完成的数据。

> [!NOTE]说明
> 手动同步中`event_id`取值范围`[0, 15]`，仅在上一次使用已被消费后才可复用同一id。手动同步属于ISASI类别的高级用法，不保证跨硬件版本兼容。

## TilingData

TilingData是把**运行时参数**——shape、stride、循环边界、算子选择器、缩放系数等——喂给已编译kernel的方式，而无需把它们固化进kernel签名。详细说明请参考[TilingData](TilingData.md)。

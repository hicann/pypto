# Cube矩阵计算编程

本文介绍如何在PyPTO Pro中使用Tile API编写基于L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer等片上存储的**矩阵计算代码**。

Cube计算单元专用于执行矩阵乘加运算，直接访问的专用缓存如下：L0A Buffer存储左矩阵，L0B Buffer存储右矩阵，L0C Buffer存储累加值及矩阵计算结果。PyPTO Pro通过`pl.section_cube()`标记Cube执行域，在域内完成数据搬运与矩阵计算。

## 矩阵编程的基本步骤

Cube矩阵计算的基本步骤为：数据搬入 → 数据加载 → 计算 → 数据搬出。当描述一个常见的Cube矩阵计算时，需要执行以下四个步骤：

1. 通过`pl.load`将数据从GM搬入L1 Buffer（Mat）
2. 通过`pl.move`将L1 Buffer数据搬入L0A Buffer（Left）和L0B Buffer（Right）
3. 通过`pl.matmul`执行矩阵乘法，结果存储在L0C Buffer（Acc）中
4. 通过`pl.store`将L0C Buffer中的结果搬出到GM

对应的数据流和硬件流水如下：

**图1** Cube矩阵计算的数据流和硬件流水

![Cube矩阵计算的数据流和硬件流水](../../../../figures/cube_matrix_computation_data_flow.png)

## 矩阵计算内存管理

### 矩阵计算内存申请

Cube矩阵计算主要通过L0A/L0B/L0C Buffer进行计算，并经L1 Buffer中转。开发者需将输入数据从GM搬入L1 Buffer（`pl.MemorySpace.Mat`），再从L1 Buffer搬入L0A（`pl.MemorySpace.Left`）/L0B（`pl.MemorySpace.Right`），最后通过`pl.matmul`完成计算，结果写入L0C（`pl.MemorySpace.Acc`）。

PyPTO Pro通过`TileType`描述Tile的shape、dtype和target_memory，再通过`make_tile_group`分配片上Buffer：

```python
TILE_M = 128
TILE_K = 128
TILE_N = 128

@pl.jit(auto_mutex=True)
def matmul_kernel(a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
                  b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
                  out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    # L1 Buffer（Mat）：GM到L0之间的矩阵暂存
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[1])
    # L0A / L0B Buffer：matmul的操作数
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    # L0C Buffer（Acc）：matmul累加结果
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])
```

各内存空间的典型角色如下：

| `pl.MemorySpace` | 物理 Buffer | 典型角色 |
|:---|:---|:---|
| `Mat` | L1 Buffer | GM与L0A/L0B之间的矩阵暂存 |
| `Left` | L0A Buffer | `matmul`左操作数 |
| `Right` | L0B Buffer | `matmul`右操作数 |
| `Acc` | L0C Buffer | `matmul`累加结果（通常为FP32/INT32） |

### 矩阵计算内存布局

Cube计算单元采用分块计算逻辑，硬件最小计算粒度为分形块（Fractal）。对于half数据类型，分形形状为16×16（即32B/sizeof(half)=16）；对于int8类型，分形形状为16×32或32×16。传统线性存储布局下，读取一个分形块需要访问多个不连续的内存地址，导致访存效率下降。

为解决该问题，昇腾引入矩阵分形存储格式，使每个分形块在物理内存中连续存放，硬件单次读取即可加载整块数据，大幅提升数据吞吐能力。

PyPTO Pro通过`TileType`的`layout`参数指定分形布局。采用"大Y小x"命名法：

- 大Y（Z/N）：表示分形矩阵之间的排列顺序（Z为行主序，N为列主序）。
- 小x（z/n）：表示分形矩阵内部元素的排列顺序（z为行主序，n为列主序）。

对于矩阵乘法 C = A × B，Ascend 950PR/Ascend 950DT要求：左矩阵A使用Nz格式，右矩阵B使用Zn格式，结果矩阵C使用Nz格式。

各内存空间的默认layout如下（Ascend 950PR/Ascend 950DT）：

| 内存空间 | 默认layout | 说明 |
|:---|:---|:---|
| `Mat`（L1） | `pl.NZ` | GM→L1搬运时随路完成ND→NZ转换 |
| `Left`（L0A） | `pl.NZ` | L1→L0A搬运时为NZ→NZ |
| `Right`（L0B） | `pl.ZN` | L1→L0B搬运时完成NZ→ZN转换 |
| `Acc`（L0C） | `pl.NZ` | 矩阵乘结果按NZ存放 |

> [!NOTE]说明
> 在同一款硬件上，如果`target_memory`确定了，`layout`和`fractal`是确定的，可以不填。仅转置搬入等特殊场景需要显式指定`layout`。

## 矩阵数据搬入

矩阵搬入分为两跳：GM → L1（`pl.load`）和 L1 → L0A/L0B（`pl.move`）。

### GM → L1搬运

通过`pl.load`将矩阵从GM搬入L1 Buffer。`load`在搬运过程中自动完成ND到NZ的格式转换，无需手动配置分形参数。

```python
with pl.section_cube():
    cur_a = a_l1.current()
    cur_b = b_l1.current()
    pl.load(cur_a, a, [i, 0])    # A矩阵搬入L1，自动ND→NZ
    pl.load(cur_b, b, [0, j])    # B矩阵搬入L1，自动ND→NZ
```

`load`的坐标参数`[row, col]`为GM Tensor上的元素偏移，表示从该位置开始搬运一个Tile大小的数据。

### L1 → L0A/L0B搬运

通过`pl.move`将L1 Buffer中的数据搬入L0A/L0B Buffer，搬运过程中自动完成Nz到Zn（L0B）的格式转换。

```python
    cur_a_left = a_left.current()
    cur_b_right = b_right.current()
    pl.move(cur_a_left, cur_a)    # L1 → L0A，NZ→NZ
    pl.move(cur_b_right, cur_b)   # L1 → L0B，NZ→ZN
```

### 转置搬入

当输入矩阵的轴序与L1 Tile的轴序相反时（如Tensor为`[K, M]`而Tile为`[M, K]`），需要在搬入时进行转置。通过`load`的`order`参数控制：`order=[1, 0]`表示转置搬入，此时L1 Tile的`layout`需设为`pl.ZN`。

以`C[M, N] = A[M, K] @ B[K, N]`为例：

| 操作数 | Tensor shape | 是否转置 | `load`的`order` | L1 Mat Tile layout |
|:---|:---|:---|:---|:---|
| 左矩阵A | `[M, K]` | 否 | `[0, 1]`（默认） | `pl.NZ`（默认） |
| 左矩阵A | `[K, M]` | 是 | `[1, 0]` | `pl.ZN` |
| 右矩阵B | `[K, N]` | 否 | `[0, 1]`（默认） | `pl.NZ`（默认） |
| 右矩阵B | `[N, K]` | 是 | `[1, 0]` | `pl.ZN` |

左矩阵转置搬入示例：

```python
@pl.jit(auto_mutex=True)
def kernel_left_transpose(
    a: pl.Tensor[[K, M], pl.DT_FP16],               # [K, M]，需转置
    b: pl.Tensor[[K, N], pl.DT_FP16],               # [K, N]，不转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0], order=[1, 0])    # 转置搬入
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0])                   # 不转置
```

## 矩阵数据搬出

通过`pl.store`将L0C Buffer中的计算结果搬出到GM。L0C→GM的搬运走FIX流水线，支持在搬运过程中进行格式转换（NZ→ND）。

```python
    pl.store(out, acc, [i, j])    # L0C → GM，自动NZ→ND
```

如果输出Tensor标注为NZ布局，`store`会将Acc的计算结果按NZ分形直接写入GM，无需额外格式转换：

```python
@pl.jit(auto_mutex=True)
def kernel(...,
           nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],   # NZ输出Tensor
          ):
    ...
    pl.store(nz_out, acc, [0, 0])    # 按NZ分形写入GM
```

## 矩阵计算

`pl.matmul`是PyPTO Pro封装NPU硬件计算能力的矩阵乘法核心接口，实现`dst_tile = lhs_tile × rhs_tile`，数据通路为L0A(Left) × L0B(Right) → L0C(Acc)。

**表 矩阵乘计算A、B、C矩阵说明（Ascend 950PR/Ascend 950DT）**

| 矩阵 | 存储位置 | 维度 | 数据格式 | 数据类型 |
|:---|:---|:---|:---|:---|
| A | L0A Buffer | M × K | Nz | FP16、BF16、FP32、INT8 |
| B | L0B Buffer | K × N | Zn | 与A一致 |
| C | L0C Buffer | M × N | Nz | FP16、BF16、FP32、INT32 |

```python
    pl.matmul(acc_tile, a_left, b_right)    # C = A × B
```

### K维分块累加

当K维度较大，无法一次装入L1/L0时，需要将K轴切分为多个分块，逐块累加。首块用`pl.matmul`写入累加器，其余块用[`pl.matmul_acc`](../../../../../api/SIMD-API/operation/matrix_computation/matmul_acc.md)累加到同一个L0C。

K维分块累加对正确性有三个硬性要求：

1. **每步matmul / matmul_acc都要传`phase`**：首块和中间块用`phase=pl.AccPhase.Partial`，末块用`phase=pl.AccPhase.Final`；写回GM的`store`也传`phase=pl.STPhase.Final`。
2. **L0C累加器设`fractal=1024`**（FP32）。
3. **cube段用`pl.system.set_mm_layout_transform(enabled=True)`开启**，段末`enabled=False`关闭。

```python
@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < K_SIZE - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```

> [!NOTE]说明
> `phase`参数控制Cube（M流水）与FixPipe（FIX流水）之间的硬件unit_flag握手。`phase`配对使用时，框架不自动插入M与FIX之间的软件同步，由硬件unit_flag保证顺序。使用不当会导致精度问题或设备卡死。详见[`phase`使用约束](../../../../../api/SIMD-API/operation/matrix_computation/phase.md)。

## 同步机制

Cube矩阵计算的四个步骤分别对应MTE2、MTE1、M、FIX四条流水线，各流水线异步并行执行，读写同一存储资源时存在数据依赖。

PyPTO Pro推荐使用`pl.make_tile_group`配合`@pl.jit(auto_mutex=True)`，由框架根据Tile的`mutex_id`自动插入跨Pipe同步，开发者无需手写`sync_src`/`sync_dst`。使用单个`pl.make_tile`并需要手工控制依赖时，可调用`pl.system.sync_src`/`pl.system.sync_dst`。

| 流水线 | 含义 | 典型操作 |
|:---|:---|:---|
| MTE2 | GM→L1/UB搬运 | `pl.load`/`pl.load_tile` |
| MTE1 | L1→L0A/L0B搬运 | `pl.move` |
| M | 矩阵计算 | `pl.matmul`/`pl.matmul_acc` |
| FIX | L0C→GM搬运 | `pl.store`/`pl.store_tile` |

> [!NOTE]说明
> 当`matmul`/`matmul_acc`使用了`phase`参数时，M流水与FIX流水之间的同步由硬件unit_flag完成，框架不会自动插入该段同步。

## 尾块处理

当GM Tensor的shape不能被Tile shape整除时，边界上会出现比Tile小的尾块。Cube场景的尾块处理通过`valid_shape=[-1, -1]`配合`pl.set_validshape`和`compact=1`完成：

- `valid_shape=[-1, -1]`：声明有效区域为运行时动态，后续通过`set_validshape`设置。
- `compact=1`：让硬件按有效行紧凑摆放，减少L1/L0占用。

```python
tt_left = pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
                      target_memory=pl.MemorySpace.Left,
                      valid_shape=[-1, -1], compact=1)
tt_right = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Right,
                       valid_shape=[-1, -1], compact=1)
tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Acc,
                     valid_shape=[-1, -1], compact=1)
```

在运行时为每个尾块设置有效尺寸：

```python
valid_m = pl.min(TILE_M, M - i * TILE_M)
valid_n = pl.min(TILE_N, N - j * TILE_N)
pl.set_validshape(cur_acc, [valid_m, valid_n])
```

详细的尾块处理参数协同请参考[尾块处理](tail_block_handling.md)。

## 完整示例

以下是一个完整的Matmul Kernel，计算`C[M, N] = A[M, K] @ B[K, N]`，使用`make_tile_group` + `auto_mutex=True`管理L1/L0A/L0B/L0C缓冲，L1用双缓冲（`next()`轮转）让搬运与计算重叠：

```python
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 128
TILE_K = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
    b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    M = a.shape[0]
    N = b.shape[1]

    # L1双缓冲（next()轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A / L0B / Acc单缓冲（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(core_id, M // TILE_M, num_cores):
            for j in pl.range(0, N // TILE_N, 1):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                pl.load_tile(cur_a, a, [i, 0])
                pl.load_tile(cur_b, b, [0, j])

                cur_a_left = a_left.current()
                cur_b_right = b_right.current()
                pl.move(cur_a_left, cur_a)
                pl.move(cur_b_right, cur_b)

                acc_tile = acc.next()
                pl.matmul(acc_tile, cur_a_left, cur_b_right)
                pl.store_tile(out, acc_tile, [i, j])


# Host端调用
device = "npu:0"
torch.npu.set_device(device)
torch.manual_seed(42)
M_SIZE, K_SIZE, N_SIZE = 8192, 128, 8192
a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

matmul_kernel[None, 32](a, b, out)
torch.npu.synchronize()

golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul kernel passed!")
```

> [!NOTE]说明
>
> - `make_tile_group`在`section_cube`外部声明，与Add等Vector示例风格一致。
> - L1使用双缓冲（`mutex_ids`长度为2），L0A/L0B/L0C使用单缓冲（`mutex_ids`长度为1）。
> - `auto_mutex=True`自动管理各Tile的mutex锁，开发者无需手写`mutex_lock`/`mutex_unlock`。
> - 多核切分通过`pl.range(core_id, M // TILE_M, num_cores)`实现跨步分配，详见[多核切分与Tiling](multi_core_partitioning_and_Tiling.md)。
> - 上例K恰好为一个Tile，无需K维分块累加。K需要分块时请参考上文[K维分块累加](#k维分块累加)。

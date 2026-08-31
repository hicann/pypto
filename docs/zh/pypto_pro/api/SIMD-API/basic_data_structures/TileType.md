# pypto_pro.language.TileType

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

描述一块Tile的“规格”——形状、数据类型、所在内存空间、排布方式等，配合[`pypto_pro.language.make_tile`](../operation/resource_management/make_tile.md)或[`pypto_pro.language.make_tile_group`](../operation/resource_management/make_tile_group.md)分配实际缓冲区。

TileType本身不分配内存，只是一个规格描述符。实际缓冲区通过`make_tile`（单块）或`make_tile_group`（多块轮转）创建。

## 函数原型

```python
pypto_pro.language.TileType(shape, dtype, target_memory=pypto_pro.language.MemorySpace.Vec, valid_shape=None,
             layout=None, fractal=None, pad=None, compact=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | tile各维大小，如`[64, 128]` |
| `dtype` | 输入 | 元素数据类型，如`pypto_pro.language.DT_FP16` |
| `target_memory` | 输入 | 目标内存空间，默认`pypto_pro.language.MemorySpace.Vec` |
| `valid_shape` | 输入 | 可选，有效形状（处理尾块/非满块场景） |
| `layout` | 输入 | 可选，排布方式（`pl.NZ`/`pl.ZN`/`pl.ND`/`pl.DN`/`pl.ZZ`/`pl.NN`） |
| `fractal` | 输入 | 可选，分形大小 |
| `pad` | 输入 | 可选，填充模式 |
| `compact` | 输入 | 可选，紧凑模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 长度为2的编译期常量整数列表，各维大小须为正整数；仅支持二维Tile<br>对齐及分形布局约束由使用该TileType的具体API检查 |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](DataType.md)枚举值<br>常用：`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_BF16`、`pypto_pro.language.DT_INT8`、`pypto_pro.language.DT_INT32` |
| `target_memory` | 输入 | [`pypto_pro.language.MemorySpace`](MemorySpace.md)枚举值<br>默认`pypto_pro.language.MemorySpace.Vec`（UB）<br>可选：`Vec`(UB)、`Mat`(L1)、`Left`(L0A)、`Right`(L0B)、`Acc`(L0C)、`Scaling`、`ScaleLeft`、`ScaleRight` |
| `valid_shape` | 输入 | 编译期常量整数列表或`None`（默认）<br>具体整数（如`[32, 64]`）：编译期确定有效形状<br>`None`：后端缺省行为等同于`[-1, -1]`（动态模式）<br>`[-1, -1]`：运行时动态设置有效形状，配合[`pypto_pro.language.set_validshape`](../operation/memory_vector_computation/transpose_and_element_access/set_validshape.md)使用 |
| `layout` | 输入 | [`pypto_pro.language.TensorLayout`](TensorLayout.md)枚举值或`None`（默认）<br>不指定时按“内存空间 + 架构”自动取默认值（见下表）<br>`Mat`、`Left`、`Right`、`Acc`、`Scaling`、`ScaleLeft`、`ScaleRight`的非法组合在构造时即报`ValueError`；`Vec`的可用布局由具体Tile API约束 |
| `fractal` | 输入 | 整数或`None`（默认）<br>`Acc`的FP32/INT32在未指定时自动设为1024<br>显式值会写入Tile硬件信息，取值要求和适用场景由具体Tile API决定 |
| `pad` | 输入 | [`pypto_pro.language.TilePad`](TilePad.md)枚举值或整数0-3<br>`null`(0)不填充 / `zero`(1)补零 / `max`(2)补最大值 / `min`(3)补最小值<br>非法值报`ValueError`，非法类型报`TypeError` |
| compact | 输入 | 可选，Tile缓冲区的紧凑布局模式，描述Tile在搬运、重排和矩阵计算路径中的布局解释方式。取值如下：<br>- **0或None**：不启用紧凑模式。<br>&nbsp;&nbsp;- 此时L1→L0A/L0B的pypto_pro.language.move完全按物理shape搬运，pypto_pro.language.set_validshape在该搬运路径上不生效。<br>- **1**：使用普通紧凑模式。<br>&nbsp;&nbsp;- 数据在valid_shape向上对齐到分形粒度的有效空间内连续排布，Tile声明的其余空间空闲在尾部，详见[普通紧凑模式的数据排布](#普通紧凑模式的数据排布)。<br>&nbsp;&nbsp;- 通常用于尾块场景，与pypto_pro.language.set_validshape搭配使用。<br>&nbsp;&nbsp;- 在L1上配置与否对结果无影响。<br>&nbsp;&nbsp;- 与phase搭配使用时，如果存在尾块，L0C必须配置compact=1，否则可能会卡死。<br>&nbsp;&nbsp;- compact不会填充无效区域。需要填充时，通过pad参数或pypto_pro.language.fillpad补齐。<br>&nbsp;&nbsp;- compact不改变缓冲区的分配大小，只改变数据在缓冲区内的排布，因此把多个Tile拼成一个更宽的操作数的写法，需要特别注意地址的使用与偏移。多Tile之间距按物理shape计算地址偏移，紧凑模式下存储尾块数据的有效分形间距按实际shape计算地址偏移，这种场景下只能使用非紧凑搬运的方式，详见[多Tile拼接的尾块处理](#多Tile拼接的尾块处理)。<br>- **2**：使用RowPlusOne紧凑模式。<br>&nbsp;&nbsp;- 仅在UB Tile中配置，用于避免以该Tile为源作搬运时的bank冲突，详见[RowPlusOne紧凑模式的数据排布](#rowplusone紧凑模式的数据排布)。<br>&nbsp;&nbsp;- NZ格式下，每个分形列会多预留一行物理空间，仅作占位，不参与计算。因此要求在申请Tile的物理shape时包含多出来的这一行，数据使用时通过pypto_pro.language.set_validshape配置实际的有效行数。ZN格式同理，多预留的是一列。 |

### 普通紧凑模式的数据排布

```python
  举例：L0A Tile的物理shape=[64, 64]，数据类型为FP16，valid_shape=[8, 24]，NZ格式，由此可知：
        1、分形粒度D=16行×16列，共512字节（0x200）；
        2、实际搬运数据的有效空间为[ceil16(8), ceil16(24)] = [16, 32]，共2个分形；

  不配置 compact：数据按 shape=[64,64] 排布

     0   16   32   48   64  ← 列
   0 ┌────┬────┬────┬────┐
     │ D1 │ D2 │    │    │   D1、D2 = 两个有效数据分形
  16 ├────┼────┼────┼────┤
     │    │    │    │    │   相邻两列的间距 = 64 行 = 4 个分形 = 0x800
  32 ├────┼────┼────┼────┤   D1 首地址 0x000，D2 首地址 0x800
     │    │    │    │    │
  48 ├────┼────┼────┼────┤
     │    │    │    │    │
  64 └────┴────┴────┴────┘

  配置 compact=1：数据在 [16,32]，有效空间内连续排布

     0   16   32
   0 ┌────┬────┐          相邻两列的间距 = 16 行 = 1 个分形 = 0x200
     │ D1 │ D2 │          D1 首地址 0x000，D2 首地址 0x200
  16 └────┴────┘
     ┌─────────────────────┐
     │ 声明的其余空间，空闲 │   ← 缓冲区仍按 shape 分配
     └─────────────────────┘
```

### RowPlusOne紧凑模式的数据排布

```python
  举例：UB Tile的物理shape=[64, 64]，NZ格式

  配置 compact=1：分形列首尾相接

     ┌────┬────┬────┐
     │ D1 │ D2 │ D3 │   每列 64 行
     └────┴────┴────┘
       ↑    ↑    ↑
     首地址间距 = 64 行，各列起始落在相同的 bank 上


  配置 compact=2：每列多留 1 行

     ┌─────┬─────┬─────┐
     │ D1  │ D2  │ D3  │   每列 64 行数据
     │ ─── │ ─── │ ─── │ ← 每列末尾多出的 1 行（占位）
     └─────┴─────┴─────┘
       ↑     ↑     ↑
     首地址间距 = 65 行，各列起始依次错开一行，落到不同 bank 上
```

## 默认布局表

`layout`不指定时，按内存空间和架构自动取默认值：

| 内存空间 | A3默认`layout` | A5默认`layout` |
|---|---|---|
| `Vec` | 无约束 | 无约束 |
| `Mat` | `pl.NZ` | `pl.NZ` |
| `Left` | `pl.ZZ` | `pl.NZ` |
| `Right` | `pl.ZN` | `pl.ZN` |
| `Acc` | `pl.NZ` | `pl.NZ` |
| `Scaling` | `pl.ND` | `pl.ND` |
| `ScaleLeft` | — | `pl.ZZ` |
| `ScaleRight` | — | `pl.NN` |

补充规则：

- `Mat`除默认`pl.NZ`外，还允许`pl.ZN`；当`dtype`为`UINT64`或`INT64`时，额外允许`pl.ND`
- `Left`跨架构兼容，同时允许`pl.ZZ`（A3默认）和`pl.NZ`（A5默认）
- `Mat`中的E8M0 scale允许`pl.ZZ`和`pl.NN`，未指定`fractal`时自动取32
- `ScaleLeft`和`ScaleRight`仅A5支持，分别用于A矩阵和B矩阵的E8M0 scale；未指定`fractal`时自动取32，Tile地址须按32字节对齐

## 调用示例

以下示例展示不同内存空间和使用场景下的`TileType`定义。`TileType`仅描述Tile规格，实际缓冲区由`make_tile`或`make_tile_group`创建。

```python
import pypto_pro.language as pl

TILE_M = 64
TILE_K = 128
TILE_N = 128

# UB Tile：target_memory 默认取 MemorySpace.Vec
vec_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP16,
)

# A5 L0A Tile：Left 的默认布局为 NZ，也可以显式指定
left_type = pl.TileType(
    shape=[TILE_M, TILE_K],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left,
    layout=pl.NZ,
)

# A5 L1 转置分形 Tile：Mat 支持显式指定 ZN
mat_zn_type = pl.TileType(
    shape=[TILE_K, TILE_N],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.ZN,
)

# L0C Tile：FP32 Acc 未指定 fractal 时自动取 1024
acc_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Acc,
)

# 动态尾块 Tile：运行时使用 set_validshape 设置实际有效形状
tail_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
    compact=1,
)

# 带填充模式的 Tile：供 fillpad 等读取 pad 属性的操作使用
mask_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
    compact=1,
    pad=pl.TilePad.min,
)
```

### compact = 1 行方向尾块

```python
import os
import pypto_pro.language as pl
import torch

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
TILE = 128
M_TAIL = 72

@pl.jit(auto_mutex=True)
def matmul_m_tail(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[M_TAIL, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ, valid_shape=[-1, -1],
        ),
        addrs=0x0,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    # L0A：matmul恒按紧凑布局取数，compact=1使move按同样的跨度写入
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
            layout=pl.NZ, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0,
        mutex_ids=[3],
    )
    # L0C：matmul恒按紧凑布局写数，compact=1使store按同样的跨度读出
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(al, [M_TAIL, TILE])
        pl.set_validshape(ac, [M_TAIL, TILE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])

if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    torch.manual_seed(42)
    a = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    b = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    out = torch.zeros([M_TAIL, TILE], device=DEVICE, dtype=torch.float32)

    matmul_m_tail(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a[:M_TAIL, :].float(), b.float())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### 多Tile拼接的尾块处理

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def matmul_wide_alias(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[40, 128], pl.DT_FP32],
):
    x_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    y_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    x_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[2, 3],
    )
    # 别名：同一基地址，作为matmul的左操作数
    x_wide = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[[2, 3]],
    )
    y_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        cur_x = x_l1.current()
        cur_y = y_l1.current()
        br = y_l0b.current()
        ac = c_l0c.current()

        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.move(br, cur_y)

        # 两块L0A Tile不设置compact=1和set_validshape，数据全量搬运，确保拼接后的整块L0A Tile中数据排布连续
        al_1 = x_l0a.next()
        pl.move(al_1, cur_x, [0, 0])
        al_2 = x_l0a.next()
        pl.move(al_2, cur_x, [0, 64])

        al = x_wide.next()
        pl.matmul(ac, al, br)

        # L0C不设置compact=1，store按matmul的写入格式搬运；set_validshape控制搬运的数据量；
        pl.set_validshape(ac, [40, 128])
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn([64, 128], device=device, dtype=torch.float16)
    y = torch.eye(128, device=device, dtype=torch.float16)
    out = torch.zeros([40, 128], device=device, dtype=torch.float32)

    matmul_wide_alias(x, y, out)
    torch.npu.synchronize()

    ref = x[:40, :].float()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"wide alias matmul passed, max diff = {(out - ref).abs().max().item()}")
```

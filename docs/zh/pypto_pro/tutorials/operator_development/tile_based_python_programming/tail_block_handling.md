# 尾块处理：valid_shape、set_validshape、pad与compact

本文档介绍PyPTO Pro编程里的**尾块（tail tile）**处理：当GM上的`pl.Tensor`
的shape **不能被tile shape整除**时，边界上会出现比tile小的"不完整块"。本文讲清
下面这几个参数是如何**协同**工作的：

| 参数 / API                    | 位置                          | 作用                                                   |
|-------------------------------|-------------------------------|--------------------------------------------------------|
| `TileType.shape`              | `TileType`（编译期常量）      | tile在片上占用的**物理**大小，决定buffer字节数。   |
| `TileType.valid_shape`        | `TileType`（编译期常量/`-1`） | 声明tile的**有效**（逻辑）区域；`-1`表示运行时决定。|
| `pl.set_validshape(tile,...)` | kernel体（运行时）           | 在运行时把每个尾块的真实有效行/列写进tile。          |
| `TileType.pad`                | `TileType`（编译期常量）      | 有效区域之外的**填充区**如何取值（null/zero/max/min）。|
| `pl.fillpad(dst, src)`        | kernel体                     | 真正把填充值写进padding区的算子。                    |
| `TileType.compact`            | `TileType`（编译期常量）      | 片上buffer的紧凑摆放模式（null/normal/row_plus_one）。|

源码参考：

- `python/pypto_pro/ir/op/block_ops.py` — `TileType`、`make_tile`、`_normalize_tile_pad`。
- `python/pypto_pro/language/_api.py` — `set_validshape`、`fillpad`系列。
- `framework/src/interface/pypto_pro/codegen/cce/cce_codegen.cpp` —
  `ExtractValidShapeInfo` / `BuildTileCtorArgs`（valid_shape如何进入Tile构造）。
- `framework/src/interface/pypto_pro/backend/backend_cce_block_out_ops.cpp` —
  `MakeBlockOutSetValidShapeCodegenCCE`、`EmitSetShapeIfDynamic`（load/store如何按有效
  形状裁剪GM搬运）。

可执行范例：

- `python/tests/st/pypto_pro/frontend/element_wise/test_eltwise_dynamic_rank.py`
  （四类尾块的完整覆盖，本文贯穿使用）。
- `python/tests/st/pypto_pro/frontend/docs/test_doc_dcci_atomic_validshape.py`
  （最小`set_validshape`示例）。
- `python/tests/st/pypto_pro/frontend/datacopy/test_pro_fillpad.py`（`pad` + `fillpad`）。

统一前置导入：

```python
import torch
import pypto_pro.language as pl
```

---

## 为什么会出现尾块

一个tile是**固定大小**的片上buffer。当你用一个`[TILE_M, TILE_N]`的tile去覆盖
一个`[H, W]`的GM tensor时，需要`ceil(H/TILE_M) * ceil(W/TILE_N)`个tile。若
`H`、`W`不是tile尺寸的整数倍，**最后一行 / 最后一列 / 右下角**的tile就装不满。

以文档标题里的例子说明：

- `pl.Tensor` shape = `[129, 129]`
- tile shape = `[64, 64]`

`129 = 2 * 64 + 1`，所以每个轴被切成 **3段**：`[0:64]`、`[64:128]`、`[128:129]`（最后
一段只有1个元素）。于是整张tensor被切成一个 **3 × 3的tile网格**，其中会出现
**4种不同的有效形状**：

```text
            列 0:64        列 64:128     列 128:129
          +-------------+-------------+-----------+
行 0:64   |  [64, 64]   |  [64, 64]   | [64, 1]   |
          +-------------+-------------+-----------+
行 64:128 |  [64, 64]   |  [64, 64]   | [64, 1]   |
          +-------------+-------------+-----------+
行128:129 |  [1, 64]    |  [1, 64]    | [1, 1]    |   <- 尾行
          +-------------+-------------+-----------+
                                          ^尾角
```

| 有效形状     | 出现位置                     | 说明                       |
|--------------|------------------------------|----------------------------|
| `[64, 64]`   | 内部块                       | 满块，physical == valid。 |
| `[1, 64]`    | 尾行（最后一行tile）        | 行方向被截断。            |
| `[64, 1]`    | 尾列（最后一列tile）        | 列方向被截断。            |
| `[1, 1]`     | 右下角（尾行 ∩ 尾列）        | 两个方向都被截断。        |

**关键点**：这4类tile的**物理**尺寸都仍是`[64, 64]`（片上buffer不变），只是
**有效区域**不同。尾块处理的本质就是：*物理形状固定，有效形状随位置变化*。

---

## `TileType.shape`与`TileType.valid_shape`的分工

```python
@dataclass
class TileType:
    shape: Sequence[int]                          # 物理大小，决定 buffer 字节数
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None   # 逻辑有效区域（<= shape）
    layout: Optional[TensorLayout] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None                      # TilePad.null/zero/max/min
    compact: Optional[int] = None
```

- **`shape`** 决定片上占用（`prod(shape) * dtype_bytes`字节）。它必须是**编译期常量**，
  始终按满块（如`[64, 64]`）分配 —— 这样一个buffer可以复用来处理满块和各种尾块。
- **`valid_shape`** 描述"里面实际有多少有效数据"。它有三种取值形态：

| `valid_shape`写法 | 含义                                          | 谁来给最终值                     |
|--------------------|-----------------------------------------------|----------------------------------|
| **不给（None）**   | 有效区域 == 物理`shape`（永远满块）。       | 编译期，取`shape`。            |
| **静态整数**，如`[64, 32]` | 每个tile有效区固定为该子矩形。      | 编译期常量。                    |
| **`-1`哨兵**，如`[-1, -1]` | 有效行/列是**运行时**的，随尾块变化。 | 运行时`set_validshape`决定。 |

> **注意**：`valid_shape`里**不能**直接写运行时标量变量（例如kernel的`rows`
> 参数）。`TileType`的shape/valid_shape会被提升到生成C++ 的函数序言处、声明tile
> 的位置，运行时变量在那里尚未定义。因此parser会拒绝它
> （`block_ops.py:_parse_tile_type_call`），要用运行时有效形状，请写`-1` + 后续调用
> `pl.set_validshape()`。

尾块处理的标准写法就是把动态轴写成`-1`：

```python
# 物理 64x64；有效行、有效列都在运行时用 set_validshape 决定
tile_type = pl.TileType(
    shape=[64, 64],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
)
```

### `-1`在codegen里发生了什么

`-1`是"模板占位"的意思。CCE codegen的`ExtractValidShapeInfo`
（`cce_codegen.cpp`）对每个valid_shape维度这样处理：

| valid_shape元素   | 生成的Tile模板参数 | 是否需要构造参数         |
|--------------------|----------------------|--------------------------|
| 不给 / `ConstInt(-1)` | `-1`（动态）       | 需要 —— 运行时值走构造函数 |
| `ConstInt(N > 0)`  | `N`（静态）          | 不需要                    |
| 运行时`Var`       | `-1`                 | 需要 —— 用变量名做构造参数 |

也就是说：写`-1`会生成一个"有效形状可在运行时设置"的Tile类型；`set_validshape`
再把真实值填进这个Tile对象（生成`tile.SetValidShape(row, col);`）。

### 三种写法对照生成的C++

三种`valid_shape`写法生成的Tile声明大致如下（示意，实际模板名以codegen为准）：

```python
# (a) 不给 valid_shape
pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
# -> Tile<half, 64, 64, ...> tile;                 // 模板即物理 shape，无构造参数

# (b) 静态子块
pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
            valid_shape=[64, 32])
# -> Tile<half, 64, 64, /*validRow=*/64, /*validCol=*/32, ...> tile;   // 有效区编进模板

# (c) 动态（-1）
pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1])
# -> Tile<half, 64, 64, -1, -1, ...> tile(rows, cols);   // -1 占位，运行时走构造参数
# 之后每次 set_validshape 生成：tile.SetValidShape(rows, cols);
```

判断"要不要构造参数"的逻辑就是`ExtractValidShapeInfo`里的`needs_ctor`：只要某个维度
是`-1`或运行时`Var`，该维就走构造参数 / `SetValidShape`；纯静态整数则完全编进模板。

---

## `pl.set_validshape` —— 运行时写入尾块有效形状

```python
def set_validshape(tile, [row, col]) -> None
```

它在kernel体里、每次处理一个tile之前调用，把该tile这一次的有效行/列写进tile
对象。后端生成的就是一句`tile.SetValidShape(row, col);`
（`backend_cce_block_out_ops.cpp:MakeBlockOutSetValidShapeCodegenCCE`）。

最小示例（照搬`test_doc_dcci_atomic_validshape.py`）：

```python
@pl.jit()
def validshape_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            valid_shape=[-1, -1])          # 两轴都动态
    tile = pl.make_tile(tile_type, addr=0x0000, size=32768)
    with pl.section_vector():
        pl.load(tile, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.set_validshape(tile, [rows, cols])             # 运行时写有效形状
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile, [0, 0])
```

### 有效形状如何裁剪GM搬运（这是尾块正确性的核心）

`load` / `store`在GM一侧要告诉硬件DMA "搬多少行、多少列"。当tile有`set_validshape`
时，后端**不是**用tile的物理shape，而是回读tile的运行时有效形状
（`GetValidRow()` / `GetValidCol()`）来生成GM的`SetShape<DIM_3, DIM_4>(...)`，见
`EmitSetShapeIfDynamic`（`backend_cce_block_out_ops.cpp`）：

```cpp
tensor.SetShape<DIM_3, DIM_4>(
    static_cast<int64_t>(tile.GetValidRow()),
    static_cast<int64_t>(tile.GetValidCol()));
```

这带来两个直接后果：

1. **不会越界**：尾块只会从GM读/写`[1, 64]`（而不是`[64, 64]`），所以哪怕
   tensor只有129行，读第128行的尾块也不会碰到第129～191行的非法地址。
2. **不会污染**：store尾块时只把有效区写回GM，padding区里的垃圾值不会写出去。

> 因为后端是**回读tile的运行时有效形状**而不是缓存"最后一次set_validshape的表达
> 式字符串"，所以`set_validshape`无论写在直线代码里、还是在`if/else`各分支里各写一
> 次，都能得到正确的DMA尺寸。

### 静态子块：编译期就知道有效区

不是所有"部分块"都要用`-1` + `set_validshape`。如果有效区在**编译期**就固定（例如你
永远只用一个`[64, 64]` buffer的左上`[64, 32]`），直接写静态`valid_shape`：

```python
# 有效区固定为 64x32，无需任何 set_validshape
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                 target_memory=pl.MemorySpace.Vec, valid_shape=[64, 32])
tile = pl.make_tile(tt, addr=0x0, size=64*64*4)
pl.load(tile, a, [0, 0])       # 直接按 64x32 搬运
```

**如何选**：

| 场景                                   | 写法                              |
|----------------------------------------|-----------------------------------|
| 永远满块                               | 不给`valid_shape`                |
| 有效区编译期固定、每个tile都一样     | 静态`valid_shape=[R, C]`         |
| 有效区随tile位置变（真正的尾块）     | `valid_shape=[-1, -1]` + `set_validshape` |
| 只有某一轴动态（如行动态、列永远满）   | `valid_shape=[-1, 64]`（混用）    |

最后一行是常见优化：如果你保证列方向总能整除、只有行方向有尾块，就写
`valid_shape=[-1, TILE_N]`，只有行走运行时，列仍是静态模板参数，生成的代码更紧。

---

## `pad`与`fillpad` —— 填充区如何取值

有效形状只是"逻辑边界"，物理buffer里有效区**之外**仍然是上一次遗留的垃圾值。对于
**逐元素**算子这无所谓（我们只store有效区）。但对**归约 / matmul** 这类"会把整块读进
去参与计算"的算子，垃圾值会污染结果 —— 比如对一个尾块做`maximum(dim=0)`，padding里的大数
会顶掉真正的最大值。

`TileType.pad`声明填充区**应该**取什么值：

| `pl.TilePad` | 语义             | 典型用途                                  |
|--------------|------------------|-------------------------------------------|
| `null`       | 不填充（默认）   | 逐元素算子，或你自己保证不读padding。   |
| `zero`       | 填0             | 求和 / 卷积等，0是加法单位元。          |
| `max`        | 填该dtype最大值| `minimum(dim=0)`前填充，避免假的最小值。       |
| `min`        | 填该dtype最小值| `maximum(dim=0)` / softmax前填充，避免假的最大值。|

`pad`只是**声明**；真正把值写进padding区要靠`pl.fillpad`：

```python
def fillpad(out, src, *, mode=pl.FillPadMode.NORMAL) -> None
```

示例（照搬`test_pro_fillpad.py`）：`src`声明动态有效形状，`dst`声明`pad=zero`；运行时把
有效形状缩到`[5, 7]`，`fillpad`就把`dst`里`[5:8, 7:8]`等padding区清零：

```python
src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                       target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                       target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)  # 声明填 0
src = pl.make_tile(src_type, addr=0x0000, size=256)
dst = pl.make_tile(dst_type, addr=0x0100, size=256)

with pl.section_vector():
    pl.load(src, x, [0, 0])
    # ... 同步 ...
    pl.set_validshape(src, [5, 7])   # 有效区 5x7，其余 3 行/1 列是 padding
    pl.fillpad(dst, src)             # dst 的 padding 区按 pad=zero 清零
    # ... store dst ...
```

### `fillpad`三种模式

| API | dst / src关系 | 用途 |
| --- | --- | --- |
| `pl.fillpad(dst, src)` | 不同址 | 默认模式，把`src`拷到`dst`并按`dst.pad`填充边界。 |
| `pl.fillpad(dst, src, mode=pl.FillPadMode.INPLACE)` | **同址**（`addr`相同） | 原地填充，省一块buffer。 |
| `pl.fillpad(dst, src, mode=pl.FillPadMode.EXPAND)` | dst列数 > src | 列方向**展开**并填充（如`[8,8]` → `[8,16]`）。 |

`EXPAND`模式例子（`test_pro_fillpad.py`）：`src`有效区`[5, 7]`，`dst`是`[8, 16]`且
`pad=zero`，展开后`dst`里`[0:5, 0:7]`是数据、其余（含新增的8列）清零：

```python
src_type = pl.TileType(shape=[8, 8],  dtype=pl.DT_INT32,
                       target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32,       # 更宽
                       target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
src = pl.make_tile(src_type, addr=0x0000, size=256)
dst = pl.make_tile(dst_type, addr=0x0100, size=512)
with pl.section_vector():
    pl.load(src, x, [0, 0])
    # ... 同步 ...
    pl.set_validshape(src, [5, 7])
    pl.fillpad(dst, src, mode=pl.FillPadMode.EXPAND)  # [8,8] 有效 5x7 -> [8,16] 其余清零
    pl.store(z, dst, [0, 0])
```

### 为什么归约 / matmul尾块必须fillpad

设想对一个`[64, 64]` tile的尾块（有效区`[3, 64]`）做`maximum(dim=0)`（每行取最大）。归约会把
**整块** 64行都读进去；第3～63行是上一轮遗留的垃圾值，可能比真实最大值还大，结果就
错了。正确做法是`pad=min` + `fillpad`，把无效行填成dtype最小值，这样它们不会顶替真正
的最大值：

```python
# softmax / maximum(dim=0) 场景：无效区填 min，不会污染最大值
t_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                     valid_shape=[-1, -1], pad=pl.TilePad.min)
src = pl.make_tile(t_type, addr=0x0, size=64*64*2)
out = pl.make_tile(pl.TileType(shape=[64, 1], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.DN), addr=0x2000, size=128)
tmp = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec), addr=0x2080, size=64*64*2)
with pl.section_vector():
    pl.load(src, a, [0, 0])
    pl.set_validshape(src, [rows, 64])   # 只有 rows 行有效
    pl.fillpad(src, src, mode=pl.FillPadMode.INPLACE)  # 无效行填 min（pad=min）
    pl.maximum(out, src, tmp, dim=0)      # 现在整块归约安全
```

对应关系（选pad的口诀）：

| 归约类型              | 无效区应填    | `pad`             |
|-----------------------|---------------|-------------------|
| `sum(dim=0)` / 求和 / 卷积 | 0（加法单位元） | `pl.TilePad.zero` |
| `maximum(dim=0)` / softmax   | 最小值        | `pl.TilePad.min`  |
| `minimum(dim=0)`             | 最大值        | `pl.TilePad.max`  |
| 逐元素（只store有效区） | 不用填        | `pl.TilePad.null`（默认） |

`pad`与`valid_shape`的配合关系一句话：**`valid_shape`划出边界，`pad`决定边界外的
值，`fillpad`负责把这个值真正写进去。**

---

## `compact` —— 片上紧凑摆放

`TileType.compact`使用整数控制tile在片上buffer里的紧凑摆放方式：

| `compact`值 | 含义 |
| --- | --- |
| `0`或不设置 | 不紧凑（默认），按物理shape常规摆放。 |
| `1` | 常规紧凑模式。 |
| `2` | “行数 +1”紧凑模式（某些算子的对齐需求）。 |

CCE CodeGen会将这三个值分别转换为`CompactMode::None`、`CompactMode::Normal`和
`CompactMode::RowPlusOne`（`type_converter.cpp:ConvertCompactModeToPTOValue`）。
Python Kernel中应直接使用整数值，不使用`pl.CompactMode`。

### 与尾块 / matmul的关系

`compact`常和`valid_shape=[-1, -1]`一起出现在 **matmul操作数**（Mat / Left / Right）
上。原因是：matmul的L1 / L0 buffer按满块shape分配，但一个尾块只有效一部分；开
`compact`让硬件按**有效行**紧凑摆放，减少L1/L0占用、提升带宽利用。真实例子
（`fa/test_fa_perf_tkv_preload_dn_tile.py`）：

```python
# Q/K/V 的 L1 暂存 tile：动态有效形状 + 紧凑摆放
q_mat_type = pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                         valid_shape=[-1, -1], compact=1)     # normal 紧凑
k_mat_type = pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
left_type  = pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
```

`compact`是较进阶的布局参数，只在特定算子对片上摆放有特殊要求时才设置；纯逐元素的尾块
场景用默认`null`即可。它与`valid_shape`/`pad`正交：`compact`影响的是buffer内部
排布，不改变"哪块数据有效"，也不改变填充值。

---

## 完整可执行示例 —— 四类尾块全覆盖

下面是`python/tests/st/pypto_pro/frontend/element_wise/test_eltwise_dynamic_rank.py`的
核心。它把`[M, N]`用`128 × 128`的tile铺满，用嵌套`if`为四类tile（满块 / 尾行 /
尾列 / 尾角）分别设置有效形状。以`[513, 513]`（`513 = 4*128 + 1`）为例，恰好命中
`128x1`、`1x128`、`1x1`三种尾块。

```python
from __future__ import annotations
from dataclasses import dataclass
import torch
import pypto_pro.language as pl

TILE_M = 128
TILE_N = 128


@dataclass
class AddTiling:
    shape: int[4]       # 每维大小；未用的前导维为 1
    opkind: int[8]      # 算子选择器，真正的值在 opkind[4]


@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    # 把 rank 2..4 的逻辑 shape 折叠成二维 [M, N]
    N = tiling.shape[3]
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]

    tensor_x = pl.make_tensor(x, [M, N], [N, 1])
    tensor_y = pl.make_tensor(y, [M, N], [N, 1])
    tensor_z = pl.make_tensor(z, [M, N], [N, 1])

    # valid_shape=[-1, -1]：两轴有效窗口都在运行时用 set_validshape 决定，
    # 这样边界 tile 能处理一个部分 [rows, cols] 块。
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000,  mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        # ceildiv 向上取整，保证边界 tile 被覆盖。
        m_tiles = (M + TILE_M - 1) // TILE_M
        n_tiles = (N + TILE_N - 1) // TILE_N
        total_tiles = m_tiles * n_tiles

        for idx in pl.range(core_id, total_tiles, num_cores):
            i = idx // n_tiles          # 行 tile 索引
            j = idx % n_tiles           # 列 tile 索引
            tile_a = a_db.next()
            tile_b = b_db.next()
            tile_c = c_db.next()

            # 本 tile 每轴的实际有效跨度。只有最后一行 / 最后一列的 tile
            # 才 < TILE；嵌套分支挑出四类有效形状之一。
            rem_r = M - i * TILE_M
            rem_c = N - j * TILE_N
            if rem_r >= TILE_M:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [TILE_M, TILE_N])   # 满块
                    pl.set_validshape(tile_b, [TILE_M, TILE_N])
                    pl.set_validshape(tile_c, [TILE_M, TILE_N])
                else:
                    pl.set_validshape(tile_a, [TILE_M, rem_c])    # 尾列
                    pl.set_validshape(tile_b, [TILE_M, rem_c])
                    pl.set_validshape(tile_c, [TILE_M, rem_c])
            else:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [rem_r, TILE_N])    # 尾行
                    pl.set_validshape(tile_b, [rem_r, TILE_N])
                    pl.set_validshape(tile_c, [rem_r, TILE_N])
                else:
                    pl.set_validshape(tile_a, [rem_r, rem_c])     # 尾角
                    pl.set_validshape(tile_b, [rem_r, rem_c])
                    pl.set_validshape(tile_c, [rem_r, rem_c])

            pl.load_tile(tile_a, tensor_x, [i, j])   # 只搬有效区，不越界
            pl.load_tile(tile_b, tensor_y, [i, j])

            if tiling.opkind[4] == 0:
                pl.add(tile_c, tile_a, tile_b)
            elif tiling.opkind[4] == 1:
                pl.sub(tile_c, tile_a, tile_b)
            else:
                pl.mul(tile_c, tile_a, tile_b)

            pl.store_tile(tensor_z, tile_c, [i, j])  # 只写回有效区
```

### 两种等价的set_validshape写法

上面用嵌套`if`分别处理四类tile。也可以先算出`valid_rows` / `valid_cols`，再统一调
一次 —— 两种写法都对，因为后端从tile的**运行时**有效形状取值，不依赖调用点：

```python
valid_rows = TILE_M
valid_cols = TILE_N
if M - i * TILE_M < TILE_M:
    valid_rows = M - i * TILE_M       # 尾行
if N - j * TILE_N < TILE_N:
    valid_cols = N - j * TILE_N       # 尾列
pl.set_validshape(tile_a, [valid_rows, valid_cols])
pl.set_validshape(tile_b, [valid_rows, valid_cols])
pl.set_validshape(tile_c, [valid_rows, valid_cols])
```

---

## 各参数如何协同 —— 一图流

针对一个边界tile，处理顺序是：

```text
物理分配          有效边界              填充                 计算/搬运
--------          --------              ----                 --------
TileType.shape -> valid_shape=[-1,-1] -> pad=zero/max/min -> load(只搬有效区)
[64,64] 固定       set_validshape         fillpad(写填充区)     计算(可读整块)
buffer            (运行时行/列)          (归约/matmul 前)       store(只写有效区)
```

- **`shape`**：定物理buffer（永远满块，可复用）。
- **`valid_shape=[-1,-1]`**：声明"有效边界运行时定"。
- **`set_validshape`**：每个tile填真实有效行/列；load/store据此裁剪GM搬运，天然
  防越界、防污染。
- **`pad` + `fillpad`**：只有当算子会读到有效区之外（归约 / matmul）时才需要；把
  padding填成计算的单位元。
- **`compact`**：正交的片上摆放优化，尾块本身不需要。

### 按场景选参数

| 场景                                        | `valid_shape`      | `pad`  | `fillpad` | `compact` |
|---------------------------------------------|--------------------|--------|-----------|-----------|
| Vec逐元素、shape整除tile                 | 不给               | 默认   | 否        | 默认      |
| Vec逐元素、shape **不整除**（尾块）        | `[-1, -1]`         | 默认   | 否        | 默认      |
| Vec归约（`maximum(dim=0)`/softmax）尾块           | `[-1, -1]`         | `min`  | **是**    | 默认      |
| Vec归约（`sum(dim=0)`）尾块                   | `[-1, -1]`         | `zero` | **是**    | 默认      |
| matmul操作数（Mat/Left/Right）尾块         | `[-1, -1]`         | 默认   | 否        | `1`（normal） |
| 有效区编译期固定                            | `[R, C]`静态      | 默认   | 否        | 默认      |

> 记忆法：**逐元素**只需`valid_shape`；**归约/matmul会读整块**，所以要么`pad+fillpad`
> 把无效区变成安全值（Vec归约），要么开`compact`让硬件只算有效行（matmul）。

---

<a id="常见坑"></a>

## 常见坑

- **把运行时变量写进`valid_shape`。** `TileType(valid_shape=[rows, cols])`里
  `rows/cols`是kernel参数时会被parser拒绝（tile声明被提升到函数序言，变量在那儿
  还没定义）。写`valid_shape=[-1, -1]`，再用`pl.set_validshape`给运行时值。
- **忘了`fillpad`而直接对尾块做归约 / matmul。** `valid_shape`只裁剪GM搬运，不清
  理片上padding。做`maximum(dim=0)` / `sum(dim=0)` / matmul前，用`pad=min/max/zero` +
  `fillpad`把padding填成单位元，否则垃圾值污染结果。
- **`size`按有效形状而不是物理shape算。** `make_tile(addr=..., size=...)`的`size`
  是**物理** buffer字节数（`prod(shape)*dtype_bytes`），跟`valid_shape`无关。
- **tile数用了`//`而不是ceildiv。** 覆盖尾块必须向上取整：
  `n_tiles = (N + TILE_N - 1) // TILE_N`。用`N // TILE_N`会漏掉尾块。
- **多tile复用同一buffer却漏设`set_validshape`。** tile的有效形状是**有状态**的：
  上一轮设过`[1, 64]`，这一轮若是满块却忘了重设，就会仍按`[1, 64]`搬运。每轮都要
  为当轮形状显式`set_validshape`（或走"统一计算一次"的写法）。

---

## 速查

```python
# --- 物理满块 + 运行时有效边界 ---
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec,
                 valid_shape=[-1, -1])        # -1 = 运行时决定
tile = pl.make_tile(tt, addr=0x0, size=64*64*2)   # size 按物理 shape

# --- 每个尾块运行时写有效形状 ---
pl.set_validshape(tile, [valid_rows, valid_cols])
pl.load_tile(tile, tensor, [i, j])           # 只搬 valid_rows x valid_cols
pl.store_tile(tensor, tile, [i, j])

# --- 归约 / matmul 前填充 padding ---
dst_tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec,
                     pad=pl.TilePad.min)      # softmax/maximum(dim=0) 用 min
pl.fillpad(dst, src)                          # 把 padding 写成 min

# --- 静态子块（编译期就知道有效区）---
tt2 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                  target_memory=pl.MemorySpace.Vec,
                  valid_shape=[64, 32])        # 固定有效区，无需 set_validshape

# --- tile 数一定向上取整，才能覆盖尾块 ---
m_tiles = (M + TILE_M - 1) // TILE_M
n_tiles = (N + TILE_N - 1) // TILE_N
```

# pypto_pro.language.shl

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

将源操作数lhs中的每个元素按照rhs对应位置的移位量向左移位，将结果写入目的操作数out。

## 函数原型

```python
pypto_pro.language.shl(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素左移的结果。必须位于UB，采用row-major布局。数据类型与lhs、rhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64或DT_UINT64。valid_shape与lhs、rhs一致。 |
| lhs | 输入 | 被移位操作数，Tile类型。必须位于UB，采用row-major布局。数据类型与out一致。 |
| rhs | 输入 | 移位量，Tile类型。必须位于UB，采用row-major布局。数据类型与out一致。移位量必须为非负数，且小于元素位宽。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

下面是一个完整Kernel：从GM载入两个DT_INT32输入到UB，使用pypto_pro.language.shl逐元素左移后写回GM。Vector Kernel开启auto_mutex，同步由make_tile_group自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shl_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shl(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

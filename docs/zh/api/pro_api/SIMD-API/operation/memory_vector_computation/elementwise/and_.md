# pypto_pro.language.and_

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

两个操作数对应位置逐元素按位与。支持Tile-Tile和Tile-Scalar两种模式，其中Scalar为标量；同时支持原地计算。

- **Tile-Tile模式**：对lhs和rhs对应位置的元素执行按位与，将结果写入out。
- **Tile-Scalar模式**：对lhs中的每个元素和rhs标量执行按位与，将结果写入out。

## 函数原型

```python
pypto_pro.language.and_(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素按位与的结果。数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32或DT_UINT32。可与lhs为同一Tile，实现原地计算。 |
| lhs | 输入 | 左操作数，Tile类型。数据类型与out一致。 |
| rhs | 输入 | 右操作数，Tile或Scalar类型。传入Tile时执行Tile-Tile计算，数据类型与out一致，且shape与out、lhs一致；传入Scalar时执行Tile-Scalar计算，支持int或Scalar类型。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### Tile-Scalar模式

下面是一个完整Kernel：从GM载入DT_INT32输入到UB，使用pypto_pro.language.and_与标量7逐元素按位与后写回GM。Vector Kernel开启auto_mutex，同步由make_tile_group自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def and_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
                      out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.and_(cur_out, cur_a, 7)
        pl.store(out, cur_out, [0, 0])
```

### Tile-Tile模式

```python
# 两个Tile对应位置按位与。
pl.and_(tile_out, tile_a, tile_b)
```

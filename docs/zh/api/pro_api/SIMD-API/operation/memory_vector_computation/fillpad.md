# pypto_pro.language.fillpad

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

填充Tile的padding区域。当Tile设置的valid_shape（有效形状）小于shape（物理形状）时，有效区域之外的部分为padding区域。fillpad根据Tile创建时指定的pad值（如pypto_pro.language.TilePad.zero）填充该区域。

mode用于选择填充模式：

- pypto_pro.language.FillPadMode.NORMAL：out和src的形状相同、地址不同。
- pypto_pro.language.FillPadMode.EXPAND：允许out的形状大于src的形状，将源Tile的有效数据复制到目标Tile，并填充扩展区域。
- pypto_pro.language.FillPadMode.INPLACE：out和src的形状相同、地址相同，直接在原地址上填充。

## 函数原型

```python
pypto_pro.language.fillpad(
    out: Tile,
    src: Tile,
    *,
    mode: FillPadMode = FillPadMode.NORMAL,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存储空间为UB或L1 Buffer，支持8、16和32 bit数据类型。必须设置有效的pad属性，例如pypto_pro.language.TilePad.zero。 |
| src | 输入 | 源操作数，Tile类型，存储空间为UB或L1 Buffer，且必须与out位于相同Buffer。数据位宽必须与out相同。填充范围由src的有效形状确定，有效形状可在TileType中指定或通过set_validshape设置。 |
| mode | 输入 | 填充模式，[FillPadMode](../../basic_data_structures/FillPadMode.md)类型。 |

## 约束说明

fillpad支持UB和L1 Buffer中的Tile，源Tile和目标Tile必须位于相同Buffer。

### pypto_pro.language.FillPadMode.NORMAL模式

- out和src的形状必须相同，地址必须不同。
- out和src位于L1 Buffer时，Tile类型必须完全相同，包括形状、有效形状和pad属性。

### pypto_pro.language.FillPadMode.EXPAND模式

- out和src的地址必须不同，out各维形状不得小于src对应维度。
- 不支持在L1 Buffer中对不同形状的源Tile和目标Tile使用EXPAND模式。

### pypto_pro.language.FillPadMode.INPLACE模式

- out和src的形状必须相同，并共享同一地址。

## 返回值说明

无。

## 调用示例

### NORMAL模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src)
        pl.store(z, cur_dst, [0, 0])
```

### EXPAND模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_expand_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.EXPAND)
        pl.store(z, cur_dst, [0, 0])
```

### INPLACE模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_inplace_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec,
                           pad=pl.TilePad.zero, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.INPLACE)
        pl.store(z, cur_dst, [0, 0])
```

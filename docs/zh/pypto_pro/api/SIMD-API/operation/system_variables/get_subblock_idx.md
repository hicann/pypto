# pypto_pro.language.get_subblock_idx

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

获取当前subblock索引。

## 函数原型

```python
val = pypto_pro.language.get_subblock_idx()
```

无参数。

## 返回值说明

返回当前subblock索引（0或1），类型为整型Expr。A5架构上每个AI Core有两个vector子核，该值用于区分子核身份。

## 典型使用场景

`pypto_pro.language.get_subblock_idx()`主要用于以下两种模式：

1. **`insert` + cube模式**：每个子核计算部分结果，用`pypto_pro.language.insert`拼入Mat tile（L1 NZ缓冲），cube侧读取合并后的完整数据。详见[insert](../memory_data_movement/insert.md)文档示例。

2. **条件执行**：根据子核号决定是否执行某段代码，例如只让sub-core 0发起`pypto_pro.language.ssbuf_store`。

> [!CAUTION]注意
> 纯vector kernel中两个子核共享MTE搬运管道，不能让每个子核独立`pypto_pro.language.store`到GM的不同区域。如需按子核切分数据搬运，应使用`insert` + cube模式。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.get_subblock_idx()`读出子核号（此处验证可调用），实际计算为64×64 FP32 element-wise加法。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def subblock_add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_sum = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        _sub_index = pl.get_subblock_idx()
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_sum = tile_sum.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_sum, cur_x, cur_y)
        pl.store(out, cur_sum, [0, 0])
```

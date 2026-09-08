# pypto_pro.language.get_block_num

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

获取本次实际启动的逻辑Block数量，用于多核控制和数据偏移计算。

## 函数原型

```python
val = pypto_pro.language.get_block_num()
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

返回设备运行时产生的整型标量值，其值为限核后实际启动的逻辑Block数，可用于Kernel内整数运算和索引。
JIT每次启动通过C++启动器查询实际Stream的有效资源限制，按Kernel执行域和配对比例限制Host请求的`block_dim`。
因此返回值可能小于Host请求值；数据切分应使用本接口返回值作为循环步长，避免遗漏任务。

仅启动Cube（AIC）或仅启动Vector（AIV）时，该值等于执行域逻辑核数。
在AIC:AIV为1:2的混合Kernel中，该值表示逻辑Block数；AIC逻辑核数为`get_block_num()`，
AIV逻辑核数为`get_block_num() * get_subblock_num()`。

## 调用示例

下面是一个仅包含Vector段的多核Kernel：用`kernel[None, NUM_CORES](...)`请求最多2个逻辑Block，
每个AIV按实际Block数跨步处理64行Tile，即使限为1个Block也能覆盖全部128行。

```python
import pypto_pro.language as pl

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def multicore_add_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        vidx = pl.get_block_idx()              # 当前AIV的全局逻辑索引
        num_blocks = pl.get_block_num()
        for tile_idx in pl.range(vidx, 2, num_blocks):
            offset = tile_idx * 64
            cur_a = tile_a.current()
            cur_b = tile_b.current()
            cur_c = tile_c.current()
            pl.load(cur_a, x, [offset, 0])
            pl.load(cur_b, y, [offset, 0])
            pl.add(cur_c, cur_a, cur_b)
            pl.store(z, cur_c, [offset, 0])
```

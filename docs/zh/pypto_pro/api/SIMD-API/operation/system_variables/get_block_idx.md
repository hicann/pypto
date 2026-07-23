# pypto_pro.language.get_block_idx

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

获取当前 AI Core 的 block 索引。返回值可直接参与偏移计算。

## 函数原型

```python
val = pypto_pro.language.get_block_idx()
```

无参数。

## 返回值说明

返回当前 AI Core 的 block 索引，类型为整型 Expr，可直接参与坐标计算。

## 调用示例

下面是一个完整多核 kernel：用 `kernel[None, NUM_CORES](...)` 启动 2 核，每核用 `pypto_pro.language.get_block_idx()` 获取当前核号算行偏移，各处理 64 行做 element-wise 加法。

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
        vidx = pl.get_block_idx()              # 当前核号
        _bnum = pl.get_block_num()             # 核总数（此处读出验证可调用）
        offset = vidx * 64                     # 第 vidx 核处理第 [vidx*64, +64) 行
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [offset, 0])
        pl.load(cur_b, y, [offset, 0])
        pl.add(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [offset, 0])
```

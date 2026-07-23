# pypto_pro.language.mrgsort2

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

多路归并排序：将 2 到 4 个已排序的源 tile 归并为一个有序输出。每个源 tile 内部已按降序排列（val-idx 对格式），`mrgsort2` 从中选取最大值依次写入 dst。

`exhausted` 参数标记某个源是否已耗尽，用于多步归并中处理长度不一致的源。

## 函数原型

```python
pypto_pro.language.mrgsort2(src0, src1, dst, tmp, *args, exhausted=False)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src0` | 输入 | 第一个已排序源 tile |
| `src1` | 输入 | 第二个已排序源 tile |
| `dst` | 输出 | 目标 tile，存放归并结果 |
| `tmp` | 输入 | 临时 tile（硬件中间计算用） |
| `*extra_srcs` | 输入 | 可选，额外的已排序源 tile（支持 3 路或 4 路归并） |
| `exhausted` | 输入 | 是否有源已耗尽 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src0`, `src1` | 输入 | 数据类型：b32（FP32 val-idx 对）<br>shape：行数为 1<br>内部已按降序排列 |
| `dst` | 输出 | 数据类型：与源一致<br>shape：与源一致 |
| `tmp` | 输入 | 数据类型：与源一致<br>shape：与源一致 |
| `*extra_srcs` | 输入 | 可选的第 3、4 个源 tile，格式同上 |
| `exhausted` | 输入 | `True` 或 `False`（默认） |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整 kernel：用 `pypto_pro.language.mrgsort2` 把两个已排序的源 tile 归并为一个有序输出。纯 vector kernel 使用 `make_tile_group` 管理 Tile 资源，并通过 `auto_mutex` 完成流水同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort2_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    sorted_out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src0_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_src1_group = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    tile_tmp_group = pl.make_tile_group(type=tt, addrs=0x0C00, mutex_ids=[3])
    with pl.section_vector():
        tile_src0 = tile_src0_group.current()
        tile_src1 = tile_src1_group.current()
        tile_dst = tile_dst_group.current()
        tile_tmp = tile_tmp_group.current()
        pl.load(tile_src0, src0_tensor, [0, 0])
        pl.load(tile_src1, src1_tensor, [0, 0])
        pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, exhausted=False)
        pl.store(sorted_out, tile_dst, [0, 0])
```

其他典型用法（节选）：

```python
# 3 路归并
pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, tile_src2, exhausted=False)

# 4 路归并
pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, tile_src2, tile_src3, exhausted=False)
```

# pypto_pro.language.load_tile

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

把GM中一块数据搬入L1/UB Tile，与[`pypto_pro.language.load`](load.md)不同，`load_tile`的偏移以**tile块索引**为单位，内部自动按`块索引 × tile_shape`换算成绝对元素坐标。在按tile规整切分的循环中，可直接用块号定位，省去手动乘tile大小的计算。

例如tile shape为`[64, 128]`时，`tile_offsets=[2, 2]`等价于[`pypto_pro.language.load`](load.md)的绝对偏移`[128, 256]`。

![load_tile按块索引从GM搬入Tile](../../../figures/load_tile_block_offset.jpg "load_tile按块索引从GM搬入Tile")

## 函数原型

```python
pypto_pro.language.load_tile(dst_tile, src_tensor, tile_offsets, *, order=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是L1、UB Tile，搬入目的地 |
| `src_tensor` | 输入 | Tensor类型，来自GM的源数据 |
| `tile_offsets` | 输入 | 以tile为单位的块索引，内部换算为`块索引 × tile_shape`的绝对元素偏移 |
| `order` | 输入 | 可选，Tile维度在GlobalTensor维度中对应哪几根轴；元素为Tensor绝对轴索引，升序表示不转置，反序表示转置；省略时默认`[ndim-2, ndim-1]`（不转置） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>尾块处理：<br>• 可通过set_validshape设置尾块大小，Tile shape需要32字节对齐，不对齐报错<br>• valid_shape可以不对齐<br>• set_validshape需要compact = 1，compact不等于1且validshape不等于shape时需要报错<br>• 支持设定padding值<br>地址配置：<br>• Tile的类型只能是L1、UB，Cube侧非L1报错<br>• Vector侧非UB报错<br>• L1、UB buffer首地址必须32字节对齐，不对齐编译报错 |
| `src_tensor` | 输入 | 数据类型：b8、b16、b32、b64<br>layout：支持`ND`、`DN`、`NZ`<br>stride：支持配置Stride，stride维度需要等于tensor维度数，默认不配置时是尾轴stride = 1的连续场景 |
| `tile_offsets` | 输入 | 单位为tile块索引，换算后的绝对偏移不超过对应维度的shape，不支持负数索引<br>被切分的维度（由`order`指定）按`块索引 × tile该维大小`换算；其余维度的取值按绝对偏移直接使用 |
| `order` | 输入 | 只支持配置tensor维度范围内的dim，只支持二维数组配置，其余配置报错<br>用于高维tensor中指定tile对应哪几个维度；order中轴索引的顺序决定是否转置：升序不转置（ND行主序），反序转置（DN列主序），需要配合Tensor的layout以及Tile的shape和stride填写<br>省略时默认取tensor的最后两维`[ndim-2, ndim-1]`（不转置） |

## 流水类型

MTE2（GM → L1/UB的搬入流水）。

## 调用示例

下面是一个完整kernel：GM输入按64×64的tile规整切分，用块索引逐块载入、翻倍后写回对应位置。`pypto_pro.language.load_tile`用块号`[ti, 0]`定位，内部自动换算为绝对偏移`[ti*64, 0]`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_kernel(
    x: pl.Tensor[[256, 64], pl.DT_FP16],   # 4 个 64x64 的块
    out: pl.Tensor[[256, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    x_db = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0, 1])
    out_db = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2, 3])

    with pl.section_vector():
        for ti in pl.range(0, 4, 1):
            cur_x = x_db.next()
            cur_out = out_db.next()
            pl.load_tile(cur_x, x, [ti, 0])
            pl.add(cur_out, cur_x, cur_x)   # 翻倍，验证 load_tile 取到了正确的块
            pl.store_tile(out, cur_out, [ti, 0])
```

其他典型用法（节选）：

```python
# 4D BSND tensor：tile 对应第 1、3 维，其余维按绝对偏移
pl.load_tile(q_buf, q, [b_idx, qi, n_idx, 0], order=[1, 3])

# 列主序载入（DN 布局）
pl.load_tile(k_mat_buf, k, [b_idx, n_idx, j, 0], order=[1, 0])
```

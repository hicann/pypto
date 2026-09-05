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

把GM中的数据搬入L1 Buffer或UB中的Tile。与[pypto_pro.language.load](load.md)不同，load_tile的偏移以Tile块索引为单位，内部按块索引乘以Tile形状换算成绝对元素坐标。

例如Tile形状为[64, 128]时，tile_offsets=[2, 2]等价于[pypto_pro.language.load](load.md)的绝对偏移[128, 256]。

![load_tile按块索引从GM搬入Tile](../../../figures/load_tile_block_offset.jpg "load_tile按块索引从GM搬入Tile")

## 函数原型

```python
pypto_pro.language.load_tile(
    dst_tile: Tile,
    src_tensor: Tensor,
    tile_offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L1 Buffer或UB，首地址必须按32字节对齐。支持DT_FP4E2M1、DT_FP4E1M2、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。可通过set_validshape设置尾块有效形状。 |
| src_tensor | 输入 | 源操作数，Tensor类型，存储空间为GM，支持的数据类型与dst_tile一致，排布支持ND、DN和NZ。 |
| tile_offsets | 输入 | 源Tensor的Tile块偏移，List[int或Scalar]类型。由order选中的维度以Tile块索引为单位，内部乘以Tile对应维度大小；未被选中的维度按绝对元素偏移使用。各项必须为非负整数或运行时整数表达式，换算后的访问范围不得越过Tensor边界。 |
| order | 输入 | 维度映射，List[int]类型，可选。表示Tile两个维度分别对应源Tensor的哪两个维度；两个轴索引必须互不重复且位于Tensor维度范围内，升序表示不转置，降序表示转置。 |

## 约束说明

当src_tensor声明为pypto_pro.language.NZ时，其物理排布和完整Tensor shape约束见[TensorLayout](../../basic_data_structures/TensorLayout.md#tensor布局)，同布局搬运、目标Tile和order约束与[load](load.md#约束说明)一致。load_tile还需满足以下NZ搬运约束：

- tile_offsets按Tile块索引寻址：最后两项分别乘以Tile的M、N shape，前导项选择batch；换算后的M、N offset需分别按16和Tensor dtype对应的C0对齐。

当前DT_FP8E8M0 Tensor搬入fractal=32的ZZ或NN排布L1 Buffer Tile，仅支持作为matmul_mx或matmul_mx_acc的缩放因子搬运。普通E8M0数据不支持使用该目标组合；满足该组合的load_tile会按MX缩放因子解释，并要求源Tensor的最后一轴是长度为2的物理phase轴。

开启auto_mutex时，若连续两次pypto_pro.language.load_tile向同一个UB或L1 Buffer Tile地址搬运数据，并且前一次搬入的数据没有被读取，则必须在两次load_tile之间调用pypto_pro.language.system.bar_mte2()，再复用该地址。

load_tile复用Tile地址的同步规则与load接口一致；详细说明请参考[load](load.md)文档中的“Tile地址复用与流水同步”。

## 返回值说明

无。

## 调用示例

### 按Tile块索引从GM搬入UB

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

### 高维Tensor与转置搬运

```python
# 4D BSND Tensor：Tile 对应第 1、3 维，其余维按绝对偏移
pl.load_tile(q_buf, q, [b_idx, qi, n_idx, 0], order=[1, 3])

# 列主序载入（DN 布局）
pl.load_tile(k_mat_buf, k, [b_idx, n_idx, j, 0], order=[1, 0])
```

# pypto_pro.language.fused_mul_add_relu

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

先将 out 中的原始数据与 a 逐元素相乘，再将乘法结果与 b 逐元素相加，最后执行 ReLU 激活，将小于 0 的元素置为 0。计算结果写回 out，并覆盖 out 中的原始数据。

## 函数原型

```python
pypto_pro.language.fused_mul_add_relu(
  out: Tile,
  a: Tile,
  b: Tile
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输入/输出 | UB Tile。输入时提供参与逐元素乘法的数据，输出时保存最终计算结果。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |
| a | 输入/输出 | 参与逐元素乘法的 UB Tile，在计算过程中用于保存中间结果。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |
| b | 输入 | 参与逐元素加法的 UB Tile。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |

## 约束说明

- out、a 和 b 的数据类型和有效 Shape 须一致。
- out、a 和 b 支持 ND、ZN 和 ZZ 布局。
- out 中的原始数据参与计算，调用后会被最终结果覆盖。
- a 在计算过程中会被修改，调用后其中的原始数据不再保留。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def fused_mul_add_relu_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.fused_mul_add_relu(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:fused_mul_add_relu:start -->
```bash
输入数据x：[[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [62 62.25 62.5 62.75 63 63.25 63.5 63.75 ...], [94 94.25 94.5 94.75 95 95.25 95.5 95.75 ...], ...]
输入数据y：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [33 33.125 33.25 33.375 33.5 33.625 33.75 33.875 ...], [49 49.125 49.25 49.375 49.5 49.625 49.75 49.875 ...], ...]
输入数据z原始值：[[3 2.9375 2.875 2.8125 2.75 2.6875 2.625 2.5625 ...], [-5 -5.0625 -5.125 -5.1875 -5.25 -5.3125 -5.375 -5.4375 ...], [-13 -13.0625 -13.125 -13.1875 -13.25 -13.3125 -13.375 -13.4375 ...], [-21 -21.0625 -21.125 -21.1875 -21.25 -21.3125 -21.375 -21.4375 ...], ...]
输出数据z：[[0 0 0 0 0 0 0.4375 1.234375 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], ...]
```
<!-- pypto-doc-output:fused_mul_add_relu:end -->

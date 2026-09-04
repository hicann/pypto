# pypto_pro.language.sub_relu

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

对两个源 Tile 执行逐元素减法，再对结果执行 ReLU 激活，将小于 0 的元素置为 0。计算公式为 out = max(lhs - rhs, 0)。

## 函数原型

```python
pypto_pro.language.sub_relu(
  out: Tile,
  lhs: Tile,
  rhs: Tile
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标 UB Tile，用于保存减法和 ReLU 激活的结果。支持的数据类型为 DT_FP16 和 DT_FP32。 |
| lhs | 输入 | 第一个源 UB Tile，作为被减数。支持的数据类型为 DT_FP16 和 DT_FP32。 |
| rhs | 输入 | 第二个源 UB Tile，作为减数。支持的数据类型为 DT_FP16 和 DT_FP32。 |

## 约束说明

- out、lhs 和 rhs 的 Shape 和数据类型须一致。
- out 可以与 lhs 或 rhs 指向同一个 Tile，以实现原地计算。
- 该接口依次执行减法和 ReLU 激活，并将 lhs 用作中间结果，因此 lhs 中的原始数据会被覆盖。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sub_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:sub_relu:start -->
```bash
输入数据a：[[2 2.5 3 3.5 4 4.5 5 5.5 ...], [34 34.5 35 35.5 36 36.5 37 37.5 ...], [66 66.5 67 67.5 68 68.5 69 69.5 ...], [98 98.5 99 99.5 100 100.5 101 101.5 ...], ...]
输入数据b：[[3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [19 19.25 19.5 19.75 20 20.25 20.5 20.75 ...], [35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [51 51.25 51.5 51.75 52 52.25 52.5 52.75 ...], ...]
输出数据out：[[0 0 0 0 0 0.25 0.5 0.75 ...], [15 15.25 15.5 15.75 16 16.25 16.5 16.75 ...], [31 31.25 31.5 31.75 32 32.25 32.5 32.75 ...], [47 47.25 47.5 47.75 48 48.25 48.5 48.75 ...], ...]
```
<!-- pypto-doc-output:sub_relu:end -->

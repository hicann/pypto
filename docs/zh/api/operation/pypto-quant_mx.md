# pypto.quant_mx

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持

## 功能说明

将1-4维ND格式的高精度浮点Tensor量化为MX（Microscaling）格式，返回量化结果和共享指数scale。

- 输入Tensor支持DT_FP16、DT_BF16、DT_FP32。
- 输出量化Tensor支持DT_FP8E4M3、DT_FP4_E2M1X2。其中DT_FP4_E2M1X2仅支持DT_FP16、DT_BF16输入。
- scale Tensor的数据类型固定为DT_FP8E8M0。
- 当前仅支持对尾轴进行量化，支持ROUND_DOWN（OCP）和ROUND_UP（NV）模式。

若输入shape记为 $[d_0, d_1, ..., d_{n-1}]$，则：

- 量化结果 `quantized` 的shape与 `input` 相同。
- scale的shape为 $[d_0, d_1, ..., d_{n-2}, \lceil d_{n-1} / 64 \rceil, 2]$。

## 函数原型

```python
quant_mx(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_FP8E4M3,
    mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
    axis: int = -1,
    performance_mode: bool = True,
) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input | 输入 | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32。<br>仅支持TILEOP_ND格式；Shape仅支持1-4维。<br>当前仅支持最后一维参与量化，且最后一维按字节数需满足256字节对齐。对于DT_FP32，通常要求最后一维长度是64的倍数；对于DT_FP16/DT_BF16，通常要求最后一维长度是128的倍数。 |
| quant_dtype | 输入 | 量化后输出Tensor的数据类型。<br>支持：DT_FP8E4M3、DT_FP4_E2M1X2。DT_FP4_E2M1X2仅支持DT_FP16、DT_BF16输入。 |
| mode | 输入 | 量化时共享指数的舍入模式。<br>支持：ROUND_DOWN（OCP）、ROUND_UP（NV）。 |
| axis | 输入 | 指定量化轴。<br>当前仅支持最后一维，即 `-1` 或 `input.shape.size() - 1`。 |
| performance_mode | 输入 | 是否启用性能模式。<br>默认值为 `True`。启用后可获得更好的性能，但仅改变内部TQuant的中间布局，不改变返回的公共 `scale` shape。<br>启用该模式时，不支持尾块场景，即运行时实际shape需要整除view shape。除此之外，view shape与TileShape的尾轴长度必须相同，且该尾轴长度需要与输入最后一维保持一致。<br> 当前只支持性能模式|

## 返回值说明

返回一个二元组 `(quantized, scale)`：

- `quantized`：量化后的Tensor，数据类型由 `quant_dtype` 指定，Shape与 `input` 相同。
- `scale`：共享指数Tensor，数据类型固定为DT_FP8E8M0，Shape为 `[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过 `set_vec_tile_shapes` 设置TileShape。

TileShape维度应和输入一致，且最后一维需要满足256字节对齐。若 `performance_mode=True`，则不支持尾块场景，运行时实际shape需要整除view shape；同时view shape与TileShape的尾轴形状必须相同，且TileShape的最后一维应与输入最后一维相同。

示例1：输入 `input` shape为 `[m, n]`，输出 `quantized` shape为 `[m, n]`，`scale` shape为 `[m, ceil(n / 64), 2]`，TileShape可设置为 `[m1, n1]`，其中 `n1` 需满足对齐约束。

```python
pypto.set_vec_tile_shapes(4, 64)
```

### 接口调用示例

```python
x = pypto.tensor([8, 64], pypto.DT_FP32)

# 默认配置：DT_FP8E4M3 + ROUND_DOWN + 最后一维量化
quantized, scale = pypto.quant_mx(x)

# 显式指定OCP参数
quantized_perf, scale_perf = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -1,
    True,
)

# 使用NV scale算法
quantized_nv, scale_nv = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    True,
)
```

结果示例如下：

```python
Input x.shape: [8, 64]
Input x.dtype: DT_FP32
Output quantized.shape: [8, 64]
Output quantized.dtype: DT_FP8E4M3
Output scale.shape: [8, 1, 2]
Output scale.dtype: DT_FP8E8M0
```

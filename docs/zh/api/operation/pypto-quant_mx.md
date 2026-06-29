# pypto.quant_mx

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

将1-4维ND格式的高精度浮点Tensor量化为MX（Microscaling）格式，返回量化结果和共享指数scale。

- 输入Tensor支持DT_FP16、DT_BF16、DT_FP32。
- 输出量化Tensor支持DT_FP8E4M3、DT_FP4_E2M1X2。其中DT_FP4_E2M1X2仅支持DT_FP16、DT_BF16输入。
- scale Tensor的数据类型固定为DT_FP8E8M0。
- 当前仅支持对尾轴进行量化，支持ROUND_DOWN（OCP）和ROUND_UP（NV）模式。
- 支持性能模式和非性能模式。性能模式要求view shape尾轴能整切实际shape尾轴，且TileShape尾轴与view shape尾轴相同；非性能模式支持更灵活的 viewshape 和 Tileshape 设置，进行更好的算子融合，单算子性能有些下降。

若输入shape记为 $[d_0, d_1, ..., d_{n-1}]$，则：

- 量化结果`quantized`的shape与`input`相同。
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
| axis | 输入 | 指定量化轴。<br>当前仅支持最后一维，即`-1`或`input.shape.size() - 1`。 |
| performance_mode | 输入 | 是否启用性能模式。<br>默认值为`True`。<br>启用该模式时，实际shape尾轴长度必须能被view shape尾轴长度整切，即无尾块；若已设置TileShape，则TileShape维度数必须与输入一致，且TileShape最后一维必须等于view shape最后一维；同时view shape最后一维需要满足256字节对齐。<br>关闭该模式时，不要求TileShape最后一维等于view shape最后一维，也不要求TileShape最后一维满足256字节对齐，但输入最后一维仍必须是64的倍数。 |

## 返回值说明

返回一个二元组`(quantized, scale)`：

- `quantized`：量化后的Tensor，数据类型由`quant_dtype`指定，Shape与`input`相同。
- `scale`：共享指数Tensor，数据类型固定为DT_FP8E8M0，Shape为`[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape维度应和输入一致。若`performance_mode=True`，则view shape尾轴需要整切实际shape尾轴，TileShape最后一维应与view shape最后一维相同；同时view shape最后一维需要满足256字节对齐。若`performance_mode=False`，则TileShape最后一维只要求为正数，输入最后一维仍必须是64的倍数。

示例1：性能模式下，输入view shape为`[m, n]`，输出`quantized` shape为`[m, n]`，`scale` shape为`[m, ceil(n / 64), 2]`；实际shape尾轴需能被`n`整切，TileShape可设置为`[m1, n]`，其中`n`需满足256字节对齐。

```python
pypto.set_vec_tile_shapes(4, 64)
```

示例2：非性能模式下，输入`input` shape为`[m, n]`，`n`需为64的倍数；TileShape可设置为`[m1, n1]`，其中`n1`为正数。

```python
pypto.set_vec_tile_shapes(2, 128)
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

# 关闭性能模式
x_non_perf = pypto.tensor([2, 512], pypto.DT_FP32)
quantized_general, scale_general = pypto.quant_mx(
    x_non_perf,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    False,
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

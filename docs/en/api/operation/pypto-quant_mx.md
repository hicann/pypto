# pypto.quant_mx

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-09-02T08:39:58.299Z pushedAt=2026-09-05T07:36:26.361Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not Supported

## Description

Quantizes a 1- to 4-dimensional ND-format high-precision floating-point tensor into the MX (Microscaling) format, and returns the quantization result and the shared exponent scale.

- The input tensor supports DT_FP16, DT_BF16, and DT_FP32.
- The output quantized tensor supports DT_FP8E4M3 and DT_FP4_E2M1X2. DT_FP4_E2M1X2 only supports DT_FP16 and DT_BF16 inputs.
- The data type of the scale tensor is fixed to DT_FP8E8M0.
- Currently, only quantization along the last axis is supported, with ROUND_DOWN (OCP) and ROUND_UP (NV) modes.
- Both performance mode and non-performance mode are supported. In performance mode, the last axis of the view shape must evenly divide the last axis of the actual shape, and the last axis of TileShape must be the same as the last axis of the view shape. Non-performance mode supports more flexible view shape and TileShape settings for better operator fusion, at the cost of some single-operator performance.

If the input shape is denoted as $[d_0, d_1, ..., d_{n-1}]$, then:

- The shape of the quantization result `quantized` is the same as that of `input`.
- The shape of scale is $[d_0, d_1, ..., d_{n-2}, \lceil d_{n-1} / 64 \rceil, 2]$.

## Prototype

```python
quant_mx(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_FP8E4M3,
    mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
    axis: int = -1,
    performance_mode: bool = True,
) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input | Input | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Only the TILEOP_ND format is supported; the shape supports only 1 to 4 dimensions.<br>Currently, only the last dimension is quantized, and the last dimension must satisfy 256-byte alignment. For DT_FP32, the last dimension length is usually required to be a multiple of 64; for DT_FP16/DT_BF16, the last dimension length is usually required to be a multiple of 128. |
| quant_dtype | Input | Data type of the quantized output tensor.<br>Supported: DT_FP8E4M3 and DT_FP4_E2M1X2. DT_FP4_E2M1X2 only supports DT_FP16 and DT_BF16 inputs. |
| mode | Input | Rounding mode of the shared exponent during quantization.<br>Supported: ROUND_DOWN (OCP) and ROUND_UP (NV). |
| axis | Input | Quantization axis.<br>Currently, only the last dimension is supported, that is, `-1` or `input.shape.size() - 1`. |
| performance_mode | Input | Whether to enable the performance mode.<br>The default value is `True`.<br>When this mode is enabled, the actual shape last-axis length must be exactly divisible by the view shape last-axis length, that is, no last block exists; if the TileShape is set, the TileShape must have the same number of dimensions as the input, and the last dimension of the TileShape must equal the last dimension of the view shape; in addition, the last dimension of the view shape must satisfy 256-byte alignment.<br>When this mode is disabled, the last dimension of the TileShape is not required to equal the last dimension of the view shape, nor is it required to satisfy 256-byte alignment, but the last dimension of the input must still be a multiple of 64. |

## Return Value

Returns a two-tuple `(quantized, scale)`:

- `quantized`: Quantized tensor, whose data type is specified by `quant_dtype` and whose shape is the same as `input`.t
- `scale`: Shared exponent tensor, whose data type is fixed to DT_FP8E8M0 and whose shape is `[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The dimensionality of `TileShape` must be consistent with that of the input. If `performance_mode=True`, the last axis of the view shape must evenly divide the last axis of the actual shape, and the last dimension of the TileShape should be the same as the last dimension of the view shape; in addition, the last dimension of the view shape must satisfy 256-byte alignment. If `performance_mode=False`, the last dimension of the TileShape is only required to be a positive number, and the last dimension of the input must still be a multiple of 64.

Example 1: In performance mode, the input view shape is `[m, n]`, the output `quantized` shape is `[m, n]`, and the `scale` shape is `[m, ceil(n / 64), 2]`; the actual shape last axis must be exactly divisible by `n`, and the TileShape can be set to `[m1, n]`, where `n` must satisfy 256-byte alignment.

```python
pypto.set_vec_tile_shapes(4, 64)
```

Example 2: In non-performance mode, the `input` shape is `[m, n]`, where `n` must be a multiple of 64; the TileShape can be set to `[m1, n1]`, where `n1` is a positive number.

```python
pypto.set_vec_tile_shapes(2, 128)
```

### API Call Example

```python
x = pypto.tensor([8, 64], pypto.DT_FP32)

# Default configuration: DT_FP8E4M3 + ROUND_DOWN + last-dimension quantization.
quantized, scale = pypto.quant_mx(x)

# Explicitly specify OCP parameters.
quantized_perf, scale_perf = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -1,
    True,
)

# Use the NV scale algorithm.
quantized_nv, scale_nv = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    True,
)

# Disable performance mode.
x_non_perf = pypto.tensor([2, 512], pypto.DT_FP32)
quantized_general, scale_general = pypto.quant_mx(
    x_non_perf,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    False,
)
```

The results are as follows:

```python
Input x.shape: [8, 64]
Input x.dtype: DT_FP32
Output quantized.shape: [8, 64]
Output quantized.dtype: DT_FP8E4M3
Output scale.shape: [8, 1, 2]
Output scale.dtype: DT_FP8E8M0
```

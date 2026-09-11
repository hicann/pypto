# pypto.matmul

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:28:39.114Z pushedAt=2026-09-05T07:36:26.351Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Implements matrix multiplication of the **input** and **mat2** matrices, with the formula: out = input @ mat2

- **input** and **mat2** are source operands, where **input** is the left matrix and **mat2** is the right matrix.
- **out** is the destination operand, which stores the matrix multiplication result.

## Precautions

- **Consistent data types between the left and right matrices**: The left and right matrix data types of matmul must be the same (for example, BF16+BF16 and FP16+FP16). Mixed input (for example, BF16+FP32) is not supported, except for the FP8 data type.
- **Low-precision input is recommended**: Using BF16/FP16 input to directly output FP32 through matmul delivers better performance than casting to FP32 first and then performing matmul, with comparable precision.
- **Avoiding unnecessary casts**: Upgrading BF16 to FP32 before matmul does not improve precision; instead, it incurs additional data movement overhead.
- **By leveraging on-the-fly transpose**: `matmul` supports the `a_trans` and `b_trans` parameters, which enable transpose operations to be performed on-the-fly during matrix multiplication, eliminating the need for separate `transpose` calls.
- **TileShape must be set first**: Before calling the matmul API, you need to call `set_cube_tile_shapes` to set the tile sizes on the M, K, and N axes.

## Prototype

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters

Table 1: API parameters

| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| **input**             | Input      | Input left matrix. Empty tensors are not supported as input.<br> **Data type**: For details, see Table 3.<br> **Matrix dimension**: 2D, 3D, or 4D, and the left and right matrices must have the same dimension count.<br> **Format**: TILEOP_ND or TILEOP_NZ (DT_FP32, DT_FP8E5M2, and DT_HF8 inputs do not support the TILEOP_NZ format).<br> **Inner/Outer axis**: When the input matrix is not transposed, the corresponding data is arranged as [M, K], where the outer axis is M and the inner axis is K. When the input matrix is transposed, the corresponding data is arranged as [K, M], where the outer axis is K and the inner axis is M. <br> **Alignment requirements**: When **Format** is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535].<br> When **Format** is TILEOP_NZ (NZ format), the shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> When using the **pypto.view** API, ensure that the shape dimension passed also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis.|
| **mat2**              | Input      | Input right matrix. Empty tensors are not supported as input.<br> **Data type**: For details, see Table 3.<br> **Matrix dimension**: 2D, 3D, or 4D, and the left and right matrices must have the same dimensions.<br> **Format**: TILEOP_ND or TILEOP_NZ (DT_FP32, DT_FP8E5M2, and DT_HF8 inputs do not support the TILEOP_NZ format).<br> **Inner/Outer axis**: When the input matrix **mat2** is not transposed, the corresponding data is arranged as [K, N], where the outer axis is K and the inner axis is N. When the input matrix **mat2** is transposed, the corresponding data is arranged as [N, K], where the outer axis is N and the inner axis is K.<br> **Alignment requirements**: When **Format** is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535].<br> When **Format** is TILEOP_NZ (NZ format), the shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> When using the **pypto.view** API, ensure that the Shape dimension passed also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| **out_dtype**         | Output      | Output matrix data type. For the supported data types of the output in the basic scenario, see Table 3. For the supported data types of the output in the dequantization and quantization scenarios, see Table 4 and Table 5.|
| **a_trans**           | Input      | Whether the input left matrix is transposed, defaulting to **False**. |
| **b_trans**           | Input      | Whether the input right matrix is transposed, defaulting to **False**. |
| **c_matrix_nz**       | Input      | Whether the output matrix uses the NZ format, defaulting to **False**. Currently only **False** is supported, meaning the output matrix supports only the ND format. |
| **extend_params**     | Input      | Includes bias, fixpipe quantization and dequantization, and TF32 rounding mode. For details, see Table 2.<br>For the supported input and output data types of bias, fixpipe quantization, and dequantization, see Table 3, Table 4, and Table 5.<br>- The data type is in dictionary format.<br>- This parameter and its internal parameters are all optional.|

Table 2: extend_params parameters

| Parameter            | Description                                                                 |
|-------------------|----------------------------------------------------------------------|
| **scale**             | Dequantization parameter for the output matrix in per-tensor quantization scenarios (where a single scaling factor is used to map high-precision values to low-precision values).<br>The input is of the float type, and 1 sign bit + 8 exponent bits + 10 mantissa bits are used in the computation.<br>For the supported input and output data types, see Table 4 and Table 5.<br>Multi-core K-splitting is not supported.|
| **scale_tensor**      | Dequantization matrix for the output matrix in per-channel quantization scenarios (where a separate set of quantization parameters is computed independently for each output channel).<br>The **scale_tensor** input is fixed as a tensor of uint64_t or int64_t. During computation, the 64-bit value is converted to the lower 32 bits of the float type, and then 1 sign bit + 8 exponent bits + 10 mantissa bits are used in the computation.<br>For the supported input and output data types, see Table 4 and Table 5.<br>The shape of the second-to-last dimension of **scale_tensor** must be set to **1**, and the N dimension must be equal to the N dimension of the **mat2** matrix.<br>**scale_tensor** supports only the ND format.<br>Multi-core K-splitting is not supported.<br>When the quantization output type is DT_INT8, call **torch_npu.npu_trans_quant_param** in advance and pass a float32 **torch.tensor** to obtain the int64 **scale_tensor**.|
| **bias_tensor**       | Bias matrix.<br>The input is of the Tensor type.<br>For the supported input and output data types, see Table 3.<br>**bias_tensor** supports only the ND format.<br>The shape of the second-to-last dimension of **bias_tensor** should be set to **1**, and the N dimension must be equal to the N dimension of the **mat2** matrix.<br>In the 4D matrix dimension scenario, **bias** allows only 2D input.<br>Multi-core K-splitting is not supported. |
| **relu_type**         | Whether the output matrix undergoes the ReLU operation.<br>The input is of the [ReLuType](../datatype/ReLuType.md) type.<br>The RELU and NO_RELU modes are supported.<br>Multi-core K-splitting is not supported.|
| **trans_mode**        | Whether to enable TF32 computation and the TF32 rounding mode.<br>The input is of the [TransMode](../datatype/TransMode.md) type, supporting the following three modes:<br>• **CAST_NONE**: Does not enable conversion of the float data type to the TF32 data type.<br>• **CAST_RINT**: Enables conversion of the float data type to the TF32 data type. Rounding rule: round to the nearest integer, with ties rounding to even.<br>• **CAST_ROUND**: Enables conversion of the float data type to the TF32 data type. Rounding rule: round to the nearest integer, with ties rounding away from zero.<br>This can be set only when the data types of the input left and right matrices and the output matrix are all DT_FP32. |

Table 3: Data types supported in the Matmul basic scenario

| **input** | **mat2** | **out_dtype** | **bias_tensor** | Product Support |
|:------|:-----|:----------|:------------|:---------|
| DT_FP16 | DT_FP16 | DT_FP16/DT_FP32 | DT_FP16/DT_FP32 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_BF16 | DT_BF16 | DT_BF16/DT_FP32 | DT_FP32 | Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_BF16 | DT_BF16 | DT_BF16/DT_FP32 | DT_BF16 | Ascend 950PR/Ascend 950DT |
| DT_FP32 | DT_FP32 | DT_FP32 | DT_FP32 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |
| DT_HF8 | DT_HF8 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |

Table 4: Data types supported in the Matmul dequantization scenario

| **input** | **mat2** | **out_dtype** | Product Support |
|:------|:-----|:----------|:----------|
| DT_INT8 | DT_INT8 | DT_FP16 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |

Table 5: Data types supported in the Matmul quantization scenario

| **input** | **mat2** | **out_dtype** | Product Support |
|:------|:-----|:----------|:----------|
| DT_BF16 | DT_BF16 | DT_INT8 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_FP16 | DT_FP16 | DT_INT8 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_FP32 | DT_FP32 | DT_INT8 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_INT8 | DT_INT8 | DT_INT8 | Ascend 950PR/Ascend 950DT <br> Atlas A2 training products/Atlas A2 inference products <br> Atlas A3 training products/Atlas A3 inference products |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 | Ascend 950PR/Ascend 950DT |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 | Ascend 950PR/Ascend 950DT |
| DT_HF8 | DT_HF8 | DT_INT8 | Ascend 950PR/Ascend 950DT |

## Return Value

Returns the **out** matrix (tensor).

## Constraints

- Atlas A2 training products/Atlas A2 inference products: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported, and the **trans_mode** parameter in **extend_params** is not supported.
- Atlas A3 training products/Atlas A3 inference products: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported, and the **trans_mode** parameter in **extend_params** is not supported.
- Before the **matmul** API is called, the tile sizes along the M, K, and N axes must be set via `pypto.set_cube_tile_shapes`.







- When the matrix dimensions are 3D or 4D, the `pypto.set_vec_tile_shapes` API must be called to set the vector tile shape. If not set, the API will internally use a default 2D vector tile shape of `[128, 128]`.
- When the input to the `matmul` API is in NZ format after a `pypto.reshape` call, the `pypto.set_matrix_size` API must be called to set the original m, k, and n values of the input before `pypto.reshape` for `matmul`.
- When the input matrices to the `matmul` API are 3D/4D and in NZ format, the `pypto.set_matrix_size` API must be called to set the original m, k, and n values of the input for `matmul`.

## Examples

```python
# Basic matrix multiplication.
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

# Batch matrix multiplication.
a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

# Batch broadcasting.
a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

# Add bias.
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)

# Dequantize.
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

# Dequantize and apply ReLU.
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

# Dequantize and apply ReLU.
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

# Quantize.
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)

# Quantize and apply ReLU.
scale_cpu = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # Generate scale_tensor.
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)

# TF32 computation mode (950PR/DT only).
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'trans_mode': pypto.TransMode.CAST_ROUND}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
```

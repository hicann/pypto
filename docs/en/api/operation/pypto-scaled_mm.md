# pypto.scaled\_mm

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:48:10.293Z pushedAt=2026-09-05T07:36:26.367Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Implements the mx quantized matrix multiplication of the **mat_a** and **mat_b** matrices. The calculation formula is: out = (mat_a \* scale_a) @ (mat_b \* scale_b)

- **mat_a**, **mat_b**, **scale_a**, and **scale_b** are source operands. **mat_a** is the left matrix, **mat_b** is the right matrix, **scale_a** is the quantization parameter of the left matrix, and **scale_b** is the quantization parameter of the right matrix.
- **out** is the destination operand, which is the matrix that stores the matrix multiplication result.

## Prototype

```python
scaled_mm(mat_a, mat_b, out_dtype, scale_a, scale_b, *, a_trans = False, b_trans = False, scale_a_trans = False, scale_b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters

Table 1: API Parameters

| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| mat_a             | Input      | Left input matrix. Empty tensors are not supported. <br> **Data type**: See Table 3. <br> **Matrix dimension**: 2D, 3D, or 4D. <br> **Format**: TILEOP_ND, TILEOP_NZ (TILEOP_NZ is not supported for DT_FP8E5M2 input).<br> **Inner axis and outer axis**: When the input matrix **mat_a** is not transposed, the corresponding data layout is [M, K], where the outer axis is M and the inner axis is K. When the input matrix **mat_a** is transposed, the corresponding data layout is [K, M], where the outer axis is K and the inner axis is M. <br> **Alignment requirement**: When Format is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535]. <br> When Format is TILEOP_NZ (NZ format), the shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> When using the **pypto.view** API, ensure that the shape dimension of the passed View also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| mat_b              | Input      | Right input matrix. Empty tensors are not supported. <br> **Data type**: See Table 3. <br> **Matrix dimension**: 2D, 3D, or 4D. <br> **Format**: TILEOP_ND, TILEOP_NZ (TILEOP_NZ is not supported for DT_FP8E5M2 input).<br> **Inner axis and outer axis**: When the input matrix **mat_b** is not transposed, the corresponding data layout is [K, N], where the outer axis is K and the inner axis is N. When the input matrix **mat_b** is transposed, the corresponding data layout is [N, K], where the outer axis is N and the inner axis is K. <br> **Alignment requirement**: When Format is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535]. <br> When Format is TILEOP_NZ (NZ format), the shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> When using the **pypto.view** API, ensure that the shape dimension of the passed View also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| out_dtype         | Output      | Output matrix data type. For the supported output data types in the basic scenario, see Table 3. For the supported output data types in the quantization scenario, see Table 4.|
| scale_a              | Input      | Quantization parameter of the left input matrix. Empty tensors are not supported. <br> **Data type**: See Table 3. <br> **Quantization parameter dimension**: 3D.<br> **Format**: TILEOP_ND.<br> **Quantization parameter shape**: When the input quantization parameter is not transposed, the corresponding input shape is [M, CeilAlign(K, 64)/64, 2]. When the input quantization parameter is transposed, the corresponding input shape is [CeilAlign(K, 64)/64, M, 2]. Here, M and K are equal to the shape values of the M and K dimensions of the input matrix **mat_a**.|
| scale_b              | Input      | Quantization parameter of the right input matrix. Empty tensors are not supported. <br> **Data Type**: See Table 3. <br> **Quantization parameter dimension**: 3D.<br> **Format**: TILEOP_ND.<br> **Quantization parameter shape**: When the input quantization parameter is not transposed, the corresponding input shape is [CeilAlign(K, 64)/64, N, 2]. When the input quantization parameter is transposed, the corresponding input shape is [N, CeilAlign(K, 64)/64, 2]. Here, N and K are equal to the shape values of the N and K dimensions of the input matrix **mat_b**.|
| a_trans           | Input      | Whether the left input matrix is transposed. Defaults to **False**. |
| b_trans           | Input      | Whether the right input matrix is transposed. Defaults to **False**. |
| scale_a_trans     | Input      | Whether the quantization parameter of the left input matrix is transposed. Defaults to **False**. |
| scale_b_trans     | Input      | Whether the quantization parameter of the right input matrix is transposed. Defaults to **False**. |
| c_matrix_nz       | Input      | Whether the output matrix uses the NZ format. Defaults to **False**. Currently only **False** is supported, meaning the output matrix supports only the ND format. |
| extend_params     | Input      | Supports the quantization functions of bias and fixpipe. The data type is a dictionary. Defaults to **None**. See Table 2. |

where:
    - The element alignment of `CeilAlign(value, align)` is implemented as `(value + align - 1) / align * align`

Table 2: extend_params Parameter Description

| Parameter            | Description                                                                 |
|-------------------|----------------------------------------------------------------------|
| scale             | Quantization parameter of the output matrix in the pertensor quantization scenario (using the same scaling factor to map high-precision numbers to low-precision numbers).<br>The input is of the float type, with 1 sign bit + 8 exponent bits + 10 mantissa bits participating in the operation.<br>For the supported input and output data types, see Table 4.<br>Multi-core K-splitting is not supported.|
| scale_tensor      | Matrix for output matrix quantization in the perchannel quantization scenario (computing an independent set of quantization parameters for each output channel).<br>The **scale_tensor** input is fixed as a uint64_t or int64_t tensor. During computation, the 64 bits are converted to the lower 32 bits of the float type, and then 1 sign bit + 8 exponent bits + 10 mantissa bits participate in the operation.<br>For the supported input and output data types, see Table 4.<br>The first dimension of **scale_tensor** must be set to 1, and the N dimension must be equal to the N dimension of the **mat_b** matrix.<br>**scale_tensor** supports only the ND format.<br>Only the 2D matrix dimension scenario is supported.<br>Multi-core K-splitting is not supported.<br>When the quantized output type is DT_INT8, call **torch_npu.npu_trans_quant_param** outside the function in advance and pass a float32 **torch.tensor** to obtain a **scale_tensor** of the int64 data type.|
| bias_tensor       | Bias matrix.<br>The input is of the Tensor type.<br>The bias matrix data type can be DT_FP16, DT_BF16, or DT_FP32.<br>**bias_tensor** supports only the ND format.<br>Only the 2D/3D/4D matrix dimension scenarios are supported.<br>When the input matrix is 3D, the bias dimension can be [B, 1, N] or [1, N], and the N dimension must be equal to the N dimension of the **mat_b** matrix.<br>When the input matrix is 4D, the bias dimension can only be [1, N], and the N dimension must be equal to the N dimension of the **mat_b** matrix.<br>Multi-core K-splitting is not supported.|
| relu_type         | Whether to perform the ReLu operation on the output matrix.<br>The input is of the [ReLuType](../datatype/ReLuType.md) type.<br>Supports the RELU and NO_RELU modes.<br>Multi-core K-splitting is not supported. |

Table 3: Data Types Supported by scaled_mm in the Basic Scenario

| mat_a | mat_b | out_dtype | scale_a | scale_b | bias_tensor | Product Support |
|:------|:------|:----------|:--------|:--------|:------------|:---------|
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 | Ascend 950PR/Ascend 950DT |

Table 4: Data types supported by scaled_mm in quantization scenarios

| mat_a | mat_b | out_dtype | Product Support |
|:------|:-----|:----------|:----------|
| DT_FP8E5M2 | DT_FP8E5M2, DT_FP8E4M3 | DT_INT8 | Ascend 950PR/Ascend 950DT |
| DT_FP8E4M3 | DT_FP8E5M2, DT_FP8E4M3 | DT_INT8 | Ascend 950PR/Ascend 950DT |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_INT8 | Ascend 950PR/Ascend 950DT |

## Return Value

Returns the out matrix, which is a tensor.

## Constraints

- When the input is in the DT_FP4_E2M1 quantization scenario, ensure that the inner axis is even.
- Before calling the scaled_mm API, use pypto.set\_cube\_tile\_shapes to set the splitting sizes on the M, K, and N axes.
- When the input to the `scaled_mm` API is in NZ format after `pypto.reshape`, you must call `pypto.set_matrix_size` to specify the original `m`, `k`, and `n` dimensions of the matrix multiplication, that is, the shape before `pypto.reshape` is applied.

## Examples

```python
# Basic matrix multiplication.
mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([64, 2, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# Add bias.
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
bias = pypto.tensor((1, 32), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

# Quantize and add RELU.
scale_cpu = pypto.tensor((1, 32), pypto.DT_UINT64, "tensor_scale")
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # Generate scale_tensor.
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

```

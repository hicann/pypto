# pypto.conv

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:34:09.594Z pushedAt=2026-09-05T08:30:53.951Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs convolution on the input **input_conv** and **weight**, and supports the **bias** parameter. The computation formula is: out = input_conv @ weight + bias (@ denotes convolution processing).

- **input_conv**, **weight**, and **bias** are source operands. **input_conv** is the input matrix, **weight** is the weight matrix, and **bias** is the input bias.
- **out** is the destination operand, which stores the matrix of convolution processing results.
- Quantization scenarios are currently not supported.
- The ReLU function is currently not supported.

## Prototype

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## Parameters

| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input_conv       | Input      | Input feature map tensor.<br>Empty tensors are not supported.<br>Supported dimensions: 3D (1D conv), 4D (2D conv), and 5D (3D conv).<br>Supported formats: NCL, NCHW, NCDHW.<br>Supported data types: DT_FP16, DT_BF16, and DT_FP32.<br>Shape constraint: The value range of each dimension is [1, 1000000]. The cin of input_conv must satisfy: cin of weight * groups = cin of input_conv. |
| weight            | Input      | Convolution kernel tensor.<br>The number of dimensions must be consistent with input_conv (3D/4D/5D).<br>The data type must be consistent with input_conv.<br>Shape constraint: The value range of each dimension is [1, 1000000]. Different models have additional constraints. For details, see [Constraints](#constraints). |
| out_dtype         | Input      | Data type of the output tensor.<br>Supported: DT_FP16, DT_BF16, and DT_FP32.<br>Must be consistent with input_conv; in fixpipe quantization scenarios, it can be specified separately. |
| strides           | Input      | Convolution stride, a unidirectional parameter.<br>- 1D (1D conv)<br>- 2D (2D conv)<br>- 3D (3D conv)<br>Value range: [1, 63]. |
| paddings          | Input      | Convolution padding, a bidirectional parameter.<br>- 2D (1D conv)<br>- 4D (2D conv)<br>- 6D (3D conv)<br>Value range: [0, 255], and the padding value of each dimension is less than the corresponding convolution kernel size. |
| dilations         | Input      | Dilation rate of dilated convolution, a unidirectional parameter.<br>- 1D (1D conv)<br>- 2D (2D conv)<br>- 3D (3D conv)<br>Value range: [1, 63]. |
| groups            | Input      | Number of groups for grouped convolution, defaulting to **1**.<br>Value range: [1, 65535].<br>Cin and Cout must be divisible by groups. |
| transposed        | Input      | Whether it is a transposed convolution (deconvolution), defaulting to **False**.<br>**True** is currently not supported. |
| output_paddings   | Input      | Output padding of transposed convolution, used only when transposed=True.<br>Currently not supported. |
| extend_params     | Input      | Extended parameter dictionary, supporting **bias**, **scale**, **relu**, and **scale_tensor**:<br>- **bias_tensor**: Optional bias tensor with shape (C_out,), supporting only the ND format. The supported data types vary by model. For details, see [Constraints](#constraints).<br>- **scale**: Floating-point type, per-tensor scaling factor.<br>- **scale_tensor**: uint64 type, per-channel scaling tensor with shape [1, Cout], supporting only the ND format.<br>- **relu_type**: Activation type, supporting RELU/NO_RELU and others (currently not supported; see the "Description" section for details). |

## Return Value

Returns the output tensor after the convolution operation:

- 1D convolution output shape: (Batch, Cout, Wout)
- 2D convolution output shape: (Batch, Cout, Hout, Wout)
- 3D convolution output shape: (Batch, Cout, Dout, Hout, Wout)

The value range of each dimension of the output shape: [1, 1000000].

## Constraints

### 1. Shape Validity Constraints

- Input feature map (**input_conv**): The **Batch**, **Cin**, **Hin**, **Win**, and **Din** dimensions must be within the range [1, 1000000].
- Convolution kernel (**weight**): The **Cout**, **Kh**, **Kw**, and **Kd** dimensions must be within the range [1, 1000000].
- Bias (**bias_tensor**): The shape must be equal to [**Cout**]; otherwise, the validation fails.
- Output feature map: The **H_out**, **W_out**, and **D_out** dimensions must be within the range [1, 1000000].

### 2. Attribute Parameter Validity Constraints

- Basic dimension matching constraints:
  - The number of dimensions of **strides** must match the convolution dimension (2D conv length = 2, 3D conv length = 3).
  - The number of dimensions of **dilations** must match the convolution dimension (2D conv length = 2, 3D conv length = 3).
  - The number of dimensions of **paddings** must be 2 × the convolution dimension (2D conv length = 4, 3D conv length = 6).
- Value range constraints:
  - The value range of **strides** is [1, 63].
  - The value range of **dilations** is [1, 63].
  - The value range of **paddings** is [0, 255], and the padding value of each dimension must be less than the corresponding convolution kernel dimension size (for example, padding_h < Kh, padding_w < Kw).
  - The value range of **groups** is [1, 65535].
- Convolution kernel constraints:
  - Kh ≤ 255 and Kw ≤ 255.
  - Kh × Kw × 32bytes/dtype ≤ 65535; **dtype** is the number of bits occupied by the data type of **input_conv**, for example, 16 for FP16 and 32 for FP32.
- Channel count constraints:
  - Cin (number of input channels) must be divisible by **groups**.
  - Cout (number of output channels) must be divisible by **groups**.
  - CinFmap = CinWeight × **groups**.
- Product-specific constraints:
  - Atlas A3 training products/Atlas A3 inference products: The N axis (cout) of **weight** divided by **groups** must be an integer multiple of **C0** (**C0** = ALIGN_SIZE_32 / sizeof(dtype), ALIGN_SIZE_32 = 32). If dynamic axis splitting is configured for **cout**, the split **cout** divided by **groups** must also be an integer multiple of **C0**. If the data type of **input_conv** is **DT_BF16**, the data type of **bias** must be **DT_FP32**.
  - Atlas A2 training products/Atlas A2 inference products: The N axis (cout) of **weight** divided by **groups** must be an integer multiple of **C0** (**C0** = ALIGN_SIZE_32 / sizeof(dtype), ALIGN_SIZE_32 = 32); if dynamic axis splitting is configured for **cout**, the split **cout** divided by **groups** must also be an integer multiple of **C0**. If the data type of **input_conv** is **DT_BF16**, the data type of **bias** must be **DT_FP32**.

### 3. Cache Space Constraints

- Before calling the **conv** API, you must set the convolution TileShape splitting size at the L1/L0 level through the **pypto.set_conv_tile_shapes** API.

### 4. Functional Support Constraints

- **transposed**=**True** (transposed convolution) is currently not supported, and calling it throws a **RuntimeError**.
- **input_conv**/**weight** supports only the **DT_FP16**, **DT_BF16**, and **DT_FP32** data types; other types throw a **ValueError**.
- The dimensions of **input_conv** and **weight** must be consistent (for example, if **input_conv** is 4D, **weight** must also be 4D); otherwise, a **RuntimeError** is thrown.

### 5. Dynamic Axis Splitting Support

The dynamic axis splitting dimensions supported by the convolution operator are as follows:

| Dimension | Supported | Splitting Method | Description |
|:-----------|:--------:|:-----------------------------------|:---------------------------------------------------------------------|
| Batch      |    √     | Frontend loop splitting | **TileL1Info.tileBatch** must be 1 (hardware constraint), and dynamic splitting is implemented through the frontend loop. |
| Cout       |    √     | **TileShape** dynamic splitting + frontend loop | Supports dynamic splitting configured by **TileShape**, and covers the complete Cout dimension in conjunction with the frontend loop (splitting is not allowed when **groups** > 1). |
| Dout       |    √     | **TileShape** dynamic splitting + frontend loop | Supported only for 3D convolution, with dynamic splitting of the Dout dimension. |
| Hout       |    √     | **TileShape** dynamic splitting + frontend loop | Dynamic splitting of the Hout dimension, covering the complete dimension in conjunction with the frontend loop. |
| Wout       |    √     | **TileShape** dynamic splitting + frontend loop | Dynamic splitting of the Wout dimension, covering the complete dimension in conjunction with the frontend loop. |
| Cin        |    ×     | -       | Dynamic axis splitting of the Cin dimension is currently not supported. Use **set_conv_tile_shapes()** to perform tile splitting of k. |

**Precautions:**
- For 1D convolution on Atlas A3 training products/Atlas A3 inference products, set **vec_tile_shapes** to {n, c, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 = ALIGN_SIZE_32 / sizeof(dtype), ALIGN_SIZE_32 = 32, and w is 32B-aligned.
- For 2D convolution on Atlas A3 training products/Atlas A3 inference products, set **vec_tile_shapes** to {n, c, h, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 is the same as above, and w is 32B-aligned.
- For 3D convolution on Atlas A3 training products/Atlas A3 inference products, set **vec_tile_shapes** to {n, c, d, h, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 is the same as above, and w is 32B-aligned.
- For 1D convolution on Atlas A2 training products/Atlas A2 inference products, set **vec_tile_shapes** to {n, c, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 = ALIGN_SIZE_32 / sizeof(dtype), ALIGN_SIZE_32 = 32, and w is 32B-aligned.
- For 2D convolution on Atlas A2 training products/Atlas A2 inference products, set **vec_tile_shapes** to {n, c, h, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 is the same as above, and w is 32B-aligned.
- For 3D convolution on Atlas A2 training products/Atlas A2 inference products, set **vec_tile_shapes** to {n, c, d, h, w}, where n is an integer multiple of 16, c is an integer multiple of C0, C0 is the same as above, and w is 32B-aligned.

### 6. Data Type Constraints

- The data types supported by Ascend 950PR/Ascend 950DT are DT_FP16, DT_BF16, and DT_FP32. The data types of input, weight, bias, and output need to be consistent.
- The data types supported by Atlas A3 training products/Atlas A3 inference products are DT_FP16, DT_BF16, and DT_FP32. For DT_FP16 and DT_FP32 types, the data types of input, weight, bias, and output need to be consistent; for the DT_BF16 type, input, weight, and output are of the BF16 type, and bias must be of the DT_FP32 type.
- The data types supported by Atlas A2 training products/Atlas A2 inference products are DT_FP16, DT_BF16, and DT_FP32. For DT_FP16 and DT_FP32 types, the data types of input, weight, bias, and output need to be consistent; for the DT_BF16 type, input, weight, and output are of the BF16 type, and bias must be of the DT_FP32 type.

## Examples

```python
# Basic 2D convolution example.
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D convolution with bias.
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias}

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D convolution example.
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```

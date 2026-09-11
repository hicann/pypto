# pypto.tanh

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:59:24.747Z pushedAt=2026-09-05T07:36:26.378Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Applies the hyperbolic tangent function (tanh) to each element of the input tensor. The computation formula is:

$$
\tanh(input) = \frac{e^{input} - e^{-input}}{e^{input} + e^{-input}} = \frac{e^{2 \cdot input} - 1}{e^{2 \cdot input} + 1}
$$

This function maps the input to the \((-1, 1)\) interval and is commonly used as a neural network activation function.

## Prototype

```python
tanh(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data types: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the results of mapping the input elements to the \((-1, 1)\) interval through the tanh function.

## Constraints

1. The TileShape must have the same dimension count as the input.
2. Due to temporary memory usage, when the input data type is DT_FP32, the TileShape size has additional constraints. Assuming the TileShape is [...,H,W] (where the last two dimensions are H and W), then:
    `input_size + output_size + 2 * (W_align8) * H * sizeof(float) + (W_align8 / 8) * H + 32 bytes < UB`
    where `W_align8 = (W + 7) / 8 * 8`
    (FP32: input + output + 2 float temp tiles + 1 compare mask tile + 32 bytes alignment)

    For DT_FP16/DT_BF16 input, the following must be satisfied:
    `input_size + output_size + 4 * (W_align8) * H * sizeof(float) + (W_align8 / 8) * H + 32 bytes < UB`
    (FP16/BF16: input + output + 4 float temp tiles + 1 compare mask tile + 32 bytes alignment)
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.tanh(x)
```

The results are as follows:

```python
Input data x: [-3.0, -1.0, 0.0, 1.0, 3.0]
Output data y: [-0.9951, -0.7616, 0.0000, 0.7616, 0.9951]
```

Computation process description:

- tanh(-3.0) ≈ -0.9951, close to -1
- tanh(-1.0) ≈ -0.7616
- tanh(0.0) = 0.0
- tanh(1.0) ≈ 0.7616
- tanh(3.0) ≈ 0.9951, close to 1

# pypto.cbrt

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:27:12.202Z pushedAt=2026-09-05T08:30:40.240Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the cube root of each element in the input tensor element-wise and returns a tensor with the same shape as the input. The computation formula is:

$$
y_i = \sqrt[3]{x_i}
$$

## Prototype

```python
cbrt(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of the input tensor.

## Constraints

1. The input tensor and the output tensor must have the same type.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([1.0, 2.0], pypto.DT_FP32)
y = pypto.cbrt(x)
```

The results are as follows:

```python
Input  x:[[8.0, -8.0]]
Output y:[[2.0, -2.0]]
```

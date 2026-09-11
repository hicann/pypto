# pypto.asin

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:15:13.000Z pushedAt=2026-09-05T08:30:09.256Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the arcsine of each element in the input tensor, operating element-wise. The result falls within the interval $[-\pi/2, \pi/2]$. When the absolute value of an input element is greater than 1, the corresponding output is NaN.

$$
y_i = \arcsin(x_i)
$$

## Prototype

```python
asin(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| **input**  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor, its data type is the same as that of the input tensor, and its elements are the arcsine values of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.asin(x)
```

The results are as follows:

```python
Input data x: [-1.0000, 0.0000, 1.0000]
Output data y: [-1.5708, 0.0000, 1.5708]
```

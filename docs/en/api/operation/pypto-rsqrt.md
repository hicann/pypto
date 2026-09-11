# pypto.rsqrt

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:46:37.858Z pushedAt=2026-09-05T07:36:26.366Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the reciprocal square root of each element in the input tensor, performing element-wise operations. Returns NaN when the input is negative and Inf when the input is zero.

## Prototype

```python
pypto.rsqrt(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| **input** | **Tensor** | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of **Tensor**: **DT_FP32**, **DT_FP16**, and **DT_BF16**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| **precision_type** | **PrecisionType**, optional | Precision mode of the reciprocal square root operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss. |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the reciprocal square roots of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

Example 1: If the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.rsqrt(x)
```

The results are as follows:

```python
Input data x: [[1.0  4.0], [16.0  9.0]]
Output data y: [[1.0  0.5], [0.25  0.33333]]
```

### High-Precision Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.INTRINSIC)
```

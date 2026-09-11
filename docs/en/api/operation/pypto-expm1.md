# pypto.expm1

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:57:14.343Z pushedAt=2026-09-05T07:36:26.325Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the natural exponential of each element of the input tensor and then subtracts 1.
$$
y_i = e^{x_i} - 1
$$

## Prototype

```python
expm1(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                             |
|----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | Input      | Source operand.<br>Supported type: Tensor.<br>The supported data type is DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor whose shape is the same as that of the input tensor. When the input data type is DT_FP32, DT_FP16, or DT_BF16, the output data type is the same as that of the input tensor. When the input data type is DT_INT32 or DT_INT16, the output data type is DT_FP32. Each element is the natural exponential of the corresponding element of the input tensor minus 1.

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
y = pypto.expm1(x)
```

The results are as follows:

```python
Input data x: [[1., 2.], [3., 4.]]
Output data y: [[1.7183, 6.3891], [19.0855, 53.5981]]
```

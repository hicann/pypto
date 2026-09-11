# pypto.exp2

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:45:55.511Z pushedAt=2026-09-05T08:31:16.295Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the base-2 exponential of each element in the input tensor element-wise, and returns a tensor with the same shape as the input.

## Prototype

```python
exp2(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns an output tensor. When the input is of type DT_FP32, DT_FP16, or DT_BF16, the output tensor has the same data type and shape as the input. When the input is of type DT_INT32, DT_INT16, DT_INT8, or DT_UINT8, the output tensor is of type DT_FP32 and has the same shape as the input.

## Constraints

1. The value range of **input** must be within \[-2^24, 2^24\] to ensure accurate conversion to float32 during computation.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
y = pypto.exp2(x)
```

The results are as follows:

```python
Input data x: [0.0    1.0    2.0]
Output data y: [1.0    2.0    4.0]
```

# pypto.abs

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-26T12:01:04.421Z pushedAt=2026-09-05T09:17:26.338Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the absolute value of each element in the input tensor, performing element-wise operations.

## Prototype

```python
abs(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: **Tensor**.<br>The supported tensor data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the absolute values of the corresponding elements of the input tensor.

## Constraints

1. The data types supported by a tensor are as follows:
   - Ascend 950PR/Ascend 950DT: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32.
   - Atlas A3 training products/Atlas A3 inference products: DT_FP16, DT_BF16, and DT_FP32.
   - Atlas A2 training products/Atlas A2 inference products: DT_FP16, DT_BF16, and DT_FP32.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.abs(x)
```

The results are as follows:

```python
Input data x: [-1.0, 2.0, -3.0, 4.0, 5.0]
Output data y: [1.0,  2.0,  3.0, 4.0, 5.0]
```

# pypto.ceil

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:29:32.335Z pushedAt=2026-09-05T08:30:43.137Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the ceiling of each element in the input tensor (returns the smallest integer not less than the element) in an element-wise manner. Integer values are returned as-is, while floating-point values are rounded up.

## Prototype

```python
ceil(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16. The shape supports only 1 to 4 dimensions. |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the ceiling values of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.ceil(x)
```

The results are as follows:

```python
Input data x: [1.2, 4.7, -1.1, 9.0, 3.9]
Output data y: [2.0, 5.0, -1.0, 9.0, 4.0]
```

# pypto.round

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:46:10.298Z pushedAt=2026-09-05T07:36:26.366Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Rounds the elements of the input tensor to the specified number of decimal places. If the value is equally distant from the two decimals at the specified position, the even number at the specified position is taken.

## Prototype

```python
round(input: Tensor, decimals: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                             |
|-----------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| decimals  | Input        | Source operand specifying the number of decimal places to round to.<br>Type: int.

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the results of rounding the corresponding elements of the input tensor to the specified number of decimal places.

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
y = pypto.round(x, decimals=1)
```

The results are as follows:

```python
Input data x: [[1.21, 2.35], [3.65, 4.76]]
Output data y: [[1.2, 2.4], [3.6, 4.8]]
```

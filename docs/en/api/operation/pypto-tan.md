# pypto.tan

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:57:43.446Z pushedAt=2026-09-05T07:36:26.377Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the tangent (trigonometric function tan) of each element in the input tensor, operating element-wise.

$$
y_i = \tan(x_i)
$$

## Prototype

```python
tan(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input  | Input      | Source operand.<br>Supported types: Tensor.<br>Supported data types for Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).|

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor, its data type is the same as that of the input tensor, and its elements are the tangent values of the corresponding elements of the input tensor.

## Constraints

1. The input tensor and the output tensor must have the same type.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.tan(x)
```

The results are as follows:

```python
Input data x: [0.0000, 0.7854, 1.0472, -0.7854]
Output data y: [0.0000, 1.0000, 1.7321, -1.0000]
```

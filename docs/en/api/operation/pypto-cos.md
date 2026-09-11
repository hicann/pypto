# pypto.cos

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:34:41.029Z pushedAt=2026-09-05T08:30:55.626Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the cosine value (trigonometric function cos\( \)) of each element in the input tensor. It supports element-wise operations and returns a tensor with the same shape as the input.

## Prototype

```python
cos(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported data types: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor with the same shape and data type as the input tensor. Each element of the output tensor is the cosine of the corresponding element in the input tensor.

## Constraints

1. The input tensor and output tensor must have the same type.
2. Due to temporary memory usage, the TileShape size has additional constraints. Assuming the TileShape is \[a,b,c,d\], then 5\*a\*b\*c\*d\*sizeof\(DT_FP32\) < UB.
3. The supported input range is [-65504.0, 65504.0].
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.cos(x)
```

The results are as follows:

```python
Input data x: [0.0000, 0.7854, 1.5708, 2.3562]
Output data y: [1.0000, 0.7071, 0.0000, -0.7071]
```

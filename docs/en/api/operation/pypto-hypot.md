# pypto.hypot

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:08:28.475Z pushedAt=2026-09-05T07:36:26.335Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported

- Atlas A3 training products/Atlas A3 inference products: Supported

- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the square root of the sum of squares of **input** and **other** element-wise (that is, the length of the hypotenuse of a right triangle). The computation formula is as follows:

$$
res_i = \sqrt{input_i^2 + other_i^2}
$$

## Prototype

```python
hypot(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (**INT32_MAX**).|
| other     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.

2. For BF16 and FP16 types, the internal computation may increase precision to avoid intermediate overflow.

3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
# Example: Compute the hypotenuse of two sets of right triangle legs.
# First set: (3, 4) -> 5
# Second set: (5, 12) -> 13
a = pypto.tensor([3.0, 5.0], pypto.DT_FP32)
b = pypto.tensor([4.0, 12.0], pypto.DT_FP32)
out = pypto.hypot(a, b)
```

The results are as follows:

```python
Input data a:   [3.0, 5.0]
Input data b:   [4.0, 12.0]
Output data out: [5.0, 13.0]
```

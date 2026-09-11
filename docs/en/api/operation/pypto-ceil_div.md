# pypto.ceil_div

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:28:39.036Z pushedAt=2026-09-05T08:30:41.721Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Divides each element of **self** by the element at the corresponding position in **other** and rounds up. The computation formula is as follows:

$$
res_i = ceil(self_i \div other_i)
$$

## Prototype

```python
ceil_div(self: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| self  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| other  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. The types of **input** and **other** must be the same.
2. Only single-axis broadcasting is supported.
3. The value ranges of **input** and **other** must be within \[-2^24, 2^24\] to ensure accurate conversion to float32 during computation. **other must not be 0**. When the divisor in integer division is 0, the result is determined by the chip and may be INT32_MAX or INT32_MIN.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.ceil_div(a, b)
```

The results are as follows:

```python
Input data a:    [[2 4 6]]
Input data b:    [[4 2 5]]
Output data out:  [[1 2 2]]
```

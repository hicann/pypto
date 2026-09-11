# pypto.add

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-26T12:01:04.906Z pushedAt=2026-09-05T09:17:33.028Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Adds the values of **input** and **other** element-wise. The calculation formula is as follows:

$$
res_i = input_i + other_i
$$

## Prototype

```python
add(input: Tensor, other: Union[Tensor, float, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: **Tensor**. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |
| other     | Input        | Source operand.<br>Supported types: **float**, **int**, and **Tensor**. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. When **other** is a scalar, if **input** is of a floating-point type, the scalar supports integer types (automatically converted to floating-point); if **input** is of an integer type, the scalar does not support floating-point types (an error is reported).
3. Tensor data type description:
   - Ascend 950PR/Ascend 950DT: DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, and DT_INT8.
   - Atlas A3 training products/Atlas A3 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
   - Atlas A2 training products/Atlas A2 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.add(a, b)
```

The results are as follows:

```python
Input data a:   [[1.0 2.0 3.0]]
Input data b:   [[2.0 3.0 4.0]]
Output data out: [[3.0 5.0 7.0]]
```

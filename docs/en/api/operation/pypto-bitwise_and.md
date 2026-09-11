# pypto.bitwise\_and

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:20:52.065Z pushedAt=2026-09-05T08:30:24.095Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs a bitwise AND operation on **input** and **other** element by element. The computation formula is as follows:

$$
\text{res}_i = \text{input\_i \& other\_i}
$$

## Prototype

```python
bitwise_and(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported types: int and Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting to the same shape along a single dimension is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. Tensor data type description:
   - Ascend 950PR/Ascend 950DT: DT_INT16, DT_UINT16, DT_INT8, DT_UINT8, and DT_INT32.
   - Atlas A3 training products/Atlas A3 inference products: DT_INT16, DT_UINT16, DT_INT8, and DT_UINT8.
   - Atlas A2 training products/Atlas A2 inference products: DT_INT16, DT_UINT16, DT_INT8, and DT_UINT8.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
a = pypto.tensor([2], pypto.DT_INT16)
b = pypto.tensor([2], pypto.DT_INT16)
out = pypto.bitwise_and(a, b)
```

The results are as follows:

```python
Input data a:  [2, 5]
Input data b: [1, 7]
Output data out: [0, 5]
```

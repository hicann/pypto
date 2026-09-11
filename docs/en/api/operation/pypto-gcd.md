# pypto.gcd

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:04:55.448Z pushedAt=2026-09-05T07:36:26.332Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported

- Atlas A3 training products/Atlas A3 inference products: Supported

- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the greatest common divisor of the elements of **input** and **other**.

## Prototype

```python
gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                                                   |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (**INT32_MAX**).           |
| other  | Input      | Source operand.<br>Supported types: int and Tensor.<br>Supported data types of Tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. **input** and **other** have the same data type.

2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**. The last axis of the TileShape must be 32-byte-aligned.

The dimensions of the TileShape must be the same as those of the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)
y = pypto.tensor([2, 3], pypto.DT_INT32)
z = pypto.gcd(x, y)
# Using a scalar
c = pypto.gcd(x, 2)
```

The results are as follows:

```python
Input data x: : [[9 9 9],
             [6 6 6]]
Input data y:   [[1 2 3],
             [1 2 3]]
Output data z:   [[1 1 3],
             [1 2 3]]
Output data c:   [[1 1 1],
             [2 2 2]]
```

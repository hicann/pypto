# pypto.floor_div

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:59:31.189Z pushedAt=2026-09-05T07:36:26.327Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Divides each element of **input** by the element at the corresponding position in **other** and rounds down. The computation formula is as follows:

$$
res_i = floor(\frac{input_{i}}{other_{i}})
$$

## Prototype

```python
def floor_div(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other  | Input     | Source operand.<br>Supported types: Tensor and int.<br>Supported data type of Tensor: DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. The data types of **input** and **other** must be the same.
2. Only single-axis broadcasting is supported.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.floor_div(a, b)
```

The results are as follows:

```python
Input data a:    [[2 4 6]]
Input data b:    [[4 2 5]]
Output data out:  [[0 2 1]]
```

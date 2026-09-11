# pypto.bitwise_left_shift

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:22:06.882Z pushedAt=2026-09-05T08:30:25.958Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Shifts **input** left by **other** bits element-wise. The computation formula is as follows:

$$
res_i = input_i << other_i
$$

## Prototype

```python
bitwise_left_shift(input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                      |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input  | Input      | Source operand.<br>Supported types: int and Tensor. The supported Tensor data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting along a single dimension to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other  | Input      | Source operand.<br>Supported types: int and Tensor. The supported tensor data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Broadcasting along a single dimension to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).     |

## Return Value

Returns an output tensor with the same data type as **input and other**, and with a shape equal to the broadcast size of **input** and **other**.

## Constraints

1. The types of **input** and **other** must be the same.
2. Tensor data type description:
   - Ascend 950PR/Ascend 950DT: DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT8, and DT_UINT8.
   - Atlas A3 training products/Atlas A3 inference products: DT_INT16, and DT_UINT16.
   - Atlas A2 training products/Atlas A2 inference products: DT_INT16, and DT_UINT16.
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
a = pypto.tensor([1, 3], pypto.DT_INT16)
b = pypto.tensor([1, 3], pypto.DT_INT16)
out = pypto.bitwise_left_shift(a, b)
```

The results are as follows:

```python
Input a:    [[1 2 3]]
Input b:    [[1 1 1]]
Output out: [[2 4 6]]
```

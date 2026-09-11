# pypto.logical\_and

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:23:34.570Z pushedAt=2026-09-05T07:36:26.346Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs an element-wise logical AND operation on two input tensors. Operation rules:

- If the input tensors are of the bool type, **True** and **True** -\> **True**; otherwise, the result is **False**.
- If the input tensors are numeric, they are automatically converted to **True**/**False**, where 0 is **False** and any non-zero value is **True**.

## Prototype

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Input tensors of different data types are supported. Single-axis broadcasting is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Input tensors of different data types are supported. Single-axis broadcasting is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type is DT\_BOOL and whose shape is the broadcast shape.

## Constraints

1. The TileShape must have the same dimension count as `input` and `other`.
2. Due to temporary memory usage, the TileShape size has additional constraints. Assuming that the TileShape is [a,b,c,d], then a*b*c*d*sizeof(input) + a*b*c*d*sizeof(other) + a*b*c*d*sizeof(BOOL) + 1.1875KB must be less than the available memory upper limit.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

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
x = pypto.tensor([2], pypto.DT_BOOL)
y1 = pypto.tensor([2], pypto.DT_BOOL)
z1 = pypto.logical_and(x, y1)
# Supports broadcasting.
y2 = pypto.tensor([2,2], pypto.DT_BOOL)
z2 = pypto.logical_and(x, y2)
```

The results are as follows:

```python
Input data x:  [True, False]
Input data y1: [True, True]
Input data y2: [[True, False], [False, True]]
Output data z1: [True, False]
Output data z2: [[True, False], [False, False]]
```

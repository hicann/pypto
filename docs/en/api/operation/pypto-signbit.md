# pypto.signbit

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:51:58.806Z pushedAt=2026-09-05T07:36:26.372Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Checks whether the sign bit of each element in the input tensor is set (that is, whether the element is negative). This is an element-wise operation.

Logic:

- If the element is negative (including −∞ and −0.0), **True** is returned.
- If the element is positive (including +∞ and +0.0) or NaN, **False** is returned.

## Prototype

```python
signbit(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of **Tensor**: **DT_FP16**, **DT_BF16**, **DT_FP32**, **DT_INT8**, **DT_INT16**, and **DT_INT32**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor whose shape is the same as that of the input tensor, with the data type **DT_BOOL**. Each element indicates whether the sign bit of the corresponding element in the input tensor is set (True for negative numbers and False for non-negative numbers).

## Constraints

1. **TileShape** must be consistent with the dimensions of **input**.
2. Due to temporary memory usage, when the input data type is **DT_FP32**, the **TileShape** size has an additional constraint. Assuming **TileShape** is [a,b,c,d], then a*b*c*d*sizeof(self) + a*b*c*d*sizeof(FP16) + a*b*c*d*sizeof(UINT8) < UB.
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
x = pypto.tensor([-5, 0, 5, 10, -2], pypto.DT_FP32)
y = pypto.signbit(x)
```

The results are as follows:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [True, False, False, False, True]
```

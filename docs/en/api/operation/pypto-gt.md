# pypto.gt

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:06:59.699Z pushedAt=2026-09-05T07:36:26.334Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs an element-wise greater-than comparison.

## Prototype

```python
gt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor. The supported Tensor data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other   | Input      | Source operand.<br>Supported types: Tensor, float, and Element.<br>When the type is float, it is automatically converted to the Element type, and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The supported Tensor and Element data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor whose shape is the same as that of the input tensor and whose data type is DT\_BOOL. If the element value at a position in **input** is strictly greater than the element value at the corresponding position in **other**, the return value at that position is **True**; otherwise, it is **False**.

## Constraints

1. The types of **input** and **other** must be the same.
2. One-dimensional broadcasting is supported.
3. Data types of Tensor and Element:
   - Ascend 950PR/Ascend 950DT: DT_FP16, DT_FP32, and DT_INT16.
   - Atlas A3 training products/Atlas A3 inference products: DT_FP16 and DT_FP32.
   - Atlas A2 training products/Atlas A2 inference products: DT_FP16 and DT_FP32.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.gt(a, b)
```

The results are as follows:

```python
Input data a: [1.0 2.0 3.0]
Input data b: [2.0 2.0 2.0]
Output data out: [False, False, True]
```

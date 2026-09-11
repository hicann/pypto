# pypto.arange

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-26T12:03:10.654Z pushedAt=2026-09-05T09:17:41.970Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a one-dimensional tensor of length $\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$, containing an arithmetic sequence within the interval \[start, end\) with **step** as the step size.

## Prototype

```python
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| start  | Input      | Source operand.<br>Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, and DT_FP32.<br>Default value: 0. |
| end    | Input      | Source operand.<br>Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, and DT_FP32.<br>This parameter is mandatory. |
| step   | Input      | Source operand.<br>Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, and DT_FP32.<br>Default value: 1. |

## Return Value

Returns a one-dimensional output tensor. If any input value is of a floating-point data type, the output tensor data type is `DT_FP32`; otherwise, it is `DT_INT32`.

## Constraints

1. **step** cannot be 0. As a floating-point number, abs\(step\)\>1e-8.

2. \(end-start\)/step must be greater than 0.

3. If **start**, **end**, and **step** are all int inputs, all three must remain within the `int32` range.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape and the output have the same dimensions, both being one-dimensional.

If the input **start** is m, **end** is n, and **step** is p, the output shape is [q], and the TileShape is set to [q1], then q1 is used to split the q axis.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

```python
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

The results are as follows:

```python
Output data y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
Output data y2: [1.0, 2.0, 3.0]
Output data y3: [0, 1, 2, 3]
```

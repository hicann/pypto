# pypto.logical\_not

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:23:44.341Z pushedAt=2026-09-05T07:36:26.347Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Converts zero values in the input tensor to `True` and all non-zero values to `False`.

## Prototype

```python
logical_not(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns an output tensor of data type DT\_BOOL with the same shape as the source operand `input`.

## Constraints

1. The TileShape must have the same dimension count as `input`.
2. Due to temporary memory usage, when the input data type is DT\_FP32, the TileShape size has additional constraints. Assuming the TileShape is \[a,b,c,d\], the following condition must be satisfied: a\*b\*c\*d\*sizeof\(input\) + a\*b\*c\*d\*sizeof\(BOOL\) + 20.25KB < UB. For other input data types, the condition is: a\*b\*c\*d\*sizeof\(input\) + a\*b\*c\*d\*sizeof\(BOOL\) + 12.54KB < UB.
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
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

The results are as follows:

```python
Input data x: [0 1 2 3 4]
Output data y: [True False False False False]
```

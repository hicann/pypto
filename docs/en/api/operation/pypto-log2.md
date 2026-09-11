# pypto.log2

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:21:46.238Z pushedAt=2026-09-05T07:36:26.345Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs a base-2 logarithm operation on **input**.

## Prototype

```python
pypto.log2(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Supported dimensions: 1 to 4.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | PrecisionType, optional | Precision mode of the logarithm operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss. |

## Return Value

Returns the output tensor, whose data type is the same as that of **input** and whose shape has the same size as **input**.

## Constraints

1. The input tensor and the output tensor must have the same type.
2. Input elements must be greater than 0; otherwise, the output is NaN.

## TileShape Setting Example

The dimensions of the TileShape must be the same as those of the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Examples

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log2(x)
```

The results are as follows:

```python
Input data x: [1.0     2.0     3.0]
Output data y: [0.0000 1.0000 1.5849]
```

### High-Precision Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.INTRINSIC)
```

# pypto.exp

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:44:12.040Z pushedAt=2026-09-05T08:31:14.514Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the exponential of **e** for each element in the input tensor element-wise, and returns a tensor with the same shape as the input.

## Prototype

```python
pypto.exp(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| precision_type | PrecisionType, optional | Precision mode of the exponential operation. The default value is **PrecisionType.INTRINSIC**.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss. |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of **input**.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp(x)
```

The results are as follows:

```python
Input data x: [0.0    1.0    2.0]
Output data y: [1.0000  2.7183  7.3891]
```

### High-Precision Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.INTRINSIC)
```

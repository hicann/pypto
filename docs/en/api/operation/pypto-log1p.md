# pypto.log1p

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:21:28.066Z pushedAt=2026-09-05T07:36:26.344Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the natural logarithm (base e) of 1 + **input**.

## Prototype

```python
log1p(input: Tensor) -> Tensor:
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input    | Input      | Source operand.<br>Supported types: Tensor.<br>Supported data types for Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type is the same as that of **input** and whose shape has the same size as **input**.
## TileShape Setting Example

The dimensions of the TileShape must be the same as those of the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log1p(x)
```

The results are as follows:

```python
Input data x: [1e-99]
Output data y: [1e-99]
```

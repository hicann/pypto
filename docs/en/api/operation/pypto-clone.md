# pypto.clone

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:31:51.129Z pushedAt=2026-09-05T08:30:49.665Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Copies the input source data and returns the copy.

## Prototype

```python
clone(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported data types: data types supported by PyPTO.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor with the same shape and data type as the input.

## Constraints

1. The input tensor and output tensor must have the same type.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.clone(x)
```

The results are as follows:

```python
input x: [[1, 2],
          [3, 4]]
output y: [[1, 2],
           [3, 4]]
```

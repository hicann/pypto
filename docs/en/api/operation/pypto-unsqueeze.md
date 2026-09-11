# pypto.unsqueeze

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T09:06:00.419Z pushedAt=2026-09-05T07:36:26.383Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Adds a dimension to the input tensor.

## Prototype

```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                          |
|------------|--------------|----------------------------------------------------------------------|
| **input**  | Input        | Source operand.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| **dim**    | Input        | Position (index) at which the new dimension is inserted.<br>Negative indexes are supported.<br>Must be within the range [-input.dim - 1, input.dim]. |

## Return Value

Returns the output tensor with a new dimension of size 1 added at the specified dimension **dim**. The output tensor shares data with the input tensor and has identical attributes.

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.unsqueeze(x, 0)
```

The results are as follows:

```python
Output data x: [[1, 2, 3],
            [4, 5, 6]]
Output data y: [[[1, 2, 3],
             [4, 5, 6]]]
```

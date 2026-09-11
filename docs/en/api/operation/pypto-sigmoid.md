# pypto.sigmoid

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:50:18.692Z pushedAt=2026-09-05T07:36:26.369Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Applies the sigmoid activation function to each element of the input tensor. The calculation formula is as follows:t

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

## Prototype

```python
sigmoid(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the results of mapping the input elements to the \(0, 1\) interval through the sigmoid function.

## Examples

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sigmoid(x)
```

The results are as follows:

```python
Input data x: [-3.0, 0.0, 2.0, 5.0]
Output data y: [0.0474, 0.5000, 0.8808, 0.9933]
```

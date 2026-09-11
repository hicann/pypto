# pypto.softmax

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:54:11.712Z pushedAt=2026-09-05T07:36:26.372Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Applies the softmax function to the input tensor along a specified dimension, normalizing the elements of that dimension into a probability distribution with values in the range \[0, 1\] (the sum of all elements is 1). The calculation formula is as follows:

$$
\text{softmax}(input)_i = \frac{e^{input_i - \text{max}(input)}}{\sum_{j=1}^{k} e^{input_j - \text{max}(input)}}
$$

## Prototype

```python
softmax(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim       | Input        | Dimension along which normalization is performed.<br>Negative indexing is supported (for example, -1 indicates the last dimension).<br>Must be within the range [-input.dim, input.dim-1]. |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and the sum of the elements along the dimension specified by **dim** is 1.

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.softmax(x, -1)
```

The results are as follows:

```python
Input data x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
Output data y: [[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
```

# pypto.atanh

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:18:39.072Z pushedAt=2026-09-05T08:30:18.565Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the inverse hyperbolic tangent value of each element in the input tensor, performing element-wise operations.

$$
y_i = \text{atanh}(x_i) = \frac{1}{2} \ln\left(\frac{1 + x_i}{1 - x_i}\right)
$$

## Prototype

```python
atanh(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|:--------|:---------|:------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**).|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the inverse hyperbolic tangent values of the corresponding elements of the input tensor.

## Constraints

When the input exceeds ±1, the output is NaN; when the input is ±1, the output is ±inf.

Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.atanh(x)
```

The results are as follows:

```python
Input data x: [0.0000, 0.5000, -0.5000, 0.8000]
Output data y: [0.0000, 0.5493, -0.5493, 1.0986]
```

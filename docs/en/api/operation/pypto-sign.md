# pypto.sign

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:51:31.546Z pushedAt=2026-09-05T07:36:26.371Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the sign value of each element in the input tensor, element by element. The calculation formula is as follows:

$$
\text{sign}(x) = \begin{cases}
-1 & \text{if } x < 0 \\
0 & \text{if } x = 0 \\
1 & \text{if } x > 0
\end{cases}
$$

## Prototype

```python
sign(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size is not greater than **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the sign values (-1, 0, or 1) of the corresponding elements of the input tensor.

## Constraints

1. The TileShape must have the same dimension count as the input.
2. Due to temporary memory usage, when the input data type is DT\_INT8, the TileShape size has an additional constraint. Assuming that TileShape is \[a,b,c,d\], then a\*b\*c\*d\*sizeof\(self\) + c\*d\*sizeof\(INT8\) < UB.
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
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sign(x)
```

The results are as follows:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [-1.0, 0.0, 1.0, 1.0, -1.0]
```

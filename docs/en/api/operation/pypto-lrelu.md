# pypto.lrelu

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:25:09.814Z pushedAt=2026-09-05T07:36:26.347Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Applies the Leaky ReLU (leaky rectified linear unit) activation function to each element of the input tensor. The computation formula is as follows:

$$
\text{res}_i =
\begin{cases}
\text{input}_i & \text{if } \text{input}_i \geq 0 \\
\text{negative\_slope} \cdot \text{input}_i & \text{if } \text{input}_i < 0
\end{cases}
$$

Here, `negative_slope` is the negative slope parameter, with a default value of `0.01`.

## Prototype

```python
lrelu(input: Tensor, negative_slope: Union[float, Element] = 0.01) -> Tensor
```

## Parameters

| Parameter      | Input/Output | Description                                                                 |
|----------------|--------------|----------------------------------------------------------------------|
| input          | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| negative_slope | Input        | Slope coefficient of the negative interval.<br>Supported type: float\Element. The default value is `0.01` (float type). When it is of the float type, it is automatically converted to the Element type, where float corresponds to DT_FP32. To use other data types, construct it through Element.<br>Must be a non-negative real number (≥ 0). Special values such as `nan` and `inf` are not supported. |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of **input**.

## Constraints

1. The data type of **input** must be DT_FP16, DT_BF16, or DT_FP32.
2. **negative_slope** must be a non-negative floating-point number (≥ 0) and must not be `nan` or `inf`.
3. It is recommended to use Element for **negative_slope** and pass a float scalar. For the fp16 scenario, correctness is not guaranteed.
4. In-place operations are not supported (that is, the output cannot share memory with the input).
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
a = pypto.tensor([[-1.0, 0.0, 1.0]], pypto.DT_FP32)
out = pypto.lrelu(a)
```

The results are as follows:

```python
Input data a:   [[-1.0  0.0  1.0]]
Output data out: [[-0.01  0.0   1.0]]
```

# pypto.remainder

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:43:35.606Z pushedAt=2026-09-05T07:36:26.363Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs a remainder operation on each element of **input** and the element at the corresponding position in **other**. The calculation formula is as follows:

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## Prototype

```python
remainder(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor:
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported types: **Tensor**, **int**, and **float**.<br>Supported data types of **Tensor**: **DT_FP32**, **DT_FP16**, **DT_BF16**, **DT_INT32**, and **DT_INT16**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and can be broadcast to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other  | Input      | Source operand.<br>Supported types: **Tensor**, **int**, and **float**.<br>Supported data types of **Tensor**: **DT_FP32**, **DT_FP16**, **DT_BF16**, **DT_INT32**, and **DT_INT16**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and can be broadcast to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | Input      | Precision mode enumeration, used to control the precision mode of the remainder calculation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to **HIGH_PRECISION** (high-precision mode). |

## Return Value

Returns the output tensor, whose shape is the size after broadcasting **input** and **other**, and whose data type is the same as that of the input tensor.

## Constraints

1. Mixed-precision inputs are not supported. That is, when all inputs are tensors, their data types must be the same; when one input is a scalar, the data type of the tensor must be the corresponding integer type (**DT_INT32** or **DT_INT16**) or floating-point type (**DT_FP32**, **DT_FP16**, or **DT_BF16**).
2. When **input** is of an integer data type, **other** must not contain 0. The result of integer remainder is determined by the chip type and may be 0 or -1.
3. If the data type of the input tensor is **DT_INT32** and the data range exceeds \[-2^24, 2^24\], precision is not guaranteed.
4. The high-precision mode is currently effective only on Ascend 950PR/Ascend 950DT. On other products, the instruction mode `INTRINSIC` is used by default at the underlying layer.
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

The results are as follows:

```python
Input data a:    [7.0, 8.0, 9.0]
Input data b:    [-3.0, -3.0, -3.0]
Output data out:  [-2.0, -1.0, 0.0]
```

### High-Precision Mode Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP16)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP16)
out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b, pypto.PrecisionType.INTRINSIC)
```

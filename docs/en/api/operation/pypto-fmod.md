# pypto.fmod

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:01:10.838Z pushedAt=2026-09-05T07:36:26.330Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs the modulo operation on each element of **input** and the element at the corresponding position in **other**. The computation formula is as follows:

$$
res_i = input_i \;\%\; other_i
$$

## Prototype

```python
fmod(input: Tensor, other: Union[Tensor, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other  | Input      | Source operand.<br>Supported types: float and Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports broadcasting to the same shape along a single dimension. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | Input | Precision mode enumeration, used to control the precision mode of the modulo operation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to **HIGH_PRECISION** (high-precision mode). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. The data types of **input** and **other** must be the same.
2. When **other** is a number, implicit conversion is not supported.
3. **other** does not support special values such as **nan** and **inf**.
4. The high-precision mode is currently effective only on Ascend 950PR/Ascend 950DT. On other products, the underlying layer uses the instruction mode `INTRINSIC` by default.
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b)
```

The results are as follows:

```python
Input data a:    [[7.0 8.0 9.0]]
Input data b:    [[3.0 3.0 3.0]]
Output data out:  [[1.0 2.0 0.0]]
```

### High-Precision Mode Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.INTRINSIC)
```

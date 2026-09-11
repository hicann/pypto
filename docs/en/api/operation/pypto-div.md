# pypto.div

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:40:53.667Z pushedAt=2026-09-05T08:31:07.601Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Divides each element of **input** by the element at the corresponding position in **other**. The computation formula is as follows:

$$
res_i = input_i \div other_i
$$

## Prototype

```python
div(input: Tensor, other: Union[Tensor, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. Supported dimensions: 1 to 4. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other  | Input      | Source operand.<br>Supported types: float and Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. Supported dimensions: 1 to 4. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | Input | Precision mode enumeration, used to control the precision mode of the division computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to **HIGH_PRECISION** (high-precision mode). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. When **other** is a scalar, if **input** is of a floating-point type, the scalar supports integer types (automatically converted to floating-point); if **input** is of an integer type, the scalar does not support floating-point types (an error is reported).
3. **Precision mode description**:
    - **HIGH_PRECISION (high-precision mode)**: Default mode. A higher-precision computation method is used in the underlying implementation. Currently, this mode is effective only on Ascend 950PR/Ascend 950DT.
    - **INTRINSIC (instruction mode)**: Directly uses chip instructions for computation.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.


## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

#### Basic Usage (High-Precision Mode by Default)

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b)  # Use the HIGH_PRECISION mode by default.
```

The results are as follows:

```python
Input data a:    [[2.0 4.0 6.0]]
Input data b:    [[2.0 2.0 2.0]]
Output data out:  [[1.0 2.0 3.0]]
```

#### Explicitly Specifying the High-Precision Mode

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

#### Using the Instruction Mode

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)
```

#### Using Operators (Automatically Using the HIGH_PRECISION Mode)

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = a / b  # Automatically use HIGH_PRECISION mode.
out = a.div(b)  # Automatically use HIGH_PRECISION mode.
```

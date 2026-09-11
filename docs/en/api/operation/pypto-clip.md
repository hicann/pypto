# pypto.clip

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:30:52.584Z pushedAt=2026-09-05T08:30:46.628Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Clips the input tensor to the specified minimum and maximum value range. Elements smaller than the minimum are replaced with the minimum, elements greater than the maximum are replaced with the maximum, and all other values remain unchanged. This API is non-in-place; it does not modify the input tensor, but returns a new tensor as the output.

## Prototype

```python
clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
)-> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Data types supported by Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The total number of elements must not exceed **INT32_MAX**. |
| min    | Input      | Source operand.<br>Supported types: int\float\Element and Tensor. The data type must be consistent with that of the input.<br>When an `int` or `float` value is passed, it is automatically converted to the `Element` type, with the same data type as that of the input tensor. When other data types are needed, they can be constructed explicitly via `Element`.<br>Data types supported by Tensor and Element: DT_FP32, DT_FP16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The total number of elements must not exceed **INT32_MAX**.<br>Optional, with a default value of **-INF**.<br>**NaN**, **INF**, and **-INF** are defined only in floating-point operations, that is, they take effect only when the data type is DT_FP16/DT_FP32. When the data type is DT_INT16 or DT_INT32, the comparison logic of the default value is skipped. |
| max    | Input      | Source operand.<br>Supported types: int\float\Element and Tensor. The data type must be consistent with that of the input.<br>When an `int` or `float` value is passed, it is automatically converted to the `Element` type, with the same data type as that of the input tensor. When other data types are needed, they can be constructed explicitly via `Element`.<br>Data types supported by Tensor and Element are: DT_FP32, DT_FP16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The total number of elements must not exceed **INT32_MAX**.<br>Optional, with a default value of **INF**.<br>**NaN**, **INF**, and **-INF** are defined only in floating-point operations, that is, they take effect only when the data type is DT_FP16/DT_FP32. When the data type is DT_INT16 or DT_INT32, the comparison logic of the default value is skipped. |

## Return Value

When the input is a scalar, the output is:

$$
Y_{i} = \text{MIN}\left( \text{MAX}\left(X_{i}, \text{min\_value}\right), \text{max\_value} \right)
$$

When the input is a tensor, the output is:

$$
Y_{i} = MIN\left( MAX(X_{i}, min\_value_{i}), max\_value_{i} \right)
$$

The data type of the output tensor is the same as that of the input.

When either min or max is NAN, the output result is NAN.

When min \> max, all elements in the output tensor are set to max.

## Constraints

1. **min** and **max** must have the same type, both being Element or both being Tensor.
2. When **min**/**max** are of the Tensor type, their shape sizes must be broadcastable to the input shape.
3. **min** and **max** support being omitted simultaneously, in which case the original value is returned.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **max** and **min** are [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **max** and **min** are [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = pypto.tensor([2,3], pypto.DT_INT32)
max = pypto.tensor([2,3], pypto.DT_INT32)
out = pypto.clip(x,min,max)
```

The results are as follows:

```python
Input data self: [[-2 1 2], [3 4 5]]
Input data min: [[-1 0 2], [0 3 5]]
Input data max: [[1 2  1], [4 4 4]]
Output data out: [[-1 1 1], [3 4 4]]
```

Example 2:

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = 1
max = 3
out = pypto.clip(x,min,max)
```

The results are as follows:

```python
Input data x: [[0 2 4], [3 4 6]]
Output data out: [[1 2 3], [3 3 3]]
```

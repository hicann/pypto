# pypto.ones

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:34:38.795Z pushedAt=2026-09-05T07:36:26.355Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a tensor of the specified `size` filled with ones. The data type is specified by `dtype` and defaults to `DT_FP32`.

## Precautions

- **TileShape must be set first**: Before calling this API, you must set the TileShape through [set_vec_tile_shapes](../config/pypto-set_vec_tile_shapes.md).
- **The dtype parameter must be passed explicitly**: When a data type needs to be specified, it must be passed explicitly using the keyword argument `dtype=`, and cannot be passed as a positional argument. For example, use `pypto.ones(2, 3, dtype=pypto.DT_INT32)` instead of `pypto.ones(2, 3, pypto.DT_INT32)`. If passed as a positional argument, the dtype value is mistakenly parsed as a dimension of `size`, causing an error.

## Prototype

```python
ones(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## Parameters

| Parameter    | Input/Output | Description                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| *size        | Input      | Source operand that defines the shape of the output tensor.<br>Supports variable-length arguments (multiple ints) or a single sequence (such as List[int] or Tuple[int]). |
| dtype        | Input      | Source operand, an optional parameter that defines the data type of the output tensor.<br>Supported data types: `DT_FP32`, `DT_INT32`, `DT_INT16`, `DT_FP16`, and `DT_BF16`.<br>The default value is `pypto.DT_FP32`. |

## Return Value

Returns an output tensor. The data type of the tensor is determined by `dtype`, its shape is the size specified by `size`, and all elements are set to `1`.

## Constraints

1. The dimensions of `tileshape` must be the same as those of the output result, and `tileshape` is used to tile the result.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`. The TileShape dimensions must be the same as those of the output.
For example, if the input size is `[m, n]`, the output is `[m, n]`, and the TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(2, 3)
```

### API Call Example

```python
# Example 1: Pass size using variable-length arguments and use the default dtype (DT_FP32).
x1 = pypto.ones(2, 3)

# Example 2: Pass size using a list and explicitly specify dtype (DT_INT32).
x2 = pypto.ones([2, 3], dtype=pypto.DT_INT32)
```

The results are as follows:

```python
x1 output data: [[1., 1., 1.],
             [1., 1., 1.]]
x2 output data: [[1, 1, 1],
             [1, 1, 1]]
```

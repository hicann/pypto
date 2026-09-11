# pypto.cumsum

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:37:52.578Z pushedAt=2026-09-05T08:31:02.009Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the cumulative sum of the input tensor along the specified dimension.

## Prototype

```python
cumsum(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                              |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input  | Input      | Source operand. Supported type: Tensor. Supported data types of Tensor: DT_FP16, DT_BF16, DT_INT16, DT_INT32, and DT_FP32. Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim    | Input      | Dimension along which the accumulation is performed. Type: int.                                                                                                                                                                       |

## Return Value

The output tensor has the same shape as the input.
When the input is of type DT_FP16, DT_BF16, DT_FP32, or similar, the output data type is the same as that of the input. When the input is of type DT_INT16 or DT_INT32, the output data type is DT_INT64.

## Constraints

1. **dim**: Specifies the dimension along which the cumulative sum is computed. It must be within the valid dimension range of the input tensor, and its value must satisfy -input.dim <= dim < input.dim.
2. The ViewShape of the input along the **dim** axis cannot be tiled, while no restrictions apply to other dimensions.
3. The TileShape has the same dimensions as the input. The total size of all TileShapes for both inputs and outputs must not exceed the UB memory capacity.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
input = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
dim = 0
y = pypto.cumsum(input, dim)
```

The results are as follows:

```python
Input data x:   [[0 1 2],
               [3 4 5]]
Output data y:   [[0 1 2],
               [3 5 7]]                             # shape (2, 3)
```

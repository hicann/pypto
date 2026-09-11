# pypto.tril

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T09:01:50.541Z pushedAt=2026-09-05T07:36:26.380Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Returns the lower triangular part of a 2D tensor or a batch of tensors. The other elements of the resulting tensor are set to 0.

## Prototype

```python
tril(input: Tensor, diagonal: SymInt = 0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                        |
| --------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32, and DT_INT8.<br>Empty tensors are not supported. The shape supports only 2 to 5 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| diagonal  | Input        | Diagonal offset, defaulting to **0** (the main diagonal).<br>SymInt type.                                                                                                                                                              |

## Return Value

Returns a tensor with the same shape and data type as the input.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)        # shape (3, 3)
diagonal = 0
out = pypto.tril(x, diagonal)
```

The results are as follows:

```python
Input data x :[[1 2 3],
             [4 5 6],
             [7 8 9]]
Output data out:[[1 0 0],
             [4 5 0],
             [7 8 9]]                             # shape (3, 3)
```

# pypto.scatter\_update

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:49:11.881Z pushedAt=2026-09-05T07:36:26.370Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Function 1: An in-place operation that updates the 4D **src** to the 4D **input** based on the 2D index **index**. The calculation formula is as follows:

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

Function 2: An in-place operation that updates the 2D **src** to the 2D **input** based on the 2D **index**. The calculation formula is as follows (where s is the size of the second dimension of **index**, that is, **index.shape[1]**):

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## Prototype

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Supported type is: **Tensor**.<br>Supported data type of Tensor is: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16.<br>Supported dimensions: 2D, 4D.<br>2D shape [blockNum * blockSize, d], 4D shape [blockNum, blockSize, 1, d].<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim     | Input      | Keep the default value -2. |
| index   | Input      | A set of indices of **input**.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT64, DT_INT32, and DT_INT16.<br>Supported dimensions: 2D.<br>Shape [b, s]. |
| src     | Input      | **src** is a set of update values.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16. The data type must be the same as that of **input**.<br>Supported dimensions: 2D, 4D.<br>2D shape [b * s, d], 4D shape [b, s, 1, d].<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the updated **input**. This is an in-place operation.

## Constraints

Broadcast constraint: broadcast is not supported.

Tensor format constraint: Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

ViewShape constraint: In 2D scenarios, the ViewShape is \[viewB \* s, d\]; in 4D scenarios, the ViewShape is \[viewB, viewS, 1, d\]. The last axis d is not tiled. In 2D scenarios, \[viewB \* s, d\] performs tiling on src, whose 0th dimension is a multiple of s, the 1st dimension of index; \[viewB, S\] performs tiling on index. In 4D scenarios, \[viewB, viewS, 1, d\] performs tiling on src, and \[viewB, viewS\] performs tiling on index.

TileShape constraint: In 2D scenarios, the TileShape is \[tileS, d\]; in 4D scenarios, the TileShape is \[tileB, tileS, 1, d\]. The last axis d is not tiled. In 2D scenarios, the TileShape performs tiling on src, and \[1,tileS\] performs tiling on index. tileS is a divisor of s, the 1st dimension of the input index. For example, if src is \[12, 64\], index is \[3, 4\], and the TileShape is \[TileS, 64\], then TileS can be 1, 2, or 4. In 4D scenarios, the TileShape performs tiling on src, and \[tileB, tileS\] performs tiling on index. Because the TileShape tiling applies to both src and index, the sum of the tile sizes must be less than the UB limit.

2D example: input: [16, 8], index: [5, 2], src: [10, 8], viewShape: [viewB \* s, 8]. viewB must be an integer, that is, the 0th dimension must be a multiple of s. tileShape: [tileS, 8]. tileS must be a divisor of s, that is, 1 or 2.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with the input src.

Both the input and the output reside in gm and do not involve tiling. The input index and the input src must be moved into ub, which involves tiling.

For example, if the input is [t, d], dim is -2, the input index is [b, s], the input src is [bs, d], where bs=b*s, and the output is [t, d], with the TileShape set to [bs1, d1], then bs1 is used to tile the bs axis, the d axis is not allowed to be tiled, and d1 must be equal to d.

```python
pypto.set_vec_tile_shapes(16, 64)
```

### API Call Example

- Update the 2D input with the 2D src according to the 2D index. Note the in-place operation syntax: the output on the left side of the equal sign must be the same as the input:

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    The results are as follows:

    ```python
    Input data x:[[0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0]]
    Input data y:[[1 2],
               [4 5]]
    Input data z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    Output data x:[[0 0 0],
               [1 2 3],
               [4 5 6],
               [0 0 0],
               [7 8 9],
               [10 11 12],
               [0 0 0],
               [0 0 0]]
    ```

- Update the 4D input with the 4D src according to the 2D index. Note the in-place operation syntax: the output on the left side of the equal sign must be the same as the input:

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    The results are as follows:

    ```python
    Input data x:[[
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ]]
    Input data y:[[1 8],
               [4 10]]
    Input data z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    Output data x:[[
                 [[0 0 0]],
                 [[1 2 3]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[7 8 9]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[4 5 6]],
                 [[0 0 0]],
                 [[10 11 12]],
                 [[0 0 0]],
               ]]
    ```

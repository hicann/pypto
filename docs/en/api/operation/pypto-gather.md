# pypto.gather

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:03:31.850Z pushedAt=2026-09-05T07:36:26.331Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Extracts values from the input tensor along the specified dimension `dim` at the given indices `index`, and returns the result. For example, for a 3D tensor, the computation is as follows:

$$
\begin{cases}
output[i,j,k] = input[index[i,j,k], j, k] & \text{if } dim = 0; \\
output[i,j,k] = input[i, index[i,j,k], k] & \text{if } dim = 1; \\
output[i,j,k] = input[i,j, index[i,j,k]] & \text{if } dim = 2.
\end{cases}
$$

## Prototype

```python
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| dim     | Input      | Dimension for which the index is specified.<br>Supports any valid dimension index, ranging from -input.dim to input.dim - 1. |
| index   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size on each axis of **index** must not exceed the corresponding shape size of **input**, and the values must be valid indices, that is, they must not exceed the shape size of **input** on the **dim** axis. |

## Return Value

Returns an output tensor. The data type of the output tensor is the same as that of `input`, and its shape is identical to that of `index`.

## Constraints

1. index.dim = input.dim, and index.shape\[i\] <= input.shape\[i\] (i != dim), with values being valid indices, that is, they must not exceed input.shape\[dim\].

2. dim: -input.dim <= dim < input.dim.

3. The **dim** axis of input.shape cannot be tiled, requiring viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \), while the shape sizes of the remaining dimensions are not restricted.

4. The dimensions of the TileShape are the same as those of **index**, and are used to tile **input** and **index**. The **dim** axis of **input** cannot be tiled, and the total size of all input and output TileShape values must not exceed the size of the UB memory.

5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

For example, if the input **input** is [x, y, z], **dim** is 1, the input **index** is [m, t, p], and the output is [m, t, p], where m <= x and p <= z, and the TileShape is set to [m1, t1, p1], then m1, t1, and p1 are used to tile the m, t, and p axes, respectively. The y axis cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

The results are as follows:

```python
Input data x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
Output data y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```

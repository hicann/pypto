# pypto.experimental.gather\_in\_ub

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:53:56.124Z pushedAt=2026-09-05T07:36:26.322Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This is a custom API with many constraints. Its stability is not guaranteed.

This operator supports the sparse attention mechanism. It loads the KV cache of the selected tokens from Global Memory (GM) to Unified Buffer (UB), and supports Page Attention.

## Prototype

```python
gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor,
                 block_size: int, axis: int) -> Tensor
```

## Parameters

| Parameter    | Input/Output | Description                                                                 |
|--------------|--------------|-----------------------------------------------------------------------------|
| param        | Input        | Source operand.<br>Supported data types: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>In practice, it represents the KV cache, whose shape is [token_size, hidden_dim]. |
| indices      | Input        | Source operand.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>In practice, it represents the top k output result, whose shape is [1, k]. |
| block_table  | Input        | Source operand.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>In practice, it represents the page table in Page Attention, whose shape is [1, block_table_size], where **block_table_size** indicates the length of the page table. |
| block_size   | Input        | Source operand.<br>Type: int.<br>Number of tokens that one block can hold in Page Attention. |
| axis         | Input        | Source operand.<br>Type: int.<br>Only the -2 axis is supported. |

## Return Value

Returns the output tensor. The data type of the tensor is the same as that of **param**, and its shape is \[k, hidden\_dim\], that is, the KV cache of the selected tokens.

## Constraints

1. This is a custom API, and its stability is not guaranteed.
2. **param**, **indices**, and **block_table** do not support empty tensors and support only two-dimensional tensors.
3. The shape of **indices** must be [1, k].
4. The **axis** parameter supports only -2.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimension settings of the TileShape must be consistent with the output tensor and are used to control the size of the output tile block.

Take the input **$ param[token_size,hidden_dim]$** , the index **$indices[1,k]$** , the axis **$\text{axis}=-2$** , and the output **$output[k,hidden_dim]$**  as an example:

Set the TileShape to **$[k_1, hidden_dim_1]$**. This configuration directly applies to each dimension of the output, and is mapped to the input and the index. Here, **$k_1$** tiles the k dimension of indices, and **$ hidden_dim_1$** tiles the feature dimension **$hidden_dim$** of **param**. The tile memory usage must satisfy the constraint **$b_1 \cdot k_1 \cdot hidden_dim_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UB_Size}$**.

### API Call Example

![](../figures/zh-cn_image_0000002524825989.png)

Consider the preceding scenario, where **indices** is the top k result, **block\_table** is the page table of Page Attention, **param** is the KV cache, and **block\_size** is **2**. The final result is the collection of the KV cache of the tokens.

Take **token id** 4 as an example (marked in red in the figure). The actual offset is computed based on **blockSize**:

blockIdx = 4 / 2; // Compute the corresponding logical block, which is the second logical block.

tail = 4 % 2;        // Compute the intra-block offset, which is 0.

slcBlockIdx = blockTable\[0, blockIdxInBatch\];  // Look up the table to obtain the actual offset of this block, which corresponds to the first physical block.

offsets = slcBlockIdx \* blockSize + tail;// Compute the actual offset, which is 2.

Move the data.

```python
param = pypto.tensor([6, 4], pypto.DT_FP32)
indices = pypto.tensor([1, 3], pypto.DT_INT32)
blockTable = pypto.tensor([1, 3], pypto.DT_INT32)
blockSize = 2
axis = -2
result = pypto.experimental.gather_in_ub(param , indices , blockTable, blockSize , axis)
```

The results are as follows:

```python
Input data param :
[
  # token 0
  [  0,  1,  2,  3],
  # token 1
  [ 10, 11, 12, 13],
  # token 2
  [ 20, 21, 22, 23],
  # token 3
  [ 30, 31, 32, 33],
  # token 4
  [ 40, 41, 42, 43],
  # token 5
  [ 50, 51, 52, 53],
]
Input data indices : [0, 4, 3]
Input data blockTable : [0, 2, 1]
Output data out:
[
   [  0,  1,  2,  3],
   [ 20, 21, 22, 23],
   [ 50, 51, 52, 53],
]
```

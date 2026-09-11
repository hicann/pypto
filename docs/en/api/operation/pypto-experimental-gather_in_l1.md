# pypto.experimental.gather\_in\_l1

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:53:54.766Z pushedAt=2026-09-05T07:36:26.321Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This is a custom API with many constraints, and its stability is not guaranteed.

Discretely copies the data of specified rows from a tensor in GM to L1, copying the first **size** elements of each row.

## Prototype

```python
gather_in_l1(src: Tensor, indices: Tensor, block_table: Tensor, block_size: int,
                 size: int, is_b_matrix: bool, is_trans: bool) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------------|-----------|----------------------------------------------------------------------|
| **src** | Input | Source operand.<br>Supported data types: DT_FP32, DT_FP16, DT_BF16, and DT_INT8.<br>Empty tensors are not supported. Two dimensions are supported. |
| **indices** | Input | Row offset of the source operand.<br>Supported data types: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. Two dimensions are supported.<br>The shape is [1,n]. |
| **block_table** | Input | Source operand.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two dimensions are supported.<br>In practice, it represents the page table in Page Attention, with a shape of [1,block_table_size], where **block_table_size** is the length of the page table. |
| **block_size** | Input | Source operand.<br>Type: int.<br>Number of tokens that one block can hold in Page Attention. |
| **size** | Input | Number of elements copied per row.<br>The number of elements must be less than the number of columns of the source operand. |
| **is_b_matrix** | Input | Whether the copied result, that is, the output tensor, is used as the B matrix of matmul. |
| **is_trans** | Input | Whether the copied result, that is, the output tensor, is transposed. |

## Return Value

Returns the output tensor.

## Constraints

None

## Example

```python
src = pypto.tensor([16, 32], pypto.DT_FP32, "tensor_src")
offset = pypto.tensor([1, 32], pypto.DT_INT32, "tensor_offset")
block_table = pypto.tensor([1, 4], pypto.DT_INT32, "block_table")
block_size = 2
size = 16
is_b_matrix = False
is_trans = False
out = pypto.experimental.gather_in_l1(src, offset, block_table, block_size, size, is_b_matrix, is_trans)
```

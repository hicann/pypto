# pypto.experimental.transposed_batchmatmul

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:56:22.583Z pushedAt=2026-09-05T07:36:26.324Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This is a custom API with many constraints. Its stability is not guaranteed.

This operator performs transposed batch matrix multiplication. The specific operations are as follows:

1. Transpose the input tensor `tensor_a` from shape (M, B, K) to (B, M, K).
2. Perform batch matrix multiplication by multiplying the transposed `tensor_a` (B, M, K) with `tensor_b` (B, K, N) to obtain the intermediate result (B, M, N).
3. Transpose the intermediate result back to shape (M, B, N) as the final output.

## Prototype

```python
transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype: dtype) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| tensor_a  | Input        | Left input tensor.<br>Supported data types: DT_FP16 and DT_BF16.<br>Empty tensors are not supported; three-dimensional tensors are supported.<br>The shape must be (M, B, K). |
| tensor_b  | Input        | Right input tensor.<br>Supported data types: DT_FP16 and DT_BF16.<br>Empty tensors are not supported; three-dimensional tensors are supported.<br>The shape must be (B, K, N). |
| out_dtype | Input        | Data type of the output tensor.<br>Supported data types: DT_FP16 and DT_BF16. |

## Return Value

Returns the output tensor, whose data type is specified by `out_dtype` and whose shape is (M, B, N).

## Examples

```python
import pypto

# Create the input tensor.
a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")

# Call the operator.
c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)

# The shape of the output tensor c is (16, 2, 64).
```

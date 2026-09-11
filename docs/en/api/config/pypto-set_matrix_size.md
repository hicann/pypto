# pypto.set\_matrix\_size

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:56.267Z pushedAt=2026-08-26T09:10:38.131Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

When an NZ-format input tensor is reshaped and then used in **matmul**/**scaled_mm** for computation, the shape values of the **m**, **k**, and **n** dimensions of the original **Tensor** (before reshaping) must be passed in so that **matmul**/**scaled_mm** can obtain the original **m**, **k**, and **n** dimension shape values.

## Prototype

```python
set_matrix_size(size: List[int])-> None
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| **size** | Input | Shape values of the **m**, **k**, and **n** dimensions of the input tensor. |

## Return Value

void

## Constraints

- When an NZ-format tensor is reshaped and then used in **matmul**/**scaled_mm** computation, this parameter must be set.

- When the inputs to **matmul**/**scaled_mm** are 3D/4D NZ-format tensors, this parameter must be set.

## Example

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) #Shape values of the m, k, and n dimensions of the corresponding input tensor.
out = pypto.matmul(a, b, pypto.DT_FP32)
```

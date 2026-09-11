# pypto.Tensor.dim

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:42:28.467Z pushedAt=2026-08-28T11:36:17.371Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the dimension of a tensor.

## Prototype

```python
dim(self) -> int
```

## Parameters

None

## Return Value

Returns the dimension of a tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((2, 3, 4), pypto.DT_FP32)
out = t.dim
```

The results are as follows:

```python
Output data out: 3
```

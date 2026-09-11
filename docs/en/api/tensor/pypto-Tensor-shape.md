# pypto.Tensor.shape

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:55:45.137Z pushedAt=2026-08-28T11:36:17.397Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Gets the tensor shape.

## Prototype

```python
shape(self) -> List[SymInt]
```

## Parameters

None

## Return Value

Returns the shape list of a tensor.

## Constraints

None

## Example

```python
t = pypto.tensor((16, 32), pypto.DT_FP32)
out = t.shape
```

The results are as follows:

```python
Output data out: [16, 32]
```

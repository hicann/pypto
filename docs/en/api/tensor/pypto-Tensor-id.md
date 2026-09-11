# pypto.Tensor.id

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:46:43.121Z pushedAt=2026-08-28T11:36:17.380Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the unique identifier of a tensor.

## Prototype

```python
id(self) -> int
```

## Parameters

None

## Return Value

Returns the unique identifier of a tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((4, 4), pypto.DT_FP32)
print(t.id)  # Output the ID of the Tensor.
```

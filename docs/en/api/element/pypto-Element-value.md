# pypto.Element.value

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:24:45.232Z pushedAt=2026-08-20T13:04:18.617Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains data.

## Prototype

```python
def value(self) -> int | float
```

## Parameters

N/A

## Return Value

Returns the data in **Element**.

## Constraints

Read-only attribute.

## Example

```python
t = pypto.element(pypto.DT_FP32, 3)
t.value
```

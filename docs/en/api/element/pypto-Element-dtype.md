# pypto.Element.dtype

<!-- md-trans-meta sourceCommit=d665ab95092c497ed3fc231aebc833bc88e9cfe6 translatedAt=2026-08-20T10:24:19.052Z pushedAt=2026-08-20T13:04:01.608Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the data type.

## Prototype

```python
def dtype(self) -> pypto.DataType
```

## Parameters

N/A

## Return Value

Returns the data type.

## Constraints

Read-only data.

## Example

```python
t = pypto.element(pypto.DT_FP32, 3)
t.dtype
```

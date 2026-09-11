# pypto.Element Constructor

<!-- md-trans-meta sourceCommit=d665ab95092c497ed3fc231aebc833bc88e9cfe6 translatedAt=2026-08-20T10:23:03.931Z pushedAt=2026-08-20T13:01:09.870Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates an element.

## Prototype

```python
def __init__(self, dtype, value) : ...
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| dtype | Input | Data type. For details, see <a href="../datatype/DataType.md">DataType</a>. |
| value | Input | Integer or floating-point number. |

## Return Value

Returns an element.

## Constraints

None

## Example

```python
t = pypto.Element(pypto.DT_FP32, 3)
```

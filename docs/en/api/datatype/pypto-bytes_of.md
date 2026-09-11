# pypto.bytes\_of

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:05:35.602Z pushedAt=2026-08-20T12:41:25.525Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Returns the size in bytes occupied by a data type.

## Prototype

```python
bytes_of(dtype: pypto.DataType) -> int
```

## Parameters

| Parameter | Input/Output | Description                              |
|-----------|--------------|------------------------------------------|
| **dtype** | Input        | Data type whose size in bytes is to be queried. |

## Return Value

Returns the size in bytes occupied by the data type.

## Constraints

None

## Example

```python
pypto.bytes_of(pypto.DT_FP32)
```

The result is as follows:

```python
Output: 4
```

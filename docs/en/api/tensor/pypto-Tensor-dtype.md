# pypto.Tensor.dtype

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:42:59.289Z pushedAt=2026-08-28T11:36:17.372Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the data type of a tensor.

## Prototype

```python
dtype(self) -> DataType
```

## Parameters

None

## Return Value

Returns the data type of a tensor.

## Constraints

None

## Example

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
out = t.dtype
```

The results are as follows:

```python
Output data out: DataType.DT_FP32
```

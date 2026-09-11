# pypto.Tensor.format

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:45:17.188Z pushedAt=2026-08-28T11:36:17.377Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the format of a tensor.

## Prototype

```python
format(self) -> TileOpFormat
```

## Parameters

None

## Return Value

**TileOpFormat**: Returns the format of a tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((4, 4), pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND)
print(t.format)
```

The results are as follows:

```text
Output: TileOpFormat.TILEOP_ND
```

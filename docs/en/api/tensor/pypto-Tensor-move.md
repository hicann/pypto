# pypto.Tensor.move

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:50:29.512Z pushedAt=2026-08-28T11:36:17.387Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Moves the data of one tensor to the current tensor.

## Prototype

```python
move(self, other: 'Tensor') -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| **other**   | Input      | Source tensor whose data is to be moved. |

## Return Value

None

## Constraints

None

## Example

```python
t1 = pypto.tensor((2, 3), pypto.DT_FP32)
t2 = pypto.tensor((2, 3), pypto.DT_FP32)
# Move the data of t2 to t1.
t1.move(t2)
```

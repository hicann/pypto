# pypto.Tensor.name

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:51:23.651Z pushedAt=2026-08-28T11:36:17.390Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains or sets the name of a tensor.

## Prototype

```python
name(self) -> str
name(self, value: str) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| value   | Input      | Name of the tensor to be set. |

## Return Value

Name of the tensor.

## Constraints

None

## Example

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
n1 = t.name
t.name = "my_tensor"
n2 = t.name
```

The results are as follows:

```python
Output n1: ""
Output n2: "my_tensor"
```

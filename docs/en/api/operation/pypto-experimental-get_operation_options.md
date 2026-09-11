# pypto.experimental.get\_operation\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-09-02T07:53:53.080Z pushedAt=2026-09-05T07:36:26.319Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the **operation** configuration.

## Prototype

```python
get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict** containing all configuration items of the **operation**.

## Constraints

1. The return value is of the **Dict** type and contains all configuration items of the **operation**.
2. Different configuration items may have different types, such as **str**, **int**, **List[int]**, and **Dict[int, int]**.

## Example

```python
pypto.get_operation_options()
```

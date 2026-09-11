# pypto.get\_pass\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:05:57.993Z pushedAt=2026-08-26T09:10:38.123Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the Pass optimization parameter information.

## Prototype

```python
get_pass_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict** containing all parameter information of **Pass**.

## Constraints

None

## Example

```python
pypto.get_pass_options()
```

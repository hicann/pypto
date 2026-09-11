# pypto.get\_verify\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:06:28.362Z pushedAt=2026-08-26T09:10:38.126Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the currently configured options of the precision debugging **Verify** feature.

## Prototype

```python
get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns the current settings of the **Verify** feature for precision debugging.

## Constraints

## Example

```python
pypto.get_verify_options()
```

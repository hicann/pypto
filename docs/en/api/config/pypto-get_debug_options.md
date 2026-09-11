# pypto.get\_debug\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:05:26.455Z pushedAt=2026-08-26T09:10:38.119Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the debug configurations.

## Prototype

```python
get_debug_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict**, containing all debug configuration items.

## Constraints

None

## Example

```python
pypto.get_debug_options()
```

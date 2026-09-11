# pypto.get\_codegen\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:04:55.599Z pushedAt=2026-08-26T09:10:38.115Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the **codegen** configuration.

## Prototype

```python
get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict** containing all configuration items of **codegen**.

## Constraints

None

## Example

```python
pypto.get_codegen_options()
```

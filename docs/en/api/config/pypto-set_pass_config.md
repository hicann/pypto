# pypto.set\_pass\_config

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:53.896Z pushedAt=2026-08-26T09:10:38.132Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Modifies the configuration information of a specified Pass.

## Prototype

```python
set_pass_config(strategy: str, identifier: str, key: PassConfigKey, value: bool)
```

## Parameters

| Parameter | Input/Output | Description |
|--------------|-----------|----------------------------------------------------------------------|
| **strategy** | Input | Pass strategy name, for example, **PVC2_OOO**. |
| **identifier** | Input | Pass name, for example, **ExpandFunction**. |
| **key** | Input | PassConfigKey enumeration. <br> KEY_DUMP_GRAPH: Dump the Pass computation graph. |
| **value** | Input | Value. |

## Return Value

None

## Constraints

- Setting timing: This API must be called before graph compilation starts.
- Scope: The configuration information is global and affects all subsequent compilation processes.

## Example

```python
pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

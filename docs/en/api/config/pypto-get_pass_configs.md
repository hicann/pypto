# pypto.get\_pass\_configs

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:05:56.891Z pushedAt=2026-08-26T09:10:38.122Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains all configuration information of a specified Pass.

## Prototype

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## Parameters

| Parameter  | Input/Output | Description                                                          |
|------------|--------------|----------------------------------------------------------------------|
| **strategy**   | Input        | Pass strategy name, for example, **PVC2_OOO**. |
| **identifier** | Input        | Pass name, for example, **ExpandFunction**. |

## Return Value

A **PassConfigs** object containing the following read-only attributes:

| Attribute          | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| **printGraph**         | Dumps the computation graph IR. |
| **dumpGraph**          | Dumps the computation graph. |
| **dumpPassTimeCost**   | Dumps the Pass execution time. |
| **preCheck**           | Performs validation before Pass execution. |
| **postCheck**          | Performs validation after Pass execution. |
| **disablePass**        | Skips the current Pass. |
| **healthCheck**        | Performs a health check and generates a report. |

## Constraints

None

## Example

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```

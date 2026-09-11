# pypto.get\_pass\_config

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:05:31.133Z pushedAt=2026-08-26T09:10:38.121Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the configuration information of the specified **Pass**.

## Prototype

```python
get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool
```

## Parameters

| Parameter       | Input/Output | Description                                                                 |
|-----------------|--------------|------------------------------------------------------------------------------|
| **strategy**    | Input        | Pass strategy name, for example, **PVC2_OOO**. |
| **identifier**  | Input        | Pass name, for example, **ExpandFunction**. |
| **key**         | Input        | **PassConfigKey** enumeration. <br> **KEY_DUMP_GRAPH** dumps the Pass computation graph. |
| **default_value** | Input      | If the configuration value of the specified key is not found, this default value is returned. |

## Return Value

Returns the configuration value of **key** in the Pass named **identifier** under the specified **strategy**.

## Constraints

**key** can only accept the specified enumeration values.

## Example

```python
pypto.get_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
```

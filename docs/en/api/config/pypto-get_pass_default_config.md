# pypto.get\_pass\_default\_config

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:05:57.223Z pushedAt=2026-08-26T09:10:38.121Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the default configuration information of a Pass.

## Prototype

```python
get_pass_default_config(key: PassConfigKey, default_value: bool) -> bool
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| key | Input | **PassConfigKey** enumeration. <br> **KEY_DUMP_GRAPH**: Dump the Pass computation graph. |
| default_value | Input | Default value returned when the configuration value for the specified key is not found. |

## Return Value

Returns the default configuration value of the Pass named **key**, or **default_value** if it does not exist.

## Constraints

**key** can only accept restricted enumerated values.

## Example

```python
pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

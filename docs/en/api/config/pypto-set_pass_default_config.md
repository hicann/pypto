# pypto.set\_pass\_default\_config

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:08:04.534Z pushedAt=2026-08-26T09:10:38.133Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Modifies the default configuration of a Pass. Its main function is to dynamically modify the runtime behavior configuration of a Pass. Currently, it supports configuring the switch for dumping computation graphs to facilitate analysis and debugging.

## Prototype

```python
set_pass_default_config(key: PassConfigKey, value: bool)
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| key       | Input        | **PassConfigKey** enumeration. <br> **KEY_DUMP_GRAPH**: Dump the Pass computation graph. |
| value     | Input        | Value. |

## Return Value

None

## Constraints

- Timing: This function must be called before graph compilation starts.
- Scope: The configuration is global and affects all subsequent compilation processes.

## Example

```python
pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

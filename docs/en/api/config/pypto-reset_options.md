# pypto.reset\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:06:29.390Z pushedAt=2026-08-26T09:10:38.125Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Resets all configuration items to their default values, including **codegen_options**, **host_options**, **pass_options**, **runtime_options**, and **verify_options**.

## Prototype

```python
reset_options() -> None
```

## Parameters

None

## Return Value

No return value. The setting takes effect immediately upon successful operation.

## Constraints

None

## Example

```python
pypto.reset_options()
```

# pypto.experimental.set\_operation\_options

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T07:55:52.424Z pushedAt=2026-09-05T07:36:26.323Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It transforms parameters that were originally statically written in the configuration file **tile\_fwk\_config.json** into dynamic, programmable instructions.

## Prototype

```python
set_operation_options(*, combine_axis: Optional[bool] = None)
```

## Parameters

| Parameter            | Input/Output | Description                                                                 |
|----------------------|--------------|-----------------------------------------------------------------------------|
| combine_axis         | Input        | **Meaning**: Implements tail-axis broadcast inline during the code generation phase.<br>**Description**: For a binary operation (32,1) + (32,128), instead of first broadcasting (32,1) to (32,128), (32,1) is expanded to (32,8) through the brcb instruction, and then (32,8) + (32,128) is performed. The prerequisite is that (32,1) must be contiguous.<br>**Type**: bool<br>**Value range**: {True, False}<br>**Default value**: False |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- Usage scenario: For tail-axis broadcasting, the tail axis of the input must be contiguous; otherwise, the feature does not take effect. If the preceding node is a tail-axis reduce, the reduce API can guarantee this. If the preceding node is COPY_IN, contiguity in gm must be ensured at the frontend.
- Type safety: The type of the passed **value** must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: The parameter setting is local and only affects the compilation process within the current jit/loop. If it is not set, the setting in the upper-level jit/loop scope is inherited.

## Example

```python
pypto.experimental.set_operation_options(combine_axis=True)
```

# pypto.set\_host\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:37.215Z pushedAt=2026-08-26T09:10:38.130Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It converts parameters originally statically configured in **tile_fwk_config.json** into dynamic, programmable instructions, primarily to control the execution of the on-board process.

## Prototype

```python
set_host_options(*, compile_stage: Optional[CompStage] = None,
                    compile_monitor_enable: Optional[int] = None,
                    compile_timeout: Optional[int] = None,
                    compile_timeout_stage: Optional[int] = None,
                    compile_monitor_print_interval: Optional[int] = None) -> None
```

## Parameters

| Parameter          | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | Input      | Meaning: Controls the compilation phase to be executed. <br> Description: <br> **ALL_COMPLETE**: No effect; compilation and execution proceed normally. <br> **TENSOR_GRAPH**: In the compilation phase, stops after the final tensor graph is generated. <br> **TILE_GRAPH**: In the compilation phase, terminates after the final tile graph is generated. <br> **EXECUTE_GRAPH**: In the compilation phase, terminates after the final execution graph is generated. <br> **CODEGEN_INSTRUCTION**: In the compilation phase, terminates after the instruction code is generated. <br> **CODEGEN_BINARY**: Terminates after the code binary is generated during compilation, marking the end of the compilation phase. <br> Value range: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY) <br> Default value: ALL_COMPLETE |
| compile_monitor_enable    | Input      | Meaning: Controls the monitoring mode of the compilation phase. <br> Description: <br> **0**: Disables Compiler Monitor. <br> **1**: Enables Compiler Monitor and disables Pass details. <br> **2**: Enables Compiler Monitor and enables Pass details. <br> Value range: int [0, 2] <br> Default value: 0 |
| compile_timeout    | Input      | Meaning: Enables compilation progress monitoring. When the total duration of the current compilation exceeds this value, a timeout warning message is printed. <br> Description: Takes effect only when **compile_monitor_enable** is **1** or **2**. Unit: seconds. A value of 0 disables warning message printing. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 600 |
| compile_timeout_stage    | Input      | Meaning: Enables compilation progress monitoring. When the duration of a single phase in the compilation process exceeds this value, a timeout warning message is printed. <br> Description: Takes effect only when **compile_monitor_enable** is **1** or **2**. Unit: seconds. A value of **0** disables warning message printing. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 0 (disabled) |
| compile_monitor_print_interval    | Input      | Meaning: Enables compilation progress monitoring. When the duration of a single phase in the compilation process exceeds 60s, progress is printed at this interval. <br> Description: Takes effect only when **compile_monitor_enable** is 1 or 2. Unit: seconds. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 60 |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- Type safety: Ensure that the type of the passed **value** exactly matches the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: Parameter settings are global and affect all subsequent compilation processes.

## Example

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH,
                       compile_monitor_enable=1,
                       compile_timeout=120,
                       compile_timeout_stage=30,
                       compile_monitor_print_interval=20
                       )
```

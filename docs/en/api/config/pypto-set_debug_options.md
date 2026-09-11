# pypto.set\_debug\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:03.029Z pushedAt=2026-08-26T09:10:38.127Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the debug options.

## Prototype

```python
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  ) -> None
```

## Parameters

| Parameter            | Input/Output | Description                                                                 |
|----------------------|--------------|------------------------------------------------------------------------------|
| **compile_debug_mode**   | Input      | Meaning: Sets the compile-phase debug mode. <br> Description:<br> **0**: indicates that the compile-phase debug mode is disabled by default; <br> **1**: indicates enabling the compile-phase debug mode for graphs, which one-click enables graph compilation-related configurations, currently including only the computation graph; <br> **2**: indicates enabling the fixed CCE feature, which one-click enables a fixed device-side code output path, does not overwrite existing CCE files, and enables single-threaded compilation; <br> Type: int <br> Value range: 0, 1, or 2 <br> Default value: 0 <br> Affected pass scope: NA |
| **runtime_debug_mode**   | Input      | Meaning: Sets the runtime-phase debug mode. <br> Description: <br>**0**: indicates that the runtime-phase debug mode is disabled by default; <br> **1**: indicates enabling the runtime-phase debug mode, which one-click enables graph execution-related configurations, currently including only the swimlane graph; <br> 2: indicates enabling the AICORE_MODEL simulation mode; <br> **3**: indicates enabling runtime dependency correctness verification data dump; <br> **4**: indicates enabling runtime GM memory out-of-bounds check; <br> Type: int <br> Value range: 0, 1, 2, 3, or 4 <br> Default value: 0 <br> Affected pass scope: NA |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon successful execution.

## Constraints

- When **compile_debug_mode** is set to **2** (fixed CCE), the device-side code output path is determined by the environment variable **ASCEND_WORK_PATH** (`$ASCEND_WORK_PATH/pypto/<name>`). If **ASCEND_WORK_PATH** is not set, the code is output to the current working directory. In this mode, **TILE_FWK_OUTPUT_DIR** does not participate in the device-side code path calculation.

## Example

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
pypto.set_debug_options(compile_debug_mode=2)
```

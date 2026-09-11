# Swimlane Diagram Issues

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:02:59.233Z pushedAt=2026-09-04T08:47:13.446Z -->

## Symptom

Swimlane diagram data collection or display is abnormal, including no file generated, empty file, excessively long first task startup, no data in ACL graph mode, and large deviation in `msprof` data.

## Possible Causes

- **No swimlane diagram file generated**: Performance data collection is not enabled (`runtime_debug_mode` is not set to **1**).
- **Swimlane diagram file is empty**: Profiling failed to be enabled.
- **First task startup is excessively long**: AI CPU starts slowly, causing the AI Core task reception to be delayed overall.
- **No data in ACL graph mode**: Profiling during the ACL graph capture phase is not currently supported (tasks are delivered during the capture phase but profiling is enabled only during the replay phase). This will be supported in a later version.
- **Large deviation between msprof and swimlane diagram data**: The AI Core duration in `msprof` also includes the waiting time for AICPU to deliver devTask and the exit time.

## Solution

1. No file generated: Ensure that `runtime_debug_mode: 1` is set in the code.
2. File is empty: Enable DEBUG logs and search for `aicore profiling is opened` / `aicore profiling is closed` to check the enable status.
3. First task startup is excessively long: Check whether the AICPU resource allocation and scheduling configuration is normal.
4. No data in the ACL graph: Avoid this scenario for now. It will be supported in a later version.
5. Large msprof deviation: Set `export DUMP_DEVICE_PERF=true` to obtain more accurate AI Core end-to-end duration.

## Output Directory Artifacts

Files related to the swimlane diagram in the `output/output_timestamp` directory:

| File | Purpose |
|---|---|
| `machine_trace_perf_data*.json` | Raw profiling data of the Machine component |
| `tilefwk_L1_prof_data_*.json` | Raw profiling data of the Machine component |
| `merged_swimlane.json` | IDE visual integrated swimlane diagram |
| `machine_runtime_operator_trace*.json` | AI CPU/AI Core swimlane diagram (joint timeline) |

> `machine_trace_perf_data*.json` and `tilefwk_L1_prof_data_*.json` can be used to determine whether underlying data collection is successful. `merged_swimlane.json` and `machine_runtime_operator_trace*.json` are used for IDE display. For any issues with IDE, please contact the IDE team lead first.

## IDE Parameter Description

### CTRL AICPU

| Phase | Description | Trace Point |
|---|---|---|
| **DEV_TASK_BUILD** | Time spent building devTask (stitch time) | After stitch |
| **Post-process** | From building all devTasks to exit | When AI CPU exits |
| **Total run time** | Total time from launch to exit | From launch to exit |

### SCHED AICPU

| Phase | Description| Trace Point |
|---|---|---|
| **ALLOC_THREAD_ID** | Thread allocation and core binding duration | After AllocThreadIdx |
| **INIT** | Sched initialization duration | After Sched init() |
| **CORE_HAND_SHAKE** | Duration of handshake between Sched and AI Core | After handshake |
| **DEV_TASK_RCV** | Duration of receiving the devTask built by Ctrl | After taskQue reads devTask |
| **Post-process** | Duration from completing all devTasks to exit | After ExecuteTask |
| **Total run time** | Total time from launch to exit | From launch to exit |

### AICORE

| Phase | Description | Trace Point |
|---|---|---|
| **End-to-End time** | Actual end-to-end execution on the AI Core | From the earliest to the latest execution of ExecCoreFunctionKernel on the AI Core |
| **Total run time** | Total time from launch to exit | From launch to exit |

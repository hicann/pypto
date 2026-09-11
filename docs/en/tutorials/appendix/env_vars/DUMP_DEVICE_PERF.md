# DUMP_DEVICE_PERF

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:55:31.463Z pushedAt=2026-09-03T12:20:11.031Z -->

## Function Description
Enables joint AI CPU/AI Core performance data collection. When enabled, the framework collects end-to-end latency data during AI CPU scheduling and AI Core execution, and prints relevant performance statistics to the device. The collection results are saved to the `output/output_timestamp/` directory.

- Type: string
- Value range: `true` (enabled), `false`, or unset (disabled)

## Configuration Example
```bash
export DUMP_DEVICE_PERF=true
python3 examples/02_intermediate/operators/softmax/softmax.py

# Analyze the data after collection.
python tools/scripts/machine_perf_trace.py analyze output/output_<timestamp>/machine_trace_perf_data_0.json
```

## Constraints
- A maximum of 200 rounds of data collection and screen output are supported. The excess will be truncated.
- Currently, only data from 200 devTask constructions can be collected. A warning will appear in the log when this limit is exceeded.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

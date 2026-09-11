# PYPTO_DATADUMP_ENABLE

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:55:42.043Z pushedAt=2026-08-20T10:38:59.918Z -->

## Function Description
Enables tensor dumping during on-device execution. When enabled, the framework automatically dumps the input and output data of leaf functions during NPU operator execution, for precision comparison against simulation results.

- Type: string
- Valid values: `true` (enabled), `false`, or unset (disabled)

## Configuration Example
```bash
# Method 1: Set via command line
export PYPTO_DATADUMP_ENABLE=true

# Method 2: Set in Python code
import os
os.environ["PYPTO_DATADUMP_ENABLE"] = "true"
```

## Constraints
- Must be used together with `enable_pass_verify` and `pass_verify_save_tensor` in `verify_options`.
- Dump data is written to the `output/output_*/tensor/` directory.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

# TILE_FWK_DEVICE_ID

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:57:53.511Z pushedAt=2026-08-20T10:39:06.532Z -->

## Function Description
Specifies the NPU device ID used for PyPTO operator execution. The framework and test cases determine the target device through this variable.

- Type: integer
- Value range: `0` to `N-1` (N is the number of available NPUs)

## Configuration Example
```bash
export TILE_FWK_DEVICE_ID=0
```

## Constraints
- The specified device ID must be available. You can run `npu-smi info` to verify this.
- In multi-process scenarios, avoid using the same device ID across multiple processes.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

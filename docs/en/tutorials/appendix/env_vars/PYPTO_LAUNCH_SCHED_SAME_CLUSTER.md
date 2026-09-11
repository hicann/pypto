# PYPTO_LAUNCH_SCHED_SAME_CLUSTER

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:55:33.330Z pushedAt=2026-08-20T10:39:01.815Z -->

## Function Description
Controls whether to forcibly allocate scheduling threads (AICPU) within the same cluster. Allocating scheduling threads within the same cluster can achieve better inter-core pipeline performance. However, in the full-network scenario, components other than PyPTO also use AICPU, and forcing same-cluster allocation may cause execution timeout due to insufficient AICPU resources.

- Type: string
- Value range: `true` (default, same-cluster allocation), `false` (not forced to the same cluster)

## Configuration Example
```bash
# Disable the same-cluster constraint in full-network scenarios to avoid insufficient AICPU resources.
export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false
```

## Constraints
- When set to `false`, you can use `launch_sched_aicpu_num` (configured through `runtime_options`) to specify the number of available AICPUs.
- When same-cluster allocation is enabled, the `launch_sched_aicpu_num` configuration does not take effect.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

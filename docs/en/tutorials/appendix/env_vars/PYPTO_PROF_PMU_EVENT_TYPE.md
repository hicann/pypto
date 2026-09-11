# PYPTO_PROF_PMU_EVENT_TYPE

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:56:30.864Z pushedAt=2026-08-20T10:39:03.138Z -->

## Function Description
Selects the PMU (Performance Monitoring Unit) data collection mode. The PMU monitors hardware performance events of the AI Core, and different modes correspond to different event groups.

- Type: integer
- Value range: `1`, `2`, `4`, `5`, `6`, `7`, `8`. The default value is `2`.

## Configuration Example
```bash
export PYPTO_PROF_PMU_EVENT_TYPE=2

# Collect PMU data.
msprof --task-time=l3 --output=./prof_data python xxx.py

# Parse data.
python tools/profiling/tilefwk_pmu_to_csv.py -p PROF_xxx/device_x/data -pe=$PYPTO_PROF_PMU_EVENT_TYPE --arch dav_3510
```

## Constraints
- For details about the complete PMU data collection process, see [Collecting PMU Data](../../debug/performance.md).

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

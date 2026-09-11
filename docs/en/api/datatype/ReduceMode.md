# ReduceMode

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:38:01.456Z pushedAt=2026-08-20T12:44:05.138Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**ReduceMode** defines the execution mode of reduction operations. It specifies the reduction computation method in multi-threaded or multi-device environments, ensuring the correctness and performance of computation results.

## Prototype

```python
class ReduceMode(enum.Enum):
     ATOMIC_ADD = ...  # Atomic addition reduction, which uses atomic operations to ensure thread-safe data accumulation.
```

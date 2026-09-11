# ScatterMode

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:38:06.715Z pushedAt=2026-08-20T12:49:55.646Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**ScatterMode** defines the reduction mode of the **scatter** function.

## Prototype

```python
class ScatterMode(enum.Enum):
     None = ...     # Only performs data movement.
     ADD = ...      # Addition mode.
     MULTIPLY = ... # Multiplication mode.
```

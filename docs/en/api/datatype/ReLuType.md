# ReLuType

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:38:05.701Z pushedAt=2026-08-20T12:46:55.185Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**ReLuType** defines the modes of the ReLU activation function, used to enable the ReLU function.

## Prototype

```python
class ReLuType(enum.Enum):
     NO_RELU= ...  # Disable the ReLU function.
     RELU= ...     # Enable the ReLU function.
```

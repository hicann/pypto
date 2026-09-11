# SaturationMode

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:38:03.725Z pushedAt=2026-08-20T12:49:39.267Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**SaturationMode** defines the overflow handling method for floating-point to integer conversion. It controls the processing strategy when the source data exceeds the representable range of the target integer type, ensuring the correctness and predictability of the conversion result.

## Prototype

```python
class SaturationMode(enum.Enum):
     OFF = ...   # Truncation mode (default). Directly truncates the excess part, which may cause overflow.
     ON = ...    # Saturation mode. Values out of range are clamped to the maximum or minimum value of the target type.
```

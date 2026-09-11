# PrecisionType

<!-- md-trans-meta sourceCommit=ef159c53aaffca68734dbf8caedc52ff7e051796 translatedAt=2026-08-20T09:05:40.045Z pushedAt=2026-08-20T12:40:10.300Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**PrecisionType** defines the precision mode of high-precision vector operators, which is used to control how precision is handled in computations such as division, modulo, exponential, and logarithm.

## Prototype

```python
class PrecisionType(enum.Enum):
    INTRINSIC = ...       # Intrinsic mode, which directly uses chip instructions.
    HIGH_PRECISION = ...  # High-precision mode.
```

## Parameters

| Name | Description |
|:-------|:-----|
| HIGH_PRECISION | High-precision mode. A higher-precision computation method is used in the underlying implementation. Currently, this mode is effective only on Ascend 950PR/Ascend 950DT. |
| INTRINSIC | Intrinsic mode. Computation is performed directly using chip instructions, delivering higher performance. |

## Usage Recommendations

1. **Default behavior**: If no precision mode is specified, `HIGH_PRECISION` mode is used by default to ensure computation precision.
2. **Scenarios with high precision requirements**: `HIGH_PRECISION` mode is recommended, as it can effectively reduce precision loss and improve the accuracy of computation results.
3. **Scenarios with low precision requirements but a focus on performance**: `INTRINSIC` mode can be used to perform computations directly using chip instructions.

## Supported Operators

The following operators support the `PrecisionType` parameter:

| Operator | Description |
|:-----|:-----|
| div | Element-wise division. |
| fmod | Element-wise modulo. |
| remainder | Element-wise remainder. |
| pow | Element-wise power. |
| exp | Exponential operation. |
| sqrt | Square root operation. |
| rsqrt | Reciprocal square root operation. |
| log | Logarithmic operation. |
| log2 | Base-2 logarithmic operation. |
| log10 | Base-10 logarithmic operation. |
| reciprocal | Reciprocal operation. |

## Example

```python
import pypto

# Create a tensor.
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)

# Use the high-precision mode.
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)

# Use the intrinsic mode.
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)

# Use the high-precision mode by default.
out = pypto.div(a, b)

# Use operators (automatically using high-precision mode).
out = a / b

# Examples of other operators.
out = pypto.exp(a, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.sqrt(a, pypto.PrecisionType.INTRINSIC)
out = pypto.log(a, pypto.LogBaseType.LOG_E, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.pow(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

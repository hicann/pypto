# CastMode

<!-- md-trans-meta sourceCommit=6026cedafe6ebf8b8df67d5702d0815a8fef162f translatedAt=2026-08-20T08:10:55.770Z pushedAt=2026-08-24T03:11:02.075Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**CastMode** defines the rounding mode used during data type conversion, controlling the precision handling of floating-point conversions to ensure the accuracy of conversion results.

## Prototype

```python
class CastMode(enum.Enum):
     CAST_NONE = ...   # No rounding mode specified. The framework automatically selects the default value.
     CAST_RINT = ...   # Round to the nearest integer, with ties rounding to even (IEEE 754 default).
     CAST_ROUND = ...  # Round to the nearest integer, with ties rounding away from zero.
     CAST_FLOOR = ...  # Round down toward negative infinity.
     CAST_CEIL = ...   # Round up toward positive infinity.
     CAST_TRUNC = ...  # Truncate toward zero.
     CAST_ODD = ...    # Round to odd (Von Neumann rounding).
```

## Rounding Mode Details

| CastMode | Description | Atlas A2 Training Products/Atlas A2 Inference Products/Atlas A3 Training Products/Atlas A3 Inference Products | Ascend 950PR/Ascend 950DT |
|----------|------|----------|--------|
| CAST_RINT | Round to the nearest integer, with ties rounding to even (IEEE 754 default). | Supported | Supported |
| CAST_ROUND | Round to the nearest integer, with ties rounding away from zero. | Supported | Supported |
| CAST_FLOOR | Round down toward negative infinity. | Supported | Supported |
| CAST_CEIL | Round up toward positive infinity. | Supported | Supported |
| CAST_TRUNC | Truncate toward zero. | Supported | Supported |
| CAST_ODD | Round to odd (Von Neumann rounding). | FP32→FP16 | FP32→FP16 |

> **NOTE**: **CAST_NONE** is not a hardware-supported rounding mode but a framework-level concept, meaning "no rounding mode is specified, and the framework automatically selects the default value."

## Usage

### Default CastMode Rules

When the user uses **CAST_NONE** or specifies a **CastMode** that is not supported by the hardware, the framework automatically adopts the following default **CastMode**:

| Conversion Type | Default CastMode |
|----------|---------------|
| Floating-point to integer (FP→INT) | CAST_TRUNC (truncation toward zero) |
| Other conversion scenarios | CAST_RINT (round to nearest even) |

### Type Extension Conversion

Type extension conversions (such as FP16→FP32, INT8→FP16, and INT16→INT32) are inherently lossless and do not require rounding. In this case, the **CastMode** parameter passed by the user is ignored.

### Handling of Unsupported CastMode

When a user specifies a **CastMode** that is not supported by the hardware for a conversion:

1. The framework does not report an error.
2. The framework automatically adopts the default **CastMode** for that conversion (following the default rules above).

## Binary Rounding Rules

Before understanding the precision conversion rules, you need to first understand the representation of floating-point numbers and the binary rounding rules:

### Floating-Point Number Representation

- **DT_FP16**: 16 bits in total, including a 1-bit sign bit (S), a 5-bit exponent field (E), and a 10-bit mantissa field (M).
  - When E is neither all 0s nor all 1s, the represented value is: (-1)^S × 2^(E-15) × (1 + M)
  - When E is all 0s, the represented value is: (-1)^S × 2^(-14) × M
  - When E is all 1s, if M is all 0s, the represented value is **±inf** (depending on the sign bit); if M is not all 0s, the represented value is **nan**.

- **DT_FP32**: A total of 32 bits, including a 1-bit sign bit (S), an 8-bit exponent field (E), and a 23-bit mantissa field (M).
  - When E is neither all 0s nor all 1s, the represented value is: (-1)^S × 2^(E-127) × (1 + M)
  - When E is all 0s, the represented value is: (-1)^S × 2^(-126) × M
  - When E is all 1s, if M is all 0s, the represented value is **±inf** (depending on the sign bit); if M is not all 0s, the represented value is **nan**.

- **DT_BF16**: A total of 16 bits, including a 1-bit sign bit (S), an 8-bit exponent field (E), and a 7-bit mantissa field (M).
  - When E is neither all 0s nor all 1s, the represented value is: (-1)^S × 2^(E-127) × (1 + M)
  - When E is all 0s, the represented value is: (-1)^S × 2^(-126) × M
  - When E is all 1s, if M is all 0s, the represented value is **±inf** (depending on the sign bit); if M is not all 0s, the represented value is **nan**.

### Specific Behavior of Each Rounding Mode

- **CAST_RINT**: If the first bit of the part to be rounded is 0, no carry occurs. If the first bit is 1 and the subsequent bits are not all 0, a carry occurs. If the first bit is 1 and the subsequent bits are all 0, no carry occurs when the last bit of M is 0, and a carry occurs when the last bit of M is 1.

- **CAST_FLOOR**: If S is 0, no carry occurs. If S is 1, no carry occurs when the part to be rounded is all 0; otherwise, a carry occurs.

- **CAST_CEIL**: If S is 1, no carry occurs. If S is 0, no carry occurs when the part to be rounded is all 0; otherwise, a carry occurs.

- **CAST_ROUND**: If the first bit of the part to be rounded is 0, no carry occurs; otherwise, a carry occurs.

- **CAST_TRUNC**: Never carries.

- **CAST_ODD**: If the part to be rounded is all 0s, no carry occurs; if the part to be rounded is not all 0s, no carry occurs when the last bit of M is 1, and a carry occurs when the last bit of M is 0.

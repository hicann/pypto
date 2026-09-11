# DataType

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:10:44.787Z pushedAt=2026-08-24T03:13:21.515Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**DataType** is an enumeration class in the PTO framework used to represent tensor data types. It defines all supported data types, including integers, floating-point numbers, and Boolean values. As the core type identifier for tensor operations, **DataType** specifies the storage format and computation precision of a tensor.

## Prototype

```python
class DataType(enum.Enum):
     ...  # Enum class definition, containing all supported data types.

 # Data type constant definitions.
 DT_INT4 = ...     # 4-bit signed integer, occupying byte memory.
 DT_INT8 = ...     # 8-bit signed integer, occupying 1 byte of memory.
 DT_INT16 = ...    # 16-bit signed integer, occupying 2 bytes of memory.
 DT_INT32 = ...    # 32-bit signed integer, occupying 4 bytes of memory.
 DT_INT64 = ...    # 64-bit signed integer, occupying 8 bytes of memory.
 DT_FP8 = ...      # 8-bit floating-point number, used for low-precision computation.
 DT_FP16 = ...     # 16-bit half-precision floating-point number, occupying 2 bytes of memory.
 DT_FP32 = ...     # 32-bit single-precision floating-point number, occupying 4 bytes of memory.
 DT_BF16 = ...     # 16-bit Brain Float format, occupying 2 bytes of memory.
 DT_HF4 = ...      # 4-bit Half Float format, occupying 1 byte of memory.
 DT_HF8 = ...      # 8-bit Half Float format, occupying 1 byte of memory.
 DT_FP4E2M1 = ...  # 4-bit floating-point number with a 2-bit exponent and a 1-bit mantissa; two elements occupy 1 byte of memory.
 DT_FP8E4M3 = ...  # 8-bit floating-point number with a 4-bit exponent and a 3-bit mantissa, occupying 1 byte of memory.
 DT_FP8E5M2 = ...  # 8-bit floating-point number with a 5-bit exponent and a 2-bit mantissa, occupying 1 byte of memory.
 DT_FP8E8M0 = ...  # 8-bit floating-point number with an 8-bit exponent and a 0-bit mantissa, occupying 1 byte of memory.
 DT_FP4_E2M1X2 = ... # MXFP4 format with a 2-bit exponent and a 1-bit mantissa; 2-element packed, occupying 1 byte of memory.
 DT_FP4_E1M2X2 = ... # MXFP4 format with a 1-bit exponent and a 2-bit mantissa; 2-element packed, occupying 1 byte of memory.
 DT_UINT8 = ...    # 8-bit unsigned integer, occupying 1 byte of memory.
 DT_UINT16 = ...   # 16-bit unsigned integer, occupying 2 bytes of memory.
 DT_UINT32 = ...   # 32-bit unsigned integer, occupying 4 bytes of memory.
 DT_UINT64 = ...   # 64-bit unsigned integer, occupying 8 bytes of memory.
 DT_BOOL = ...     # Boolean type, occupying 1 byte of memory.
 DT_DOUBLE = ...   # 64-bit double-precision floating-point number, occupying 8 bytes of memory.
```

## Constraints

- Atlas A3 training products/Atlas A3 inference products do not support the DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 types.
- Atlas A2 training products/Atlas A2 inference products do not support the DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 types.

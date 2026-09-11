# TransMode

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:23:09.802Z pushedAt=2026-08-20T12:59:47.452Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not Supported
- Atlas A2 training products/Atlas A2 inference products: Not Supported

## Description

In the scenario where both the input and output matrices of **matmul** are of the FP32 data type, sets whether to enable TF32 computation and the rounding mode for TF32. After TF32 is enabled, the FP32 data type is converted to the TF32 data type during matrix multiplication. <br> TF32 uses 1 sign bit, 8 exponent bits, and 10 mantissa bits, totaling 19 bits for computation. Fewer mantissa bits reduce the hardware computation workload and accelerate computation, but this also leads to precision loss. <br> The input data format remains FP32. The rounding mode is determined by the **TransMode** parameter. In general, the **CAST_ROUND** mode (round to the nearest integer, with ties rounding away from zero) is used.

## Prototype

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # Disable the conversion from the float data type to the TF32 data type.
     CAST_RINT = ...   # Round to the nearest integer, with ties rounding to even.
     CAST_ROUND = ...  # Round to the nearest integer, with ties rounding away from zero.
```

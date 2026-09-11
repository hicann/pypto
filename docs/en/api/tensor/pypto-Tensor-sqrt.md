# pypto.Tensor.sqrt

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:57:18.372Z pushedAt=2026-08-28T11:36:17.402Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Prototype

```python
sqrt(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| **precision_type**  | Input      | Precision type. <br> Supported types: **PrecisionType**. <br> Default value: **PrecisionType.INTRINSIC**. <br> **INTRINSIC** directly uses chip instructions for computation, which is faster; **HIGH_PRECISION** uses higher-precision computation to reduce precision loss. |

## Description

See [pypto.sqrt](../operation/pypto-sqrt.md).

# pypto.Tensor.reciprocal

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:52:47.085Z pushedAt=2026-08-28T11:36:17.391Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Prototype

```python
reciprocal(self, precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | Input      | Precision type. <br> Supported types: **PrecisionType**. <br> Default value: **PrecisionType.HIGH_PRECISION**. <br> **HIGH_PRECISION** uses higher-precision computation to reduce precision loss; **INTRINSIC** directly uses chip instructions for computation, which is faster. |

## Description

See [pypto.reciprocal](../operation/pypto-reciprocal.md).

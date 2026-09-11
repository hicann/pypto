# pypto.Tensor.div

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:42:51.883Z pushedAt=2026-08-28T11:36:17.370Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Prototype

```python
div(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Divisor. <br> Supported types: **Tensor**, **int**, **float**. |
| precision_type  | Input      | Precision type. <br> Supported type: **PrecisionType**. <br> Default value: **PrecisionType.HIGH_PRECISION**. <br> **HIGH_PRECISION** uses higher-precision computation to reduce precision loss; **INTRINSIC** directly uses chip instructions. |

## Description

See [pypto.div](../operation/pypto-div.md).

# pypto.Tensor.fmod

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:44:54.423Z pushedAt=2026-08-28T11:36:17.376Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Prototype

```python
fmod(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| **other**   | Input      | Divisor. <br> Supported types: **Tensor**, **int**, **float**. |
| **precision_type** | Input | Precision mode enumeration type, used to control the precision mode of the modulo computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br> Defaults to **HIGH_PRECISION** (high-precision mode). |

## Detailed Description

See [pypto.fmod](../operation/pypto-fmod.md).

# pypto.scatter

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:49:31.166Z pushedAt=2026-09-05T07:36:26.369Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Prototype

```python
scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

non-inplace version of scatter\_. For details, see [pypto.scatter\_](pypto-scatter_.md)

## Parameters

For details about the parameters, see [pypto.scatter_](pypto-scatter_.md).

## Return Value

For details about the return value, see [pypto.scatter_](pypto-scatter_.md).

## Constraints

1. For details about the constraints, see [pypto.scatter_](pypto-scatter_.md).
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

For details about the example, see [pypto.scatter_](pypto-scatter_.md).

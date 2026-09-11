# pypto.index\_add

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:13:21.880Z pushedAt=2026-09-05T07:36:26.338Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This is the non-inplace version of [pypto.index_add_](pypto-index_add_.md).

## Prototype

```python
index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters

See [pypto.index_add_](pypto-index_add_.md) for parameter descriptions.

## Return Value

See [pypto.index_add_](pypto-index_add_.md) for the return value description.

## Constraints

1. See [pypto.index_add_](pypto-index_add_.md) for constraints.

## Example

See [pypto.index_add_](pypto-index_add_.md) for the call example.

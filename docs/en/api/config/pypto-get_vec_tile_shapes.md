# pypto.get\_vec\_tile\_shapes

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-18T12:06:24.919Z pushedAt=2026-08-26T09:10:38.125Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the **TileShape** size in vector computation.

## Prototype

```python
get_vec_tile_shapes() -> List[int]
```

## Parameters

None

## Return Value

Returns the per-dimension **TileShape** size.

## Constraints

None

## Example

```python
pypto.get_vec_tile_shapes()
```

# pypto.get\_cube\_tile\_shapes

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:04:55.302Z pushedAt=2026-08-26T09:10:38.116Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the **TileShape** size configured in cube computation and the enable/disable switch status of the multi-core K-splitting feature.

## Prototype

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool]
```

## Parameters

None

## Return Value

Returns the **TileShape** sizes in the m, k, and n directions, and whether the multi-core K-splitting feature is enabled.

## Constraints

None

## Example

```python
pypto.get_cube_tile_shapes()
```

# pypto.set_vec_tile_shapes

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:08:28.596Z pushedAt=2026-08-26T09:10:38.135Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the TileShape size in vector computation.

## Prototype

```python
set_vec_tile_shapes(*args: int) -> None
```

## Parameters

| Parameter  | Input/Output | Description                  |
|---------|-----------|-----------------------|
| *args   | Input      | TileShape size of each dimension, with a maximum of 4 inputs. |

## Return Value

void

## Constraints

TileShape must meet the following constraints:

Each dimension must be greater than 0.

Assume that TileShape is two-dimensional \{m, n\}; then:

- (m > 0) && (n > 0)

## Example

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```

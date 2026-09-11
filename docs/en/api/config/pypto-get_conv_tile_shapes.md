# pypto.get\_conv\_tile\_shapes

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:04:57.516Z pushedAt=2026-08-26T09:10:38.117Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the **TileShape** size configured for convolution (conv) computation and the enable/disable status of the **L0TileInfo** switch.

## Prototype

```python
def get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## Parameters

None

## Return Value

Returns the **TileShape** sizes on L0 and L1, and whether the **L0TileInfo** switch is enabled.

## Constraints

None

## Example

```python
pypto.get_conv_tile_shapes()
```

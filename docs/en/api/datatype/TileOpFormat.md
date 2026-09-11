# TileOpFormat

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:38:08.690Z pushedAt=2026-08-20T12:51:22.030Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**TileOpFormat** defines the tile operation format of a tensor, which is used to optimize memory access and computation efficiency in different computation modes. It mainly distinguishes between tile formats for dense computation and sparse computation.

## Prototype

```python
class TileOpFormat(enum.Enum):
     TILEOP_ND = ...  # N-dimensional tensor that supports standard multi-dimensional array operations.
     TILEOP_NZ = ...  # Same as FRACTAL_NZ/NZ, a format obtained by padding, reshaping, and transposing the lowest two dimensions of a tensor (for all dimensions of a tensor, the right side is the lower dimension and the left side is the higher dimension).
```

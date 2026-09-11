# Last Dimension Fails 32-Byte Alignment in set_xxx_tile_shapes

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:03:48.438Z pushedAt=2026-08-21T03:19:12.175Z -->

## Symptom

```python
with pypto.function("TENSOR_SUM_FP32", [x], [res]):
    for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
        pypto.set_vec_tile_shapes(4, 8)
        res.move(x.sum())
```

When setting the TileShape size using `pypto.set_xxx_tile_shapes`, the last dimension must be 32-byte aligned. Otherwise, a validation error occurs:

```text
C++ exception with description "ASSERTION FAILED: vecTile[lastDim] % alignNum == 0
Sum op: the tileShape of last axis need to 32Byte align!, func Sum, file reduction.cpp, line 374
libtile_fwk_interface.so(npu::tile_fwk::Sum(npu::tile_fwk::Tensor const&, int, bool)+0x620) [0xffff9ff2e090]
```

## Possible Causes

Due to hardware instruction limitations, the data being processed must be 32-byte aligned.

## Procedure

When setting the TileShape size using `pypto.set_xxx_tile_shapes`, ensure that the last dimension size is set to a value that is 32-byte aligned, that is, `TileShape[-1] * sizeof(dtype) % 32 == 0`.

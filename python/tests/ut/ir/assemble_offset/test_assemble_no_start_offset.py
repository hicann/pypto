"""Regression tests for issue #3: assembling via a slice without a start
offset ("out[:, :h] = v") must write at offset 0, not stop - shape[axis]."""

from pathlib import Path

import pypto
from pypto import pil

from ..test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent

IR1 = _GOLDEN_DIR / "test_assemble_no_start_offset1.pypto"


def _compile_assemble_kernel():

    def foo(a, b, out):
        n, h = a.shape
        tile = 16
        for k in pypto.loop((n + tile - 1) // tile, name="Loop_B", idx_name="k", unroll_list=[8, 1]):
            valid = pypto.min(n - k * tile, tile)
            a_view = pypto.view(a, [tile, h], [k * tile, 0], valid_shape=[valid, h])
            b_view = pypto.view(b, [tile, h], [k * tile, 0], valid_shape=[valid, h])
            pypto.set_vec_tile_shapes(tile, h)
            left = pypto.neg(a_view)
            right = pypto.abs(b_view)
            out[k * tile:(k + 1) * tile, :h] = left
            out[k * tile:(k + 1) * tile, h:] = right

    a = pypto.Tensor((-1, 64), pypto.DT_FP32, 'a')
    b = pypto.Tensor((-1, 64), pypto.DT_FP32, 'b')
    out = pypto.Tensor((-1, 128), pypto.DT_FP32, 'out')
    return pil.compile(foo, a, b, out)


def test_assemble_slice_without_start_offset():
    check_snapshot(_compile_assemble_kernel(), IR1)

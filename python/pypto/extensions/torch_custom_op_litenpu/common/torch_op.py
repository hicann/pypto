#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""torch.library op registration for a declared pypto operator.

* ``_QUALNAME_TO_OP`` maps each ``torch_op_qualname`` to its current owner; ``_impl``/``_fake`` resolve the
  owner at CALL time, so a later op reusing a qualname is reachable, not shadowed (last-declared wins).
* ``_EXPORTING`` is a thread-local export signal: ``torch.jit.is_tracing()`` is False inside a custom_op impl
  during onnx export, so ``is_in_onnx_export`` cannot replace it; a call on another thread is unaffected.

Imports DOWN only (``return_annotations``).
"""
import contextlib
import inspect
import logging
import threading

import torch
import torch.library

from . import return_annotations

__all__ = ("exporting_scope", "_is_exporting")

logger = logging.getLogger(__name__)


# A declared attr's .type -> the torch.library schema type of its trailing scalar operand.
_ATTR_SCHEMA_TYPE = {"Int": "int", "Float": "float", "String": "str", "ListInt": "int[]"}


# torch_op_qualname -> the declaring op currently owning it.
_QUALNAME_TO_OP: dict = {}

_EXPORTING = threading.local()


def _is_exporting() -> bool:
    """True while an :func:`exporting_scope` is active on the current thread."""
    return getattr(_EXPORTING, "active", False)


@contextlib.contextmanager
def exporting_scope():
    """Mark the current thread as exporting for the duration of the block.

    While set, an op with no ``torch_defn`` returns a shaped-empty stub from its impl instead of
    raising. Nesting-safe.
    """
    prev = getattr(_EXPORTING, "active", False)
    _EXPORTING.active = True
    try:
        yield
    finally:
        _EXPORTING.active = prev


def _infer_shape_call_args(op, tensor_shapes, attr_values):
    """Build ``infer_shape``'s positional args: the tensor shapes, then the shape-affecting attr values.

    ``infer_shape`` names as trailing params only the subset of declared attrs that affect shape, so each
    is looked up BY NAME -- their positions in ``_attr_specs`` need not line up. Returns just the tensor
    shapes when no attr is shape-affecting (the common case).
    """
    specs = getattr(op, "_attr_specs", ())
    tensor_shapes = list(tensor_shapes)
    if not specs:
        return tensor_shapes
    infer_params = list(inspect.signature(op._infer_shape_fn).parameters)
    shape_attr_names = infer_params[len(tensor_shapes):]  # trailing (non-tensor) infer_shape params
    if not shape_attr_names:
        return tensor_shapes
    name_to_val = {s.name: v for s, v in zip(specs, attr_values)}
    return tensor_shapes + [name_to_val[nm] for nm in shape_attr_names]


def _shape_stub(op, inputs, attr_values=()):
    """The infer-driven shaped-empty output for *op*: right shape/dtype/device, no compute.

    Used as the fake always, and as the impl when *op* has no ``torch_defn`` and we are exporting.
    *inputs* are the tensor operands; *attr_values* the trailing declared-attr operands, whose
    shape-affecting subset is passed to ``infer_shape`` (``infer_dtype`` never takes attrs).
    """
    infer_shape, infer_dtype = op._infer_shape_fn, op._infer_dtype_fn
    shapes = infer_shape(*_infer_shape_call_args(op, [t.shape for t in inputs], attr_values))
    dtypes = infer_dtype(*[t.dtype for t in inputs])
    device = inputs[0].device
    n_out = return_annotations.infer_shape_output_arity(infer_shape)
    if n_out == 1:
        return torch.empty(shapes, dtype=dtypes, device=device)
    return tuple(torch.empty(s, dtype=d, device=device) for s, d in zip(shapes, dtypes))


def _op_ns_name(torch_op_qualname: str):
    """Split a ``<namespace>::<op_name>`` torch op qualname into its two halves."""
    ns, sep, name = torch_op_qualname.partition("::")
    if not sep or not ns or not name:
        raise ValueError(
            f"torch_op_qualname must be a '<namespace>::<op_name>' qualified name "
            f"(e.g. 'pypto::add_pypto'), got {torch_op_qualname!r}"
        )
    return ns, name


def _synthesize_torch_op_qualname(op) -> None:
    """Register a ``torch.library`` custom op + fake for *op* from its infer_shape/infer_dtype.

    Uses an explicit ``schema`` string built from the derived input/output arity, since the generic
    ``*inputs`` closure carries no per-parameter annotations for ``torch.library`` to infer one from.
    """
    ns, name = _op_ns_name(op._torch_op_qualname)
    infer_shape, infer_dtype = op._infer_shape_fn, op._infer_dtype_fn
    qualname = op._torch_op_qualname

    prior = _QUALNAME_TO_OP.get(qualname)
    if prior is not None and prior is not op:
        logger.warning(
            "torch op qualname %r re-declared by a different op; most-recent declaration wins",
            qualname,
        )
    _QUALNAME_TO_OP[qualname] = op

    already = hasattr(getattr(torch.ops, ns, None), name)
    if already and prior is None:
        # A hand-written @torch.library.* op (escape hatch) already owns this qualname -- prior is absent
        # yet torch.ops has it. Leave it alone, register nothing, and drop the transient entry: nothing
        # would ever read it.
        del _QUALNAME_TO_OP[qualname]
        return
    if already:
        return  # pypto already registered this qualname; only the dict update above is needed

    # The tensor-input count is infer_dtype's param count: infer_dtype takes exactly one dtype per tensor
    # input and never takes attrs, so it is the drift-free tensor-arity source. (infer_shape's param count
    # is NOT usable here: a shape-affecting attr adds a trailing param to infer_shape.)
    n_in = len(inspect.signature(infer_dtype).parameters)
    n_out = return_annotations.infer_shape_output_arity(infer_shape)
    if n_in < 1:
        raise ValueError(
            f"{qualname}: infer_dtype must take >=1 input (its parameter "
            "count is the synthesized torch op's tensor-input arity)"
        )
    # Declared compute attrs become trailing scalar operands, after the tensor inputs: the op's arity is
    # n_in tensors then n_attrs scalars, split off in _impl/_fake before the infer calls.
    attr_specs = getattr(op, "_attr_specs", ())
    in_names = [f"x{i}" for i in range(n_in)]
    out_schema = "Tensor" if n_out == 1 else "(" + ", ".join(["Tensor"] * n_out) + ")"
    schema_params = [f"Tensor {nm}" for nm in in_names] + [
        f"{_ATTR_SCHEMA_TYPE[s.type]} {s.name}" for s in attr_specs
    ]
    schema = "(" + ", ".join(schema_params) + f") -> {out_schema}"

    def _impl(*args):
        cur = _QUALNAME_TO_OP[qualname]
        # Split the tensor inputs from the trailing attr scalars. infer_shape/torch_defn's
        # shape check see only the tensors.
        inputs = args[:n_in]
        attr_values = args[n_in:]
        fn = cur._torch_defn_fn
        if fn is not None:
            out = fn(*args)  # torch_defn consumes the tensor inputs + the trailing attr scalars
            cur._assert_torch_defn_output(out, inputs, attr_values)
            return out
        if _is_exporting():
            return _shape_stub(cur, inputs, attr_values)
        raise RuntimeError(
            f"{qualname}: this op has no torch_defn; pass torch_defn= when declaring the op to run it on CPU"
        )
    _impl.__name__ = name

    def _fake(*args):
        return _shape_stub(_QUALNAME_TO_OP[qualname], args[:n_in], args[n_in:])
    _fake.__name__ = name

    torch.library.custom_op(qualname, mutates_args=(), schema=schema)(_impl)
    torch.library.register_fake(qualname)(_fake)

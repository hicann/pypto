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
"""The framework-agnostic pypto op authoring surface: the ``ExportedCustomOp`` class and its export machinery.

Holds the op handle (``ExportedCustomOp``, whose ``__init__`` takes the kernel + inference fns by
reference) and ``_build_node_meta_completer``, the shared export closure every framework symbolic calls.
The declare-then-finalize registry lives in the sibling ``common.finalize``, whose ``finalize_pending_ops``
this module re-exports. The onnx symbolic and its node-meta encoding live in the ``onnx`` subpackage and
consume the config declared here (``onnx.spec.OnnxSymbolicSpec``). This module never imports a subpackage
at module scope.
"""
import ast
import inspect
import json
import logging
import textwrap

from . import finalize, kernel_snippet, return_annotations, torch_op
from .attr_spec import AttrSpec
from .authoring import (
    _DEFAULT_DOMAIN,
    _DEFAULT_DOMAIN_OPSET_VERSION,
    _detect_factory_signature,
    _direct_declared_annotations,
    _is_jit_kernel,
    _validate_direct_decorator_no_soc,
    _validate_factory_returns_jit_kernel,
    validate_op_type_identifier,
)
from .finalize import finalize_pending_ops
from .node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__KERNEL_COMPILE_SNIPPET,
    _META_KEY__OP_EXPORT_RECORD,
    _META_KEY__OP_TYPE,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
    _RESERVED_META_KEYS,
)
from .torch_op import _is_exporting, exporting_scope

__all__ = (
    "ExportedCustomOp",
    "AttrSpec",
    "finalize_pending_ops",
    "exporting_scope",
    "_is_exporting",
)

logger = logging.getLogger(__name__)


def _factory_inner_kernel_param_count(factory):
    """Tensor-parameter count of the inner jit kernel *factory* defines, or ``None`` when uncountable.

    Mirrors the inner-def scan rule of ``authoring._validate_factory_returns_jit_kernel`` (the first
    nested def whose decorator list carries a ``.jit`` attribute) but counts its tensor parameters;
    it is kept local so the arity guard does not widen this module's authoring import surface.
    ``None`` means "do not check": unreadable source, no inner jit def, or a ``*args``/``**kwargs``
    signature whose tensor count is not knowable from the source alone.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(factory)))
    except (OSError, TypeError, SyntaxError):
        return None
    top = tree.body[0] if tree.body else None
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    for node in ast.walk(top):
        if node is top or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(call, ast.Attribute) and call.attr == "jit":
                args = node.args
                if args.vararg is not None or args.kwarg is not None:
                    return None
                return len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    return None


def _build_node_meta_completer(
    *,
    op,
    encode_node_meta,
    input_dtypes,
    framework_kind: str,
):
    """Build the kernel export callable that runs the pipeline and returns the node-meta kwargs.

    *framework_kind* names which framework flow authored the node ("onnx"); it is
    recorded in the node's export record so downstream consumers can branch per flow.
    """
    def complete_node_meta(
        *input_nodes,
        op_type: str,
        domain: str = _DEFAULT_DOMAIN,
        domain_opset_version: int = _DEFAULT_DOMAIN_OPSET_VERSION,
    ):
        """Populate the node meta (compile snippet, export record) and return them as ONNX kwargs.

        op_type
            The effective, already-qualified op identifier; for the onnx flow, also the node's type string.
        domain, domain_opset_version
            Must match the prefix the symbolic passes to ``g.op(...)``: the CANN parser keys dispatch on
            ``<domain>::<domain_opset_version>::<op_type>`` with no bare-name fallback. The version is the
            pypto custom *domain*'s opset, not the base ai.onnx opset.
        """
        validate_op_type_identifier(op_type)
        node_meta = op._node_meta
        # Stash op_type in the node meta so the exported node carries the op identifier (redundant for onnx
        # since node.op_type already holds it). Lets a reader match on the same op_type string.
        node_meta[_META_KEY__OP_TYPE] = op_type
        dtypes = input_dtypes(*input_nodes)

        def _fn_name(fn):
            return fn.__name__ if fn is not None else None

        n_inputs = len(dtypes)

        # Kernel-source-of-truth embedded on the node: the self-contained compile snippet (import it
        # to reconstruct the live capture-set funcs) + the per-op export record (decided here by which
        # framework flow authored the node). Downstream consumers import the snippet to reconstruct the
        # live funcs; nothing downstream runs at export time.
        if op._bare_kernel_fn is not None:
            # DIRECT: an already-built @pypto.frontend.jit kernel, no factory and no synthesis. The
            # kernel source is emitted verbatim (with its @pypto.frontend.jit decorator); at op-compile
            # the host builds a CompileEntry(jit_kernel=...) over it.

            # Author-pinned literal shapes/dtypes from the DIRECT kernel's annotations (None if fully
            # erased). Fed to BOTH the embedded snippet and the op_export_record field so the
            # build-time regeneration reproduces the same mismatch guard; the two must agree.
            declared_annotations = _direct_declared_annotations(op)
            node_meta[_META_KEY__KERNEL_COMPILE_SNIPPET] = kernel_snippet.build_kernel_compile_snippet(
                mode="jit_kernel",
                jit_kernel_func=op._bare_kernel_fn,
                infer_shape_func=op._infer_shape_fn,
                infer_dtype_func=op._infer_dtype_fn,
                n_inputs=n_inputs,
                declared_annotations=declared_annotations,
            )
            mode = "jit_kernel"
            factory_signature = None  # a factory call shape does not apply to jit_kernel mode
            # The top-level jit-kernel name the stripped source emits (the build step's _fn("kernel_name")
            # resolves it). MUST equal the emitted `def` name, so keep it in sync with kernel_snippet's
            # create_name. Held under a distinct "kernel_name" key so the "create_kernel_name" field is
            # never abused to carry a non-factory name (create_kernel_name is None in jit_kernel mode).
            kernel_name_direct = op._bare_kernel_fn._original_func.__name__
            create_kernel_name = None
        else:
            # FROM_FACTORY: a hand-written create_*_kernel factory returning a jit kernel. The factory's
            # calling convention (single / lists / full) is read from its signature so multi-input
            # factories with different per-input shapes/dtypes work (factory_signature="lists").
            factory_signature = _detect_factory_signature(op._create_kernel_fn)
            node_meta[_META_KEY__KERNEL_COMPILE_SNIPPET] = kernel_snippet.build_kernel_compile_snippet(
                create_kernel_func=op._create_kernel_fn,
                infer_shape_func=op._infer_shape_fn,
                infer_dtype_func=op._infer_dtype_fn,
                n_inputs=n_inputs,
                factory_signature=factory_signature,
            )
            mode = "factory"
            create_kernel_name = _fn_name(op._create_kernel_fn)
            kernel_name_direct = None  # preserve schema shape; unused for factory mode
            declared_annotations = None  # factory kernels never pin literal annotations (bind a dtype var)

        export_record = {
            "dtypes": [str(d) for d in dtypes],
            "domain": domain,
            "domain_opset_version": domain_opset_version,
            "framework_kind": framework_kind,
            "mode": mode,
            "create_kernel_name": create_kernel_name,
            "kernel_name": kernel_name_direct,
            "infer_shape_name": _fn_name(op._infer_shape_fn),
            "infer_dtype_name": _fn_name(op._infer_dtype_fn),
            "n_inputs": n_inputs,
            "factory_signature": factory_signature,
            # Author-pinned literal shapes/dtypes (DIRECT kernels that spell out shape/dtype, e.g.
            # attention): the sole source of truth for the build-time regenerated snippet's mismatch
            # guard. None for erased DIRECT kernels and all factory kernels. Agrees with the embedded snippet.
            "declared_annotations": declared_annotations,
        }
        # Declared compute attributes (ordered), the sole carrier of the attr contract to build-time
        # codegen. The key is present ONLY when the op declares attrs; an attr-free op's record has no
        # ``attrs`` field.
        if op._attr_specs:
            export_record["attrs"] = [s.to_json_dict() for s in op._attr_specs]
        node_meta[_META_KEY__OP_EXPORT_RECORD] = json.dumps(export_record)

        return encode_node_meta(node_meta)
    return complete_node_meta


class ExportedCustomOp:
    """A pypto custom op: kernel, shape/dtype helpers, optional torch defn, and framework symbolics.

    Construct with the framework-export config (``torch_op_qualname`` + an ``*_spec``) plus the kernel and
    inference functions -- all plain undecorated module functions passed by reference.
    ``attrs=[AttrSpec(...)]`` declares ordered typed compute attributes, each riding the exported node as
    a trailing scalar operand read at GE runtime. Helpers a kernel body calls are traced from its source,
    never passed here. The framework symbolic is always synthesized from this config; a FOREIGN op's
    manual ONNX override goes through ``torch.onnx.register_custom_op_symbolic``.
    """

    def __init__(
        self,
        *,
        kernel,
        infer_shape,
        infer_dtype,
        torch_defn=None,
        torch_op_qualname=None,
        onnx_spec=None,
        attrs=None,
    ):
        # Ordered, typed compute-attribute declaration: index in this list == REG_OP order ==
        # ExtractAttrs index == InferShape trailing-param order. Empty means no attr fields are serialized.
        self._attr_specs: tuple = tuple(attrs) if attrs else ()
        for _spec in self._attr_specs:
            if not isinstance(_spec, AttrSpec):
                raise TypeError(
                    f"ExportedCustomOp(attrs=...) entries must be AttrSpec instances, got {_spec!r}"
                )
        # Two attrs sharing a name cannot be caught by AttrSpec, which sees one spec at a time, and the
        # torch schema parser accepts the repeat. The ONNX symbolic keys its attribute kwargs by name,
        # so the second would silently overwrite the first.
        _names = [_spec.name for _spec in self._attr_specs]
        if len(set(_names)) != len(_names):
            raise ValueError(
                f"ExportedCustomOp(attrs=...) declares duplicate attr name(s) in {_names}; each attr "
                "name is the key of an exported ONNX attribute and must be unique"
            )

        # A declared attr name matching a node-meta key collides on the exported node: the attr surfaces
        # as the same ``<name>_<type-suffix>`` attribute (a same-suffix clash dies at ``g.op``, a
        # different-suffix clash serializes duplicate ONNX attribute names readers misread). Declared
        # attrs and the node meta share one node namespace, so reject the collision here.
        _clashes = sorted(s.name for s in self._attr_specs if s.name in _RESERVED_META_KEYS)
        if _clashes:
            raise ValueError(
                f"ExportedCustomOp: attr name(s) {_clashes} collide with the reserved node-meta key(s) "
                "pypto writes itself; rename them."
            )

        self._node_meta: dict = {}
        # Namespaced marker identifying this as a pypto node + gating layout compatibility.
        self._node_meta[_META_KEY__PYPTO_NODE_LAYOUT_VERSION] = _LOCAL_NODE_LAYOUT_VERSION
        # The pypto package version that exported the node.
        self._node_meta[_META_KEY__PYPTO_PACKAGE_VERSION] = _LOCAL_PYPTO_VERSION

        self._bare_kernel_fn = None  # DIRECT: a @pypto.frontend.jit kernel (built directly at op-compile)
        self._infer_shape_fn = None
        self._infer_dtype_fn = None
        self._torch_defn_fn = None  # optional eager-torch reference (the run-path compute)
        self._torch_defn_output_checked = False  # first-call shape/dtype assert guard (see _assert_torch_defn_output)
        self._create_kernel_fn = None  # FROM_FACTORY: a plain create_*_kernel factory returning a jit kernel
        self._onnx_symbolic_attached = False

        # Config-driven finalization (declare then finalize). When ``torch_op_qualname`` / a framework
        # spec is given, pypto synthesizes the torch op + symbolic at export time (finalize_pending_ops)
        # instead of the demo hand-writing them. An ``*_spec`` requires ``torch_op_qualname``; a bare
        # ``torch_op_qualname`` is fine.
        self._torch_op_qualname = torch_op_qualname
        self._onnx_spec = onnx_spec
        self._finalized = False
        if onnx_spec is not None and torch_op_qualname is None:
            raise ValueError(
                "ExportedCustomOp(onnx_spec=...) requires torch_op_qualname=... (the torch op qualified name)"
            )

        # Attach the kernel + inference fns. Order matters: infer_shape/infer_dtype before torch_defn
        # (its arity check reads _infer_dtype_fn), and torch-op auto-registration runs last (once every
        # fn is attached) so it sees the complete op.
        self._attach_kernel(kernel)

        # Attrs reach the kernel ONLY through a factory_signature="full" factory: the DIRECT form and the
        # narrower factory signatures have no attrs parameter, so a declared attr would ride the ONNX
        # node and be dropped at deploy, giving wrong numerics with no error anywhere.
        if self._attr_specs:
            if self._create_kernel_fn is None:
                raise ValueError(
                    "ExportedCustomOp(attrs=...) requires a factory kernel: a DIRECT "
                    "@pypto.frontend.jit kernel has no parameter to receive attrs, so they would be "
                    "silently ignored by the deployed kernel. Pass a create_*_kernel factory taking "
                    "(shapes, dtypes, attrs, soc_version)."
                )
            _takes = _detect_factory_signature(self._create_kernel_fn)
            if _takes != "full":
                raise ValueError(
                    f"ExportedCustomOp(attrs=...) requires a factory taking "
                    f"(shapes, dtypes, attrs, soc_version); {self._create_kernel_fn.__name__} takes "
                    f"{_takes!r}, which never receives attrs."
                )

        # infer_shape: each shape parameter is torch.Size; the return is torch.Size (single output) or a
        # fixed-length tuple of torch.Size. infer_dtype: each parameter and the return are torch.dtype.
        self._infer_shape_fn = infer_shape
        self._infer_dtype_fn = infer_dtype
        # A trailing ``infer_shape`` param (beyond the tensor count = ``infer_dtype``'s param count)
        # is looked up BY NAME in the declared attrs at dispatch, so a name matching no attr would
        # otherwise surface as a bare KeyError deep inside torch fake-tensor tracing.
        if self._infer_shape_fn is not None and self._infer_dtype_fn is not None:
            _n_tensors = len(inspect.signature(self._infer_dtype_fn).parameters)
            _trailing = list(inspect.signature(self._infer_shape_fn).parameters)[_n_tensors:]
            _unknown = sorted(set(_trailing) - {s.name for s in self._attr_specs})
            if _unknown:
                raise ValueError(
                    f"infer_shape declares trailing (attr) parameter(s) {_unknown} that match no "
                    "declared attr name; trailing infer_shape params are resolved by attr name at "
                    "dispatch, so each must name an AttrSpec."
                )
            # A DIRECT kernel is called as kernel(*inputs, *outputs); nothing else checks its arity
            # against infer_dtype/infer_shape (_check_declared_mismatches zips, so it truncates
            # silently). Skipping this check lets a mismatch surface only at runtime compile, possibly
            # on-device, as a raw-tensor-count FeError.
            if self._bare_kernel_fn is not None:
                _n_outputs = return_annotations.infer_shape_output_arity(self._infer_shape_fn)
                _defs = self._bare_kernel_fn._cached_signature[0]
                if len(_defs) != _n_tensors + _n_outputs:
                    raise ValueError(
                        f"kernel {self._bare_kernel_fn.__name__!r} declares {len(_defs)} tensor "
                        f"parameter(s), but infer_dtype/infer_shape declare {_n_tensors} input(s) and "
                        f"{_n_outputs} output(s) ({_n_tensors + _n_outputs} total); a DIRECT kernel is "
                        "called as kernel(*inputs, *outputs)."
                    )
            # The factory counterpart: the inner jit kernel a factory returns is called the same way,
            # so its tensor parameters must also number inputs + outputs. Best-effort -- the count is
            # read from the factory's source, and an unreadable or *args signature just skips the check.
            elif self._create_kernel_fn is not None:
                _inner_count = _factory_inner_kernel_param_count(self._create_kernel_fn)
                if _inner_count is not None:
                    _n_outputs = return_annotations.infer_shape_output_arity(self._infer_shape_fn)
                    if _inner_count != _n_tensors + _n_outputs:
                        raise ValueError(
                            f"factory {self._create_kernel_fn.__name__!r}'s inner jit kernel declares "
                            f"{_inner_count} tensor parameter(s), but infer_dtype/infer_shape declare "
                            f"{_n_tensors} input(s) and {_n_outputs} output(s) "
                            f"({_n_tensors + _n_outputs} total); the inner kernel is called as "
                            "kernel(*inputs, *outputs)."
                        )
        if torch_defn is not None:
            self._attach_torch_defn(torch_defn)
        if torch_op_qualname is not None:
            # Auto-registers once qualname + both infer fns are attached (checked above). This is the
            # only torch-op registration path -- a registration error (e.g. an unresolvable infer_shape
            # return annotation) surfaces here, not at export time.
            torch_op._synthesize_torch_op_qualname(self)
            finalize._register_pending_op(self)

    def _attach_kernel(self, fn):
        """Store the op's kernel, distinguishing the DIRECT (jit) and FROM_FACTORY forms.

        - DIRECT: ``fn`` is a ``@pypto.frontend.jit`` object. Use erased annotations
          (``pypto.Tensor([...])``) so the dummies drive shape+dtype, and set ONLY ``run_mode``
          (validated here). Pin ``run_mode`` to SIM: the packer forces SIM anyway, but the frontend
          defaults it per-box at import, so pinning avoids an "NPU is not available" surprise.
        - FROM_FACTORY: ``fn`` is a plain function that defines and returns an inner
          ``@pypto.frontend.jit`` kernel, binding ``soc_version`` as a closure param.
        """
        if _is_jit_kernel(fn):
            _validate_direct_decorator_no_soc(fn)
            self._bare_kernel_fn = fn
        else:
            _validate_factory_returns_jit_kernel(fn)
            self._create_kernel_fn = fn

    def _attach_torch_defn(self, fn):
        """Store the op's eager-torch defn: the run-path (CPU) compute and reference.

        *fn* takes the op's input tensors and returns the output tensor(s) matching the synthesized
        schema. The run path dispatches ``torch.ops.<ns>.<op>`` to it, and it also runs during onnx
        export-trace with its return discarded (the symbolic drives the artifact). Optional: without
        it a real run raises a clear error while export still works. Arity is checked against
        ``infer_dtype``'s param count -- the drift-free tensor arity, since a shape-affecting attr
        adds a trailing ``infer_shape`` param -- plus the declared attrs.
        """
        if self._infer_dtype_fn is not None:
            n_inputs = len(inspect.signature(self._infer_dtype_fn).parameters)
            n_expected = n_inputs + len(self._attr_specs)
            n_got = len(inspect.signature(fn).parameters)
            if n_got != n_expected:
                raise ValueError(
                    f"torch_defn takes {n_got} params but infer_dtype declares {n_inputs} inputs "
                    f"+ {len(self._attr_specs)} declared attrs = {n_expected}; they must match "
                    "(n inputs + n attrs -> n outputs)"
                )
        self._torch_defn_fn = fn

    def _assert_torch_defn_output(self, out, inputs, attr_values=()) -> None:
        """First-call check that the ``torch_defn`` output matches infer_shape/infer_dtype.

        Cheap and idempotent-guarded (runs once per op); surfaces an author error as a clear message rather
        than a downstream graph divergence. No-op if either infer helper is absent. *attr_values* are the
        trailing declared-attr operands; the shape-affecting subset is fed to ``infer_shape``.
        """
        if self._torch_defn_output_checked:
            return
        self._torch_defn_output_checked = True
        infer_shape, infer_dtype = self._infer_shape_fn, self._infer_dtype_fn
        if infer_shape is None or infer_dtype is None:
            return
        exp_shape = infer_shape(*torch_op._infer_shape_call_args(self, [t.shape for t in inputs], attr_values))
        exp_dtype = infer_dtype(*[t.dtype for t in inputs])
        n_out = return_annotations.infer_shape_output_arity(infer_shape)
        outs = (out,) if n_out == 1 else tuple(out)
        exp_shapes = (exp_shape,) if n_out == 1 else tuple(exp_shape)
        exp_dtypes = (exp_dtype,) if n_out == 1 else tuple(exp_dtype)
        for i, (o, s, d) in enumerate(zip(outs, exp_shapes, exp_dtypes)):
            if tuple(o.shape) != tuple(s) or o.dtype != d:
                raise RuntimeError(
                    f"{self._torch_op_qualname}: torch_defn output {i} has shape "
                    f"{tuple(o.shape)}/dtype {o.dtype} but infer_shape/infer_dtype declare {tuple(s)}/{d}"
                )

# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# golden_template.py
# Starter for `custom/<op>/<op>_golden.py` (golden authoring, mathematician-owned)
# AND for `custom/<op>/modules/<op>_module<suffix_k>_golden.py` (per-module
# scaffolding step A, verifier-owned).
#
# Covers the torch reference side: Layer A (utilities), Layer B (math
# building blocks), Layer C (forward reference), Layer D (host-side
# constants), Layer E (decomposed backward helpers), Layer F (golden
# backward single entry).
#
# This file is pure torch — NO PyPTO imports, NO `import pypto`. Lint
# rule OL15 enforces this.
# =============================================================================
#
# Reference implementation constraints (OL15 + golden normalization rules)
#
#   ALLOWED:
#     - torch.matmul(a.float(), b.float()), elementwise ops (+ - * /),
#       torch.sigmoid, torch.softmax
#       matmul inputs MUST be .float() first to match pypto.matmul's FP32
#       accumulation on NPU Cube L0C. Output dtype via .to(target) after.
#       ❌ torch.matmul(a_bf16, b_bf16).float() — BF16 accumulation, precision lost.
#     - torch.sum / torch.mean over named or last dim with explicit dim=
#     - explicit Python loops over batch / head / chunk on the HOST
#       (this is golden code, not kernel code — the OL18 ban does not apply
#       here because golden files do not import pypto)
#     - torch.transpose(t, dim0, dim1) instead of `.T` / `.t()`
#     - host-side prefix-sum-via-matmul (D layer constants), explicit reshape
#
#   DISALLOWED (these complicate NPU lowering or differ from the PyPTO path):
#     - .T  /  .t()   /  any implicit transpose form
#     - torch.cumsum
#     - torch.masked_fill
#     - torch.tril  /  torch.triu  /  any factory mask op
#     - torch.flip
#     - rank > what the device supports (clamp at the SPEC.md p0_shapes rank)
#
# Cumulative-golden contract (per-module goldens only)
#
#   Each `<op>_module<suffix_k>_golden.py` exports a single function
#       `<op>_module<suffix_k>_golden(*primary_inputs)`
#   that returns the M1..M_k composed output as a tuple in the order the
#   module chain produces it. The body is SELF-CONTAINED — do NOT
#   `from <op>_module<suffix_{k-1}>_golden import ...`. Inline every step
#   from M1 to M_k by reading `module_interfaces.yaml` and partitioning
#   the math from `<op>_golden.py`. Header comment:
#       # Derived from module_interfaces.yaml — do not hand-edit.
#       # On YAML changes, regenerate.
# =============================================================================

import torch


# =============================================================================
# Layer A — Utilities (optional)
# Diagnostics / logging helpers used only inside this golden file.
# =============================================================================

def tensor_compare_report(actual: torch.Tensor, expected: torch.Tensor, *, name: str) -> None:
    """Optional helper. Lives here only when the reference itself wants to
    self-check intermediate tensors during development."""
    raise NotImplementedError


# =============================================================================
# Layer B — Math building blocks (pure torch)
# Reusable fragments that mirror PyPTO sub-kernel decomposition. Keeping
# them here (rather than inline in Layer C) lets golden helpers be
# unit-tested without running the full forward.
# =============================================================================

def norm_fwd(x: torch.Tensor) -> torch.Tensor:
    """Example math helper. Implement op-specific helpers as needed."""
    raise NotImplementedError


# =============================================================================
# Layer C — Forward reference
# Ground-truth forward in PyTorch. Must follow the constraints listed at
# the top of this file (no .T, no cumsum, etc.). Use stage labels
# (# ===== (A) stage description =====) so they map 1-to-1 to PyPTO
# sub-kernels in `_module<suffix_k>_impl.py`.
# =============================================================================

def forward_ref(*primary_inputs) -> torch.Tensor:
    """Forward reference. Same math as `<op>_golden`, but split by stages
    that align with the module decomposition."""
    # ===== (A) input normalization =====
    # ===== (B) main computation =====
    # ===== (C) output assembly =====
    raise NotImplementedError


# =============================================================================
# Layer D — Host-side constants
# Replace forbidden ops (cumsum, masked_fill, tril/triu) with explicit
# matrices computed once on the host. Examples:
#   - prefix-sum mask via lower-triangular @-mask matmul
#   - chunk-boundary indicator built from arange + comparison
# =============================================================================

def make_host_constants(B: int, T: int, D: int) -> dict:
    """Build masks / matrices used by Layer E and Layer C. Pure torch on host."""
    raise NotImplementedError


# =============================================================================
# Layer E — Backward reference (decomposed)
# One private helper per loop-body / pipeline stage. Each helper mirrors a
# `pypto_*` sub-kernel in Layer H of the impl file so a precision drift
# can be localized to a single stage.
# =============================================================================

def _stage_alpha_backward(*args) -> tuple:
    """One decomposed backward stage. Mirrors `pypto_stage_alpha`."""
    raise NotImplementedError


def _stage_beta_backward(*args) -> tuple:
    """One decomposed backward stage. Mirrors `pypto_stage_beta`."""
    raise NotImplementedError


# =============================================================================
# Layer F — Golden backward (single entry)
# The one public name tests grep for. Compose the Layer E helpers to form
# the full backward.
# =============================================================================

def torch_golden_<op>_backward_ref(*args, **kwargs) -> tuple:
    """Single entry-point golden backward. Composes stage helpers from
    Layer E. Tests compare its outputs against the PyPTO `<op>_module*_impl.py`
    backward (when the op has a backward)."""
    raise NotImplementedError


# =============================================================================
# Public golden entry — the function the verifier and tests import
#
# For `<op>_golden.py` (mathematician):
#   def <op>_golden(*primary_inputs) -> torch.Tensor | tuple
#
# For `modules/<op>_module<suffix_k>_golden.py` (scaffolding step A, verifier):
#   def <op>_module<suffix_k>_golden(*primary_inputs) -> torch.Tensor | tuple
#       """Cumulative golden returning the M1..M_k composed output."""
# =============================================================================

def <op>_golden(*primary_inputs):
    """Public golden entry. Composes Layer C / D / F as needed."""
    raise NotImplementedError

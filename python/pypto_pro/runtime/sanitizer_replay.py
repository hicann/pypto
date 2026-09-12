#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host replay for the PyPTO Pro sanitizer (v2).

Scans the single ``sanitizer_report.bin`` file (metadata header + device-side
log records) produced by a kernel launch and computes the detections:

    1. GM_OUT_OF_BOUNDS   - GM access window must fit each tensor dimension.
    2. TILE_OUT_OF_BOUNDS - tile access window must fit the declared tile dims.
    3. TILE_OVERLAP       - no two tile slots may overlap in address space.
    4. MUTEX_*            - mutex lock/unlock pairing per (region, id, pipe).

Each detection registers its own record format and check logic in
``_DETECTIONS``; see ``/workspace/doc/sanitizer_v2_design.md``.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Callable

# Record detection ids (must match C++ SanitizerDetection enum).
DET_GM_ACCESS = 1
DET_TILE_ACCESS = 2
DET_MUTEX_ACCESS = 3
DET_VIEW_SHAPE = 4
DET_TILE_SCALAR = 5
DET_TILE_DECL = 6

# Pipe name mapping, mirroring ir::PipeType (MTE1=0 … ALL=7), used for
# human-readable pipe names in reports (e.g. "MTE2" instead of "1").
_MAX_RECORD_DIMS = 5  # max logical dims per GM record (matches C++)

_PIPE_NAMES = ["MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX", "ALL"]


def _pipe_name(pipe: int) -> str:
    if 0 <= pipe < len(_PIPE_NAMES):
        return _PIPE_NAMES[pipe]
    return f"PIPE({pipe})"

# Per-detection record formats (independent, not uniform).
# GM (det_id=1, 15 u32 = 60 bytes):
#   [0] det_id, [1] ndim, [2..6] off[5], [7..11] shape[5] (runtime values,
#   aligned with the offsets' trailing dims), [12] acc_row, [13] acc_col,
#   [14] span_id. For ndim = 0 (linear access) acc_row carries the
#   whole-tensor element count.
_GM_RECORD = struct.Struct(
    "<"
    "II"     # det_id, ndim
    "iiiii"  # off[0..4]
    "iiiii"  # shape[0..4]
    "III"    # acc_row, acc_col, span_id
)
# Tile (det_id=2, 8 u32 = 32 bytes):
#   [0] det_id, [1] dim_row, [2] dim_col (the access-site declared dims --
#   the runtime truth of the access), [3] off_row, [4] off_col,
#   [5] acc_row, [6] acc_col, [7] span_id
_TILE_RECORD = struct.Struct(
    "<"
    "III"    # det_id, dim_row, dim_col
    "ii"     # off_row, off_col (signed)
    "III"    # acc_row, acc_col, span_id
)
# Mutex (det_id=3, 5 u32 = 20 bytes):
#   [0] det_id, [1] mutex_id, [2] pipe, [3] is_lock, [4] span_id
_MUTEX_RECORD = struct.Struct(
    "<"
    "IIIII"  # det_id, mutex_id, pipe, is_lock, span_id
)
# ViewShape (det_id=4, 9 u32 = 36 bytes): declaration-time view-over-source
#   check, all fields runtime expressions. [0] det_id, [1..5] shape[0..4],
#   [6] byte footprint sum((dim-1)*stride)+1 scaled by the view dtype width,
#   [7] source byte capacity (-1 = raw pointer: no known bound, skip),
#   [8] span_id.
_VIEW_SHAPE_RECORD = struct.Struct(
    "<"
    "I"      # det_id
    "iiiii"  # shape[0..4]
    "ii"     # fp_bytes, src_bytes
    "I"      # span_id
)
# TileScalar (det_id=5, 5 u32 = 20 bytes):
#   [0] det_id, [1] dim0, [2] dim1, [3] linear offset, [4] span_id.
_TILE_SCALAR_RECORD = struct.Struct(
    "<"
    "III"    # det_id, dim0, dim1
    "iI"     # offset, span_id
)
# TileDecl (det_id=6, 6 u32 = 24 bytes): a make_tile declaration for the
#   overlap scan. [0] det_id, [1] addr, [2] size (bytes), [3] memory space
#   id, [4] element count, [5] span_id. Every field is one u32 slot (the
#   codegen writes one u32 per field); on-chip addresses/sizes fit int32.
_TILE_DECL_RECORD = struct.Struct(
    "<"
    "I"      # det_id
    "ii"     # addr, size
    "II"     # space, numel
    "I"      # span_id
)

_RECORD_SIZE: dict[int, int] = {
    DET_GM_ACCESS: _GM_RECORD.size,
    DET_TILE_ACCESS: _TILE_RECORD.size,
    DET_MUTEX_ACCESS: _MUTEX_RECORD.size,
    DET_VIEW_SHAPE: _VIEW_SHAPE_RECORD.size,
    DET_TILE_SCALAR: _TILE_SCALAR_RECORD.size,
    DET_TILE_DECL: _TILE_DECL_RECORD.size,
}
# Maximum record size in bytes and u32 words; per-region capacity is a
# multiple of the latter. The GM record (15 u32) is always the largest.
_MAX_RECORD_SIZE = max(_RECORD_SIZE.values())
MAX_RECORD_U32 = _MAX_RECORD_SIZE // 4


@dataclasses.dataclass
class DetectionSpec:
    det_id: int | None          # None for detections without runtime records
    name: str                   # finding kind
    record: struct.Struct | None  # decoding format (None: no records)
    level: str                  # "ERROR" / "WARNING"
    check: Callable             # (records, ctx) -> [SanitizerFinding]


class SanitizerFinding:
    def __init__(self, kind: str, message: str, location: str = "<unknown>",
                 hint: str = "", level: str = "ERROR"):
        self.kind = kind
        self.message = message
        self.location = location
        self.hint = hint
        self.level = level

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        msg = f"{self.kind}: {self.message}{loc}"
        if self.hint:
            msg += f" | Hint: {self.hint}"
        return msg


class SanitizerReplayError(RuntimeError):
    """Aggregated sanitizer findings from one replay pass."""

    def __init__(self, findings: list[SanitizerFinding]):
        self.findings = findings
        lines = [f"\nPyPTO Sanitizer Replay: {len(findings)} issue(s) detected\n"]
        for i, f in enumerate(findings, 1):
            lines.append(f"  [{i}/{len(findings)}] {f}")
            lines.append("")
        self.detail = "\n".join(lines)
        super().__init__(f"PyPTO Sanitizer Replay: {len(findings)} issue(s) detected")


class ReplayContext:
    """State the checks need: the kernel's source file and the tensor names
    (report display). Every check value -- offsets, windows, shapes, tile
    declaration ranges, source lines -- travels in the records themselves."""

    def __init__(self, source_file: str, tensor_names: list | None = None):
        self.tensor_names = tensor_names or []
        self.source_file = source_file

    def location(self, line: int) -> str:
        return f"{self.source_file}:{line}" if line > 0 else "<unknown>"

    def tensor_name(self, idx: int) -> str:
        if 0 <= idx < len(self.tensor_names):
            return self.tensor_names[idx]
        return f"tensor#{idx}"


# ---------------------------------------------------------------------------
# Record decode
# ---------------------------------------------------------------------------

def _decode_record(raw: bytes, offset: int) -> dict | None:
    """Decode one record at *offset* (bytes) by its leading det_id."""
    if offset + 4 > len(raw):
        return None
    det_id = struct.unpack_from("<I", raw, offset)[0]
    size = _RECORD_SIZE.get(det_id)
    if size is None or offset + size > len(raw):
        return None
    if det_id == DET_GM_ACCESS:
        (_, ndim, off0, off1, off2, off3, off4, sh0, sh1, sh2, sh3, sh4,
         acc_row, acc_col, span_id) = _GM_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, ndim=ndim,
                    off0=off0, off1=off1, off2=off2, off3=off3, off4=off4,
                    sh0=sh0, sh1=sh1, sh2=sh2, sh3=sh3, sh4=sh4,
                    acc_row=acc_row, acc_col=acc_col, line=span_id)
    if det_id == DET_TILE_ACCESS:
        (_, dim_row, dim_col, off_row, off_col, acc_row, acc_col,
         span_id) = _TILE_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, dim_row=dim_row, dim_col=dim_col,
                    off_row=off_row, off_col=off_col,
                    acc_row=acc_row, acc_col=acc_col, line=span_id)
    if det_id == DET_VIEW_SHAPE:
        (_, d0, d1, d2, d3, d4, fp_bytes, src_bytes, span_id) = _VIEW_SHAPE_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, dims=(d0, d1, d2, d3, d4),
                    fp_bytes=fp_bytes, src_bytes=src_bytes, line=span_id)
    if det_id == DET_TILE_SCALAR:
        (_, dim0, dim1, linear_off, span_id) = _TILE_SCALAR_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, dim0=dim0, dim1=dim1,
                    offset=linear_off, line=span_id)
    if det_id == DET_TILE_DECL:
        (_, addr, size, space, numel, span_id) = _TILE_DECL_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, addr=addr, size=size, space=space,
                    numel=numel, line=span_id)
    if det_id == DET_MUTEX_ACCESS:
        (_, mutex_id, pipe, is_lock, span_id) = _MUTEX_RECORD.unpack_from(raw, offset)
        return dict(det_id=det_id, mutex_id=mutex_id, pipe=pipe,
                    is_lock=is_lock, line=span_id)
    return None


def decode_buffer(raw: bytes, capacity_u32: int, region_count: int) -> list[dict]:
    """Decode the raw device buffer into a flat list of records.

    Each sub-block region: [ctr u32, records...]; ctr = number of u32 words
    written after the ctr word. Region stride = (capacity_u32 + 1) * 4 bytes.
    Every record is tagged with its region index (the executing sub-block),
    which is the per-block key dimension for pairing detections (e.g. mutex:
    different sub-blocks own independent mutex hardware slots).
    """
    records: list[dict] = []
    region_stride = (capacity_u32 + 1) * 4
    for region_idx in range(region_count):
        base = region_idx * region_stride
        if base + 4 > len(raw):
            break
        (ctr,) = struct.unpack_from("<I", raw, base)
        end = base + 4 + ctr * 4
        pos = base + 4
        while pos + 4 <= end:
            rec = _decode_record(raw, pos)
            if rec is None:
                pos += _MAX_RECORD_SIZE
                continue
            rec["region"] = region_idx
            records.append(rec)
            pos += _RECORD_SIZE[rec["det_id"]]
    return records


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_gm_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """GM_OUT_OF_BOUNDS: per-dim offset + window must fit the tensor shape.

    Both the offsets and the shape travel in the record (runtime values: the
    shape fields are the access-site TensorType dims -- constants, or the
    dynamic-dim ABI scalars resolved at execution). The tile's 2-D valid
    window applies to the innermost two dimensions; the leading ndim-2
    dimensions use a window of 1 (offset only). ndim = 0 marks a linear
    access (getval/setval): off0 is an element index, acc_row the
    whole-tensor element count."""
    findings: list[SanitizerFinding] = []
    for rec in records:
        if rec["det_id"] != DET_GM_ACCESS:
            continue
        ndim = rec["ndim"]
        off = [rec["off0"], rec["off1"], rec["off2"], rec["off3"], rec["off4"]]
        shape = [rec["sh0"], rec["sh1"], rec["sh2"], rec["sh3"], rec["sh4"]]
        acc_row, acc_col = rec["acc_row"], rec["acc_col"]
        if ndim == 0:
            # Linear scalar access: acc_row carries the element count.
            if acc_row <= 0:
                continue  # unknown (dynamic) bound
            if off[0] < 0 or off[0] + 1 > acc_row:
                findings.append(
                    SanitizerFinding(
                        "GM_OUT_OF_BOUNDS",
                        f"scalar access at linear offset {off[0]} exceeds "
                        f"{acc_row} elements",
                        ctx.location(rec["line"]),
                        hint="getval/setval offsets must stay inside the tensor element count",
                    )
                )
            continue
        if ndim <= 0 or ndim > _MAX_RECORD_DIMS:
            continue

        def window(i: int) -> int:
            """Access window of dimension i: the tile's 2-D window applies to
            the innermost two dimensions, 1 elsewhere (offset-only dims)."""
            if i == ndim - 2:
                return acc_row
            if i == ndim - 1:
                return acc_col
            return 1

        # The offsets and shape are aligned (same trailing-dim layout).
        details = []
        for i in range(ndim):
            dim_size = shape[i]
            if dim_size <= 0:
                continue  # unknown (dynamic) dim
            dim_off = off[i]
            dim_win = window(i)
            if dim_off < 0:
                details.append(f"dim{i}: negative offset {dim_off}")
            elif dim_off + dim_win > dim_size:
                over = dim_off + dim_win - dim_size
                details.append(
                    f"dim{i}: off {dim_off}+win {dim_win}={dim_off + dim_win} "
                    f"> {dim_size} (over by {over})"
                )
        if not details:
            continue
        access = f"offsets {off[:ndim]} window [{acc_row},{acc_col}]"
        msg = (f"access: {access} exceeds logical shape "
               f"{tuple(shape[:ndim])} — {'; '.join(details)}")
        findings.append(
            SanitizerFinding(
                "GM_OUT_OF_BOUNDS",
                msg,
                ctx.location(rec["line"]),
                hint=(
                    "The tile valid_shape plus the loop offsets must stay inside "
                    "the tensor logical shape on every dimension"
                ),
            )
        )
    return findings


def _check_tile_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OUT_OF_BOUNDS: tile accesses at a runtime offset (block.move /
    block.insert operands, dynamic set_validshape windows) must keep
    offset + window within the declared dims.  The dims come from the record
    itself (the access-site tile type -- what the codegen instantiates the
    transfer with, i.e. the runtime truth of the access), not from the tile
    table; the table only names the tile in the report.  Whole-tile accesses
    (load/store/compute) with constant windows are bounded by the compile-time
    set_validshape check."""
    findings: list[SanitizerFinding] = []
    for rec in records:
        if rec["det_id"] != DET_TILE_ACCESS:
            continue
        dim0, dim1 = rec["dim_row"], rec["dim_col"]
        off_row, off_col = rec["off_row"], rec["off_col"]
        acc_row, acc_col = rec["acc_row"], rec["acc_col"]
        violations = []
        # Per-dimension logical bounds (0 = unknown, skipped).
        if dim0 > 0 and off_row + acc_row > dim0:
            violations.append(f"dim0: offset {off_row}+window {acc_row}="
                              f"{off_row + acc_row} > declared {dim0}")
        if dim1 > 0 and off_col + acc_col > dim1:
            violations.append(f"dim1: offset {off_col}+window {acc_col}="
                              f"{off_col + acc_col} > declared {dim1}")
        if not violations:
            continue
        findings.append(
            SanitizerFinding(
                "TILE_OUT_OF_BOUNDS",
                f"tile access at [{off_row},{off_col}] window "
                f"[{acc_row},{acc_col}]: " + "; ".join(violations),
                ctx.location(rec["line"]),
                hint="Keep the tile's valid shape within the declared TileType dims",
            )
        )
    return findings


def _check_mutex_pairing(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """MUTEX pairing: every lock must be matched by an unlock on the same
    (region, mutex_id, pipe) key, in record order. Region distinguishes
    independent sub-block mutex hardware slots (prevents cross-block
    false matches). Pending is a list to support multiple outstanding
    locks on the same key (prevents overwrite-based false negatives)."""
    findings: list[SanitizerFinding] = []
    key_span_ids: dict[tuple, list[int]] = {}
    for rec in records:
        if rec["det_id"] != DET_MUTEX_ACCESS:
            continue
        key = (rec.get("region", 0), rec["mutex_id"], rec["pipe"])
        if rec["is_lock"]:
            key_span_ids.setdefault(key, []).append(rec["line"])
        else:
            spans = key_span_ids.get(key)
            if spans:
                spans.pop()
                if not spans:
                    del key_span_ids[key]
            else:
                findings.append(
                    SanitizerFinding(
                        "MUTEX_UNLOCK_BEFORE_LOCK",
                        f"mutex_unlock(mutex_id={rec['mutex_id']}, pipe={_pipe_name(rec['pipe'])}, "
                        f"region={rec.get('region', 0)}) without a preceding mutex_lock",
                        ctx.location(rec["line"]),
                        hint=(
                            "Every mutex_unlock must be paired with a mutex_lock on the "
                            "same (region, mutex_id, pipe)"
                        ),
                    )
                )
    for key, spans in key_span_ids.items():
        region, mutex_id, pipe = key
        count = len(spans)
        msg = f"mutex_lock(mutex_id={mutex_id}, pipe={_pipe_name(pipe)}, region={region})"
        if count > 1:
            msg += f" never unlocked ({count} outstanding locks)"
        else:
            msg += " never unlocked"
        findings.append(
            SanitizerFinding(
                "UNPAIRED_MUTEX_LOCK",
                msg,
                ctx.location(spans[-1]),
                hint=(
                    "Every mutex_lock must be followed by a mutex_unlock on the same "
                    "(region, mutex_id, pipe)"
                ),
            )
        )
    return findings


_MEMORY_SPACES = {0: "DDR", 1: "Vec", 2: "Mat", 3: "Left", 4: "Right", 5: "Scaling",
                  6: "Acc", 7: "Bias", 8: "ScaleLeft", 9: "ScaleRight"}


def _check_tile_overlap(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OVERLAP (declaration records): two make_tile declarations with
    intersecting address ranges in the same memory space. The declarations
    travel in the TileDecl records; identical re-registrations (the same
    statement copied into both programs of one kernel) share (addr, size,
    space, span) and collapse, while distinct statements with the same
    range keep both records -- exactly what the scan must report."""
    findings: list[SanitizerFinding] = []
    decls = {}
    for rec in records:
        if rec["det_id"] != DET_TILE_DECL:
            continue
        if rec["addr"] < 0 or rec["size"] <= 0:
            continue
        key = (rec["addr"], rec["size"], rec["space"], rec["line"])
        decls.setdefault(key, rec)
    items = list(decls.values())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ta, tb = items[i], items[j]
            if ta["space"] != tb["space"]:
                continue
            a0, sz_a = ta["addr"], ta["size"]
            b0, sz_b = tb["addr"], tb["size"]
            if not (a0 + sz_a <= b0 or b0 + sz_b <= a0):  # ranges intersect
                sp = _MEMORY_SPACES.get(ta["space"], f"space{ta['space']}")
                findings.append(
                    SanitizerFinding(
                        "TILE_OVERLAP",
                        f"tile range [{sp} 0x{a0:x}, 0x{a0 + sz_a:x}) overlaps "
                        f"another tile [{sp} 0x{b0:x}, 0x{b0 + sz_b:x})",
                        ctx.location(ta["line"]),
                        hint="Two tiles must not share on-chip address space",
                        level="WARNING",
                    )
                )
    return findings


def _check_tile_scalar_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OUT_OF_BOUNDS (scalar): getval/setval on a Tile -- the linear
    element index must stay inside dim0 * dim1."""
    findings: list[SanitizerFinding] = []
    for rec in records:
        if rec["det_id"] != DET_TILE_SCALAR:
            continue
        dim0, dim1, offset = rec["dim0"], rec["dim1"], rec["offset"]
        numel = dim0 * dim1
        if offset < 0 or offset + 1 > numel:
            findings.append(
                SanitizerFinding(
                    "TILE_OUT_OF_BOUNDS",
                    f"scalar tile access at linear offset {offset} "
                    f"exceeds {numel} elements (dims [{dim0},{dim1}])",
                    ctx.location(rec["line"]),
                    hint="getval/setval offsets must stay inside the tile element count",
                )
            )
    return findings


def _check_view_over_source(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """GM_OUT_OF_BOUNDS (declaration-time): a ptr.make_tensor view whose byte
    footprint exceeds its source's byte capacity. All values travel in the
    ViewShape record (runtime expressions resolved at execution): the
    declared shape, the footprint sum((dim-1)*stride)+1 scaled by the view
    dtype width, and the source capacity (-1 = raw pointer: no known bound,
    skip). The view shares the source's storage, so an oversized declaration
    reads/writes past the source even for in-bounds accesses."""
    findings: list[SanitizerFinding] = []
    for rec in records:
        if rec["det_id"] != DET_VIEW_SHAPE:
            continue
        fp_bytes, src_bytes = rec["fp_bytes"], rec["src_bytes"]
        if src_bytes < 0 or fp_bytes < 0:
            continue  # pointer source or unknown bound
        if fp_bytes > src_bytes:
            findings.append(
                SanitizerFinding(
                    "GM_OUT_OF_BOUNDS",
                    f"view footprint {fp_bytes} bytes exceeds its source "
                    f"{src_bytes} bytes; the view shares the source's storage "
                    f"(the op contract requires accesses to stay inside the source)",
                    ctx.location(rec["line"]),
                    hint="Keep the view's shape/stride footprint within the source allocation",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Detection registry: the unified registration point. Adding a detection =
# appending one DetectionSpec here (plus the C++ instrumentation).
# ---------------------------------------------------------------------------

_DETECTIONS: list[DetectionSpec] = [
    DetectionSpec(DET_GM_ACCESS, "GM_OUT_OF_BOUNDS", _GM_RECORD, "ERROR", _check_gm_bounds),
    DetectionSpec(DET_TILE_SCALAR, "TILE_OUT_OF_BOUNDS", _TILE_SCALAR_RECORD, "ERROR", _check_tile_scalar_bounds),
    DetectionSpec(None, "GM_OUT_OF_BOUNDS", None, "ERROR", _check_view_over_source),
    DetectionSpec(DET_TILE_ACCESS, "TILE_OUT_OF_BOUNDS", _TILE_RECORD, "ERROR", _check_tile_bounds),
    DetectionSpec(DET_MUTEX_ACCESS, "MUTEX", _MUTEX_RECORD, "ERROR", _check_mutex_pairing),
    DetectionSpec(None, "TILE_OVERLAP", None, "WARNING", _check_tile_overlap),
]


def run_replay(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """Run every registered check over the decoded records."""
    findings: list[SanitizerFinding] = []
    for spec in _DETECTIONS:
        findings.extend(spec.check(records, ctx))
    # Collapse repeats of the same defect raised by every loop iteration /
    # block touching the same statement: keyed by (kind, location).
    collapsed: list[SanitizerFinding] = []
    by_key: dict[tuple, SanitizerFinding] = {}
    counts: dict[tuple, int] = {}
    for f in findings:
        key = (f.kind, f.location)
        counts[key] = counts.get(key, 0) + 1
        by_key.setdefault(key, f)
    for key, f in by_key.items():
        n = counts[key]
        if n > 1:
            f.message = f"{f.message}  (same defect hit {n} times at this location)"
        collapsed.append(f)
    return collapsed


# ---------------------------------------------------------------------------
# Single-file format (sanitizer_report.bin)
#
#   u32 magic "PSPT" | u32 version | u32 header_size
#   u32 region_count | u32 capacity_u32
#   u32 num_spans | [u16 len + utf8]...
#   u32 tile_count | [u16 name_len + name, u16 space_len + space,
#                     i64 addr, i64 size, i64 dim0, i64 dim1]...
#   u32 tensor_count | [u16 name_len + name, u8 rank, i64 shape[rank]]...
#   bytes raw buffer (record regions, decode_buffer layout)
# ---------------------------------------------------------------------------

_MAGIC = 0x54505350  # "PSPT"
_VERSION = 2


def build_report_file(raw_buffer: bytes, region_count: int, capacity_u32: int,
                      tensor_names: list, tensor_shapes: list, path: str) -> None:
    """Write the single sanitizer report file: metadata header + raw records."""
    buf = bytearray()
    buf += struct.pack("<III", _MAGIC, _VERSION, 0)  # header_size patched below
    buf += struct.pack("<II", region_count, capacity_u32)
    buf += struct.pack("<I", len(tensor_names))
    for name, shape in zip(tensor_names, tensor_shapes):
        n_enc = str(name).encode("utf-8")
        buf += struct.pack("<H", len(n_enc)) + n_enc
        dims = [int(d) for d in shape][:5]
        buf += struct.pack("<B", len(dims))
        buf += struct.pack("<" + "q" * len(dims), *dims)
    header_size = len(buf)
    struct.pack_into("<I", buf, 8, header_size)
    buf += raw_buffer
    with open(path, "wb") as f:
        f.write(buf)


def _read_header(data: bytes):
    """Parse the metadata header; returns (region_count, capacity_u32,
    tensor_names, tensor_shapes, header_size)."""
    magic, version, header_size = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC:
        raise ValueError(f"sanitizer report: bad magic {magic:#x} (not a report file)")
    if version != _VERSION:
        raise ValueError(f"sanitizer report: unsupported version {version}")
    pos = 12
    region_count, capacity_u32 = struct.unpack_from("<II", data, pos)
    pos += 8
    tensor_count = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    tensor_names = []
    tensor_shapes = []
    for _ in range(tensor_count):
        (n,) = struct.unpack_from("<H", data, pos)
        pos += 2
        tensor_names.append(data[pos:pos + n].decode("utf-8"))
        pos += n
        (rank,) = struct.unpack_from("<B", data, pos)
        pos += 1
        if rank:
            dims = struct.unpack_from("<" + "q" * rank, data, pos)
            pos += 8 * rank
            tensor_shapes.append(tuple(int(d) for d in dims))
        else:
            tensor_shapes.append(())
    return region_count, capacity_u32, tensor_names, tensor_shapes, header_size


def scan_report_file(path: str, source_file: str) -> list[SanitizerFinding]:
    """Scan one ``sanitizer_report.bin`` and return the findings. The records
    carry source lines; *source_file* renders the report locations."""
    with open(path, "rb") as f:
        data = f.read()
    (region_count, capacity_u32,
     tensor_names, tensor_shapes, header_size) = _read_header(data)
    raw = data[header_size:]
    records = decode_buffer(raw, capacity_u32, region_count)
    ctx = ReplayContext(source_file, tensor_names)
    return run_replay(records, ctx)

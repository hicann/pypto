#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""Profile a PyTorch NPU golden script and write the standard golden report."""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

_EMPTY = inspect.Parameter.empty
_PARAM_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
}
logger = logging.getLogger(__name__)

DEFAULT_INPUT_SHAPE = (8, 1024, 4096)
DEFAULT_DTYPE = "float32"
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 5


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype_name: str


@dataclass
class ProfileConfig:
    golden_path: Path
    function_name: str | None
    tensor_specs: list[TensorSpec]
    scalar_args: dict[str, Any]
    output_dir: Path | None
    warmup: int
    iters: int
    device_id: int | None = None


@dataclass
class PerfReportConfig:
    golden_path: Path
    output_dir: Path
    trace_path: Path | None
    n_warmup: int
    tensor_specs: list[TensorSpec]
    device: Any
    prof_dir: Path
    kernel_e2e: dict[str, float]


def _is_bracketed(value: str) -> bool:
    for open_b, close_b in [("(", ")"), ("[", "]")]:
        if value.startswith(open_b) and value.endswith(close_b):
            return True
    return False


def parse_shape(text: str) -> tuple[int, ...]:
    value = text.strip()
    if _is_bracketed(value):
        value = value[1:-1].strip()
    if value in {"", "()", "scalar"}:
        return ()
    parts = [p for p in re.split(r"[xX,]", value) if p]
    try:
        shape = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"invalid shape: {text!r}") from exc
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive: {text!r}")
    return shape


def parse_tensor_spec(text: str, default_dtype: str = DEFAULT_DTYPE) -> TensorSpec:
    parts = text.split(":")
    if len(parts) == 1:
        name, shape_text, dtype_name = "", parts[0], default_dtype
    elif len(parts) == 2:
        name, shape_text = parts
        dtype_name = default_dtype
    elif len(parts) == 3:
        name, shape_text, dtype_name = parts
    else:
        raise ValueError(
            "input spec must be SHAPE, NAME:SHAPE, or NAME:SHAPE:DTYPE"
        )
    return TensorSpec(name=name.strip(), shape=parse_shape(shape_text),
                      dtype_name=dtype_name.strip() or default_dtype)


def parse_named_json(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"argument must use NAME=VALUE format: {text!r}")
    name, raw_value = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"argument name is empty: {text!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return name, value


def import_npu_stack() -> tuple[ModuleType, ModuleType]:
    try:
        import torch
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "npu_import_error: torch_npu is not installed. Install torch_npu or run the "
            "pypto-environment-setup skill before profiling."
        ) from exc
    return torch, torch_npu


def has_npu_hardware(torch: ModuleType) -> bool:
    try:
        return bool(torch.npu.is_available() and torch.npu.device_count() > 0)
    except Exception:
        return False


def load_python_module(path: Path) -> ModuleType:
    module_name = f"_pypto_golden_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def find_golden_function(module: ModuleType, path: Path,
                         function_name: str | None) -> Callable[..., Any]:
    if function_name:
        fn = getattr(module, function_name, None)
        if callable(fn):
            return fn
        raise RuntimeError(f"function {function_name!r} not found in {path}")

    stem_fn = getattr(module, path.stem, None)
    if callable(stem_fn):
        return stem_fn

    candidates = [
        obj for name, obj in vars(module).items()
        if callable(obj) and name.endswith("_golden") and not name.startswith("_")
    ]
    if len(candidates) == 1:
        return candidates[0]
    names = [
        name for name, obj in vars(module).items()
        if callable(obj) and name.endswith("_golden") and not name.startswith("_")
    ]
    raise RuntimeError(
        "cannot infer golden function; pass --function. "
        f"Candidates: {', '.join(names) or '(none)'}"
    )


def resolve_dtype(torch: ModuleType, dtype_name: str) -> Any:
    attr = dtype_name.strip().replace("torch.", "")
    dtype = getattr(torch, attr, None)
    if dtype is None:
        raise ValueError(f"unsupported torch dtype: {dtype_name!r}")
    return dtype


def make_tensor(torch: ModuleType, spec: TensorSpec, device: Any) -> Any:
    dtype = resolve_dtype(torch, spec.dtype_name)
    size = spec.shape if spec.shape else ()
    if (
        getattr(dtype, "is_floating_point", False)
        or getattr(dtype, "is_complex", False)
    ):
        return torch.randn(size, dtype=dtype, device=device)
    if dtype is torch.bool:
        return torch.randint(0, 2, size, device=device).to(dtype)
    low = 0 if str(dtype).replace("torch.", "").startswith("uint") else -3
    return torch.randint(low, 4, size, dtype=dtype, device=device)


def default_tensor_specs(fn: Callable[..., Any]) -> list[TensorSpec]:
    sig = inspect.signature(fn)
    for param in sig.parameters.values():
        if param.kind in _PARAM_KINDS and param.default is _EMPTY:
            return [TensorSpec(param.name, DEFAULT_INPUT_SHAPE, DEFAULT_DTYPE)]
    return [TensorSpec("x", DEFAULT_INPUT_SHAPE, DEFAULT_DTYPE)]


def build_call(
    torch: ModuleType,
    fn: Callable[..., Any],
    tensor_specs: list[TensorSpec],
    scalar_args: dict[str, Any],
    device: Any,
) -> Callable[[], Any]:
    sig = inspect.signature(fn)
    values = dict(scalar_args)
    specs = tensor_specs or default_tensor_specs(fn)

    unnamed_specs = [spec for spec in specs if not spec.name]
    named_specs = [spec for spec in specs if spec.name]
    required_params = []
    for p in sig.parameters.values():
        if p.default is _EMPTY and p.kind in _PARAM_KINDS:
            required_params.append(p)

    for spec in named_specs:
        values[spec.name] = make_tensor(torch, spec, device)

    unnamed_iter = iter(unnamed_specs)
    for param in required_params:
        if param.name in values:
            continue
        try:
            spec = next(unnamed_iter)
        except StopIteration:
            continue
        values[param.name] = make_tensor(
            torch,
            TensorSpec(param.name, spec.shape, spec.dtype_name),
            device,
        )

    missing = [p.name for p in required_params if p.name not in values]
    if missing:
        raise RuntimeError(
            "missing required golden arguments: "
            f"{', '.join(missing)}. Use --input NAME:SHAPE[:DTYPE] or "
            "--arg NAME=JSON_VALUE."
        )

    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    for param in sig.parameters.values():
        if param.name not in values:
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(values[param.name])
        elif param.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            keyword_args[param.name] = values[param.name]
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            extra = values[param.name]
            if not isinstance(extra, (list, tuple)):
                raise RuntimeError(f"*{param.name} must be a list or tuple")
            positional_args.extend(extra)
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            extra = values[param.name]
            if not isinstance(extra, dict):
                raise RuntimeError(f"**{param.name} must be a dict")
            keyword_args.update(extra)

    return lambda: fn(*positional_args, **keyword_args)


def repair_trace_json_if_truncated(trace_path: Path) -> None:
    if not trace_path.exists():
        return
    try:
        json.loads(trace_path.read_text())
        return
    except json.JSONDecodeError:
        pass

    content = trace_path.read_bytes().rstrip()
    if not content:
        return

    candidates: list[bytes] = []
    stripped = content.lstrip()
    if stripped.startswith(b"[") and not content.endswith(b"]"):
        candidates.append(content + b"]")
    elif stripped.startswith(b"{"):
        candidates.extend([content + b"]}", content + b"}", content + b"]"])

    for candidate in candidates:
        try:
            json.loads(candidate.decode())
        except json.JSONDecodeError:
            continue
        trace_path.write_bytes(candidate + b"\n")
        return


def load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    data = json.loads(trace_path.read_text())
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def extract_e2e_from_trace(trace_path: Path) -> dict[str, float]:
    perf_data: dict[str, list[dict]] = {"aicore_e2e": [], "aicpu_kernel": []}
    sync_events: list[float] = []
    events = load_trace_events(trace_path)
    for event in events:
        name = event.get("name", "")
        dur = event.get("dur")
        ts = event.get("ts")
        if dur is None or ts is None:
            continue
        event["end_time"] = float(ts) + float(dur)
        ts_f = float(ts)
        args = event.get("args")
        task_type = ""
        if isinstance(args, dict):
            task_type = args.get("Task Type", "")

        if "SynchronizeDevice" in name:
            sync_events.append(ts_f)
        elif name == "KERNEL_AICPU":
            perf_data["aicpu_kernel"].append(event)
        elif task_type and task_type not in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
            perf_data["aicore_e2e"].append(event)

    if not perf_data["aicore_e2e"]:
        return {"aicore_e2e": 0.0, "aicore_e2e_jitter": 0.0, "aicpukernel_gap": 0.0}

    perf_data["aicore_e2e"].sort(key=lambda e: float(e["ts"]))
    sync_events.sort()

    groups: list[list[dict]] = []
    if sync_events:
        ai = 0
        aicores_sorted = perf_data["aicore_e2e"]
        for sync_ts in sync_events:
            group: list[dict] = []
            while ai < len(aicores_sorted) and float(aicores_sorted[ai]["ts"]) < sync_ts:
                group.append(aicores_sorted[ai])
                ai += 1
            if group:
                groups.append(group)
        remaining = aicores_sorted[ai:]
        if remaining:
            groups.append(remaining)
    else:
        groups = [perf_data["aicore_e2e"]]

    per_iter_totals = [sum(evt["dur"] for evt in g) for g in groups if g]
    per_iter_sizes = [len(g) for g in groups if g]

    if per_iter_totals and len(per_iter_totals) > 1:
        median_total = sorted(per_iter_totals)[len(per_iter_totals) // 2]
        min_total = median_total * 0.2
        min_size = max(2, int(sum(per_iter_sizes) / len(per_iter_sizes) * 0.4))
        valid_indices = [
            i for i, (t, s) in enumerate(zip(per_iter_totals, per_iter_sizes))
            if t >= min_total and s >= min_size
        ]
        if valid_indices:
            per_iter_totals = [per_iter_totals[i] for i in valid_indices]
            groups = [groups[i] for i in valid_indices]

    if len(per_iter_totals) > 1:
        min_total = min(per_iter_totals)
        threshold = 1.5 * min_total
        filtered = [t for t in per_iter_totals if (t - min_total) < threshold]
    else:
        filtered = list(per_iter_totals)
    aicore_e2e = round(sum(filtered) / len(filtered), 2) if filtered else 0.0

    n_stable = max(min(5, len(per_iter_totals)), len(per_iter_totals) * 2 // 5)
    jitter_samples = per_iter_totals[-n_stable:]
    aicore_e2e_jitter = (
        (max(jitter_samples) - min(jitter_samples)) / min(jitter_samples)
        if min(jitter_samples) > 0 else 0.0
    )

    aicore_e2e_time_list: list[list[float]] = [
        [float(g[0]["ts"]), g[-1]["end_time"]] for g in groups if g
    ]
    for ak in perf_data["aicpu_kernel"]:
        s = float(ak["ts"])
        e = ak["end_time"]
        for et in aicore_e2e_time_list:
            if s <= et[0] and e >= et[-1]:
                et.append(e)
                break
    gap_list: list[float] = []
    for et in aicore_e2e_time_list:
        gap = 0.0 if len(et) == 2 else max(et[-1] - et[-2], 0.0)
        gap_list.append(gap)
    aicpukernel_gap = round(sum(gap_list) / len(gap_list), 2) if gap_list else 0.0

    return {
        "per_iter_totals": per_iter_totals,
        "aicore_e2e": aicore_e2e,
        "aicore_e2e_jitter": round(aicore_e2e_jitter, 2),
        "aicpukernel_gap": aicpukernel_gap,
    }


def summarize_input_shapes(specs: list[TensorSpec]) -> str:
    if len(specs) == 1:
        return str(specs[0].shape)
    return "{" + ", ".join(f"{s.name or '?'}: {s.shape}" for s in specs) + "}"


def summarize_dtypes(specs: list[TensorSpec]) -> str:
    if len(specs) == 1:
        return f"torch.{specs[0].dtype_name.replace('torch.', '')}"
    return "{" + ", ".join(
        f"{s.name or '?'}: torch.{s.dtype_name.replace('torch.', '')}"
        for s in specs
    ) + "}"


def write_perf_report(config: PerfReportConfig) -> Path:
    op_durs: dict[str, list[float]] = {}
    parse_error: str | None = None
    if config.trace_path is not None:
        try:
            events = load_trace_events(config.trace_path)
            for event in events:
                name = event.get("name", "")
                dur = event.get("dur")
                args = event.get("args")
                task_type = ""
                if isinstance(args, dict):
                    task_type = args.get("Task Type", "")
                if not task_type or task_type in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
                    continue
                if name and dur is not None:
                    op_durs.setdefault(name, []).append(float(dur))
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            parse_error = str(exc)
    else:
        parse_error = "trace_view.json not found"

    op_name = config.golden_path.stem.replace("_golden", "")
    report_path = config.output_dir / "GOLDEN_PERF_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {op_name} Golden NPU Performance Report",
        "",
        f"- **Device**: {config.device}",
        f"- **Input Shape**: {summarize_input_shapes(config.tensor_specs)}",
        f"- **dtype**: {summarize_dtypes(config.tensor_specs)}",
        f"- **Timestamp**: {datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}",
        f"- **Profiling Data**: `prof/{config.prof_dir.name}/`",
        "",
        "## Op Performance",
        "",
        "| op | warmup_avg | stable_avg | stable_min | stable_max |",
        "|----|-----------|-----------|-----------|-----------|",
    ]

    for name in sorted(op_durs):
        durs = op_durs[name]
        if len(durs) <= config.n_warmup:
            continue
        warm = durs[:config.n_warmup]
        stable = durs[config.n_warmup:]
        lines.append(
            f"| {name} "
            f"| {sum(warm)/len(warm):.1f}us "
            f"| {sum(stable)/len(stable):.1f}us "
            f"| {min(stable):.1f}us "
            f"| {max(stable):.1f}us |"
        )

    if parse_error:
        lines += ["", f"Trace parse warning: {parse_error}"]

    fn_name = config.golden_path.stem
    per_iter = config.kernel_e2e.get("per_iter_totals", [])

    e2e_lines: list[str] = []
    if per_iter:
        warm = per_iter[:config.n_warmup]
        stable = per_iter[config.n_warmup:]
        e2e_lines = [
            "",
            "## Golden Kernel Performance (from profiler trace)",
            "",
            "Per-call total AICore kernel time, grouped by `SynchronizeDevice` "
            "boundaries in `trace_view.json`.",
            "",
            "| metric | warmup_avg | stable_avg | stable_min | stable_max |",
            "|--------|-----------|-----------|-----------|-----------|",
            f"| {fn_name} (kernel E2E) "
            f"| {sum(warm)/len(warm):.1f}us "
            f"| {sum(stable)/len(stable):.1f}us "
            f"| {min(stable):.1f}us "
            f"| {max(stable):.1f}us |"
            if warm and stable else (
            f"| {fn_name} (kernel E2E) "
            f"| {sum(per_iter)/len(per_iter):.1f}us "
            f"| {sum(per_iter)/len(per_iter):.1f}us "
            f"| {min(per_iter):.1f}us "
            f"| {max(per_iter):.1f}us |"
            ),
        ]
    else:
        e2e_lines = [
            "",
            "## Golden Kernel Performance (from profiler trace)",
            "",
            "No AICore kernel events found.",
        ]

    lines += e2e_lines
    lines += [
        "",
        "## Notes",
        "",
        f"- `warmup_avg`: first {config.n_warmup} iterations "
        "(includes JIT/compile overhead)",
        "- `stable_avg/min/max`: subsequent iterations "
        "(steady-state kernel execution)",
        "- Kernel E2E extracted from profiler trace, not host-side wall-clock",
        f"- Full trace: `prof/{config.prof_dir.name}/trace_view.json` "
        "(chrome://tracing)",
        f"- CANN analysis: `prof/{config.prof_dir.name}/` "
        "(operator_memory.csv, device counters)",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def profile_golden(config: ProfileConfig) -> int:
    torch, torch_npu = import_npu_stack()
    if not has_npu_hardware(torch):
        logger.info("NPU profiling skipped: no NPU hardware available.")
        return 0

    golden_path = config.golden_path.resolve()
    output_dir = (config.output_dir or golden_path.parent).resolve()
    prof_dir = output_dir / "prof" / golden_path.stem
    prof_dir.mkdir(parents=True, exist_ok=True)

    module = load_python_module(golden_path)
    fn = find_golden_function(module, golden_path, config.function_name)
    specs = config.tensor_specs or default_tensor_specs(fn)
    device = torch.device(f"npu:{config.device_id}") if config.device_id is not None else torch.device("npu")
    call_golden = build_call(torch, fn, specs, config.scalar_args, device)

    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(prof_dir), analyse_flag=True
        ),
    )

    with profiler as prof:
        for _ in range(config.warmup + config.iters):
            _ = call_golden()
            torch.npu.synchronize()
            prof.step()

    trace_view_path: Path | None = None
    for root, _, files in os.walk(prof_dir):
        for name in files:
            if name == "trace_view.json":
                trace_view_path = Path(root) / name
                break
        if trace_view_path:
            break

    if trace_view_path:
        repair_trace_json_if_truncated(trace_view_path)

    logger.info(f"Profiling output: {prof_dir}")

    kernel_e2e = extract_e2e_from_trace(trace_view_path) if trace_view_path else {
        "per_iter_totals": [], "aicore_e2e": 0.0, "aicore_e2e_jitter": 0.0, "aicpukernel_gap": 0.0,
    }

    report_config = PerfReportConfig(
        golden_path=golden_path,
        output_dir=output_dir,
        trace_path=trace_view_path,
        n_warmup=config.warmup,
        tensor_specs=specs,
        device=device,
        prof_dir=prof_dir,
        kernel_e2e=kernel_e2e,
    )
    report_path = write_perf_report(report_config)
    logger.info(f"Performance report: {report_path}")
    return 0


def run_self_test() -> None:
    def _check(condition: bool, msg: str = "") -> None:
        if not condition:
            raise AssertionError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trace = base / "chrome_trace.json"
        trace.write_text(
            json.dumps({
                "traceEvents": [
                    {"name": "aten::mul", "dur": 10,
                     "args": {"Task Type": "AI_VECTOR_CORE"}},
                    {"name": "aten::mul", "dur": 20,
                     "args": {"Task Type": "AI_VECTOR_CORE"}},
                    {"name": "aten::mul", "dur": 30,
                     "args": {"Task Type": "AI_VECTOR_CORE"}},
                ]
            })
        )
        _check(len(load_trace_events(trace)) == 3,
               f"expected 3 trace events, got {len(load_trace_events(trace))}")

        truncated = base / "trace_view.json"
        truncated.write_text('[{"name":"aten::add","dur":1}')
        repair_trace_json_if_truncated(truncated)
        _check(len(load_trace_events(truncated)) == 1,
               f"expected 1 trace event, got {len(load_trace_events(truncated))}")

        obj_truncated = base / "trace_object.json"
        obj_truncated.write_text('{"traceEvents":[{"name":"aten::sub","dur":2}')
        repair_trace_json_if_truncated(obj_truncated)
        _check(len(load_trace_events(obj_truncated)) == 1,
               f"expected 1 trace event, got {len(load_trace_events(obj_truncated))}")

        e2e_trace = base / "e2e_trace.json"
        e2e_trace.write_text(json.dumps([
            {"name": "kernel1", "dur": 100, "ts": 0,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
            {"name": "kernel2", "dur": 110, "ts": 200,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
            {"name": "kernel3", "dur": 105, "ts": 400,
             "args": {"Task Type": "AI_CORE"}},
            {"name": "kernel4", "dur": 500, "ts": 600,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
        ]))
        result = extract_e2e_from_trace(e2e_trace)
        _check(result["per_iter_totals"] == [815.0],
               f"expected [815.0], got {result['per_iter_totals']}")
        _check(math.isclose(result["aicore_e2e"], 815.0),
               f"expected 815.0, got {result['aicore_e2e']}")

        e2e_trace2 = base / "e2e_trace2.json"
        e2e_trace2.write_text(json.dumps([
            {"name": "aclnnCast_CastAiCore_Cast", "dur": 5, "ts": 0,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
            {"name": "aclnnMatmul_BatchMatMulNd_BatchMatMulV2", "dur": 35, "ts": 50,
             "args": {"Task Type": "AI_CORE"}},
            {"name": "AscendCL@aclrtSynchronizeDeviceWithTimeout", "dur": 1, "ts": 100},
            {"name": "aclnnCast_CastAiCore_Cast", "dur": 3, "ts": 200,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
            {"name": "aclnnMatmul_BatchMatMulNd_BatchMatMulV2", "dur": 38, "ts": 250,
             "args": {"Task Type": "AI_CORE"}},
            {"name": "AscendCL@aclrtSynchronizeDeviceWithTimeout", "dur": 1, "ts": 300},
            {"name": "aclnnCast_CastAiCore_Cast", "dur": 6, "ts": 400,
             "args": {"Task Type": "AI_VECTOR_CORE"}},
            {"name": "aclnnMatmul_BatchMatMulNd_BatchMatMulV2", "dur": 34, "ts": 450,
             "args": {"Task Type": "AI_CORE"}},
        ]))
        result2 = extract_e2e_from_trace(e2e_trace2)
        _check(result2["per_iter_totals"] == [40.0, 41.0, 40.0],
               f"expected [40.0, 41.0, 40.0], got {result2['per_iter_totals']}")

        spec = parse_tensor_spec("x:2x3:float32")
        _check(spec == TensorSpec("x", (2, 3), "float32"),
               f"expected TensorSpec('x', (2, 3), 'float32'), got {spec}")
        _check(parse_shape("(2, 3)") == (2, 3),
               f"expected (2, 3), got {parse_shape('(2, 3)')}")
        _check(parse_named_json("eps=1e-5") == ("eps", 1e-5),
               f"expected ('eps', 1e-5), got {parse_named_json('eps=1e-5')}")

        import torch
        _check(make_tensor(
            torch, TensorSpec("idx", (2, 3), "int64"), torch.device("cpu")
        ).dtype == torch.int64, "expected torch.int64")
        _check(make_tensor(
            torch, TensorSpec("mask", (2, 3), "bool"), torch.device("cpu")
        ).dtype == torch.bool, "expected torch.bool")

        golden_path = base / "sample_golden.py"
        golden_path.write_text("def sample_golden(x):\n    return x\n")
        report_config = PerfReportConfig(
            golden_path=golden_path,
            output_dir=base,
            trace_path=trace,
            n_warmup=1,
            tensor_specs=[spec],
            device="npu",
            prof_dir=base / "prof" / "sample_golden",
            kernel_e2e={"per_iter_totals": [180.0, 170.0, 175.0, 172.0, 168.0,
                           160.0, 162.0, 158.0, 161.0, 159.0],
                       "aicore_e2e": 165.0, "aicore_e2e_jitter": 0.02, "aicpukernel_gap": 0.0},
        )
        report = write_perf_report(report_config)
        content = report.read_text()
        _check("Golden Kernel Performance" in content,
               "'Golden Kernel Performance' not in report")
        _check("| aten::mul | 10.0us | 25.0us | 20.0us | 30.0us |" in content,
               "expected table row not in report")

    logger.info("profile_golden self-test ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile a PyTorch NPU golden script and write "
                    "GOLDEN_PERF_REPORT.md plus prof/<golden_stem>/ data."
    )
    parser.add_argument("golden", nargs="?", help="Path to *_golden.py")
    parser.add_argument("--function", help="Golden function name")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME:SHAPE[:DTYPE]",
        help="Tensor input spec. Example: x:8x1024x4096:float32. "
             "Repeat for multiple tensor inputs.",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Non-tensor golden argument. Example: eps=1e-5 or axis=-1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for GOLDEN_PERF_REPORT.md and prof/. "
             "Defaults to the golden file directory.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="DEVICE_ID",
        help="NPU device ID (integer). Defaults to 0 (system default). "
             "Use this to target a specific NPU card, e.g. --device 3.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--self-test", action="store_true",
                        help="Run built-in helper tests without NPU.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if not args.golden:
        parser.error("golden path is required unless --self-test is used")
    if args.warmup <= 0:
        parser.error("--warmup must be > 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")

    try:
        tensor_specs = [parse_tensor_spec(item) for item in args.input]
        scalar_args = dict(parse_named_json(item) for item in args.arg)
        config = ProfileConfig(
            golden_path=Path(args.golden),
            function_name=args.function,
            tensor_specs=tensor_specs,
            scalar_args=scalar_args,
            output_dir=args.output_dir,
            warmup=args.warmup,
            iters=args.iters,
            device_id=args.device,
        )
        return profile_golden(config)
    except Exception as exc:
        logger.error(f"{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

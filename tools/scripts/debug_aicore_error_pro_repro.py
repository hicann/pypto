#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use this
# file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
# IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
# full text of the License.
# -----------------------------------------------------------------------------------------------------------

r"""PyPTO Pro AICORE Error 离线复现工具。

从异常 dump 的 extra-info/data-dump 目录中读取 tensor 数据和编译产物，
构建单算子复现脚本并执行，验证 AICORE error 是否重现。

用法示例
--------
python debug_aicore_error_pro_repro.py -p <work_dir> [-d <device_id>]
"""

import argparse
import dataclasses
import glob
import inspect
import logging
import os
import re
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")


_STR_TO_TORCH = {
    "float16": "torch.float16", "float32": "torch.float32", "float64": "torch.float64",
    "int8": "torch.int8", "int16": "torch.int16", "int32": "torch.int32", "int64": "torch.int64",
    "uint8": "torch.uint8", "uint16": "torch.int16", "uint32": "torch.int32", "uint64": "torch.int64",
    "bfloat16": "torch.bfloat16", "bool": "torch.bool",
    "fp8e4m3fn": "torch.float8_e4m3fn", "fp8e5m2": "torch.float8_e5m2",
}

_STR_TO_NP = {
    "float16": "np.float16", "float32": "np.float32", "float64": "np.float64",
    "int8": "np.int8", "int16": "np.int16", "int32": "np.int32", "int64": "np.int64",
    "uint8": "np.uint8", "uint16": "np.int16", "uint32": "np.int32", "uint64": "np.int64",
    "bfloat16": "np.int16", "bool": "np.bool_",
    "fp8e4m3fn": "np.uint8", "fp8e5m2": "np.uint8",
}

_DTYPE_TO_SIZE = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "bfloat16": 2, "bool": 1,
    "fp8e4m3fn": 1, "fp8e5m2": 1,
}

_DT_NAME_TO_STR = {
    "FLOAT": "float32", "FLOAT16": "float16", "FLOAT64": "float64",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
    "BOOL": "bool", "BFLOAT16": "bfloat16",
    "FP8E5M2": "fp8e5m2", "FP8E4M3FN": "fp8e4m3fn",
}

_SIZE_TO_DTYPE = {1: "uint8", 2: "float16", 4: "float32", 8: "float64"}


@dataclasses.dataclass
class TensorInfo:
    addr: int
    size: int
    dtype_str: str
    shape: tuple
    io_type: str
    index: int


def _parse_dump_file(dump_file: str) -> Tuple[str, List[TensorInfo]]:
    """从 dump 文件中解析 kernel name 和 tensor 信息。

    dump 文件格式:
    - 8 字节 uint64 total_size
    - protobuf 二进制头: 每个 tensor 的 dataType / shape / size / index
    - 文本元数据: [Dump][Exception] 行，记录 type / index / shape / dtype / address / size
    - 纯 tensor 数据（按 index 顺序连续排列）
    """
    with open(dump_file, "rb") as f:
        content = f.read()

    text = content.decode("utf-8", errors="replace")

    tensors: List[TensorInfo] = []
    for m in re.finditer(
        r"type=(\w+);\s*index=(\d+);\s*shape=\[([\d,]*)\];\s*format=\w+;\s*dtype=DT_(\w+);\s*address=(0x[0-9a-fA-F]+);\s*size=(\d+)\s*bytes",
        text
    ):
        io_type, index = m.group(1), int(m.group(2))
        shape = tuple(int(x) for x in m.group(3).split(",")) if m.group(3) else ()
        addr, size = int(m.group(5), 16), int(m.group(6))

        # dtype: trust the dump header; fall back to size/shape inference
        # (bytes per element) for unknown names or size mismatches.
        elems = 1
        for s in shape:
            elems *= s
        per_elem = size // elems if elems else 0
        dtype_str = _DT_NAME_TO_STR.get(m.group(4))
        if dtype_str is None:
            dtype_str = _SIZE_TO_DTYPE.get(per_elem, "float16") if elems else "uint8"
        elif elems and _DTYPE_TO_SIZE.get(dtype_str, -1) != per_elem:
            fixed = _SIZE_TO_DTYPE.get(per_elem, dtype_str)
            logging.warning(f"  tensor[{index}] dtype {dtype_str} mismatch with size/shape, using {fixed}")
            dtype_str = fixed

        tensors.append(TensorInfo(addr, size, dtype_str, shape, io_type, index))
        logging.info(f"  tensor[{index}] {io_type} addr=0x{addr:x} size={size} dtype={dtype_str} shape={shape}")

    basename = os.path.basename(dump_file)
    parts = basename.split(".")
    kernel_name = parts[0] if parts else "unknown"

    return kernel_name, tensors


def _find_call_kernel_so(work_dir: str, dump_dir: Optional[str] = None,
                         kernel_name: Optional[str] = None) -> Optional[str]:
    """查找 call_kernel.so。

    优先使用异常回调复制到 dump 目录的 <kernel_name>_call_kernel.so 副本
    （随 dump 产物一起保存，不依赖 build 目录存活）；dump 目录没有副本时，
    再回退到 build 目录下搜索 call_kernel*.so（JIT 编译产物带 content digest 后缀）。
    """
    if dump_dir and os.path.isdir(dump_dir):
        if kernel_name:
            exact = os.path.join(dump_dir, f"{kernel_name}_call_kernel.so")
            if os.path.isfile(exact):
                return exact
        copies = sorted(f for f in os.listdir(dump_dir) if f.endswith("_call_kernel.so"))
        if copies:
            return os.path.join(dump_dir, copies[0])
    search_dirs = [
        os.path.join(work_dir, "build"),
        os.path.join(os.path.dirname(os.path.abspath(work_dir)), "build"),
        os.path.join(os.getcwd(), "build"),
        os.path.join(os.environ.get("ASCEND_WORK_PATH", ""), "build"),
    ]
    for build_dir in search_dirs:
        if os.path.isdir(build_dir):
            hits = glob.glob(os.path.join(build_dir, "**", "call_kernel*.so"), recursive=True)
            if hits:
                return hits[0]
    return None


def _elf_sections(data: bytes) -> Dict[str, Tuple[int, int]]:
    """Parse a little-endian ELF64's section table: name -> (file offset, size)."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return {}
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    if not e_shoff or not e_shnum:
        return {}

    def shdr(i):
        base = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from("<I", data, base)[0]
        sh_offset = struct.unpack_from("<Q", data, base + 0x18)[0]
        sh_size = struct.unpack_from("<Q", data, base + 0x20)[0]
        sh_type = struct.unpack_from("<I", data, base + 4)[0]
        sh_link = struct.unpack_from("<I", data, base + 0x28)[0]
        sh_entsize = struct.unpack_from("<Q", data, base + 0x38)[0]
        return sh_name, sh_offset, sh_size, sh_type, sh_link, sh_entsize

    str_off = shdr(e_shstrndx)[1]
    out: Dict[str, Tuple[int, int]] = {}
    for i in range(e_shnum):
        sh_name, sh_offset, sh_size, sh_type, sh_link, sh_entsize = shdr(i)
        end = data.index(b"\0", str_off + sh_name)
        name = data[str_off + sh_name:end].decode("utf-8", "replace")
        out[name] = (sh_offset, sh_size)
    return out


def _find_bisheng_tools() -> Optional[Tuple[str, str]]:
    """Locate the bisheng toolchain's ld.lld + llvm-objdump used by the CCE JIT.

    The pair relocations in the embedded device ELF's .debug_line are applied at
    link time by this lld, so linking the embedded ELF once through it yields a
    file whose line table stock LLVM consumers (llvm-objdump -d -l) can decode.
    Returns (ld_lld, llvm_objdump) or None when the toolchain is unavailable.
    """
    override = os.environ.get("PYPTO_BISHENG_BIN")
    roots = []
    if override:
        roots.append(override)
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if ascend:
        roots.append(os.path.join(ascend, "tools", "bisheng_compiler", "bin"))
        roots.append(os.path.join(ascend, "..", "tools", "bisheng_compiler", "bin"))
        roots.append(os.path.join(ascend, "bin"))
    candidates = []
    for root in roots:
        ld = os.path.join(root, "ld.lld")
        objdump = os.path.join(root, "llvm-objdump")
        if os.path.isfile(ld) and os.path.isfile(objdump):
            candidates.append((ld, objdump))
    return candidates[0] if candidates else None


def _resolve_pc_to_source_via_tools(so_path: str, pc: int, tools: Tuple[str, str],
                                    symbol: str = "", symbol_offset: int = 0) -> Optional[Tuple[str, int]]:
    """Resolve a device Error-PC to (file, line) via the bisheng toolchain.

    1. Cut the embedded device ELF (section __aicore_rel_binary) out of the .so.
    2. ld.lld -shared -Ttext=0 links it, applying the CCE pair relocations to
       .debug_line and placing .text at address 0 (PC offsets need no rebase).
    3. When the plog carries symbol+offset, rebase the PC through the symbol
       table: fatobj (mix) kernels have their AIV entry symbol at a non-zero
       .text offset, which shifts fixedPCOffset by exactly that amount.
    4. llvm-objdump -d -l interleaves source markers with instruction
       addresses; replay the (marker, address) pairs and pick the last row
       whose address is <= pc.

    The linked device ELF is persisted next to the .so as
    <kernel>_device_linked.so (relocations applied, .text at 0), so the dump
    directory keeps a directly-symbolizable device image for offline analysis.
    """
    import shutil
    import tempfile

    ld, objdump = tools
    data = open(so_path, "rb").read()
    outer = _elf_sections(data)
    rel = outer.get("__aicore_rel_binary")
    if rel is None:
        return None
    tmp = tempfile.mkdtemp(prefix="pc_resolve_")
    try:
        obj = os.path.join(tmp, "device.o")
        with open(obj, "wb") as f:
            f.write(data[rel[0]:rel[0] + rel[1]])
        linked = os.path.join(tmp, "device_linked.so")
        try:
            subprocess.run([ld, "-shared", "-Ttext=0", obj, "-o", linked],
                           capture_output=True, timeout=120, check=True)
        except (subprocess.SubprocessError, OSError):
            return None
        base = os.path.basename(so_path)
        stem = base[:-len("_call_kernel.so")] if base.endswith("_call_kernel.so") else "device"
        try:
            shutil.copyfile(linked, os.path.join(os.path.dirname(so_path), f"{stem}_device_linked.so"))
        except OSError:
            pass
        if symbol:
            try:
                syms = subprocess.run([objdump, "--syms", linked],
                                      capture_output=True, text=True, timeout=120, check=True)
                for line in syms.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 6 and parts[3] == ".text" and parts[5] == symbol:
                        pc = int(parts[0], 16) + symbol_offset
                        logging.info(f"symbol {symbol}+0x{symbol_offset:x} -> PC 0x{pc:x}")
                        break
            except (subprocess.SubprocessError, OSError, ValueError):
                pass
        try:
            result = subprocess.run(
                [objdump, "-d", "-l", linked],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if result.returncode != 0:
            return None
        rows: List[Tuple[int, str, int]] = []
        marker = re.compile(r"^(.+[^()]):(\d+)$")
        addr_re = re.compile(r"^\s+([0-9a-fA-F]+):")
        cur_file: Optional[str] = None
        cur_line = 0
        for raw in result.stdout.splitlines():
            line = raw.rstrip()
            m = marker.match(line)
            if m and "/" in m.group(1):
                cur_file, cur_line = m.group(1), int(m.group(2))
                continue
            ma = addr_re.match(line)
            if ma and cur_file is not None:
                rows.append((int(ma.group(1), 16), cur_file, cur_line))
        valid = [r for r in rows if r[2] and r[0] <= pc]
        if not valid:
            return None
        return valid[-1][1], valid[-1][2]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve_pc_to_source(so_path: str, pc: int, symbol: str = "", symbol_offset: int = 0) -> Optional[Tuple[str, int]]:
    """Map a device Error-PC offset to (file, line) via the bisheng toolchain.

    The device code's .debug_line carries CCE pair relocations that stock DWARF
    consumers cannot apply. The toolchain's own ld.lld applies them at link
    time, so the embedded device ELF is linked once (-Ttext=0, no rebase
    needed) and decoded with the toolchain's llvm-objdump. Returns None when
    the toolchain is unavailable.
    """
    tools = _find_bisheng_tools()
    if tools is None:
        logging.warning("bisheng toolchain not found, Error-PC source line unavailable")
        return None
    return _resolve_pc_to_source_via_tools(so_path, pc, tools, symbol=symbol, symbol_offset=symbol_offset)


def _source_snippet(src_file: str, src_line: int, context: int = 3) -> str:
    """Render a few source lines around src_line for the localization report."""
    try:
        with open(src_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    start = max(src_line - context, 1)
    end = min(src_line + context, len(lines))
    out = []
    for n in range(start, end + 1):
        marker = ">" if n == src_line else " "
        out.append(f"  {marker} {n:5d} | {lines[n - 1].rstrip()}")
    return "\n".join(out) + "\n"


def _locate_error_pc(so_path: str, work_dir: str, since: float) -> Optional[Tuple[int, str, Optional[Tuple[str, int]]]]:
    """Locate the Error PC from freshly written plogs and resolve it to source.

    Embedded verbatim (together with the resolver helpers) into generated
    repro scripts so they can print the localization themselves. Returns
    (pc, symbol, (file, line) | None) or None.

    The symbol line carries the true PC as symbol+offset; it is preferred over
    fixedPCOffset, whose zero-based normalization assumes a single kernel
    entry at .text 0. On fatobj (mix) kernels the AIV entry symbol sits at a
    non-zero .text offset, and fixedPCOffset is then shifted by exactly that
    amount, pointing at the wrong source row.
    """
    pc_line_re = re.compile(r"Error PC information\..*?fixedPCOffset=0x([0-9a-fA-F]+)")
    sym_re = re.compile(r"Error symbol information\..*?symbol=(\S+?)(?:\+0x([0-9a-fA-F]+))?\.?\s*$", re.MULTILINE)
    plogs: List[Tuple[float, str]] = []
    base = os.path.join(work_dir, "log")
    if os.path.isdir(base):
        for root, _, fnames in os.walk(base):
            for fname in fnames:
                fpath = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    continue
                if mtime >= since:
                    plogs.append((mtime, fpath))
    for _, plog in sorted(plogs, reverse=True):
        try:
            with open(plog, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        m = pc_line_re.search(text)
        if m:
            sym, sym_off = "", 0
            ms = sym_re.search(text)
            if ms:
                sym = ms.group(1)
                sym_off = int(ms.group(2), 16) if ms.group(2) else 0
            pc = int(m.group(1), 16)
            src = _resolve_pc_to_source(so_path, pc, symbol=sym, symbol_offset=sym_off)
            return pc, sym, src
    return None


def codegen_test_script(
    call_kernel_so: str,
    kernel_name: str,
    dump_file: str,
    tensors: List[TensorInfo],
    device_id: int,
    output_path: str,
    block_dim: int = 1,
) -> None:
    """生成单算子复现脚本。"""
    with open(dump_file, "rb") as f:
        all_data = f.read()

    # Tensor data sits after the [Dump] text header (which ends right after
    # the last "bytes\n" line within its first 4KB).
    dump_text_start = all_data.find(b"[Dump]")
    region = all_data[dump_text_start:dump_text_start + 4096] if dump_text_start != -1 else b""
    last_bytes = region.rfind(b"bytes\n")
    text_end = dump_text_start + last_bytes + len(b"bytes\n") if last_bytes != -1 else 0

    tensor_data_list = []
    offset = text_end
    for t in sorted(tensors, key=lambda x: x.index):
        tensor_data = all_data[offset:offset + t.size]
        tensor_data_list.append(tensor_data)
        offset += t.size

    bin_dir = os.path.dirname(output_path)
    load_lines: List[str] = []
    tensor_vars: List[str] = []
    tensors_sorted = sorted(tensors, key=lambda x: x.index)

    for t, data in zip(tensors_sorted, tensor_data_list):
        bin_path = os.path.join(bin_dir, f"tensor_{t.io_type}_{t.index}.bin")
        with open(bin_path, "wb") as f:
            f.write(data)

        var = f"t_{t.io_type}_{t.index}"
        torch_dtype = _STR_TO_TORCH.get(t.dtype_str, "torch.float16")
        np_dtype = _STR_TO_NP.get(t.dtype_str, "np.float16")
        shape_repr = repr(t.shape) if t.shape else "(-1,)"
        load_lines.append(f"{var}_bin = r'{bin_path}'")
        load_lines.append(f"{var}_np = np.fromfile({var}_bin, dtype={np_dtype})")
        load_lines.append(f"{var}_np = {var}_np.reshape({shape_repr})")
        load_lines.append(f"{var} = torch.tensor(")
        load_lines.append(f"    {var}_np, device=device).view({torch_dtype})")
        tensor_vars.append(var)

    # Dynamic-shape tail args: the launch ABI is
    # call_kernel(block_dim, stream, ptr..., int32 dyn...) where each
    # __pypto_dyn_<param>_<k> arg takes the k-th dim of that tensor param's
    # runtime shape. Parse the call_kernel.cpp copy shipped in the dump dir to
    # bind them exactly; without it the C side would read garbage registers.
    dyn_vals: List[str] = []
    ck_cpp = os.path.join(os.path.dirname(call_kernel_so), "call_kernel.cpp")
    if os.path.isfile(ck_cpp):
        try:
            with open(ck_cpp, encoding="utf-8", errors="replace") as f:
                ck_text = f.read()
            m = re.search(r'extern "C" void call_kernel\((.*?)\)', ck_text, re.S)
            params = [p.strip() for p in m.group(1).split(",")] if m else []
            arg_names = [p.split()[-1].lstrip("*") for p in params[2:]]
            ptr_names = [n for n in arg_names if not n.startswith("__pypto_dyn_")]
            var_by_param = dict(zip(ptr_names, tensor_vars))
            for name in arg_names:
                dm = re.fullmatch(r"__pypto_dyn_(.+)_(\d+)", name)
                if dm and dm.group(1) in var_by_param:
                    dyn_vals.append(f"{var_by_param[dm.group(1)]}.shape[{dm.group(2)}]")
        except (OSError, AttributeError, IndexError):
            dyn_vals = []
    if not dyn_vals:
        logging.warning("call_kernel.cpp signature not usable, launching without dynamic-shape args")

    lines = [
        "#!/usr/bin/env python3",
        "# Auto-generated by debug_aicore_error_pro_repro.py",
        f"# Kernel: {kernel_name}",
        f"# Device: {device_id}",
        f"# block_dim: {block_dim}",
        "",
        "import ctypes",
        "import logging",
        "import os",
        "import re",
        "import struct",
        "import subprocess",
        "import sys",
        "import time",
        "from typing import Dict, List, Optional, Tuple",
        "",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "",
        "",
    ]

    # Embed the Error-PC line resolver so this script is self-contained: on
    # reproduction it prints the source file/line itself, without needing the
    # repro tool or the build directory. Sourced live via inspect so the copy
    # can never drift from the tool's own implementation.
    lines.append(inspect.getsource(_elf_sections).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_find_bisheng_tools).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_to_source_via_tools).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_to_source).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_locate_error_pc).rstrip())
    lines.append("")
    lines.append("")

    lines += [
        f"device = torch.device('npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        f"_so_path = r'{call_kernel_so}'",
        "_SO = ctypes.CDLL(_so_path)",
        "",
    ]
    lines.extend(load_lines)
    lines += [
        "",
        f"_args = [{', '.join(tensor_vars)}]",
        "_ctypes_args = [ctypes.c_void_p(t.data_ptr()) for t in _args]",
        f"_dyn_args = [{', '.join(dyn_vals)}]",
        "_stream = torch.npu.current_stream()",
        f"_block_dim = {block_dim}",
        "",
        "print(f'Launching kernel with block_dim={_block_dim}')",
        "_SO.call_kernel(_block_dim, getattr(_stream, '_as_parameter_'), *_ctypes_args, *_dyn_args)",
        "try:",
        "    torch.npu.synchronize()",
        "    print('PASS: No AICORE error — execution succeeded')",
        "    sys.exit(0)",
        "except Exception as e:",
        "    print(f'FAIL: AICORE error reproduced: {e}')",
        "    _work = os.environ.get('ASCEND_WORK_PATH', os.getcwd())",
        "    _hit = _locate_error_pc(_so_path, _work, time.time() - 120.0)",
        "    if _hit:",
        "        _pc, _sym, _src = _hit",
        "        print(f'ErrorPC: 0x{_pc:x}')",
        "        print(f'Symbol: {_sym or \"<unknown>\"}')",
        "        if _src:",
        "            _local = os.path.join(os.path.dirname(_so_path), os.path.basename(_src[0]))",
        "            if os.path.isfile(_local):",
        "                _src = (_local, _src[1])",
        "            print(f'Source: {_src[0]}:{_src[1]}')",
        "    else:",
        "        print('ErrorPC: <not found in plog>')",
        "    sys.exit(1)",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Test script generated: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyPTO Pro AICORE Error Offline Reproduction Tool")
    parser.add_argument("-p", type=str, required=True,
                        help="Directory holding the dump artifacts for ONE device "
                             "(ASCEND_WORK_PATH/extra-info/data-dump/<device_id>, "
                             "where the dump data file and _call_kernel.so live)")
    parser.add_argument("-d", type=str, default=None, help="Device ID")
    parser.add_argument("-out", type=str, default=None,
                        help="Output directory (default: <work_dir>/aicore_error_debug)")
    parser.add_argument("-t", type=int, default=600, help="Timeout (seconds)")
    return parser.parse_args()


def _work_dir_from_dump_dir(dump_dir: str) -> str:
    """dump 目录位于 <work>/extra-info/data-dump/<device_id>,上溯三级得到 work 目录。"""
    return os.path.abspath(os.path.join(dump_dir, os.pardir, os.pardir, os.pardir))


def main():
    args = parse_args()

    dump_dir = os.path.abspath(args.p)
    if not os.path.isdir(dump_dir):
        logging.error(f"dump directory not found: {dump_dir}")
        sys.exit(1)
    work_dir = _work_dir_from_dump_dir(dump_dir)

    # Multi-card safe: the device id comes from the dump directory itself
    # (.../data-dump/<device_id>/), never defaulted to card 0. -d overrides.
    dev_name = os.path.basename(dump_dir.rstrip(os.sep))
    device_id = int(args.d) if args.d is not None else int(dev_name) if dev_name.isdigit() else 0
    logging.info(f"device_id = {device_id}")
    logging.info(f"dump dir = {dump_dir}")

    if args.out is None:
        args.out = os.path.join(work_dir, "aicore_error_debug")
    os.makedirs(args.out, exist_ok=True)
    logging.info(f"output dir = {args.out}")

    dump_file = None
    for fname in sorted(os.listdir(dump_dir)):
        fpath = os.path.join(dump_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith(("_host.o", "_debug.o", ".so", ".cpp", ".h")):
            dump_file = fpath
            break
    if dump_file is None:
        logging.error("No dump data file found")
        sys.exit(1)
    logging.info(f"dump file = {dump_file}")

    kernel_name, tensors = _parse_dump_file(dump_file)
    if not tensors:
        logging.error("No tensors parsed from dump file")
        sys.exit(1)
    logging.info(f"kernel_name = {kernel_name}, {len(tensors)} tensors")

    call_kernel_so = _find_call_kernel_so(work_dir, dump_dir=dump_dir, kernel_name=kernel_name)
    if not call_kernel_so:
        logging.error("call_kernel.so not found in build directory")
        sys.exit(1)
    logging.info(f"call_kernel.so = {call_kernel_so}")

    script_path = os.path.join(args.out, "test_single_op.py")
    codegen_test_script(call_kernel_so, kernel_name, dump_file, tensors, device_id, script_path, block_dim=1)

    logging.info("========== Executing Single-Operator Test ==========")
    repro_start = time.time()
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=args.t)
    except subprocess.TimeoutExpired:
        logging.error(f"Test script timed out ({args.t}s)")
        sys.exit(1)

    output = result.stdout or ""
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    print(output)

    pc_info = ""
    if result.returncode != 0:
        logging.info("FAIL: AICORE error reproduced")
        pc_hit = _locate_error_pc(call_kernel_so, work_dir, repro_start)
        if pc_hit is None:
            pc_info = "Error PC not found in plog (symbol locator output absent)\n"
            logging.warning(pc_info.strip())
        else:
            pc, symbol, src = pc_hit
            if src is not None:
                src_file, src_line = src
                # The kernel sources are copied into the dump dir by the
                # exception callback; prefer the local copy over the (possibly
                # deleted) build tree so the dump stays self-contained.
                local_copy = os.path.join(dump_dir, os.path.basename(src_file))
                if os.path.isfile(local_copy):
                    src_file = local_copy
                pc_info = (f"ErrorPC: 0x{pc:x}\nSymbol: {symbol or '<unknown>'}\n"
                           f"Source: {src_file}:{src_line}\n")
                logging.info(f"ErrorPC 0x{pc:x} ({symbol or '?'}) -> {src_file}:{src_line}")
                snippet = _source_snippet(src_file, src_line)
                if snippet:
                    pc_info += snippet
            else:
                pc_info = (f"ErrorPC: 0x{pc:x}\nSymbol: {symbol or '<unknown>'}\n"
                           f"Source: <line table not resolvable in call_kernel.so>\n")
                logging.warning(f"ErrorPC 0x{pc:x} ({symbol or '?'}) -> source line not resolvable")
    else:
        logging.info("PASS: No AICORE error — execution succeeded")

    report_path = os.path.join(args.out, "reproduction_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Kernel: {kernel_name}\nDevice: {device_id}\n"
                f"call_kernel.so: {call_kernel_so}\ndump file: {dump_file}\n"
                f"Tensors: {len(tensors)}\n")
        for t in tensors:
            f.write(f"  {t.io_type}[{t.index}] addr=0x{t.addr:x} size={t.size} dtype={t.dtype_str} shape={t.shape}\n")
        if pc_info:
            f.write(f"\nError PC localization:\n{pc_info}\n")
        f.write(f"\nExecution result (exit code={result.returncode}):\n{output}")
    logging.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

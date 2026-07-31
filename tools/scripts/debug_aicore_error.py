#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------
r"""
自动串联 msaicerr.py 解析 AIC Error → 解析 info.txt → 生成 test_pypto_single_op.py
用于基于 dump tensor 数据的单算子复现。

用法示例
--------
python pypto_aicerr_repro.py -p /path/to/report -out /tmp/out -d 0
python pypto_aicerr_repro.py -p /path/to/report -out /tmp/out -d 0 -kernel_src /path/to/kernel_dir
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# 日志输出（与 msaicerr 格式一致），同时追加到 CWD 的 debug_info.txt
# -------------------------------------------------------------------

_DEBUG_LOG_PATH = "debug_info.txt"  # 写 CWD，main() 最后合并到 msaicerr_out 目录


def _print_log(level: str, msg: str) -> None:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    line = current_time + " (" + str(pid) + ") - [" + level + "] " + msg
    print(line)
    sys.stdout.flush()
    # 同步追加到 CWD 的 debug_info.txt（与 msaicerr 一致）
    try:
        with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass  # 写文件失败不影响屏幕输出


def _flush_debug_log(msaicerr_out: str) -> None:
    """将 CWD 的 debug_info.txt 内容合并到 msaicerr_out/debug_info.txt（参考 msaicerr 的 mv 操作）。"""
    if not os.path.isfile(_DEBUG_LOG_PATH):
        return
    target = os.path.join(msaicerr_out, "debug_info.txt")
    try:
        with open(_DEBUG_LOG_PATH, 'r', encoding='utf-8', errors='replace') as src:
            content = src.read()
        if content.strip():
            with open(target, 'a', encoding='utf-8') as dst:
                dst.write(content)
        os.remove(_DEBUG_LOG_PATH)
    except Exception as e:
        print(f"[WARN] _flush_debug_log 失败: {e}")


# -------------------------------------------------------------------
# 1. dtype 映射表
# -------------------------------------------------------------------

_STR_TO_NP: Dict[str, str] = {
    "float16":  "np.float16",
    "float32":  "np.float32",
    "float64":  "np.float64",
    "int8":     "np.int8",
    "int16":    "np.int16",
    "int32":    "np.int32",
    "int64":    "np.int64",
    "uint8":    "np.uint8",
    "bfloat16": "np.int16",   # numpy 不支持 bfloat16，用 int16 先读
    "bool":     "np.bool_",
}

_STR_TO_TORCH: Dict[str, str] = {
    "float16":  "torch.float16",
    "float32":  "torch.float32",
    "float64":  "torch.float64",
    "int8":     "torch.int8",
    "int16":    "torch.int16",
    "int32":    "torch.int32",
    "int64":    "torch.int64",
    "uint8":    "torch.uint8",
    "bfloat16": "torch.bfloat16",
    "bool":     "torch.bool",
}


# -------------------------------------------------------------------
# 2. 获取 CANN 目录
# -------------------------------------------------------------------

def get_ascend_home() -> str:
    """通过环境变量 ASCEND_HOME_PATH 获取 CANN 包目录。"""
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        raise RuntimeError(
            "ASCEND_HOME_PATH 环境变量未设置，请 source set_env.sh 后重试"
        )
    if not os.path.isdir(ascend_home):
        raise RuntimeError(f"ASCEND_HOME_PATH 指向的目录不存在: {ascend_home}")
    return ascend_home


# -------------------------------------------------------------------
# 3. 调用 msaicerr.py 解析
# -------------------------------------------------------------------

def run_msaicerr(report_path: str, output_path: str, device_id: int, ascend_home: str) -> str:
    """
    调用 CANN 包下的 msaicerr.py 解析 AIC Error 报告。

    返回 msaicerr 输出目录（即 info_<timestamp> 目录）。
    """
    msaicerr_script = os.path.join(ascend_home, "tools", "msaicerr", "msaicerr.py")
    if not os.path.isfile(msaicerr_script):
        raise RuntimeError(f"msaicerr.py 未找到: {msaicerr_script}")

    # 先列出将要输出的目录，用于事后定位 info_<timestamp>
    before_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            before_items.add(item)

    cmd = [
        sys.executable,
        msaicerr_script,
        "-p", report_path,
        "-out", output_path,
        "-dev", str(device_id),
    ]
    _print_log("INFO", f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    if result.returncode != 0:
        _print_log("WARNING", f"msaicerr.py 返回码: {result.returncode}（继续尝试定位 info.txt）")

    # 定位 info_<timestamp> 目录
    after_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            after_items.add(item)

    new_items = after_items - before_items
    for item in new_items:
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            _print_log("INFO", f"msaicerr 输出目录: {full}")
            return full

    # 备选：在 output_path 下找最新创建的 info_* 目录
    best = None
    best_mtime = 0
    for item in os.listdir(output_path):
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            mtime = os.path.getmtime(full)
            if mtime > best_mtime:
                best_mtime = mtime
                best = full
    if best:
        _print_log("INFO", f"msaicerr 输出目录（mtime）: {best}")
        return best

    raise RuntimeError(
        f"未能在 {output_path} 中找到 msaicerr 输出的 info_<timestamp> 目录"
    )


def find_info_txt(msaicerr_out_dir: str) -> str:
    """在 msaicerr 输出目录下查找 info.txt。"""
    for root, dirs, files in os.walk(msaicerr_out_dir):
        for fname in files:
            if fname == "info.txt":
                return os.path.join(root, fname)
    raise RuntimeError(f"在 {msaicerr_out_dir} 中未找到 info.txt")


def find_debug_info_txt(msaicerr_out: str) -> Optional[str]:
    """在 msaicerr 输出顶层目录下查找 debug_info.txt。如不存在则返回 None。

    msaicerr.py 内部将 debug_info.txt 写入 CWD 后 mv 到 collect_path
    （即 info_<timestamp> 顶层），而 info.txt 可能在子目录中，
    因此必须用顶层目录而非 info.txt 同级目录来查找。
    """
    debug_path = os.path.join(msaicerr_out, "debug_info.txt")
    return debug_path if os.path.isfile(debug_path) else None


# -------------------------------------------------------------------
# 4. 解析 info.txt
# -------------------------------------------------------------------

class TensorInfo:
    """单个 dump tensor 的元信息。"""
    __slots__ = ("path", "shape", "dtype", "io_type", "index")

    def __init__(self, path: str, shape: tuple, dtype: str, io_type: str, index: int):
        self.path = path
        self.shape = shape
        self.dtype = dtype      # 原始 dtype 字符串，如 "float16"
        self.io_type = io_type  # "input" / "output" / "workspace"
        self.index = index


def parse_info_txt(info_txt_path: str, debug_msgs: Optional[List[str]] = None) -> Tuple[str, List[TensorInfo]]:
    """
    解析 info.txt，返回 (kernel_func_name, tensor_list)。

    kernel_func_name: 从 "PyPTO_xxx_kernel_N_mix_aic" 中提取 "xxx_kernel"
    tensor_list:      按 input/output/workspace 顺序排列的 TensorInfo 列表
    debug_msgs:       可选的调试信息收集列表
    """
    def _log(msg: str):
        m = re.match(r'^\[(\w+)\]\s+(.*)', msg)
        if m:
            _print_log(m.group(1), m.group(2))
        else:
            print(msg)
        if debug_msgs is not None:
            debug_msgs.append(msg)

    with open(info_txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # --- 提取 kernel name (section 1) ---
    kernel_name_match = re.search(r"kernel name\s+:\s*(.+)", content)
    if not kernel_name_match:
        raise RuntimeError("info.txt 中未找到 'kernel name' 字段")
    kernel_name = kernel_name_match.group(1).strip()

    # 从 "PyPTO_add_kernel_0_mix_aic" 提取 "add_kernel"
    func_match = re.match(r"PyPTO_(.+)_kernel_\d+_mix_aic", kernel_name)
    if func_match:
        kernel_func_name = func_match.group(1) + "_kernel"
    else:
        # 兜底：尝试去掉 PyPTO_ 前缀
        if kernel_name.startswith("PyPTO_"):
            kernel_func_name = kernel_name[len("PyPTO_"):]
        else:
            kernel_func_name = kernel_name

    _log(f"[INFO] kernel name = {kernel_name}")
    _log(f"[INFO] 推断 pypto 函数名 = {kernel_func_name}")

    # --- 提取 section 5: Operator Dump File Parsing ---
    sec5_marker = "5. Operator Dump File Parsing"
    sec5_pos = content.find(sec5_marker)
    if sec5_pos < 0:
        raise RuntimeError("info.txt 中未找到 '5. Operator Dump File Parsing' 段")
    sec5_content = content[sec5_pos:]

    # 正则匹配每个 tensor 的 shape/dtype 行 + 文件路径行
    tensor_pattern = re.compile(
        r"shape:\s*\(([^)]*)\)\s+size:\s*\d+\s+dtype:\s*(\S+)\s*\n"
        r"(.+?)\n",
    )
    raw_matches = list(tensor_pattern.finditer(sec5_content))

    tensors: List[TensorInfo] = []
    for i, m in enumerate(raw_matches):
        shape_str = m.group(1)
        dtype_str = m.group(2)
        file_path = m.group(3).strip()

        # 解析 shape
        if shape_str.strip():
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        else:
            shape = ()

        # 从文件名推断 io_type 和 index
        basename = os.path.basename(file_path)
        parts = basename.split(".")
        io_type = "unknown"
        index = 0
        if len(parts) >= 5:
            io_field = parts[-4]
            if io_field.lower() in ("input", "output", "workspace"):
                io_type = io_field.lower()
            try:
                index = int(parts[-3])
            except ValueError:
                pass

        tensors.append(TensorInfo(
            path=file_path, shape=shape, dtype=dtype_str,
            io_type=io_type, index=index,
        ))
        _log(f"[DEBUG]   tensor[{i}] {io_type}[{index}] shape={shape} dtype={dtype_str} "
             f"file={basename}")

    # 排序：input 在前，output 在后，workspace 放最后；同类按 index 排
    order = {"input": 0, "output": 1, "workspace": 2}
    tensors.sort(key=lambda t: (order.get(t.io_type, 9), t.index))

    _log(f"[INFO] 解析到 {len(tensors)} 个 tensor "
         f"({sum(1 for t in tensors if t.io_type == 'input')} input, "
         f"{sum(1 for t in tensors if t.io_type == 'output')} output, "
         f"{sum(1 for t in tensors if t.io_type == 'workspace')} workspace)")

    return kernel_func_name, tensors


# -------------------------------------------------------------------
# 5. 生成 test_pypto_single_op.py
# -------------------------------------------------------------------

def _resolve_kernel_import(
    kernel_func_name: str,
    kernel_src: Optional[str],
) -> Tuple[str, List[str]]:
    """
    决定生成的脚本如何 import kernel。

    返回 (import_block_code, [extra_header_lines])。
    - kernel_src 为 None: 尝试 import pypto 后 getattr
    - kernel_src 是文件: 直接 import 该模块
    - kernel_src 是目录: 扫描目录下定义 kernel_func_name 的 .py 模块后 import
    """
    if kernel_src is None:
        # 不在 python lib，也没有指定路径 → 假设在 pypto 下
        return textwrap.dedent(f"""\
            # 尝试从 pypto 导入 kernel
            try:
                from pypto import {kernel_func_name} as _kernel_func
            except ImportError:
                # fallback: 作为独立模块 import
                import {kernel_func_name} as _kernel_func
        """), []

    abs_src = os.path.abspath(kernel_src)

    if os.path.isfile(abs_src):
        # 具体 .py 文件 → 将文件所在目录加入 sys.path
        src_dir = os.path.dirname(abs_src)
        mod_name = os.path.splitext(os.path.basename(abs_src))[0]
        headers = [
            f"_kernel_src_dir = r'{src_dir}'",
            "if _kernel_src_dir not in sys.path:",
            "    sys.path.insert(0, _kernel_src_dir)",
        ]
        import_code = textwrap.dedent(f"""\
            # Kernel 源码文件
            import {mod_name}
            _kernel_func = {mod_name}.{kernel_func_name}
        """)
        return import_code, headers
    else:
        # 目录 → 扫描
        headers = [
            f"_kernel_src = r'{abs_src}'",
            "if _kernel_src not in sys.path:",
            "    sys.path.insert(0, _kernel_src)",
        ]
        import_code = textwrap.dedent(f"""\
            # 扫描 kernel_src 目录，查找定义 {kernel_func_name} 的模块
            _kernel_func = None
            for _entry in os.scandir(_kernel_src):
                if _entry.is_file() and _entry.name.endswith('.py') and not _entry.name.startswith('_'):
                    _mod_name = os.path.splitext(_entry.name)[0]
                    try:
                        _mod = __import__(_mod_name)
                        if hasattr(_mod, '{kernel_func_name}'):
                            _kernel_func = getattr(_mod, '{kernel_func_name}')
                            break
                    except ImportError:
                        pass
            if _kernel_func is None:
                raise ImportError(
                    f"在 {{_kernel_src}} 中未找到函数 {kernel_func_name}"
                )
        """)
        return import_code, headers


def _tensor_load_code(tensors: List[TensorInfo]) -> Tuple[List[str], List[str], List[str]]:
    """
    生成 tensor 加载代码。

    返回 (load_lines, input_var_names, output_var_names)。
    """
    load_lines: List[str] = []
    input_vars: List[str] = []
    output_vars: List[str] = []

    for t in tensors:
        var_name = f"t_{t.io_type}_{t.index}"

        if t.path.endswith(".npy"):
            # workspace: np.load
            load_lines.append(f"{var_name}_np = np.load(r'{t.path}')")
            load_lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")
        elif t.path.endswith(".bin"):
            np_dtype = _STR_TO_NP.get(t.dtype, "np.float16")
            torch_dtype = _STR_TO_TORCH.get(t.dtype, "torch.float16")
            shape_repr = repr(t.shape)

            if t.dtype == "bfloat16":
                # bfloat16: 用 int16 读 → view as bfloat16
                load_lines.append("# bfloat16: 用 int16 加载后 view 为 bfloat16")
                load_lines.append(f"{var_name}_np = np.fromfile(r'{t.path}', dtype=np.int16).reshape({shape_repr})")
                load_lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device).view({torch_dtype})")
            else:
                load_lines.append(f"{var_name}_np = np.fromfile(r'{t.path}', dtype={np_dtype}).reshape({shape_repr})")
                load_lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")

        load_lines.append("")

        if t.io_type == "input":
            input_vars.append(var_name)
        elif t.io_type == "output":
            output_vars.append(var_name)
        # workspace 不加入 input/output 参数列表

    return load_lines, input_vars, output_vars


def codegen_test_script(
    kernel_func_name: str,
    tensors: List[TensorInfo],
    kernel_src: Optional[str],
    device_id: int,
    output_path: str,
    debug_msgs: Optional[List[str]] = None,
) -> None:
    """生成 test_pypto_single_op.py。"""
    def _log(msg: str):
        m = re.match(r'^\[(\w+)\]\s+(.*)', msg)
        if m:
            _print_log(m.group(1), m.group(2))
        else:
            print(msg)
        if debug_msgs is not None:
            debug_msgs.append(msg)

    import_code, extra_headers = _resolve_kernel_import(kernel_func_name, kernel_src)
    load_lines, input_vars, output_vars = _tensor_load_code(tensors)

    _log("")
    _log(f"[INFO] codegen kernel import 策略: "
         f"{'from kernel_src' if kernel_src else 'from pypto'}")

    _log("[INFO] 共加载 tensor:")
    for t in tensors:
        _log(f"       {t.io_type}[{t.index}] shape={t.shape} dtype={t.dtype} "
             f"-> {os.path.basename(t.path)}")

    lines = [
        "#!/usr/bin/env python3",
        "# coding: utf-8",
        "# Auto-generated by pypto_aicerr_repro.py",
        f"# Kernel: {kernel_func_name}",
        f"# Device: {device_id}",
        "#",
        "",
        "import os",
        "import sys",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "import pypto",
        "",
    ]

    if extra_headers:
        lines.extend(extra_headers)
        lines.append("")

    lines += [
        "# ---- Import kernel ----",
        "",
    ]
    lines.append(import_code)
    lines.append("")

    lines += [
        "# ---- 设置 device ----",
        f"device = torch.device(f'npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        "# ---- 加载 dump tensor ----",
        "",
    ]
    lines.extend(load_lines)

    # 调用 kernel
    lines += [
        "# ---- 调用 kernel ----",
        "",
        f"print('Calling {kernel_func_name}...')",
    ]

    non_empty_vars = [vn for vn in input_vars + output_vars if vn]
    if non_empty_vars:
        args_str = ", ".join(non_empty_vars)
        lines.append(f"_kernel_func({args_str})")
    else:
        lines.append("_kernel_func()")

    lines += [
        "",
        "print('Kernel execution completed.')",
        "torch.npu.synchronize()",
        "print('sync done.')",
        "",
    ]

    content = "\n".join(lines) + "\n"

    with open(output_path, "w") as f:
        f.write(content)

    _log(f"[OK] 已生成: {output_path}")
    _log(f"     Input tensors:  {len(input_vars)}")
    _log(f"     Output tensors: {len(output_vars)}")
    _log("")
    _log("Next step:")
    _log(f"  python {output_path}")


# -------------------------------------------------------------------
# 6. 追加调试信息到 debug_info.txt / 执行并覆盖 info.txt 第 6 段
# -------------------------------------------------------------------

_INFO_SECTION_PATTERN = re.compile(
    r'^\*{3,}\d+\.\s+.+?\*{3,}$', re.MULTILINE
)
_TARGET_SECTION_KEYWORD = "6. Execution Result of the Single-Operator Test Case"


def append_to_file(filepath: str, header: str, content: str):
    """向文件追加带分隔头的文本块。"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n{'=' * 72}\n"
        f"{header}  [{timestamp}]\n"
        f"{'=' * 72}\n"
        f"{content}"
        f"{'=' * 72}\n"
    )
    with open(filepath, "a") as f:
        f.write(block)
        _print_log("INFO", f"已将内容追加到: {filepath}")



def _parse_info_txt_to_sections(info_txt_path: str):
    """
    将 info.txt 解析为 (preamble, sections_dict, header_order)。

    preamble:    第一个 section header 之前的内容（根因结论 + 空行）
    sections:    {header_line: content}  例如
                 {"********************6. ... ***********************": "执行结果内容"}
    header_order: 保持原始顺序的 header 列表
    """
    with open(info_txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    matches = list(_INFO_SECTION_PATTERN.finditer(content))
    if not matches:
        return content, {}, []

    preamble = content[:matches[0].start()]
    sections: Dict[str, str] = {}
    header_order: List[str] = []

    for i, m in enumerate(matches):
        header = m.group().strip()
        header_order.append(header)
        start = m.end() + 1  # +1 跳过 header 后的换行符
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[header] = content[start:end].rstrip()

    return preamble, sections, header_order


def _find_section6_key(sections: Dict[str, str]) -> Optional[str]:
    """在 sections 中查找第 6 段 Single-Operator Test Case 的 header key。"""
    for header in sections:
        if _TARGET_SECTION_KEYWORD in header:
            return header
    return None


def _rewrite_info_txt(info_txt_path: str, preamble: str,
                      sections: Dict[str, str], header_order: List[str]):
    """用更新后的 sections 重写 info.txt。"""
    lines = [preamble.rstrip()]
    for header in header_order:
        lines.append("")
        lines.append(header)
        lines.append(sections.get(header, ""))
    lines.append("")  # 文件末尾换行
    with open(info_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        _print_log("INFO", f"已更新 {info_txt_path}")



def _enrich_from_plog(msaicerr_out: str, sections: Dict[str, str]):
    """
    从 plog 中获取信息补充到 info.txt：
    - coreType：grep 'error info:' 取时间戳最早的内容，根据异常类型判断
      （"exception of fftsplus aicore error" → 0，"exception of fftsplus aivector error" → 1）
    - fixedStartPC / fixedCurrentPC / fixedPCOffset：从 kernel_symbol_locator.cpp 匹配 core id + core type
    """
    # 1. 从 section 1 提取 core id
    sec1_key = None
    sec1_content = ""
    for header in sections:
        if "1. Basic information" in header:
            sec1_key = header
            sec1_content = sections[header]
            break

    if sec1_key is None:
        _print_log("WARNING", "未找到 section 1 (Basic information)，跳过 plog 补充")
        return

    core_id_match = re.search(r'core\s+id\s*:\s*(\d+)', sec1_content, re.IGNORECASE)
    if not core_id_match:
        _print_log("WARNING", "section 1 中未找到 core id，跳过 plog 补充")
        return
    core_id = core_id_match.group(1)
    _print_log("INFO", f"section 1 core id = {core_id}")


    plog_dir = os.path.join(msaicerr_out, "collection", "plog")
    if not os.path.isdir(plog_dir):
        _print_log("WARNING", f"plog 目录不存在: {plog_dir}")
        return

    # 2. grep 'error info:' 取时间戳最早的内容，从中判断 coreType
    core_type = _parse_core_type_from_earliest_error_info(plog_dir)

    # 3. grep kernel_symbol_locator.cpp 获取 fixedPC 信息（同时匹配 core id + core type）
    pc_match = _parse_fixed_pc_from_plog(plog_dir, core_id, core_type)

    if core_type is None and pc_match is None:
        return

    # 4. 追加 coreType 到 section 1
    if core_type is not None:
        sections[sec1_key] = (
            sec1_content.rstrip()
            + f"\ncore type          : {core_type}"
        )

    # 5. 追加 fixed PC 信息到 section 3，并用 llvm-symbolizer 解析符号
    if pc_match is not None:
        for header in sections:
            if "3. Operator Error Line Number" in header:
                sec3_content = sections[header]
                kernel_file = _extract_kernel_file(sec1_content, msaicerr_out)
                extra = (
                    f"\nfixedStartPC       : {pc_match['fixedStartPC']}"
                    f"\nfixedCurrentPC     : {pc_match['fixedCurrentPC']}"
                    f"\nfixedPCOffset      : {pc_match['fixedPCOffset']}"
                )
                symbol_info = _run_llvm_symbolizer(kernel_file, pc_match['fixedPCOffset'])
                if symbol_info:
                    extra += f"\n{symbol_info}"
                sections[header] = sec3_content.rstrip() + extra
                break
        else:
            _print_log("WARNING", "未找到 section 3 (Operator Error Line Number)")
def _parse_core_type_from_earliest_error_info(plog_dir: str) -> Optional[str]:
    """
    参考 msaicerr 逻辑：grep 'error info:' 取时间戳最早的那一条，
    根据异常类型判断 coreType：
      - "exception of fftsplus aicore error"   → 0
      - "exception of fftsplus aivector error"  → 1
    """
    try:
        result = subprocess.run(
            ["grep", "-rnE", "error info:", plog_dir],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        _print_log("WARNING", f"grep 'error info:' 失败: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", "plog 中未找到 'error info:'")
        return None

    # 提取时间戳，参考 msaicerr 直接用字符串排序
    # （YYYY-MM-DD-HH:MM:SS.xxxx 格式天然保证字典序即时间序，无需转 datetime）
    _ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d+\.\d+)')
    ts_lines = []
    for line in result.stdout.strip().split("\n"):
        m = _ts_pattern.search(line)
        if m:
            ts_lines.append((m.group(1), line))

    if not ts_lines:
        _print_log("WARNING", "'error info:' 行中未解析到有效时间戳")
        return None

    ts_lines.sort(key=lambda x: (x[0] is None, x[0]))
    earliest_line = ts_lines[0][1]
    _print_log("INFO", f"最早的 error info: {earliest_line.strip()[:200]}...")


    if "exception of fftsplus aicore error" in earliest_line.lower():
        _print_log("INFO", "coreType 判断为 0 (aicore error)")
        return "0"
    elif "exception of fftsplus aivector error" in earliest_line.lower():
        _print_log("INFO", "coreType 判断为 1 (aivector error)")
        return "1"
    else:
        _print_log("WARNING", "最早的 error info 中未识别到 fftsplus aicore/aivector error，coreType 未知")
        return None


def _parse_fixed_pc_from_plog(plog_dir: str, core_id: str, core_type: Optional[str]) -> Optional[Dict[str, str]]:
    """
    grep kernel_symbol_locator.cpp，只解析含 "Error PC information" 的行，
    同时匹配 core id + core type，返回 fixedStartPC / fixedCurrentPC / fixedPCOffset。
    """
    try:
        result = subprocess.run(
            ["grep", "-rn", "kernel_symbol_locator.cpp", plog_dir],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        _print_log("WARNING", f"grep kernel_symbol_locator.cpp 失败: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", "plog 中未找到 kernel_symbol_locator.cpp")
        return None

    _pc_pattern = re.compile(
        r'coreId=(\d+),\s*coreType=(\d+).*?'
        r'fixedStartPC=(0x[0-9a-fA-F]+).*?'
        r'fixedCurrentPC=(0x[0-9a-fA-F]+).*?'
        r'fixedPCOffset=(0x[0-9a-fA-F]+)',
    )
    parsed = []
    for line in result.stdout.strip().split("\n"):
        if "Error PC information" not in line:
            continue
        m = _pc_pattern.search(line)
        if not m:
            continue
        parsed.append({
            "coreId": m.group(1),
            "coreType": m.group(2),
            "fixedStartPC": m.group(3),
            "fixedCurrentPC": m.group(4),
            "fixedPCOffset": m.group(5),
        })

    if not parsed:
        _print_log("WARNING", "plog 中未找到 'Error PC information' 的 kernel_symbol_locator 行")
        return None

    # 优先匹配 coreId + coreType，其次仅 coreId
    match = None
    if core_type is not None:
        for p in parsed:
            if p["coreId"] == core_id and p["coreType"] == core_type:
                match = p
                break
    if match is None:
        for p in parsed:
            if p["coreId"] == core_id:
                match = p
                break
    if match is None:
        match = parsed[0]
        _print_log("WARNING", f"未找到 coreId={core_id}, coreType={core_type} 的匹配项"
                   f"（共 {len(parsed)} 条），使用 coreId={match['coreId']},"
                   f" coreType={match['coreType']}")


    _print_log("INFO", f"从 plog 提取 fixed PC: "
              f"fixedStartPC={match['fixedStartPC']}, "
              f"fixedCurrentPC={match['fixedCurrentPC']}, "
              f"fixedPCOffset={match['fixedPCOffset']}")
    return {
        "fixedStartPC": match["fixedStartPC"],
        "fixedCurrentPC": match["fixedCurrentPC"],
        "fixedPCOffset": match["fixedPCOffset"],
    }


def _extract_kernel_file(sec1_content: str, msaicerr_out: str) -> Optional[str]:
    """从 section 1 中提取 kernel file 路径（`.o` 文件）。"""
    m = re.search(r'kernel\s+file\s*:\s*(\S+)', sec1_content, re.IGNORECASE)
    if not m:
        _print_log("WARNING", "section 1 中未找到 kernel file 路径")
        return None
    kernel_file = m.group(1)
    # 如果路径不包含 collection/compile，尝试拼接
    if not os.path.isabs(kernel_file) and "collection/compile" not in kernel_file:
        candidates = []
        compile_dir = os.path.join(msaicerr_out, "collection", "compile")
        if os.path.isdir(compile_dir):
            for fname in os.listdir(compile_dir):
                if fname.endswith(".o") and kernel_file in fname:
                    candidates.append(os.path.join(compile_dir, fname))
        if candidates:
            kernel_file = candidates[0]
    if not os.path.isfile(kernel_file):
        _print_log("WARNING", f"kernel file 不存在: {kernel_file}")
        return None
    return kernel_file


def _run_llvm_symbolizer(kernel_file: Optional[str], pc_offset: str) -> Optional[str]:
    """执行 llvm-symbolizer --obj=<kernel_file> <pc_offset>，返回解析后的符号行。"""
    if not kernel_file:
        return None
    try:
        result = subprocess.run(
            ["llvm-symbolizer", f"--obj={kernel_file}", pc_offset],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        _print_log("WARNING", f"llvm-symbolizer 执行失败: {e}")
        return None

    if result.returncode != 0 or not result.stdout.strip():
        _print_log("WARNING", f"llvm-symbolizer 无输出 (rc={result.returncode})")
        return None

    # 直接追加：指令 + 原始输出
    cmd_str = f"llvm-symbolizer --obj={kernel_file} {pc_offset}"
    # 去掉末尾多余空行，保留原始内容中的换行
    output = result.stdout.rstrip('\n')
    symbol_str = cmd_str + "\n" + output
    _print_log("INFO", f"llvm-symbolizer 结果:\n{output}")
    return symbol_str


def run_test_script_and_update_info(script_path: str, info_txt_path: str,
                                    device_id: int, msaicerr_out: str) -> bool:
    """
    执行 test_pypto_single_op.py，将执行结果覆盖写入 info.txt 的第 6 段
    (Execution Result of the Single-Operator Test Case)。
    同时从 plog 补充 coreType / fixedPC 信息到 section 1 和 3。
    返回 True 表示单算子测试通过（returncode == 0），False 表示失败。
    """
    _print_log("INFO", f"执行 test_pypto_single_op.py (device={device_id})...")
    _print_log("INFO", f"脚本: {script_path}")

    test_passed = False

    # 1. 执行测试脚本
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            timeout=300,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        test_passed = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        output = "[ERROR] 脚本执行超时（300s）"
    except Exception as e:
        output = f"[ERROR] 脚本执行异常: {e}"

    # 2. 解析 info.txt 为结构化 dict
    preamble, sections, header_order = _parse_info_txt_to_sections(info_txt_path)

    # 2.5 从 plog 补充 coreType (→section 1) 和 fixedPC (→section 3)
    _enrich_from_plog(msaicerr_out, sections)

    # 3. 覆盖更新第 6 段
    sec6_key = _find_section6_key(sections)
    if sec6_key:
        sections[sec6_key] = output
    else:
        # info.txt 中没有第 6 段（不应出现），退化为追加
        _print_log("WARNING", "info.txt 中未找到 section 6，退化为追加模式")
        append_to_file(info_txt_path, "test_pypto_single_op.py Execution Result", output)
        return False

    # 4. 重写 info.txt
    _rewrite_info_txt(info_txt_path, preamble, sections, header_order)

    return test_passed


# -------------------------------------------------------------------
# 7. 排查框架 vs 算子 CCE 问题
# -------------------------------------------------------------------

_SECTION7_TITLE = "7. Framework vs Operator (CCE) Root Cause Analysis"

_AICORE_ENTRY_SEARCH_DIR = "/opt/conda/envs/py310/lib/python3.10/site-packages"


def _find_aicore_entry_h(search_dir: str) -> Optional[str]:
    """find search_dir -name aicore_entry.h，返回第一个匹配路径。"""
    try:
        result = subprocess.run(
            ["find", search_dir, "-name", "aicore_entry.h"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        _print_log("WARNING", f"find aicore_entry.h 失败: {e}")
        return None

    paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    if not paths:
        return None
    for p in paths:
        if "pypto" in p:
            return p
    return paths[0]


def _comment_exec_core_function(header_path: str) -> Optional[str]:
    """在 aicore_entry.h 中注释 ExecCoreFunctionKernel 调用，返回备份路径。"""
    try:
        with open(header_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        _print_log("WARNING", f"读取 {header_path} 失败: {e}")
        return None

    backup_path = header_path + ".bak"
    try:
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        _print_log("WARNING", f"备份 {header_path} 失败: {e}")
        return None

    # 注释 ExecCoreFunctionKernel(&ctx, curTaskIdx, lastMixResourceType);
    pattern = re.compile(r'(ExecCoreFunctionKernel\s*\([^)]+\)\s*;)')
    if not pattern.search(content):
        _print_log("WARNING", f"在 {header_path} 中未找到 ExecCoreFunctionKernel 调用")
        # 即使没找到也返回备份路径，后续恢复用
        return backup_path

    new_content = pattern.sub(
        r'// \1  // [COMMENTED BY pypto_aicerr_repro section 7]',
        content,
    )
    try:
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        _print_log("WARNING", f"写入 {header_path} 失败: {e}")
        return None

    _print_log("INFO", f"已注释 ExecCoreFunctionKernel → {header_path}")
    return backup_path


def _restore_header(header_path: str, backup_path: str):
    """从备份恢复 aicore_entry.h。"""
    if not backup_path or not os.path.isfile(backup_path):
        return
    try:
        with open(backup_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.remove(backup_path)
        _print_log("INFO", f"已恢复 {header_path}（删除备份 {backup_path}）")
    except Exception as e:
        _print_log("WARNING", f"恢复 {header_path} 失败: {e}")


def run_section7_framework_vs_cce(info_txt_path: str, script_path: str, device_id: int):
    """
    Section 7: 排查框架 vs 算子 CCE 问题。
    注释 aicore_entry.h 中的 ExecCoreFunctionKernel 调用后重新执行单算子脚本，
    根据结果判断根因归属。
    """
    _print_log("INFO", "========== Section 7: 排查框架 vs 算子 CCE 问题 ==========")

    output_lines: List[str] = []

    # 1. pip3 show pypto
    _print_log("INFO", "执行: pip3 show pypto")
    try:
        result = subprocess.run(
            ["pip3", "show", "pypto"],
            capture_output=True, text=True, timeout=30,
        )
        pypto_info = result.stdout.strip() if result.returncode == 0 else f"[pip3 show 失败] {result.stderr}"
    except Exception as e:
        pypto_info = f"[pip3 show 异常] {e}"
    output_lines.append("--- pip3 show pypto ---")
    output_lines.append(pypto_info)
    output_lines.append("")

    # 2. find aicore_entry.h
    search_dir = _AICORE_ENTRY_SEARCH_DIR
    cmd_str = f"find {search_dir} -name aicore_entry.h"
    _print_log("INFO", cmd_str)
    header_path = _find_aicore_entry_h(search_dir)
    output_lines.append("--- find aicore_entry.h ---")
    output_lines.append(cmd_str)
    if header_path:
        output_lines.append(header_path)
    else:
        output_lines.append("[ERROR] 未找到 aicore_entry.h")
    output_lines.append("")

    if not header_path:
        _write_section7(info_txt_path, "\n".join(output_lines))
        return

    # 3. 注释 ExecCoreFunctionKernel
    _print_log("INFO", f"注释 {header_path} 中的 ExecCoreFunctionKernel")
    backup_path = _comment_exec_core_function(header_path)

    output_lines.append("--- 注释 ExecCoreFunctionKernel ---")
    if backup_path:
        output_lines.append(f"备份: {backup_path}")
        output_lines.append("已注释: ExecCoreFunctionKernel(&ctx, curTaskIdx, lastMixResourceType);")
    else:
        output_lines.append("[ERROR] 无法注释 ExecCoreFunctionKernel")
    output_lines.append("")

    # 4. 再次执行 test_pypto_single_op.py
    _print_log("INFO", "再次执行 test_pypto_single_op.py（已注释 ExecCoreFunctionKernel）...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
        )
        test_output = ""
        if result.stdout:
            test_output += result.stdout
        if result.stderr:
            test_output += "\n[stderr]\n" + result.stderr
        test_output += f"\nreturncode = {result.returncode}"
        test_passed = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        test_output = "[ERROR] 脚本执行超时（300s）"
        test_passed = False
    except Exception as e:
        test_output = f"[ERROR] 脚本执行异常: {e}"
        test_passed = False

    output_lines.append("--- 再次执行 test_pypto_single_op.py（ExecCoreFunctionKernel 已注释）---")
    output_lines.append(test_output)
    output_lines.append("")

    # 5. 恢复 aicore_entry.h
    _print_log("INFO", f"恢复 {header_path}")
    _restore_header(header_path, backup_path if backup_path else "")
    output_lines.append(f"--- 已恢复 {header_path} ---")
    output_lines.append("")

    # 6. 结论
    if test_passed:
        conclusion = "结论: 注释 ExecCoreFunctionKernel 后执行通过 → 算子 CCE 问题"
    else:
        conclusion = "结论: 注释 ExecCoreFunctionKernel 后仍然失败 → 框架问题"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _write_section7(info_txt_path, "\n".join(output_lines))


def _write_section7(info_txt_path: str, content: str):
    """将内容写入 info.txt 的 section 7。"""
    preamble, sections, header_order = _parse_info_txt_to_sections(info_txt_path)

    sec7_key = None
    for header in sections:
        if _SECTION7_TITLE in header:
            sec7_key = header
            break

    if sec7_key:
        sections[sec7_key] = content
    else:
        sec7_header = f"********************{_SECTION7_TITLE}***********************"
        sec7_key = sec7_header.strip()
        header_order.append(sec7_key)
        sections[sec7_key] = content

    _rewrite_info_txt(info_txt_path, preamble, sections, header_order)


# -------------------------------------------------------------------
# 8. 排除核间同步问题
# -------------------------------------------------------------------

_SECTION8_TITLE = "8. Inter-Core Synchronization Diagnosis"

_MSNPUREPORT = "/usr/local/Ascend/driver/tools/msnpureport"


def _is_docker_env():
    """判断当前是否为 docker 环境（通过 /.dockerenv 或 /proc/1/cgroup 检测）。"""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    except (IOError, PermissionError):
        pass
    return False


def _msnpureport_set_singlecommit(enable: bool, device_id: int, is_docker: bool = False):
    """开启/关闭 singlecommit 模式。返回 (success, cmd_str, output)。"""
    val = "1" if enable else "0"
    cmd = [_MSNPUREPORT, "config", "--set", "--singlecommit", val, "-d", str(device_id)]
    if is_docker:
        cmd.append("--docker")
    cmd_str = " ".join(cmd)
    action = "开启" if enable else "恢复"
    _print_log("INFO", f"{action} singlecommit: {cmd_str}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.stderr:
            output += "\n" + result.stderr.strip()
        success = (result.returncode == 0)
    except Exception as e:
        output = f"[ERROR] {e}"
        success = False
    return success, cmd_str, output


def run_section8_intercore_sync(info_txt_path: str, script_path: str, device_id: int):
    """
    Section 8: 通过 msnpureport 关闭核间同步（singlecommit=1）排除核间同步问题。
    """
    _print_log("INFO", "========== Section 8: 排除核间同步问题 ==========")

    output_lines: List[str] = []

    # 0. 判断是否为 docker 环境
    is_docker = _is_docker_env()
    _print_log("INFO", f"Docker 环境: {is_docker}")

    # 1. 开启 singlecommit=1
    success, cmd_str, msn_output = _msnpureport_set_singlecommit(True, device_id, is_docker)
    output_lines.append("--- msnpureport 开启单步执行模式 ---")
    output_lines.append(cmd_str)
    output_lines.append(msn_output)
    output_lines.append("")
    if not success:
        _print_log("WARNING", f"msnpureport 开启 singlecommit 失败: {msn_output}")

    # 2. 再次执行 test_pypto_single_op.py
    _print_log("INFO", "再次执行 test_pypto_single_op.py（单步执行模式）...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
        )
        test_output = ""
        if result.stdout:
            test_output += result.stdout
        if result.stderr:
            test_output += "\n[stderr]\n" + result.stderr
        test_passed = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        test_output = "[ERROR] 脚本执行超时（300s）"
        test_passed = False
    except Exception as e:
        test_output = f"[ERROR] 脚本执行异常: {e}"
        test_passed = False

    output_lines.append("--- 再次执行 test_pypto_single_op.py（singlecommit=1）---")
    output_lines.append(test_output)
    output_lines.append("")

    # 3. 恢复 singlecommit=0
    success2, cmd_str2, msn_output2 = _msnpureport_set_singlecommit(False, device_id, is_docker)
    output_lines.append("--- msnpureport 恢复 ---")
    output_lines.append(cmd_str2)
    output_lines.append(msn_output2)
    output_lines.append("")

    # 4. 结论
    if test_passed:
        conclusion = "结论: 单步执行模式下执行通过 → 核间同步问题"
    else:
        conclusion = "结论: 单步执行模式下仍然失败 → 与核间同步无关"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _write_section8(info_txt_path, "\n".join(output_lines))


def _write_section8(info_txt_path: str, content: str):
    """将内容写入 info.txt 的 section 8。"""
    preamble, sections, header_order = _parse_info_txt_to_sections(info_txt_path)

    sec8_key = None
    for header in sections:
        if _SECTION8_TITLE in header:
            sec8_key = header
            break

    if sec8_key:
        sections[sec8_key] = content
    else:
        sec8_header = f"********************{_SECTION8_TITLE}***********************"
        sec8_key = sec8_header.strip()
        header_order.append(sec8_key)
        sections[sec8_key] = content

    _rewrite_info_txt(info_txt_path, preamble, sections, header_order)


# -------------------------------------------------------------------
# 9. CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动串联 msaicerr 解析 → 生成 pypto 单算子复现脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", type=str, required=True,
        help="AIC Error 报告路径（传给 msaicerr.py -p）",
    )
    parser.add_argument(
        "-out", type=str, required=True,
        help="msaicerr 输出目录（传给 msaicerr.py -out）",
    )
    parser.add_argument(
        "-d", type=int, required=True,
        help="device id（传给 msaicerr.py -dev，同时设置 torch.npu.set_device）",
    )
    parser.add_argument(
        "-kernel_src", type=str, default=None,
        help="kernel 源码路径：可以是目录或 .py 文件（kernel 不在 python lib 时需要）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 获取 CANN 目录
    ascend_home = get_ascend_home()
    _print_log("INFO", f"ASCEND_HOME_PATH = {ascend_home}")


    # 2. 调用 msaicerr.py 解析
    msaicerr_out = run_msaicerr(args.p, args.out, args.d, ascend_home)

    # 定位 info.txt
    info_txt = find_info_txt(msaicerr_out)
    _print_log("INFO", f"info.txt: {info_txt}")


    # 3. 解析 info.txt + 收集 debug 信息
    debug_msgs: List[str] = []
    kernel_func_name, tensors = parse_info_txt(info_txt, debug_msgs=debug_msgs)

    if not tensors:
        _print_log("ERROR", "未解析到任何 dump tensor，无法生成复现脚本")
        _flush_debug_log(msaicerr_out)
        sys.exit(1)

    # 4. codegen test_pypto_single_op.py
    output_script = os.path.join(msaicerr_out, "test_pypto_single_op.py")
    codegen_test_script(
        kernel_func_name=kernel_func_name,
        tensors=tensors,
        kernel_src=args.kernel_src,
        device_id=args.d,
        output_path=output_script,
        debug_msgs=debug_msgs,
    )

    # 5. 合并 CWD debug_info.txt → msaicerr_out/debug_info.txt
    _flush_debug_log(msaicerr_out)

    # 6. 执行 test_pypto_single_op.py，结果追加到 info.txt
    section6_passed = run_test_script_and_update_info(output_script, info_txt, args.d, msaicerr_out)

    if section6_passed:
        _print_log("INFO", "Section 6 单算子测试通过，跳过 Section 7（框架 vs 算子 CCE）和 Section 8（核间同步诊断）")
    else:
        # 7. 排查框架 vs 算子 CCE 问题
        run_section7_framework_vs_cce(info_txt, output_script, args.d)

        # 8. 排除核间同步问题
        run_section8_intercore_sync(info_txt, output_script, args.d)

    # 9. 再次合并（step 6/7/8 产生的日志）
    _flush_debug_log(msaicerr_out)


if __name__ == "__main__":
    main()

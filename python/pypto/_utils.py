#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import sys
import math
import shlex
import shutil
import ctypes
import tempfile
import threading
import functools
import dataclasses
import subprocess
from pathlib import Path
from typing import Sequence, Union, List, Any, Optional
from importlib import metadata

from . import pypto_impl
from .enum import DataType
from .symbolic_scalar import SymbolicScalar, SymInt


def to_sym(value) -> pypto_impl.SymbolicScalar:
    if isinstance(value, int):
        return pypto_impl.SymbolicScalar(value)
    if isinstance(value, pypto_impl.SymbolicScalar):
        return value
    if isinstance(value, SymbolicScalar):
        return value.base()
    raise ValueError("Invalid value type")


def to_syms(value: Union[Sequence[int], Sequence[SymbolicScalar]]) -> List[pypto_impl.SymbolicScalar]:
    return [to_sym(v) for v in value]


def ceildiv(a: SymInt, b: SymInt) -> SymInt:
    return (a + b - 1) // b

# only outer takes effect void avoid tensor.py hide source_location of user code
_source_location_depth = 0


def set_source_location(level: int = 1):
    global _source_location_depth
    if _source_location_depth == 0:
        frame = sys._getframe(level + 1)
        pypto_impl.SetLocation(frame.f_code.co_filename, frame.f_lineno, "")
    _source_location_depth += 1


def clear_source_location():
    global _source_location_depth
    _source_location_depth -= 1
    if _source_location_depth == 0:
        pypto_impl.ClearLocation()


def source_location(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()
        return out
    return wrapper


def bytes_of(dtype: DataType) -> int:
    """ return the number of bytes of the current datatype

    Parameters
    ----------
    dtype: pypto.DataType
        datatype to be determined the number of bytes

    Returns
    -------
    int: the size of bytes the datatype contains

    Examples
    --------
    >>> print(pypto.bytes_of(pypto.DT_FP32))
        4
    """
    # implementation
    return pypto_impl.BytesOf(dtype)


class BuildOnlineManager:
    _instance: Any = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    @dataclasses.dataclass
    class _CMakeContext:
        cmake: Optional[Path] = None

        @dataclasses.dataclass
        class CompileContext:
            src_dir: Path  # 源码根目录, 对应 CMAKE_SOURCE_DIR
            tmp_dir: Path  # 临时目录
            cfg_cmd_ext: Optional[str] = None  # CMake Configure 阶段的扩展命令行
            build_type: str = "Release"  # 编译类型, 默认是 Release
            build_job_num: int = 4  # 4 为默认编译并行度
            capture_output: bool = True  # 拦截调用 CMake 过程的输出

            def __init__(self, src_dir: Path, tmp_dir: Path):
                self.src_dir = src_dir
                self.tmp_dir = tmp_dir

        def __init__(self):
            self.cmake = self._which_cmake()
            if self.cmake is None:
                raise RuntimeError("Can not find cmake, please check your envirionment.")

        @classmethod
        def _which_cmake(cls) -> Optional[Path]:
            """查找系统级 CMake 可执行文件路径
            排除 cmake pip 包的干扰, 通过遍历 PATH 环境变量查找 ELF 格式的 CMake 可执行文件.
            :return: 系统 CMake 可执行文件路径, 找不到则返回 None
            :rtype: Optional[Path]
            """
            # 拆分 PATH 环境变量为单个目录列表(排除空目录)
            path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]
            # 遍历每个 PATH 目录, 逐个调用 shutil.which 检查, 限定 shutil.which 只在当前单个目录下查找 cmake
            valid_path_lst = []
            for path_dir in path_dir_lst:
                # 避免 PATH 环境变量中有重复的单元
                if path_dir in valid_path_lst:
                    continue
                valid_path_lst.append(path_dir)
                # 检查当前目录
                cmake_str = shutil.which("cmake", path=path_dir)
                if not cmake_str:
                    continue
                cmake_file = Path(cmake_str).resolve()
                if not cmake_file.exists() or not cmake_file.is_file():
                    continue
                if cmake_file.stat().st_size <= 4:  # 下文读取前 4 字节判断文件是否是 ELF 文件
                    continue
                with open(cmake_file, 'rb') as fh:
                    header = fh.read(4)  # 前 4 字节是 ELF 文件标识
                if header != b'\x7fELF':
                    continue
                return cmake_file
            return None

        def compile(self, ctx: CompileContext) -> Path:
            """执行编译流程(包含 CMake 的 Configure, Build 及 Install 阶段)

            本函数会在临时目录 ctx.tmp_dir 路径下以下临时目录:
            1. 创建 build 子目录作为 CMAKE_BINARY_DIR
            2. 创建 install 子目录作为 CMAKE_INSTALL_PREFIX;

            :param ctx: 编译上下文
            :return: 安装路径, 对应 CMAKE_INSTALL_PREFIX
            :rtype: Path
            """
            # 路径准备
            build_dir = Path(ctx.tmp_dir, "build")
            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True)
            install_dir = Path(ctx.tmp_dir, "install")
            # CMake Configure
            cfg_cmd_ext = ctx.cfg_cmd_ext if ctx.cfg_cmd_ext else ""
            cmd = f"{self.cmake} -S {ctx.src_dir} -B {build_dir} -DCMAKE_BUILD_TYPE={ctx.build_type}"
            cmd += f" -DCMAKE_INSTALL_PREFIX={install_dir} {cfg_cmd_ext}"
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            # CMake Build
            cmd = f"{self.cmake} --build {build_dir}" + (f" -j {ctx.build_job_num}" if ctx.build_job_num else "")
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            # CMake Install
            cmd = f"{self.cmake} --install {build_dir} --prefix {install_dir}"
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            return install_dir

    @dataclasses.dataclass
    class _TorchContext:
        torch_version: str = ""
        torch_root_dir: str = ""
        torch_c_use_cxx11_abi: int = 1

        def __init__(self):
            os_env = os.environ.copy()
            try:
                os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                import torch
                self.torch_version = str(torch.__version__)
                self.torch_root_dir = str(Path(torch.__file__).parent)
                self.torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
            except (ModuleNotFoundError or ImportError) as e:
                raise RuntimeError(f"Can not import torch, please check your python environment. Error: {e}") from e
            finally:
                os.environ = os_env

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        cur_dir: Path = Path(__file__).parent
        pkg_dir: Path = Path(str(metadata.distribution("pypto").locate_file("pypto"))).resolve()
        pkg_dir = pkg_dir if pkg_dir == cur_dir else cur_dir  # 适配 edit 模式
        self.pkg_lib_dir: Path = Path(pkg_dir, "lib")
        self.calculator_loaded: bool = False
        self._initialized = True

    def build_and_load_calculator(self):
        if self.calculator_loaded:
            return
        torch_ctx = self._TorchContext()
        cmake_ctx = self._CMakeContext()
        with tempfile.TemporaryDirectory() as _tmp_dir:
            # 编译
            ext = f"-DPY3_MOD_TORCH_VERSION={torch_ctx.torch_version}"
            ext += f" -DPY3_MOD_TORCH_ROOT_PATH={torch_ctx.torch_root_dir}"
            ext += f" -DPY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI={torch_ctx.torch_c_use_cxx11_abi}"
            compile_ctx = self._CMakeContext.CompileContext(src_dir=Path(self.pkg_lib_dir, "calculator"),
                                                            tmp_dir=Path(_tmp_dir))
            compile_ctx.cfg_cmd_ext = ext
            install_prefix = cmake_ctx.compile(ctx=compile_ctx)
            # 加载
            calc_shared = Path(install_prefix, "lib/libtile_fwk_calculator.so")
            if not calc_shared.exists():
                raise RuntimeError(f"{calc_shared} not exists.")
            ctypes.CDLL(str(calc_shared), mode=ctypes.RTLD_GLOBAL)
        self.calculator_loaded = True

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
"""
在线编译基础设施模块.

本模块不 import pypto_impl 或任何依赖 pypto_impl 的子模块, 以确保在 pypto_impl.so 尚不存在时也可正常加载.

包含两种在线编译管理器:
- BuildOnlinePyptoImplManager: pypto_impl.so 在线编译
- BuildOnlineCalculatorManager: tile_fwk_calculator.so 在线编译
"""

import importlib.machinery
import importlib.util
import ctypes
import dataclasses
import logging
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path
from typing import Optional, Any, Dict


_log = logging.getLogger(__name__)


class _BuildOnlineManager:
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    @dataclasses.dataclass
    class _CMakeContext:
        cmake: Optional[Path] = None

        @dataclasses.dataclass
        class CompileContext:
            src_dir: Path  # 源码根目录, 对应 CMAKE_SOURCE_DIR
            tmp_dir: Path  # 临时目录
            cfg_cmd_ext: Optional[str] = None  # CMake Configure 阶段的扩展命令行
            build_type: str = "Release"  # 编译类型, 默认是 Release
            build_job_num: int = 32  # 32 为默认编译并行度
            install_prefix: Optional[Path] = None
            capture_output: bool = True  # 拦截调用 CMake 过程的输出

            @property
            def install_dir(self) -> Path:
                return self.install_prefix or Path(self.tmp_dir, "install")

        def __init__(self):
            self.cmake = self._which_cmake()
            if self.cmake is None:
                raise RuntimeError(
                    "Can not find cmake, please check your environment.\n"
                    "Hint: install system-level cmake (e.g. `apt install cmake` or `yum install cmake`), "
                    "the cmake pip package is NOT a valid substitute."
                )

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
            # CMake Configure
            cfg_cmd_ext = ctx.cfg_cmd_ext if ctx.cfg_cmd_ext else ""
            cmd = f"{self.cmake} -S {ctx.src_dir} -B {build_dir} -DCMAKE_BUILD_TYPE={ctx.build_type}"
            cmd += f" -DCMAKE_INSTALL_PREFIX={ctx.install_dir} {cfg_cmd_ext}"
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            # CMake Build
            cmd = f"{self.cmake} --build {build_dir}" + (f" -j {ctx.build_job_num}" if ctx.build_job_num else "")
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            # CMake Install
            cmd = f"{self.cmake} --install {build_dir} --prefix {ctx.install_dir}"
            subprocess.run(shlex.split(cmd), capture_output=ctx.capture_output, check=True, text=True, encoding='utf-8')
            return ctx.install_dir

    def __new__(cls):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        # 路径识别
        from importlib import metadata
        cur_dir: Path = Path(__file__).parent
        pkg_dir: Path = Path(str(metadata.distribution("pypto").locate_file("pypto"))).resolve()
        pkg_dir = pkg_dir if pkg_dir == cur_dir else cur_dir  # 适配 edit 模式
        # 变量赋值
        self.pkg_dir: Path = pkg_dir
        self.pkg_lib_dir: Path = Path(pkg_dir, "lib")
        self._initialized: bool = True


class BuildOnlineCalculatorManager(_BuildOnlineManager):

    @dataclasses.dataclass
    class _TorchContext:
        torch_version: str = ""
        torch_root_dir: str = ""
        torch_c_use_cxx11_abi: int = 1

        def __init__(self):
            old_val = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")
            try:
                os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                import torch
                self.torch_version = str(torch.__version__)
                self.torch_root_dir = str(Path(torch.__file__).parent)
                self.torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
            except ImportError as e:
                raise RuntimeError(f"Can not import torch, please check your python environment. Error: {e}") from e
            finally:
                if old_val is None:
                    os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
                else:
                    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = old_val

    def __init__(self):
        super().__init__()
        self._target_loaded: bool = False  # 目标二进制已加载进进程空间

    def build_and_load_calculator(self):
        if self._target_loaded:
            return
        torch_ctx = self._TorchContext()
        cmake_ctx = self._CMakeContext()
        with tempfile.TemporaryDirectory() as _tmp_dir:
            # 编译
            ext = f"-DPY3_MOD_TORCH_VERSION={torch_ctx.torch_version}"
            ext += f" -DPY3_MOD_TORCH_ROOT_PATH={torch_ctx.torch_root_dir}"
            ext += f" -DPY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI={torch_ctx.torch_c_use_cxx11_abi}"
            compile_ctx = self._CMakeContext.CompileContext(src_dir=Path(self.pkg_lib_dir, "framework/src/calculator"),
                                                            tmp_dir=Path(_tmp_dir))
            compile_ctx.cfg_cmd_ext = ext
            install_prefix = cmake_ctx.compile(ctx=compile_ctx)
            # 加载
            calc_shared = Path(install_prefix, "lib/libtile_fwk_calculator.so")
            if not calc_shared.exists():
                raise RuntimeError(f"{calc_shared} not exists.")
            ctypes.CDLL(str(calc_shared), mode=ctypes.RTLD_GLOBAL)
        self._target_loaded = True


class BuildOnlinePyptoImplManager(_BuildOnlineManager):

    class _PythonContext:

        def __init__(self):
            self.minor: int = 0
            self.pybind11_cmake_dir: Optional[Path] = None
            self._init_minor_version()
            self._init_development_component()
            self._init_pip_mod_pybind11()

        @staticmethod
        def _init_development_component():
            python_h = Path(sysconfig.get_path("include")) / "Python.h"
            if not python_h.exists():
                raise RuntimeError(
                        f"Python development headers not found (expected {python_h}).\n"
                        "Hint: install python3-dev (e.g. `apt install python3-dev` or `yum install python3-devel`)."
                )

        def _init_minor_version(self):
            minor = int(sys.version_info.minor)
            if minor < 9:
                raise RuntimeError(
                    f"Python version 3.{minor} is not supported for online compilation, require >= 3.9.\n"
                    "Hint: use a Python 3.9+ interpreter."
                )
            self.minor = minor

        def _init_pip_mod_pybind11(self):
            pybind11_dir = None
            try:
                import pybind11
                pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
            except ImportError as e:
                raise RuntimeError(
                    "pybind11 pip package not found.\n"
                    "Hint: install it with `pip install pybind11`."
                ) from e
            if not pybind11_dir or not pybind11_dir.exists():
                raise RuntimeError(
                    "pybind11 cmake dir empty.\n"
                    "Hint: install it with `pip install pybind11`."
                )
            self.pybind11_cmake_dir = pybind11_dir

    def __init__(self):
        super().__init__()
        self._target_compiled: bool = self._find_pypto_impl_so()

    def ensure_pypto_impl(self):
        if self._target_compiled:
            return

        src_dir = Path(self.pkg_lib_dir, "pypto_impl/src")
        if not src_dir.exists():
            # 当需要在线编译 pypto_impl 但是 whl 包内没有相关源码时, 不做在线编译
            # 靠外部 import pypto_impl 报错时暴露原始问题, 避免此处截取问题导致的原始问题的掩盖.
            return

        pyenv_ctx = self._PythonContext()
        cmake_ctx = self._CMakeContext()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 编译
            ext = f"-DPython3_EXECUTABLE={sys.executable}"
            ext += f" -DPython3_EXECUTABLE_VERSION=3.{pyenv_ctx.minor}"
            ext += f" -DPython3_MOD_PYBIND11_CMAKE_DIR={pyenv_ctx.pybind11_cmake_dir}"
            ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
            if ascend_home:
                ext += f" -DASCEND_CANN_PACKAGE_PATH={ascend_home}"
            compile_ctx = self._CMakeContext.CompileContext(src_dir=src_dir,
                                                            tmp_dir=Path(tmp_dir))
            compile_ctx.cfg_cmd_ext = ext
            compile_ctx.install_prefix = self.pkg_dir
            install_prefix = cmake_ctx.compile(ctx=compile_ctx)
        # 检查
        if not self._find_pypto_impl_so():
            raise RuntimeError(
                "Online compilation succeeded but pypto_impl.so not found in install output.\n"
                f"Expected location: {self.pkg_dir}\n"
                f"Searched suffixes: {importlib.machinery.EXTENSION_SUFFIXES}"
            )
        self._target_compiled = True

    def _find_pypto_impl_so(self) -> bool:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            so_path = self.pkg_dir / f"pypto_impl{suffix}"
            if so_path.exists():
                _log.info("Found pypto_impl.so: %s", so_path)
                return True
        return False


def ensure_pypto_impl():
    BuildOnlinePyptoImplManager().ensure_pypto_impl()

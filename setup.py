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
"""PyPTO setuptools 配置模块

本模块提供了集成 CMake 构建系统的 setuptools 配置, 支持:
    - 使用 CMake 进行扩展模块的构建
    - 支持可编辑安装模式 (editable install)
    - 支持自定义 CMake 生成器、构建类型和选项
    - 支持 CMake 安装文件的自动追踪
"""
import argparse
import importlib
import logging
import hashlib
import math
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import site
import sysconfig
import warnings
from pathlib import Path
from typing import Optional, Any, List, Tuple, Union
from importlib import metadata

from setuptools import setup, Extension
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake 扩展模块的占位类

    本类不执行实际的扩展构建工作, 仅作为 setuptools 扩展模块的占位符. 实际的构建工作由 CMakeBuild 类通过调用 CMake 完成.
    """

    def __init__(self):
        """初始化 CMakeExtension 实例

        创建空的 Extension 对象, 实际构建由 CMake 处理.
        """
        super().__init__(name="", sources=[])  # 源文件列表为空, 因为实际构建由 CMake 处理


class EditModeHelper:
    """可编辑安装模式的辅助工具类

    提供获取 pip 可编辑模式实际安装路径的工具方法.
    """

    @staticmethod
    def get_pip_edit_mode_install_path() -> Path:
        """获取 pip 可编辑模式下的实际安装包路径

        以可编辑模式 (editable) 执行时, editable_wheel 命令结束后,
        whl 包的安装由 pip 接管, 无法通过自定义 setuptools 子命令 ('install') 的方式获取 whl 安装路径.

        本方法通过遍历 site-packages 路径, 优先返回包含 "dist-packages" 的路径
        (Debian/Ubuntu 系统) , 否则返回第一个 site-packages 路径.

        :return: pip 可编辑模式下的安装路径
        :rtype: Path
        """
        # 优先取用户级路径, 再取系统级路径
        site_paths = site.getsitepackages()
        # 遍历找到包含 "dist-packages" 的路径(Debian/Ubuntu), 无则取第一个 site-packages
        for path in site_paths:
            if "dist-packages" in path:
                return Path(path)
        # 非 Debian 系统(如 CentOS/Windows), 返回默认 platlib
        return Path(site_paths[0]) if site_paths else Path(sysconfig.get_path("platlib"))


class CustomEditableWheel(editable_wheel, EditModeHelper):
    """自定义 editable_wheel 命令

    继承自 setuptools 的 editable_wheel 命令, 提供以下功能:

    1. 感知 -e (可编辑) 模式, 传递标记给 build_ext 以便其处理 CMake 安装路径
    2. 接收 build_ext 传递的 CMake 安装文件列表, 并将其写入 whl 包的 RECORD 文件
       以便在 -e 模式下对应文件可以随 uninstall 流程删除

    关于 RECORD 文件写入:
        RECORD 文件回写功能暂未使能, 原因如下:
        - 插入到 RECORD 记录内的 CMake 安装文件本质不受 whl 包管理
        - 在多次执行 "pip install -e" 的场景下, 会执行:
          先编译 (Install 或 Rewrite 对应 CMake 安装文件) -> 再卸载 -> 再安装 的流程
        - 卸载阶段因 RECORD 文件内有相关文件记录, 会导致对应文件被删除,
          从而导致重复 "pip install -e" 结束后, 对应文件被删除的问题
    """

    def run(self):
        """执行 editable_wheel 命令

        传递 -e 模式标记给 build_ext 命令, 然后继续执行标准的命令流程. 这会触发 build_ext、egg_info 等子命令.
        """
        # 传递 -e 模式标记给 build_ext 命令
        build_ext_cmd = self.distribution.get_command_obj("build_ext")
        build_ext_cmd.pypto_editable_mode = True  # 设置标记
        # 继续执行标准的命令流程(这会触发 build_ext, egg_info)
        super().run()

    def _insert_cmake_install_files_to_whl_record_file(self):
        """接收 build_ext 传递的 CMake 安装文件并写入 whl 包的 RECORD 文件

        该方法会:
        1. 获取 CMake 安装文件列表
        2. 为每个文件计算 SHA256 哈希和文件大小
        3. 生成符合 setuptools 标准格式的 RECORD 条目
        4. 将新增条目插入 whl 包的 RECORD 文件
        """
        # 获取 RECORD 新增条目字符串
        record_str, record_num = self._get_cmake_install_files_record_info()
        if not record_str:
            return

        # 获取 setuptools 内置的 wheel 包, 避免直接使用 wheel 包, 减少因与 setuptools 内置的 wheel 包版本不一致导致兼容性问题
        try:
            vendor_wheel = importlib.import_module('setuptools._vendor.wheel.wheelfile')
            setuptools_wheel = getattr(vendor_wheel, 'WheelFile')
        except ImportError as e:
            raise ImportError("Could not import setuptools wheel module") from e

        # 找到 whl 包的 RECORD 文件路径(如: pypto-0.0.1.dist-info/RECORD)
        whl_file = self._get_editable_whl_file()
        with setuptools_wheel(whl_file, "r") as wf:
            record_file_lst = [Path(p) for p in wf.namelist() if p.endswith(".dist-info/RECORD")]
        if not record_file_lst:
            raise RuntimeError(f"Can't find RECORD file in {whl_file}")
        record_file = record_file_lst[0]

        # 将新增条目插入 whl 包的 RECORD 文件
        logging.info("Overwrite RECORD(%s), will insert %s entries.", record_file, record_num)
        with setuptools_wheel(whl_file, "a") as wf:
            wf.writestr(str(record_file), record_str.encode("utf-8"))

    def _get_cmake_install_files_record_info(self) -> Tuple[str, int]:
        """获取 CMake 安装文件的 RECORD 条目信息

        生成符合 setuptools 标准格式的 RECORD 条目字符串, 包括:
        1. 相对于 site-packages 的路径
        2. SHA256 哈希值
        3. 文件大小

        :return: RECORD 条目字符串和条目数量
        :rtype: Tuple[str, int]
        """
        # 获取 cmake install 文件列表
        install_files = self._get_cmake_install_files()
        if not install_files:
            return "", 0
        # 生成 cmake install 文件的 RECORD 条目(符合 setuptools 标准格式)
        record_entries = []
        site_pkg = self.get_pip_edit_mode_install_path()  # site-packages 绝对路径
        for abs_file in install_files:
            # 跳过不存在的文件
            if not abs_file.exists():
                continue
            # 计算 RECORD 条目
            # 格式1: 相对于 site-packages 的路径, RECORD 中必须用此路径, 否则 pip 无法识别
            # 格式2: SHA256 哈希(格式: sha256=xxx)
            # 格式3: 文件大小(字节数)
            rel_path = os.path.relpath(str(abs_file), str(site_pkg))
            with open(abs_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            file_hash = f"sha256={file_hash}"  # 与 setuptools 原生格式一致
            file_size = os.path.getsize(abs_file)
            # 拼接 RECORD 条目(格式: 路径,哈希,大小)
            entry = f"{rel_path},{file_hash},{file_size}"
            record_entries.append(entry)
        return "\n".join(record_entries), len(record_entries)

    def _get_cmake_install_files(self) -> List[Path]:
        """获取 CMake 安装文件列表

        从 build_ext 命令传递的 manifest 中读取 CMake 安装的文件列表.

        :return: CMake 安装文件路径列表
        :rtype: List[Path]
        """
        install_files = []
        if not (hasattr(self, 'pypto_install_manifest_lst') and self.pypto_install_manifest_lst):
            logging.warning("Can't get any CMake install manifest.")
        else:
            install_files = [Path(p) for p in getattr(self, "pypto_install_manifest_lst", [])]
        return install_files

    def _get_editable_whl_file(self) -> Path:
        """获取可编辑模式的 whl 文件路径

        根据分布信息名称查找对应 editable whl 文件.

        :return: 可编辑模式的 whl 文件路径
        :rtype: Path
        :raises RuntimeError: 如果找不到对应的 whl 文件
        """
        dist_info = self.get_finalized_command("dist_info")
        dist_name = getattr(dist_info, "name", "pypto")
        whl_pattern = f"{dist_name}-*.editable-*.whl"
        whl_file_lst = list(Path(self.dist_dir).glob(whl_pattern))
        if not whl_file_lst:
            raise RuntimeError(f"Can't get whl file, Dir: {self.dist_dir}, pattern: {whl_pattern}")
        return Path(whl_file_lst[0])


class CMakeUserOption:
    """CMake 用户选项配置类

    管理从命令行或环境变量传递的 CMake 配置选项.

    支持的配置选项:
        - cmake-generator: CMake 生成器 (如 "Unix Makefiles" 或 "Ninja")
        - cmake-build-type: 构建类型 (如 "Debug" 或 "Release")
        - cmake-options: 额外的 CMake 选项
        - cmake-verbose: 是否启用 CMake 详细输出

    配置来源:
        1. 命令行参数
        2. 环境变量 PYPTO_BUILD_EXT_ARGS

    优先级:
        命令行配置优先于环境变量配置.
        --cmake-generator 和 --cmake-build-type 优先于 --cmake-options 中冲突的选项.
    """
    # 额外的命令行配置, 格式: 长选项, 短选项, 描述, 默认值
    USER_OPTION = [
        ('cmake-generator=', None, 'CMake Generator', None),
        ('cmake-build-type=', None, 'CMake Build Type', None),
        ('cmake-options=', None, 'CMake Options', None),
        ('cmake-verbose', None, 'Enable CMake Verbose Output', None),
    ]

    def __init__(self):
        """初始化 CMakeUserOption 实例"""
        self.cmake_generator: Optional[str] = None
        self.cmake_build_type: Optional[str] = None
        self.cmake_options: Optional[str] = None
        self.cmake_verbose: bool = False
        # 获取 CMake 路径
        self.cmake: Optional[Path] = None

    def __str__(self) -> str:
        """返回 CMake 配置的字符串表示

        :return: 格式化的配置信息字符串
        :rtype: str
        """
        ver = sys.version_info
        ver1 = metadata.version("setuptools")
        ver2 = metadata.version("pybind11")

        desc = f"\nEnviron"
        desc += f"\n    Python3               : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\n        pip"
        desc += f"\n               setuptools : {ver1}"
        desc += f"\n                 pybind11 : {ver2}"
        desc += f"\n    CMake                 : {self.cmake}"
        desc += f"\n{self.__class__.__name__}"
        desc += f"\n    cmake_generator       : {self.cmake_generator}"
        desc += f"\n    cmake_build_type      : {self.cmake_build_type}"
        desc += f"\n    cmake-options         : {self.cmake_options}"
        desc += f"\n    cmake-verbose         : {self.cmake_verbose}"
        desc += f"\n"
        return desc

    @staticmethod
    def which_cmake() -> Optional[Path]:
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

    def initialize_options_cmake(self):
        """初始化 CMake 选项

        赋初值, 从环境变量 PYPTO_BUILD_EXT_ARGS 中获取配置值作为默认值. 环境变量中的配置会被后续命令行参数覆盖.

        :raises RuntimeError: 如果找不到 CMake 可执行文件
        """
        # 赋初值, 此处需赋初值, 否则 setuptools 会丢失对应参数
        self.cmake_generator = None
        self.cmake_build_type = None
        self.cmake_options = None
        self.cmake_verbose = False
        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError(f"Can't find cmake")

        # 从环境变量(如有)中获取配置值作为默认值, 后续命令行中如果也设置了对应配置则会覆盖对应值, 达到命令行配置优先生效的效果.
        env_build_ext_args = os.environ.get("PYPTO_BUILD_EXT_ARGS", "")
        if not env_build_ext_args:
            return
        pattern = r'(?:[^\s\"\']|\"[^\"]*\"|\'[^\']*\')+'
        env_build_ext_args_split = re.findall(pattern, env_build_ext_args)
        parser = argparse.ArgumentParser(description=f"Setuptools CMakeBuild Ext.", add_help=False)
        parser.add_argument("--cmake-generator", nargs="?", type=str, default=None, dest="cmake_generator")
        parser.add_argument("--cmake-build-type", nargs="?", type=str, default=None, dest="cmake_build_type")
        parser.add_argument("--cmake-options", nargs="?", type=str, default="", dest="cmake_options")
        parser.add_argument("--cmake-verbose", action="store_true", default=False, dest="cmake_verbose")
        args, _ = parser.parse_known_args(env_build_ext_args_split)
        self.cmake_generator = args.cmake_generator
        self.cmake_build_type = args.cmake_build_type
        self.cmake_options = args.cmake_options
        self.cmake_verbose = args.cmake_verbose

    def finalize_options_cmake(self):
        """处理并修正 CMake 选项

        处理命令行传递的参数值, 修正冲突的选项:
        - --cmake-generator 和 --cmake-build-type 优先于 --cmake-options 中冲突的选项
        - 为生成器名称添加引号以支持带空格的名称
        """
        # 赋传参值
        self.cmake_generator = None if not self.cmake_generator else self.cmake_generator
        if self.cmake_generator:
            self.cmake_generator = self.cmake_generator.replace(r'"', "")
            self.cmake_generator = f"\"{self.cmake_generator}\""
        self.cmake_build_type = None if not self.cmake_build_type else self.cmake_build_type
        self.cmake_options = self.cmake_options.replace("'", "").replace('"', "") if self.cmake_options else None
        self.cmake_verbose = True if self.cmake_verbose else False
        # CMake Options 修正
        cmake_option_lst = [o.replace(" ", "") for o in (self.cmake_options.split(" ") if self.cmake_options else [])]
        if self.cmake_generator:
            for option in cmake_option_lst:
                if option.startswith("-DCMAKE_GENERATOR="):
                    cmake_option_lst.remove(option)
                    logging.warning("Configuration via --cmake-generator has higher priority than --cmake-options; "
                                    "in case of conflict, the former prevails.")
        if self.cmake_build_type:
            for option in cmake_option_lst:
                if option.startswith("-DCMAKE_BUILD_TYPE="):
                    cmake_option_lst.remove(option)
                    logging.warning("Configuration via --cmake-build-type has higher priority than --cmake-options; "
                                    "in case of conflict, the former prevails.")
        self.cmake_options = " ".join(cmake_option_lst) if cmake_option_lst else self.cmake_options


class CMakeBuild(build_ext, CMakeUserOption, EditModeHelper):
    """自定义 build_ext 命令, 调用 CMake 构建系统

    继承自 setuptools 的 build_ext 命令, 重写构建流程以使用 CMake.

    主要功能:
        - 调用 CMake 执行 Configure、Build、Install 流程
        - 支持可编辑安装模式 (editable mode)
        - 支持自定义 CMake 生成器、构建类型和选项
        - 将 CMake 安装的文件列表传递给 editable_wheel 命令

    流程:
        1. CMake Configure: 配置构建参数
        2. CMake Build: 执行编译
        3. CMake Install: 安装文件到目标目录
        4. 如果是可编辑模式, 传递安装文件清单给 editable_wheel
    """
    user_options = build_ext.user_options + CMakeUserOption.USER_OPTION

    @staticmethod
    def _get_job_num(job_num: Optional[int], generator: Optional[str]) -> Optional[int]:
        """获取构建并行任务数

        根据系统 CPU 核数和构建生成器类型确定合适的并行任务数. 如果使用 Ninja 生成器, 则由 Ninja 自动决定并行度.

        :param job_num: 用户指定的并行任务数
        :type job_num: Optional[int]
        :param generator: 构建生成器名称
        :type generator: Optional[str]
        :return: 并行任务数, None 表示由构建工具自动决定
        :rtype: Optional[int]
        """
        def_job_num = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 128)  # 128 为缺省最大核数
        def_job_num = None if generator and generator.lower() in ["ninja", ] else def_job_num  # ninja 自身决定缺省核数
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_cmake_install_manifest(build_dir: Path, file_name: str = "install_manifest.txt") -> List[str]:
        """获取 CMake 安装文件清单

        从指定文件中读取 CMake 安装的文件列表.

        :param build_dir: 构建目录
        :type build_dir: Path
        :param file_name: 清单文件名称
        :type file_name: str
        :return: 安装文件列表
        :rtype: List[str]
        """
        installed_files = []
        install_manifest_file = Path(build_dir, file_name)
        if install_manifest_file.exists():
            with open(install_manifest_file, 'r', encoding="utf-8") as fh:
                installed_files = [line.strip() for line in fh if line.strip()]
        return installed_files

    def initialize_options(self):
        """初始化构建选项

        通过控制命令行选项初始化顺序, 实现实际命令行选项优先生效.

        先调用父类的 initialize_options(), 再初始化 CMake 特定选项.
        """
        super().initialize_options()
        self.initialize_options_cmake()

    def finalize_options(self):
        """处理并最终确定构建选项

        先调用父类的 finalize_options(), 再处理 CMake 特定选项.
        """
        super().finalize_options()
        self.finalize_options_cmake()

    def run(self):
        """执行构建流程

        执行完整的 CMake 构建流程:
        1. CMake Configure: 配置构建参数
        2. CMake Build: 执行编译
        3. CMake Install: 安装文件到目标目录
        4. 如果是可编辑模式, 传递安装文件清单给 editable_wheel
        """
        logging.info("%s", self)
        # 源码根目录
        src = Path(__file__).parent.resolve()
        env = os.environ.copy()
        env["CCACHE_BASEDIR"] = str(src)  # 在 ccache 场景支持路径归一化
        build_dir = self._get_cmake_build_prefix()
        build_dir.mkdir(parents=True, exist_ok=True)
        # 获取 cmake install prefix
        cmake_install_prefix = self._get_cmake_install_prefix()

        # CMake Configure
        cmd = f"{self.cmake} -S {src} -B {build_dir}"
        cmd += f" -G {self.cmake_generator}" if self.cmake_generator else ""
        cmd += f" -DCMAKE_BUILD_TYPE={self.cmake_build_type}" if self.cmake_build_type else ""
        cmd += f" -DPython3_EXECUTABLE={sys.executable} -DCMAKE_INSTALL_PREFIX={cmake_install_prefix}"
        cmd += f" {self.cmake_options}" if self.cmake_options else ""
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8', env=env)
        ret.check_returncode()

        # CMake Build
        job_num = self._get_job_num(job_num=self.parallel, generator=self.cmake_generator)
        cmd = f"{self.cmake} --build {build_dir}" + (f" -j {job_num}" if job_num else "")
        cmd += f" --verbose" if self.cmake_verbose else ""
        logging.info("CMake Build, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8', env=env)
        ret.check_returncode()

        # CMake Install
        cmake_install_prefix: Path = self._get_cmake_install_prefix()  # 重复获取触发提示
        cmd = f"{self.cmake} --install {build_dir} --prefix {cmake_install_prefix}"
        logging.info("CMake Install, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        if self._edit_mode():
            installed_files = self._get_cmake_install_manifest(build_dir=build_dir)
            if installed_files:
                # 向 editable_wheel 命令传递
                editable_wheel_cmd = self.distribution.get_command_obj("editable_wheel")
                editable_wheel_cmd.pypto_install_manifest_lst = installed_files
                logging.info("Command build_ext passes %s CMake install files to editable_wheel command",
                             len(editable_wheel_cmd.pypto_install_manifest_lst))

    def _edit_mode(self) -> bool:
        """判断是否为可编辑安装模式

        :return: 如果是可编辑模式则返回 True
        :rtype: bool
        """
        if hasattr(self, 'pypto_editable_mode') and self.pypto_editable_mode:
            return True
        return False

    def _get_cmake_install_prefix(self) -> Path:
        """获取 CMake 安装前缀路径

        在可编辑安装模式下, 设置为源码相关路径；否则使用构建库路径.

        :return: CMake 安装前缀路径
        :rtype: Path
        """
        cmake_install_prefix = Path(self.build_lib)
        if self._edit_mode():
            # 可编辑安装模式下, 设置 CMake Install Prefix 为源码相关路径
            src_root = Path(__file__).parent.resolve()  # -e 模式下不会 copy 源码到 tmp 目录
            cmake_install_prefix = Path(src_root, "python")
            logging.warning("Run in editable mode, use %s as cmake install prefix.", cmake_install_prefix)
        return cmake_install_prefix.resolve()

    def _get_cmake_build_prefix(self) -> Path:
        """获取 CMake 构建前缀路径

        在可编辑模式下使用源码目录下的 build 目录, 否则使用临时构建目录.

        :return: CMake 构建前缀路径
        :rtype: Path
        """
        build_dir = Path(self.build_temp)
        if self._edit_mode():
            src_root = Path(__file__).parent.resolve()
            build_dir = src_root / "build"
        return build_dir.resolve()


class SetupCtrl:
    """Setuptools 配置流程控制器

    负责配置和启动 setuptools 的构建流程.

    主要功能:
        - 配置自定义的命令类 (editable_wheel、build_ext)
        - 配置扩展模块 (CMakeExtension)
        - 过滤 setuptools 的警告信息
    """

    @classmethod
    def main(cls):
        """主处理流程

        配置并启动 setuptools 构建流程.

        配置项:
        - 扩展模块: CMakeExtension (占位类)
        - 自定义命令:
            - editable_wheel: CustomEditableWheel, 处理可编辑安装模式
            - build_ext: CMakeBuild, 调用 CMake 进行构建
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="setuptools.command.build_py")
        # Setuptools 配置
        setup(
            # 扩展模块配置
            ext_modules=[
                CMakeExtension(),
            ],
            cmdclass={
                'editable_wheel': CustomEditableWheel,  # setuptools>=58.0.0, pip install -e 会触发 editable_wheel
                'build_ext': CMakeBuild,
            },
        )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    SetupCtrl.main()

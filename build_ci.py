#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO 项目 CI 场景构建控制总入口

本文件提供 PyPTO 项目 CI 场景的统一构建入口, 支持多种构建模式和配置选项.

主要功能:
    - 支持 whl 包的常规编译和可编辑模式编译
    - 支持 UTest/STest/Examples 等测试用例的执行
    - 支持构建超时控制和超时后自动清理子进程

使用方式:
    通过命令行参数配置构建选项, 执行脚本即可触发构建流程:

        python build_ci.py [选项]

    常用选项:
        -f/--frontend: 指定前端类型 (python3/cpp)
        -b/--backend: 指定后端类型 (npu/cost_model)
        -t/--targets: 指定编译目标
        -j/--job_num: 指定编译并行度
        --build_type: 指定构建类型 (Debug/Release/MinSizeRel/RelWithDebInfo)
        -u/--utest: 启用 UTest 测试
        -s/--stest: 启用 STest 测试
        -c/--clean: 清理构建目录和安装目录

示例:
    # 使用默认配置构建
    python build_ci.py

    # 指定前端和后端类型构建
    python build_ci.py -f python3 -b npu

    # 启用测试并指定并行度
    python build_ci.py -u -s -j 8

    # 清理并重新构建
    python build_ci.py -c --build_type Debug
"""
import abc
import argparse
import dataclasses
import logging
import math
import multiprocessing
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from importlib import metadata
from packaging import requirements


class CMakeParam(abc.ABC):
    """CMake 参数抽象基类

    定义所有需要向 CMake 传递 Option 的参数类的通用接口.
    子类需要实现 reg_args() 方法来注册命令行参数, 实现 get_cfg_cmd() 方法来生成 CMake 配置命令.
    """

    @staticmethod
    def get_system_processor() -> str:
        """获取系统处理器架构名称

        通过 platform.machine() 获取当前系统的处理器架构, 并将常见的别名映射到标准名称.

        :return: 标准化的处理器架构名称, 如 x86_64 或 aarch64
        :rtype: str
        """
        machine = platform.machine().lower()
        arch_map = {  # 直接映射常见架构
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
        }
        return arch_map.get(machine, machine)

    @staticmethod
    @abc.abstractmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册命令行参数

        向参数解析器注册当前类支持的命令行参数. 子类必须实现此方法以定义各自的参数选项.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 用于子类特殊实现扩展时使用
        :type ext: Optional[Any]
        """
        pass

    @classmethod
    def _cfg_require(cls, opt: str, ctr: bool = True, tv: str = "ON", fv: str = "OFF") -> str:
        """获取 CMake Configure 阶段的必选 Option 配置

        根据 ctr 控制变量的值, 返回对应的 CMake Option 配置字符串. 该方法会始终返回一个非空的配置字符串.

        :param opt: CMake Option 名称, 会最终体现到 CMake -D 传入的参数中
        :type opt: str
        :param ctr: 控制变量, 标识 CMake Option 布尔值
        :type ctr: bool
        :param tv: ctr 为 True 时设置的值, 默认为 "ON"
        :type tv: str
        :param fv: ctr 为 False 时设置的值, 默认为 "OFF"
        :type fv: str
        :return: CMake 配置字符串, 格式如 " -DOPT_NAME=VALUE"
        :rtype: str
        """
        return f" -D{opt}=" + (tv if ctr else fv)

    @classmethod
    def _cfg_optional(cls, opt: str, ctr: bool, v: str) -> str:
        """获取 CMake Configure 阶段的可选 Option 配置

        根据 ctr 控制变量的值, 返回对应的 CMake Option 配置字符串. 当 ctr 为 False 时, 返回空字符串.

        :param opt: CMake Option 名称, 会最终体现到 CMake -D 传入的参数中
        :type opt: str
        :param ctr: 控制变量, 标识 CMake Option 布尔值
        :type ctr: bool
        :param v: 控制变量为 True 时设置的值
        :type v: str
        :return: CMake 配置字符串, 格式如 " -DOPT_NAME=VALUE", ctr 为 False 时返回空字符串
        :rtype: str
        """
        return (f" -D{opt}=" + v) if ctr else ""

    @abc.abstractmethod
    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据当前参数配置, 生成对应的 CMake 配置命令字符串. 子类必须实现此方法以定义具体的配置参数.

        :param ext: 扩展信息, 用于子类特殊实现扩展时使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        pass


@dataclasses.dataclass
class FeatureParam(CMakeParam):
    """特性控制相关参数

    管理构建过程中的特性选项, 包括前端类型, 后端类型和 whl 包编译模式.
    """
    whl_name: str = "pypto"
    frontend_type: Optional[str] = None  # 前端类型, 支持 python3, cpp
    backend_type: Optional[str] = None  # 后端类型, 支持 npu, cost_model
    whl_plat_name: Optional[str] = None  # python3 whl 包 plat-name
    whl_isolation: bool = False  # 以 isolation 模式编译 whl 包
    whl_editable: bool = False  # 以 editable 模式编译 whl 包

    def __init__(self, args):
        """初始化 FeatureParam 实例

        从命令行参数中解析前端类型, 后端类型和 whl 包编译模式.
        如果后端类型为 npu 但未设置 ASCEND_HOME_PATH 环境变量, 则自动回退到 cost_model 后端.

        :param args: 命令行参数解析结果
        """
        self.frontend_type = "python3" if args.frontend is None else args.frontend
        self.backend_type = "npu" if args.backend is None else args.backend
        if not os.environ.get("ASCEND_HOME_PATH") and self.backend_type in ["npu"]:
            logging.warning("Environment variable ASCEND_HOME_PATH is unset/empty, falling back to cost_model backend.")
            self.backend_type = "cost_model"
        self.whl_plat_name = f"{args.plat_name}_{CMakeParam.get_system_processor()}" if args.plat_name else ""
        self.whl_isolation = args.isolation
        self.whl_editable = args.editable

    def __str__(self) -> str:
        """返回特性参数的字符串表示

        :return: 格式化的特性参数字符串
        :rtype: str
        """
        desc = ""
        desc += f"\nFeature"
        desc += f"\n    Frontend                : {self.frontend_type}"
        if self.frontend_type_python3:
            if self.whl_plat_name:
                desc += f"\n    PlatName                : {self.whl_plat_name}"
            desc += f"\n    Isolation               : {self.whl_isolation}"
            desc += f"\n    Editable                : {self.whl_editable}"
        desc += f"\n    Backend                 : {self.backend_type}"
        return desc

    @property
    def frontend_type_python3(self) -> bool:
        """判断前端类型是否为 Python3

        :return: 如果前端类型为 "python" 或 "python3", 返回 True
        :rtype: bool
        """
        return self.frontend_type in ["python", "python3"]

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册特性相关的命令行参数

        向参数解析器注册前端类型, 后端类型, whl 编译模式等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("-f", "--frontend", nargs="?", type=str, default="python3",
                            choices=["python3", "cpp"],
                            help="frontend, such as python3/cpp etc.")
        parser.add_argument("--plat_name", nargs="?", type=str, default="",
                            choices=["manylinux2014", "manylinux_2_24", "manylinux_2_28"],
                            help="whl plat_name, such as manylinux2014/manylinux_2_24/manylinux_2_28 etc.")
        parser.add_argument("--no_isolation", action="store_false", default=True, dest="isolation",
                            help="Disable building the project(whl) in an isolated virtual environment. "
                                 "Build dependencies must be installed separately when this option is used.")
        parser.add_argument("--editable", action="store_true", default=False,
                            help="Install whl in editable mode (i.e. setuptools \"editable_wheel\")")
        parser.add_argument("-b", "--backend", nargs="?", type=str, default="npu",
                            choices=["npu", "cost_model"],
                            help="backend, such as npu/cost_model etc.")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据前端类型和后端类型生成对应的 CMake 配置参数.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        cmd = ""
        cmd += self._cfg_require(opt="ENABLE_FEATURE_PYTHON_FRONT_END", ctr=self.frontend_type_python3)
        cmd += self._cfg_require(opt="BUILD_WITH_CANN", ctr=self.backend_type in ["npu"])
        return cmd


@dataclasses.dataclass
class BuildParam(CMakeParam):
    """构建相关参数

    管理构建过程的配置选项, 包括 CMake 配置参数和构建执行参数.
    """
    # Configure
    generator: Optional[str] = None  # Generator
    build_type: Optional[str] = None  # 构建类型
    asan: bool = False  # 使能 AddressSanitizer
    ubsan: bool = False  # 使能 UndefinedBehaviorSanitizer
    gcov: bool = False  # 使能 GNU Coverage
    gcov_incr: bool = False  # 使能增量覆盖率 GCov 计算
    clang_install_path: Optional[Path] = None  # Clang 安装位置
    compile_dependency_check: bool = False  # 使能编译依赖关系检查
    # Build
    targets: Optional[List[str]] = None  # 编译目标
    job_num: Optional[int] = None  # 编译阶段使用核数

    def __init__(self, args):
        """初始化 BuildParam 实例

        从命令行参数中解析构建相关的配置选项.

        :param args: 命令行参数解析结果
        """
        self.targets = args.targets
        self.job_num = self._get_job_num(job_num=args.job_num, generator=args.generator)
        self.generator = self._get_generator(generator=args.generator)
        self.build_type = args.build_type
        self.asan = args.asan
        self.ubsan = args.ubsan
        self.gcov = args.gcov
        self.gcov_incr = args.gcov_increment
        self.clang_install_path = self._get_clang_install_path(opt=args.clang)
        self.compile_dependency_check = args.compile_dependency_check

    def __str__(self) -> str:
        """返回构建参数的字符串表示

        :return: 格式化的构建参数字符串
        :rtype: str
        """
        desc = f"\nBuild"
        desc += f"\n    CMake"
        desc += f"\n        Configure"
        desc += f"\n                  Generator : {self.generator}"
        desc += f"\n                  BuildType : {self.build_type}"
        desc += f"\n                       ASan : {self.asan}"
        desc += f"\n                      UbSan : {self.ubsan}"
        desc += f"\n                       GCov : {self.gcov}, Increment: {self.gcov_incr}"
        desc += f"\n           ClangInstallPath : {self.clang_install_path}"
        desc += f"\n            CompileDepCheck : {self.compile_dependency_check}"
        desc += f"\n        Build"
        desc += f"\n                    Targets : {self.targets}"
        desc += f"\n                    Job Num : {self.job_num}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册构建相关的命令行参数

        向参数解析器注册构建生成器, 构建类型, Sanitizer 选项等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        # Configure
        parser.add_argument("--generator", nargs="?", type=str, default="",
                            help="Specify a build system generator.")
        parser.add_argument("--build_type", "--build-type", nargs="?", type=str, default="Release",
                            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
                            help="build type.")
        parser.add_argument("--asan", action="store_true", default=False,
                            help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument("--gcov", action="store_true", default=False,
                            help="Enable GNU Coverage Instrumentation Tool.")
        parser.add_argument("--gcov_increment", action="store_true", default=False,
                            help="Enable increment coverage calculation based on latest commit.")
        parser.add_argument("--clang", nargs="?", type=str, default="",
                            help="Specify clang install path, such as /usr/bin/clang")
        parser.add_argument("--compile_dependency_check", action="store_true", default=False,
                            help="Enable compile dependency relation check.")
        # Build
        parser.add_argument("-t", "--targets", nargs="?", type=str, action="append",
                            help="targets, specific build targets, "
                                 "If you specify more than one, all targets within the specified range are built.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=-1,
                            help="job num, specific job num of build.")

    @staticmethod
    def _get_clang_install_path(opt: Optional[str]) -> Optional[Path]:
        """获取 Clang 安装目录

        根据指定的 clang 参数或自动查找来确定 Clang 安装路径.

        :param opt: Clang 参数, 可以是 None (自动查找) , 空字符串 (不使用 Clang) 或具体路径
        :type opt: Optional[str]
        :return: Clang 安装目录路径, 如果不使用 Clang 则返回 None
        :rtype: Optional[Path]
        """
        if opt is None:  # 指定 clang 参数, 但未指定具体路径, 此时需尝试寻找
            cmd = "which clang"
            ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
            ret.check_returncode()
            clang_install_path = Path(ret.stdout).resolve()
        elif opt == "":  # 未指定 clang 参数
            clang_install_path = None
        else:  # 指定 clang 参数, 并指定具体路径
            clang_install_path = Path(opt)
        if clang_install_path is not None:
            clang_install_path = Path(clang_install_path).resolve().parent
            if not clang_install_path.exists():
                raise ValueError(f"Clang install path not exist, path={clang_install_path}")
        return clang_install_path

    @staticmethod
    def _get_job_num(job_num: Optional[int], generator: Optional[str]) -> Optional[int]:
        """获取构建并行任务数

        根据系统 CPU 核数和构建生成器类型确定合适的并行任务数. 如果使用 Ninja 生成器, 则由 Ninja 自动决定并行度.

        :param job_num: 用户指定的并行任务数
        :type job_num: Optional[int]
        :param generator: 构建生成器名称
        :type generator: Optional[str]
        :return: 最终的并行任务数, None 表示由构建工具自动决定
        :rtype: Optional[int]
        """
        def_job_num = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 128)  # 128 为缺省最大核数
        def_job_num = None if generator and generator.lower() in ["ninja", ] else def_job_num  # ninja 自身决定缺省核数
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_generator(generator: Optional[str]) -> Optional[str]:
        """获取构建生成器名称

        如果指定了生成器, 则在名称外添加引号以支持带空格的生成器名称.

        :param generator: 构建生成器名称
        :type generator: Optional[str]
        :return: 处理后的构建生成器名称
        :rtype: Optional[str]
        """
        return f"\"{generator}\"" if generator else generator

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据构建配置参数生成对应的 CMake 配置命令字符串.
        支持构建类型, Sanitizer 选项, 覆盖率统计和 Clang 工具链等配置.

        :param ext: 扩展信息, 如果为 True 则不包含构建类型
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        inc_build_type = bool(ext) if ext is not None else True
        cmd = (self._cfg_require(opt="CMAKE_BUILD_TYPE", tv=self.build_type) if inc_build_type else "")
        cmd += self._cfg_require(opt="ENABLE_ASAN", ctr=self.asan)
        cmd += self._cfg_require(opt="ENABLE_UBSAN", ctr=self.ubsan)
        cmd += self._cfg_require(opt="ENABLE_GCOV", ctr=self.gcov)

        def _check_clang_toolchain(_opt: str, _b: str) -> Tuple[bool, str]:
            """检查 Clang 工具链是否存在并生成配置命令"""
            _p = Path(self.clang_install_path, _b)
            if _p.exists():
                return True, self._cfg_require(opt=_opt, tv=str(_p))
            logging.error("Clang Toolchain %s not exist.", _p)
            return False, ""

        def _gen_clang_cmd() -> Tuple[bool, str]:
            """生成 Clang 相关的 CMake 配置命令"""
            _bin_opt_lst = [["clang", "CMAKE_C_COMPILER"], ["clang++", "CMAKE_CXX_COMPILER"]]
            _rst = True
            _cmd = ""
            for _bin_opt in _bin_opt_lst:
                _sub_bin, _sub_opt = _bin_opt
                _sub_rst, _sub_cmd = _check_clang_toolchain(_opt=_sub_opt, _b=_sub_bin)
                _rst = _rst and _sub_rst
                _cmd = _cmd + _sub_cmd
            return _rst, _cmd if _rst else ""

        # Clang
        if self.clang_install_path is not None:
            ret, clang_cmd = _gen_clang_cmd()
            if not ret:
                raise RuntimeError(f"Clang({self.clang_install_path}) not complete.")
            cmd += clang_cmd

        # Others
        cmd += self._cfg_require(opt="ENABLE_COMPILE_DEPENDENCY_CHECK", ctr=self.compile_dependency_check)
        return cmd

    def get_build_cmd_lst(self, cmake: Path, binary_path: Path) -> List[str]:
        """生成 CMake 构建命令列表

        根据指定的构建目标生成对应的 CMake 构建命令.

        :param cmake: CMake 可执行文件路径
        :type cmake: Path
        :param binary_path: 二进制构建目录路径
        :type binary_path: Path
        :return: CMake 构建命令列表
        :rtype: List[str]
        """
        cmd_list = []
        if self.targets:
            for t in self.targets:
                cmd = f"{cmake} --build {binary_path} --target {t}"
                cmd += f" -j {self.job_num}" if self.job_num else ""
                cmd_list.append(cmd)
        else:
            cmd = f"{cmake} --build {binary_path}"
            cmd += f" -j {self.job_num}" if self.job_num else ""
            cmd_list.append(cmd)
        return cmd_list


@dataclasses.dataclass
class TestsExecuteParam(CMakeParam):
    """测试执行相关参数

    管理测试执行的配置选项, 包括自动执行, 并行执行, 超时控制和耗时缓存等.
    """
    changed_file: Optional[Path] = None  # 修改文件路径
    auto_execute: bool = False  # 用例自动执行
    auto_execute_parallel: bool = False  # 用例并行执行
    case_execute_timeout: Optional[int] = None  # 用例执行时, 单个用例超时时长
    case_execute_cpu_rank_size: Optional[int] = None  # 用例并行执行时, CPU 亲和性 Rank Size
    dump_case_duration_json: Optional[Path] = None  # 用例耗时缓存文件路径
    dump_case_duration_max_num: Optional[int] = None  # 用例耗时缓存最大数量
    dump_case_duration_min_secends: Optional[int] = None  # 用例耗时缓存最小秒数

    def __init__(self, args):
        """初始化 TestsExecuteParam 实例

        从命令行参数中解析测试执行相关的配置选项.

        :param args: 命令行参数解析结果
        """
        self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
        self.auto_execute = args.disable_auto_execute
        self.auto_execute_parallel = self.auto_execute and self.ci_model
        timeout = args.case_execute_timeout
        self.case_execute_timeout = timeout if timeout and timeout > 0 else None  # 单个用例执行超时时长
        self.case_execute_cpu_rank_size = args.cpu_rank_size
        duration_json = args.dump_case_duration_json
        self.dump_case_duration_json = Path(duration_json).resolve() if duration_json else None
        self.dump_case_duration_max_num = args.dump_case_duration_max_num
        self.dump_case_duration_min_secends = args.dump_case_duration_min_secends

    def __str__(self) -> str:
        """返回测试执行参数的字符串表示

        :return: 格式化的测试执行参数字符串
        :rtype: str
        """
        desc = f"\n    Execute"
        desc += f"\n               Changed File : {self.changed_file}"
        desc += f"\n                       Auto : {self.auto_execute}"
        desc += f"\n                   Parallel : {self.auto_execute_parallel}"
        desc += f"\n                CaseTimeout : {self.case_execute_timeout}"
        desc += f"\n        CaseDuration"
        desc += f"\n                       Json : {self.dump_case_duration_json}"
        desc += f"\n                     MaxNum : {self.dump_case_duration_max_num}"
        desc += f"\n                     MinSec : {self.dump_case_duration_min_secends}"
        return desc

    @property
    def ci_model(self) -> bool:
        """判断是否为 CI 模式

        :return: 如果指定了修改文件, 则返回 True (表示 CI 模式)
        :rtype: bool
        """
        return True if self.changed_file else False

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册测试执行相关的命令行参数

        向参数解析器注册增量测试, 自动执行, 超时控制等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("--changed_files", nargs="?", type=Path, default=None,
                            help="Specify the file of files changed, "
                                 "so that the corresponding test cases can be triggered incrementally.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")
        parser.add_argument("--case_execute_timeout", nargs="?", type=int, default=None,
                            help="Case execute timeout.")
        parser.add_argument("--cpu_rank_size", nargs="?", type=int, default=None,
                            help="Specify the rank size for CPU affinity grouping.")
        parser.add_argument("--dump_case_duration_json", nargs="?", type=Path, default=None,
                            help="Specify the path to the case duration json cache file.")
        parser.add_argument("--dump_case_duration_max_num", nargs="?", type=int, default=None,
                            help="Maximum number of cases to dump to duration json cache.")
        parser.add_argument("--dump_case_duration_min_secends", nargs="?", type=int, default=None,
                            help="Minimum duration (in seconds) for cases to dump to duration json cache.")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据测试执行配置生成对应的 CMake 配置参数.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_TESTS_EXECUTE", ctr=self.auto_execute)
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_PARALLEL", ctr=self.auto_execute_parallel)
        changed = self.changed_file and self.changed_file.exists() and self.changed_file.suffix.lower() == ".txt"
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_CHANGED_FILE", ctr=changed, tv=str(self.changed_file))
        return cmd


@dataclasses.dataclass
class TestsGoldenParam(CMakeParam):
    """Golden 测试相关参数

    管理系统测试 (STest) 的 Golden 标准数据相关配置.
    """
    clean: bool = False  # 清理 Golden 标记
    path: Optional[Path] = None  # 指定 Golden 路径

    def __init__(self, args):
        """初始化 TestsGoldenParam 实例

        从命令行参数中解析 Golden 测试相关配置.

        :param args: 命令行参数解析结果
        """
        self.clean = args.golden_clean
        if args.golden_path:
            # 传参且指定具体路径时, 使用指定路径, 否则具体缺省路径由 CMake 侧决定
            self.path = Path(args.golden_path).resolve()

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册 Golden 测试相关的命令行参数

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("--golden_path", "--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific Tests golden path.", dest="golden_path")
        parser.add_argument("--golden_clean", "--golden_path_clean", "--stest_golden_path_clean",
                            action="store_true", default=False,
                            help="Clean Tests golden.", dest="golden_clean")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据 Golden 测试配置生成对应的 CMake 配置参数.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH_CLEAN", ctr=self.clean)
        cmd += self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH", ctr=bool(self.path), tv=str(self.path))
        return cmd


@dataclasses.dataclass
class TestsFilterParam(CMakeParam):
    """测试过滤参数

    用于按条件过滤测试用例, 支持多种测试类型和过滤模式.
    """
    cmake_option: str = ""
    enable: bool = False
    filter_str: Optional[str] = None

    def __init__(self, argv: Optional[str], opt: str = ""):
        """初始化 TestsFilterParam 实例

        根据命令行参数值确定过滤选项的启用状态和过滤字符串.

        :param argv: 命令行参数值, None 表示启用默认过滤, 空字符串表示禁用, 其他值表示指定过滤字符串
        :type argv: Optional[str]
        :param opt: CMake 选项名称
        :type opt: str
        """
        self.cmake_option = opt
        if argv is None:
            self.enable, self.filter_str = True, "ON"  # 指定 对应参数, 但未指定内容
        elif argv == "":
            self.enable, self.filter_str = False, "OFF"  # 未指定 对应参数
        else:
            self.enable, self.filter_str = True, argv  # 指定 对应参数 且指定内容

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册测试过滤相关的命令行参数

        根据扩展信息生成对应的命令行参数选项.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 用于生成参数名称和帮助信息
        :type ext: Optional[Any]
        """
        mark = str(ext).lower()
        mark_lst = mark.split("_")
        have_char = len(mark_lst) <= 1
        mark_word = mark.replace("_", " ")
        help_str = f"Enable {mark_word} scene, specific {mark_word} filter, multiple cases are separated by ','"
        if have_char:
            mark_char = mark_lst[0][0] if have_char else None
            parser.add_argument(f"-{mark_char}", f"--{mark}", nargs="?", type=str, default="", help=help_str)
        else:
            parser.add_argument(f"--{mark}", nargs="?", type=str, default="", help=help_str)

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据过滤配置生成对应的 CMake 配置参数.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        cmd = ""
        if self.cmake_option:
            cmd += self._cfg_require(opt=f"{self.cmake_option}", ctr=self.enable, tv=f"{self.filter_str}")
        return cmd

    def get_filter_str(self, def_filter: str) -> str:
        """获取测试过滤字符串

        根据配置和默认过滤条件生成最终的过滤字符串.

        :param def_filter: 默认过滤条件
        :type def_filter: str
        :return: 过滤字符串, 如果未启用则返回空字符串
        :rtype: str
        """
        if not self.enable:
            return ""
        if self.filter_str not in ["ON"]:
            return self.filter_str
        if def_filter:
            return def_filter
        return self.filter_str


@dataclasses.dataclass
class STestExecuteParam(CMakeParam):
    """STest 执行相关参数

    管理系统测试 (STest) 的执行配置, 包括设备 ID, JSON 导出等.
    """
    auto_execute_device_id: str = ""
    interpreter_config: bool = False
    enable_binary_cache: bool = False
    dump_json: bool = False

    def __init__(self, args, enable_binary_cache: bool):
        """初始化 STestExecuteParam 实例

        从命令行参数中解析 STest 执行相关配置.

        :param args: 命令行参数解析结果
        :param enable_binary_cache: 是否启用二进制缓存
        :type enable_binary_cache: bool
        """
        devs = ["0"]
        if args.device is not None:
            devs = [str(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        self.auto_execute_device_id = ":".join(devs)
        self.dump_json = args.stest_dump_json
        self.interpreter_config = args.enable_interpreter_config
        self.enable_binary_cache = enable_binary_cache

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册 STest 执行相关的命令行参数

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Device ID, default 0.")
        parser.add_argument("--stest_dump_json", action="store_true", default=False,
                            help="Dump json files.")
        parser.add_argument("--enable_interpreter_config", action="store_true", default=False,
                            help="enable STest Interpreter Config")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        根据 STest 执行配置生成对应的 CMake 配置参数.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: CMake 配置参数字符串
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_STEST_EXECUTE_DEVICE_ID", tv=self.auto_execute_device_id)
        cmd += self._cfg_require(opt="ENABLE_STEST_DUMP_JSON", ctr=self.dump_json)
        cmd += self._cfg_require(opt="ENABLE_STEST_INTERPRETER_CONFIG", ctr=self.interpreter_config)
        cmd += self._cfg_require(opt="ENABLE_STEST_BINARY_CACHE", ctr=self.enable_binary_cache)
        return cmd


class TestsParam(CMakeParam):
    """测试参数总控类

    聚合所有测试相关的参数配置, 包括执行参数, Golden 参数, 过滤参数等.
    """

    def __init__(self, args):
        """初始化 TestsParam 实例

        从命令行参数中解析并初始化所有测试相关的参数配置.

        :param args: 命令行参数解析结果
        """
        self.exec: TestsExecuteParam = TestsExecuteParam(args=args)
        self.golden: TestsGoldenParam = TestsGoldenParam(args=args)
        self.utest: TestsFilterParam = TestsFilterParam(argv=args.utest, opt="ENABLE_UTEST")
        self.utest_module: TestsFilterParam = TestsFilterParam(argv=args.utest_module, opt="ENABLE_UTEST_MODULE")
        self.stest_exec: STestExecuteParam = STestExecuteParam(args=args, enable_binary_cache=False)
        self.stest: TestsFilterParam = TestsFilterParam(argv=args.stest, opt="ENABLE_STEST")
        self.stest_group: TestsFilterParam = TestsFilterParam(argv=args.stest_group, opt="ENABLE_STEST_GROUP")
        self.stest_distributed: TestsFilterParam = TestsFilterParam(argv=args.stest_distributed,
                                                                    opt="ENABLE_STEST_DISTRIBUTED")
        self.models: TestsFilterParam = TestsFilterParam(argv=args.models)
        self.example: TestsFilterParam = TestsFilterParam(argv=args.example)

    def __str__(self) -> str:
        """返回测试参数的字符串表示

        :return: 格式化的测试参数字符串
        :rtype: str
        """
        if not self.enable:
            return ""
        desc = f"\nTests"
        desc += f"{self.exec}"
        if self.utest.enable:
            desc += f"\n    Utest"
            desc += f"\n                     Enable : {self.utest.enable}"
            desc += f"\n                     Filter : {self.utest.filter_str}"
        if self.stest.enable or self.stest_distributed.enable:
            desc += f"\n    Golden"
            desc += f"\n                      Clean : {self.golden.clean}"
            desc += f"\n                       Path : {self.golden.path}"
            desc += f"\n    Stest Execute"
            desc += f"\n                     Device : {self.stest_exec.auto_execute_device_id}"
            desc += f"\n                   DumpJson : {self.stest_exec.dump_json}"
            desc += f"\n         Interpreter Config : {self.stest_exec.interpreter_config}"
            desc += f"\n        Enable Binary Cache : {self.stest_exec.enable_binary_cache}"
        if self.stest.enable:
            desc += f"\n    Stest"
            desc += f"\n                     Enable : {self.stest.enable}"
            desc += f"\n                     Filter : {self.stest.filter_str}"
            desc += f"\n                     Group  : {self.stest_group.filter_str}"
        if self.stest_distributed.enable:
            desc += f"\n    Stest Distributed"
            desc += f"\n                     Enable : {self.stest_distributed.enable}"
            desc += f"\n                     Filter : {self.stest_distributed.filter_str}"
        if self.models.enable:
            desc += f"\n    Models"
            desc += f"\n                     Enable : {self.models.enable}"
            desc += f"\n                     Filter : {self.models.filter_str}"
        if self.example.enable:
            desc += f"\n    Example"
            desc += f"\n                     Enable : {self.example.enable}"
            desc += f"\n                     Filter : {self.example.filter_str}"
        return desc

    @property
    def enable(self) -> bool:
        """判断是否启用任意测试

        :return: 如果启用了任意类型的测试, 返回 True
        :rtype: bool
        """
        tests_enable = self.utest.enable or self.stest.enable or self.stest_distributed.enable
        return tests_enable or self.example.enable or self.models.enable

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册所有测试相关的命令行参数

        向参数解析器注册测试执行, Golden 测试, 过滤选项等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息 (子命令解析器)
        :type ext: Optional[Any]
        """
        TestsExecuteParam.reg_args(parser=parser)
        TestsGoldenParam.reg_args(parser=parser)
        TestsFilterParam.reg_args(parser=parser, ext="utest")
        TestsFilterParam.reg_args(parser=parser, ext="utest_module")
        STestExecuteParam.reg_args(parser=parser)
        TestsFilterParam.reg_args(parser=parser, ext="stest")
        TestsFilterParam.reg_args(parser=parser, ext="stest_group")
        TestsFilterParam.reg_args(parser=parser, ext="stest_distributed")
        TestsFilterParam.reg_args(parser=parser, ext="models")
        TestsFilterParam.reg_args(parser=parser, ext="example")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self.utest.get_cfg_cmd()
        cmd += self.stest.get_cfg_cmd()
        cmd += self.stest_distributed.get_cfg_cmd()
        cmd += self.models.get_cfg_cmd()
        cmd += self.example.get_cfg_cmd()
        if self.enable:
            cmd += self.exec.get_cfg_cmd()
            if self.utest.enable:
                cmd += self.utest_module.get_cfg_cmd()
            if self.stest.enable or self.stest_distributed.enable:
                cmd += self.golden.get_cfg_cmd()
                cmd += self.stest_exec.get_cfg_cmd()
            if self.stest.enable:
                cmd += self.stest_group.get_cfg_cmd()
        return cmd


class BuildCtrl(CMakeParam):
    """构建过程控制类

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行. 是整个构建流程的入口和控制器, 负责协调整个构建过程.
    """
    _PYTHONPATH: str = "PYTHONPATH"

    def __init__(self, args):
        """初始化 BuildCtrl 实例

        从命令行参数中解析并初始化所有构建相关的配置.

        :param args: 命令行参数解析结果
        """
        self.clean: bool = args.clean  # 强制清理 Build-Tree 及 Install-Tree 标记
        self.origin_timeout: Optional[int] = args.timeout if args.timeout and args.timeout > 0 else None  # 超时时长
        self.remain_timeout: Optional[int] = self.origin_timeout
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "build_out")
        self.feature: FeatureParam = FeatureParam(args=args)
        self.build: BuildParam = BuildParam(args=args)
        self.tests: TestsParam = TestsParam(args=args)
        self.third_party_path: Optional[Path] = Path(args.third_party_path).resolve() if args.third_party_path else None
        self.verbose: bool = args.verbose
        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError(f"Can't find cmake")
        # 表示 pip 版本是否支持传递 --config-setting 这种 pep 标准参数传递方式
        self.pip_dependence_desc: Dict[str, str] = {"pip": ">=22.1"}
        self.pip_support_config_setting = self.check_pip_dependencies(deps=self.pip_dependence_desc,
                                                                      raise_err=False, log_err=False)

    def __str__(self) -> str:
        """返回构建控制参数的字符串表示

        :return: 格式化的构建控制参数字符串
        :rtype: str
        """
        py3_ver = sys.version_info
        pip_ver = metadata.version("pip")
        desc = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3                 : {sys.executable} ({py3_ver.major}.{py3_ver.minor}.{py3_ver.micro})"
        desc += f"\n    pip3                    : {pip_ver}"
        desc += f"\nPath"
        desc += f"\n    Source  Dir             : {self.src_root}"
        desc += f"\n    Build   Dir             : {self.build_root}"
        desc += f"\n    Install Dir             : {self.install_root}"
        desc += f"\n    3rd     Dir             : {self.third_party_path}"
        desc += f"\nFlag"
        desc += f"\n    Clean                   : {self.clean}"
        desc += f"\n    Verbose                 : {self.verbose}"
        desc += f"\nOthers"
        desc += f"\n    Timeout                 : {self.origin_timeout}"
        desc += f"{self.feature}"
        desc += f"{self.build}"
        desc += f"{self.tests}"
        desc += f"\n"
        return desc

    @staticmethod
    def which_cmake() -> Optional[Path]:
        """查找系统级 CMake 可执行文件路径

        实现本函数是为了排除 cmake pip 包的干扰, 否则在 Python 中直接调用 cmake 会调用到 cmake pip 包.
        通过遍历 PATH 环境变量中的目录, 查找 ELF 格式的 CMake 可执行文件.

        :return: 系统级 cmake 可执行文件绝对路径, 找不到则返回 None
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

    @staticmethod
    def find_match_whl(name: str, path: Path) -> Optional[Path]:
        """
        在指定路径下, 查找对应匹配的 whl 包文件

        :param name: 包名
        :type name: str
        :param path: 指定路径
        :type path: Path
        :return: 指定路径
        :rtype: Path | None
        """
        cpp_desc = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pattern = f"{name}-*-{cpp_desc}-{cpp_desc}-*.whl"
        whl_glob = path.glob(pattern=pattern)
        whl_files = [Path(f) for f in whl_glob]
        whl_file = whl_files[0] if whl_files else None
        if whl_file:
            logging.info("Success find match %s from %s", whl_file, path)
        else:
            logging.error("Failed to find match %s whl from %s, pattern=%s", name, path, pattern)
        return whl_file

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("--timeout", nargs="?", type=int, default=None,
                            help="Total timeout.")
        parser.add_argument("--cann_3rd_lib_path", "--third_party_path",
                            nargs="?", type=str, default="", dest="third_party_path",
                            help="Specify 3rd Libraries Path")
        parser.add_argument("--verbose", action="store_true", default=False,
                            help="verbose, enable verbose output.")

    @classmethod
    def check_pip_dependencies(cls, deps: Dict[str, str], raise_err: bool = False, log_err: bool = True) -> bool:
        info_lst = []
        for pkg, ver in deps.items():
            info = cls._check_pip_pkg(pkg=pkg, ver=ver)
            info_lst.extend(info)
        if info_lst:
            if log_err:
                logging.error("%s", info_lst)
                install_cmd = " ".join([f'{pkg}{deps[pkg]}' for pkg in deps])
                logging.error(f"Please install the missing dependencies first [{install_cmd}]")
            if raise_err:
                raise RuntimeError("\n".join(info_lst))
            return False
        return True

    @classmethod
    def main(cls):
        ts = datetime.now(tz=timezone.utc)
        try:
            cls._main()
        except KeyboardInterrupt as e:
            logging.error("Operation cancelled by user")
            raise e
        except subprocess.TimeoutExpired as e:
            logging.error("Operation timeout, %s", e)
            raise e
        # 计算总耗时
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("Build[CI] Finish, Duration %s secs.", duration)

    @classmethod
    def _check_pip_pkg(cls, pkg: str, ver: str) -> List[str]:
        info_lst = []
        requirement_str = f"{pkg}{ver}"
        try:
            req = requirements.Requirement(requirement_str)
            try:
                installed_version = metadata.version(pkg)
                if ver and not req.specifier.contains(installed_version, prereleases=True):
                    info_lst.append(f"{pkg}: version {installed_version} not satisfy {ver}")
            except metadata.PackageNotFoundError:
                info_lst.append(f"package {pkg} has not been installed")
        except Exception as e:
            info_lst.append(f"package {pkg} check fail {e}")
        return info_lst

    @classmethod
    def _main(cls):
        """主处理流程
        """
        parser = argparse.ArgumentParser(description=f"PyPTO Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # 子命令
        # 参数注册
        FeatureParam.reg_args(parser=parser)
        BuildParam.reg_args(parser=parser)
        TestsParam.reg_args(parser=parser, ext=sub_parser)
        BuildCtrl.reg_args(parser=parser)

        # 参数处理
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        # 流程处理
        if ctrl.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        # 区分 python3 前端和 cpp 前端
        logging.info("%s", ctrl)
        if ctrl.feature.frontend_type_python3:
            logging.info("Front-end(python3), start process")
            ctrl.py_clean()
            ctrl.py_build()
            ctrl.py_tests()
        else:
            logging.info("Front-end(cpp), start process with CMake")
            if 'func' in args:
                args.func(args=args, ctrl=ctrl)
            ctrl.cmake_clean()
            ctrl.cmake_configure()
            ctrl.cmake_build()

    def run_build_cmd(self, cmd: str, update_env: Optional[Dict[str, str]] = None,
                      check: bool = True, pg_desc: str = "CMake") -> Tuple[subprocess.CompletedProcess, str]:
        """执行具体 build 命令行

        因以下原因, 设置本函数, 而非调用原生 subprocess.run
            1. 支持多 target 构建, 各 target 构建时长共享公共 timeout 配置;
            2. UTest/STest 并行执行场景下, 执行时进程调用关系为:
                   build_ci.py(主进程) -> 进程1(CMake) -> 进程2(CMake Generator, make/ninja) -> 进程3(Python)-> 进程4(exe)
               此时若 进程1 超时, 需要触发其子/孙进程感知, 进而结束

        本函数内支持 timeout 重计算, 仅执行成功时会进行重计算

        :param cmd: Build 命令行
        :param update_env: 环境变量(额外更新内容)
        :param check: 检查返回值
        :param pg_desc: Process Group Desc, 进程组描述
        """

        def _stop_pg(_msg: str, _p: subprocess.Popen):
            """通过 SIGINT 信号通知所有子/孙进程结束, python 并行脚本内会捕获该信号进行结算处理
            """
            _pgid = os.getpgid(_p.pid)
            logging.info("%s. Send terminate event to %s[%s]", _msg, pg_desc, _pgid)
            os.killpg(_pgid, signal.SIGINT)

        ts = datetime.now(tz=timezone.utc)
        stdout = None
        stderr = None
        env = os.environ.copy()
        env.update(update_env if update_env else {})
        with subprocess.Popen(shlex.split(cmd), env=env, text=True, encoding='utf-8',
                              start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.remain_timeout)
            except subprocess.TimeoutExpired as e:
                _stop_pg(_msg=f"Timeout({self.remain_timeout})", _p=process)
                raise e
            except KeyboardInterrupt as e:
                _stop_pg(_msg="KeyboardInterrupt", _p=process)
                raise e
            except Exception as e:
                process.kill()
                raise e
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if check and ret_code:
                raise subprocess.CalledProcessError(ret_code, process.args, output=stdout, stderr=stderr)
        # 超时时长更新
        duration = self._duration(ts=ts)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr), duration

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """生成 CMake Configure 命令

        BuildCtrl 类不直接生成 CMake 配置命令, 返回空字符串.

        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        :return: 空字符串
        :rtype: str
        """
        return ""

    def get_cfg_update_env(self) -> Dict[str, str]:
        """获取 CMake Configure 阶段的环境变量

        根据配置生成需要传递给 CMake Configure 的环境变量.

        :return: 环境变量字典
        :rtype: Dict[str, str]
        """
        env = {}
        if self.third_party_path:
            env.update({"PYPTO_THIRD_PARTY_PATH": self.third_party_path})
        return env

    def get_cmake_build_update_env(self) -> Dict[str, str]:
        """获取 CMake Build 阶段的环境变量

        根据配置生成需要传递给 CMake Build 和测试执行的环境变量.

        :return: 环境变量字典
        :rtype: Dict[str, str]
        """
        env = {}
        if self.build.job_num:
            env["PYPTO_TESTS_PARALLEL_NUM"] = str(self.build.job_num)
        if self.build.gcov_incr:
            env["PYPTO_BUILD_GCOV_INCREMENT"] = "True"
        # Tests exec
        tests_exec = self.tests.exec
        if tests_exec.auto_execute:
            case_timeout = tests_exec.case_execute_timeout
            if case_timeout and case_timeout > 0:
                env["PYPTO_TESTS_CASE_EXECUTE_TIMEOUT"] = str(case_timeout)
            rank_size = tests_exec.case_execute_cpu_rank_size
            if rank_size and rank_size > 0:
                env["PYPTO_TESTS_CASE_EXECUTE_CPU_RANK_SIZE"] = str(rank_size)
            # Dump case duration json
            duration_json = tests_exec.dump_case_duration_json
            if duration_json:
                env["PYPTO_TESTS_DUMP_CASE_DURATION_JSON"] = str(duration_json)
            max_num = tests_exec.dump_case_duration_max_num
            if max_num and max_num > 0:
                env["PYPTO_TESTS_DUMP_CASE_DURATION_MAX_NUM"] = str(max_num)
            min_sec = tests_exec.dump_case_duration_min_secends
            if min_sec and min_sec > 0:
                env["PYPTO_TESTS_DUMP_CASE_DURATION_MIN_SECONDS"] = str(min_sec)
        return env

    def pip_install(self, whl: Path, dest: Optional[Path] = None, opt: str = "",
                    update_env: Optional[Dict[str, str]] = None):
        """安装指定的 whl 包

        使用 pip 命令安装指定的 whl 包, 支持自定义安装路径和参数.

        :param whl: whl 包文件路径
        :type whl: Path
        :param dest: 安装路径, 未指定时使用默认路径
        :type dest: Optional[Path]
        :param opt: 额外安装参数
        :type opt: str
        :param update_env: 环境变量 (额外更新内容)
        :type update_env: Optional[Dict[str, str]]
        """
        edit_str = "-e " if self.feature.whl_editable else ""
        cmd = f"{sys.executable} -m pip install {edit_str}" + f"{whl} {opt}" + (" -vvv " if self.verbose else "")
        cmd += f" --target={dest}" if dest else ""
        logging.info("Install %s, Cmd: %s, Timeout: %s", whl, cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pip")
        logging.info("Install %s%s success, %s", whl, f" to {dest}" if dest else "", duration)

    def pip_uninstall(self, name: str, path: Optional[Path] = None):
        """卸载指定的 whl 包

        根据是否指定安装路径, 选择使用 pip 卸载或直接删除文件.

        :param name: 包名
        :type name: str
        :param path: 指定安装路径, 如果指定则直接删除对应路径下的文件
        :type path: Optional[Path]
        """
        if path:
            del_lst = [Path(f) for f in path.glob(pattern=f"{name}-*.dist-info")]
            pkg_dir = Path(path, name)
            if pkg_dir.exists() and pkg_dir.is_dir():
                del_lst.append(pkg_dir)
            for p in del_lst:
                shutil.rmtree(p)
        else:
            cmd = f"{sys.executable} -m pip uninstall -v -y {name}"
            logging.info("Uninstall %s package, Cmd: %s, Timeout: %s", name, cmd, self.remain_timeout)
            _, _ = self.run_build_cmd(cmd=cmd, pg_desc="pip")
        logging.info("Uninstall %s package%s success", name, f" from {path}" if path else "")

    def cmake_clean(self):
        """清理 CMake 构建中间结果

        清理内容包括构建树, 安装树全部内容以及 ast 数据缓存. 仅在 clean 标记为 True 时执行.
        """
        if self.clean:
            if self.build_root.exists():
                logging.info("Clean Build-Tree(%s)", self.build_root)
                shutil.rmtree(self.build_root)
            if self.install_root.exists():
                logging.info("Clean Install-Tree(%s)", self.install_root)
                shutil.rmtree(self.install_root)
            home_dir = os.environ.get('HOME')
            astdata_folder = os.path.join(home_dir, 'ast_data')
            if os.path.exists(astdata_folder):
                logging.info("Clean ast data cache folder(%s)", astdata_folder)
                shutil.rmtree(astdata_folder)

    def py_clean(self):
        """清理 Python 前端构建的中间结果

        清理包括 CMake 构建目录, Python 缓存文件, 输出目录等. 仅在 clean 标记为 True 时执行额外清理.
        """
        self.cmake_clean()
        if not self.clean:
            return
        pkg_src = Path(self.src_root, "python/pypto")
        path_lst = [
            Path(Path.cwd(), "output"),
            Path(Path.cwd(), "kernel_meta"),
            Path(self.src_root, "python/pypto.egg-info"),
            Path(pkg_src, "__pycache__"),
            Path(pkg_src, "op/__pycache__"),
            Path(pkg_src, "lib"),  # edit 模式
        ]
        so_glob = pkg_src.glob(pattern=f"*.so")
        so_path = [Path(p) for p in so_glob]
        path_lst.extend(so_path)
        for cache_dir in path_lst:
            if not cache_dir.exists():
                continue
            logging.info("Clean Cache/Output Path(%s)", cache_dir)
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            else:
                os.remove(cache_dir)

    def cmake_configure(self):
        """执行 CMake Configure 阶段流程

        生成 CMake 构建配置, 包括设置生成器, Python 解释器路径, 编译选项等.
        """
        # 基本配置, 当前 CMake 中有调用 python3 的情况, 传入 python3 解释器, 保证所使用的 python3 版本一致
        cmd = f"{self.cmake} -S {self.src_root} -B {self.build_root}"
        cmd += f" -G {self.build.generator}" if self.build.generator else ""
        cmd += f" -DPython3_EXECUTABLE={sys.executable}"
        cmd += self.feature.get_cfg_cmd()
        cmd += self.build.get_cfg_cmd()
        cmd += self.tests.get_cfg_cmd()
        # 执行
        update_env = self.get_cfg_update_env()
        update_env["CCACHE_BASEDIR"] = str(self.src_root)
        logging.info("CMake Configure, Cmd: %s, Timeout: %s", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env)
        logging.info("CMake Configure success, %s", duration)

    def cmake_build(self):
        """执行 CMake Build 阶段流程

        根据 BuildParam 配置执行实际的编译过程, 支持多 target 构建.
        """
        update_env = self.get_cmake_build_update_env()
        update_env["CCACHE_BASEDIR"] = str(self.src_root)
        cmd_list = self.build.get_build_cmd_lst(cmake=self.cmake, binary_path=self.build_root)
        for i, c in enumerate(cmd_list, start=1):
            c += " --verbose" if self.verbose else ""
            logging.info("CMake Build(%s/%s), Cmd: %s, Timeout: %s", i, len(cmd_list), c, self.remain_timeout)
            try:
                _, duration = self.run_build_cmd(cmd=c, update_env=update_env)
            except subprocess.CalledProcessError as e:
                logging.info(f"CMake Build(%s/%s) failed, ERROR CODE: %s", i, len(cmd_list), e.returncode)
                raise e
            logging.info("CMake Build(%s/%s) success, %s", i, len(cmd_list), duration)

    def py_build(self):
        """whl 包编译处理

        支持两种编译模式:
            1. 正式编译: 调用 build 库触发 setuptools(bdist_wheel 命令) 进而触发 CMake 完成编译
            2. pip 编译: 调用 pip install 命令触发 setuptools(editable_wheel 命令) 进而触发 CMake 完成编译
               pip 编译有两种模式:
               - 常规安装: 适用于生产环境或代码稳定后使用, 安装后对源码的修改不会反映到已安装的包中
               - 可编辑安装: 便于开发调试, 在 site-packages 中创建指向本地的链接,
                 对 Python 源码的修改会即时生效, 无需重新安装
        """
        update_env = self.get_cfg_update_env()
        if self._use_pip_install_mode() or self.feature.whl_editable:
            opt = f" --no-compile --no-deps"
            opt += f" --no-build-isolation" if not self.feature.whl_isolation else ""

            cmd_config_setting, env_config_setting = self._get_setuptools_build_ext_config_setting()
            if self.feature.whl_editable:
                update_env["PYPTO_BUILD_EXT_ARGS"] = env_config_setting
            else:
                if self.pip_support_config_setting:
                    opt += f" {cmd_config_setting}" if cmd_config_setting else ""
                else:
                    # pip 低版本无 --config-setting 参数, 此时以环境变量方式传入
                    update_env["PYPTO_BUILD_EXT_ARGS"] = env_config_setting

            # 重装 whl 包
            dist = self._get_pip_install_dist()
            self.pip_uninstall(name=self.feature.whl_name, path=dist)
            self.pip_install(whl=self.src_root, dest=dist, opt=opt, update_env=update_env)
        else:
            # 检查 build 包版本是否符合要求, 之所以将其放在此处检查, 是因为 pyproject.toml 中 build-system.requires 的检查功能
            # 就是 build 包实现的, 所以将其写在 pyproject.toml 中并无法提前检查
            self.check_pip_dependencies(deps={"build": ">=1.0.3"}, raise_err=True, log_err=True)
            cmd = f"{sys.executable} -m build --outdir={self.install_root}"
            cmd += f" --no-isolation" if not self.feature.whl_isolation else ""
            cmd += f" {self._get_setuptools_bdist_wheel_config_setting()}"
            logging.info("Build whl, Cmd: %s, Timeout: %s", cmd, self.remain_timeout)
            _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="build")
            logging.info("Build whl success, %s", duration)

    def py_tests(self):
        """执行 Python 前端测试

        包括单元测试 (UTest) , 系统测试 (STest) , 模型测试 (Models) 和示例测试 (Examples) .
        如果未使用 pip 安装模式, 会先卸载并重新安装 whl 包.
        """
        tests_enable = self.tests.utest.enable or self.tests.stest.enable
        if not tests_enable and not self.tests.example.enable and not self.tests.models.enable:
            return
        dist = self._get_pip_install_dist()
        if not self._use_pip_install_mode():
            # 此时需查找重装对应 whl 包
            self.pip_uninstall(name=self.feature.whl_name, path=dist)  # 卸载 whl 包
            whl = self.find_match_whl(name=self.feature.whl_name, path=dist)  # 查找 whl 包
            if not whl:
                raise RuntimeError(f"Can't find {self.feature.whl_name} whl file from {dist}")
            self.pip_install(whl=whl, dest=dist, opt="--no-compile --no-deps")  # 安装 whl 包

        # 执行用例, UTest
        # 在 Python 3.12 中, pytest-xdist 通过 os.fork() 创建子进程时会产生 DeprecationWarning.
        # 使用 -W ignore::DeprecationWarning 参数来忽略该警告.
        if self.build.job_num is not None and self.build.job_num > 0:
            n_workers = str(self.build.job_num)
        else:
            n_workers = "auto"
        self.py_tests_run_pytest(dist=dist, params=[(self.tests.utest, "python/tests/ut")],
                                 ext=f"-n {n_workers} -W ignore::DeprecationWarning")

        # 执行用例, Models/STest, 支持混合执行
        dev_lst = [int(d) for d in self.tests.stest_exec.auto_execute_device_id.split(":")]
        dev_ext = " ".join(f"{d}" for d in dev_lst)
        ext_str = f"-n {len(dev_lst)} --device {dev_ext}"
        self.py_tests_run_pytest(dist=dist, params=[(self.tests.models, "models"),
                                                    (self.tests.stest, "python/tests/st")],
                                 ext=ext_str)

        # 执行用例, Examples
        dev_ext_comma = ",".join(f"{d}" for d in dev_lst)
        self.py_run_examples(dist=dist, tests=self.tests.example,
                             def_filter=str(Path(self.src_root, "examples")),
                             dev_ext_comma=dev_ext_comma, n_workers=n_workers)

    def py_tests_run_pytest(self, dist: Optional[Path], params: List[Tuple[TestsFilterParam, str]], ext: str = ""):
        """调用 pytest 执行测试用例

        支持多路径下用例混跑, 可以根据配置并行执行.

        :param dist: 二进制分发包安装路径
        :type dist: Optional[Path]
        :param params: 参数列表, 支持多路径下用例混跑, 每个元素为 (TestsFilterParam, 测试路径)
        :type params: List[Tuple[TestsFilterParam, str]]
        :param ext: 扩展命令参数
        :type ext: str
        """
        # filter 处理
        filter_str = ""
        for cur_tests, cur_filter_str in params:
            cur_filter_str = cur_tests.get_filter_str(def_filter=cur_filter_str)
            if cur_filter_str:
                filter_str += f" {cur_filter_str}"
        if not filter_str:
            return
        # 执行 pytest
        self._py_tests_run_pytest(dist=dist, filter_str=filter_str, ext=ext)

    def py_run_examples(self, dist: Optional[Path], tests: TestsFilterParam, def_filter: str,
                        dev_ext_comma: str = "0", n_workers: str = "auto"):
        """运行示例测试用例

        根据 backend_type 决定执行模式 (NPU 或 SIM) , 支持设备分配和超时控制.

        :param dist: 二进制分发包安装路径
        :type dist: Optional[Path]
        :param tests: 测试过滤参数
        :type tests: TestsFilterParam
        :param def_filter: 默认过滤条件
        :type def_filter: str
        :param dev_ext_comma: 设备 ID 列表 (逗号分隔)
        :type dev_ext_comma: str
        :param n_workers: 并行工作数
        :type n_workers: str
        """
        if not tests.enable:
            return
        if not self.tests.exec.auto_execute:
            return
        # filter 处理
        filter_str = tests.get_filter_str(def_filter=def_filter).replace(',', ' ')

        # 根据 backend_type 决定执行模式
        origin_env = os.environ.copy()
        update_env = {}
        if dist:
            ori_env_python_path = origin_env.get(self._PYTHONPATH, "")
            act_env_python_path = f"{dist}:{ori_env_python_path}" if ori_env_python_path else f"{dist}"
            update_env.update({self._PYTHONPATH: act_env_python_path})

        # 获取 case_timeout 参数
        case_timeout = self.tests.exec.case_execute_timeout
        timeout_arg = f" --timeout {case_timeout}" if case_timeout and case_timeout > 0 else ""

        if self.feature.backend_type == "npu":
            # NPU 模式
            cmd = f"{sys.executable} examples/validate_examples.py -t {filter_str} -d {dev_ext_comma}{timeout_arg}"
            logging.info("examples --run_mode npu, Cmd: %s", cmd)
            ret, duration = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
            ret.check_returncode()
            logging.info("examples --run_mode npu, Cmd: %s, Duration %s sec", cmd, duration)
        else:
            # SIM 模式
            n_workers_val = int(n_workers) if n_workers != "auto" else 16
            cmd = f"{sys.executable} examples/validate_examples.py -t {filter_str} \
                  --run_mode sim -w {n_workers_val}{timeout_arg} --no-serial-fallback"
            logging.info("examples --run_mode sim, Cmd: %s", cmd)
            ret, duration = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
            ret.check_returncode()
            logging.info("examples --run_mode sim, Cmd: %s, Duration %s sec", cmd, duration)

    def _py_tests_run_pytest(self, dist: Optional[Path], filter_str: str, ext: str = ""):
        if not self.tests.exec.auto_execute:
            return
        # filter 处理
        filter_str = filter_str.replace(',', ' ')
        # cmd 拼接
        cmd = f"{sys.executable} -m pytest {filter_str} -v --durations=0 -s --capture=no"
        cmd += f" --rootdir={self.src_root} {ext} --forked"
        if self.check_pip_dependencies(deps={"pytest-xdist": ">=3.8.0"}, raise_err=False, log_err=False):
            cmd += " --dist=loadscope"
            cmd += " --no-loadscope-reorder"
        # cmd 执行
        origin_env = os.environ.copy()
        update_env = {}
        if dist:
            ori_env_python_path = origin_env.get(self._PYTHONPATH, "")
            act_env_python_path = f"{dist}:{ori_env_python_path}" if ori_env_python_path else f"{dist}"
            update_env.update({self._PYTHONPATH: act_env_python_path})
        logging.info("pytest run, Cmd: %s, Timeout: %s", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pytest")
        logging.info("pytest run success, %s", duration)

    def _tests_enable(self) -> bool:
        return self.tests.utest.enable or self.tests.stest.enable

    def _use_pip_install_mode(self) -> bool:
        return self.tests.utest.enable or self.tests.stest.enable

    def _get_pip_install_dist(self) -> Optional[Path]:
        # pip install -e 场景需直接安装到 site-packages 默认路径(与指定 --target 参数逻辑冲突), 其他场景安装到自定义目录
        return None if self._use_pip_install_mode() and self.feature.whl_editable else self.install_root

    def _get_setuptools_build_ext_config_setting(self) -> Tuple[str, str]:
        cmake_args = f"{self.build.get_cfg_cmd(ext=False)}"
        env_setting = ""
        env_setting += f" --cmake-generator={self.build.generator}" if self.build.generator else ""
        env_setting += f" --cmake-build-type={self.build.build_type}" if self.build.build_type else ""
        env_setting += f" --cmake-options=\"{cmake_args}\"" if cmake_args else ""
        env_setting += f" --cmake-verbose" if self.verbose else ""
        cmd_setting = ""
        if env_setting:
            cmd_setting = f" --config-setting=--build-option='build_ext {env_setting}'"
        return cmd_setting, env_setting

    def _get_setuptools_bdist_wheel_config_setting(self) -> str:
        cmd = f" bdist_wheel --plat-name={self.feature.whl_plat_name}" if self.feature.whl_plat_name else ""
        cmd += f" build --build-base={self.build_root.name}"
        cmd += f" --parallel={self.build.job_num}" if self.build.job_num else ""
        _, ext = self._get_setuptools_build_ext_config_setting()
        if ext:
            cmd += f" build_ext {ext}"
        cmd = f" --config-setting=--build-option='{cmd}'"
        return cmd

    def _duration(self, ts: datetime) -> str:
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        duration_str = f"Duration {duration} secs"
        if self.remain_timeout:
            self.remain_timeout = max(self.remain_timeout - duration, 0)
            duration_str += f" Remain {self.remain_timeout} secs"
        return duration_str


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    BuildCtrl.main()

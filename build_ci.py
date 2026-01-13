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
"""构建总入口.
"""
import abc
import argparse
import dataclasses
import json
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


if str(Path(Path(__file__).parent, "tools")) not in sys.path:
    sys.path.append(str(Path(Path(__file__).parent, "tools")))

import work_flow as wf


class CMakeParam(abc.ABC):
    """需要向 CMake 传入 Option 的参数
    """

    @staticmethod
    def get_system_processor() -> str:
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
        pass

    @classmethod
    def _cfg_require(cls, opt: str, ctr: bool = True, tv: str = "ON", fv: str = "OFF") -> str:
        """获取 CMake Config 阶段的必选 Option 配置

        :param opt: CMake 选项, 会最终体现到 CMake -D传入的参数中
        :param ctr: 控制变量
        :param tv: 控制变量为 True 时, 设置的值
        :param fv: 控制变量为 False 时, 设置的值
        :return: 设置的值
        """
        return f" -D{opt}=" + (tv if ctr else fv)

    @classmethod
    def _cfg_optional(cls, opt: str, ctr: bool, v: str):
        """获取 CMake Config 阶段的可选 Option 配置

        :param opt: CMake 选项, 会最终体现到 CMake -D传入的参数中
        :param ctr: 控制变量
        :param v: 控制变量为 True 时, 设置的值
        """
        return (f" -D{opt}=" + v) if ctr else ""

    @abc.abstractmethod
    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        pass


@dataclasses.dataclass
class FeatureParam(CMakeParam):
    """特性控制相关参数
    """
    whl_name: str = "pypto"
    frontend_type: Optional[str] = None  # 前端类型, 支持 python3, cpp
    backend_type: Optional[str] = None  # 后端类型, 支持 npu, cost_model
    whl_plat_name: Optional[str] = None  # python3 whl 包 plat-name
    whl_isolation: bool = False  # 以 isolation 模式编译 whl 包
    whl_editable: bool = False  # 以 editable 模式编译 whl 包

    def __init__(self, args):
        self.frontend_type = "python3" if args.frontend is None else args.frontend
        self.backend_type = "npu" if args.backend is None else args.backend
        if not os.environ.get("ASCEND_HOME_PATH") and self.backend_type in ["npu"]:
            logging.warning("Environment variable ASCEND_HOME_PATH is unset/empty, falling back to cost_model backend.")
            self.backend_type = "cost_model"
        self.whl_plat_name = f"{args.plat_name}_{CMakeParam.get_system_processor()}" if args.plat_name else ""
        self.whl_isolation = args.isolation
        self.whl_editable = args.editable

    def __str__(self):
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
        return self.frontend_type in ["python", "python3"]

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
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
        cmd = ""
        cmd += self._cfg_require(opt="ENABLE_FEATURE_PYTHON_FRONT_END", ctr=self.frontend_type_python3)
        cmd += self._cfg_require(opt="BUILD_WITH_CANN", ctr=self.backend_type in ["npu"])
        return cmd


@dataclasses.dataclass
class BuildParam(CMakeParam):
    """构建相关参数
    """
    clean: bool = False  # 强制清理 Build-Tree 及 Install-Tree 标记
    timeout: Optional[int] = None  # 构建超时时长
    # Configure
    generator: Optional[str] = None  # Generator
    build_type: Optional[str] = None  # 构建类型
    asan: bool = False  # 使能 AddressSanitizer
    ubsan: bool = False  # 使能 UndefinedBehaviorSanitizer
    gcov: bool = False  # 使能 GNU Coverage
    clang_install_path: Optional[Path] = None  # Clang 安装位置
    compile_dependency_check: bool = False  # 使能编译依赖关系检查
    # Build
    targets: Optional[List[str]] = None  # 编译目标
    job_num: Optional[int] = None  # 编译阶段使用核数

    def __init__(self, args):
        self.targets = args.targets
        self.job_num = self._get_job_num(job_num=args.job_num, generator=args.generator)
        self.clean = args.clean
        self.timeout = None if args.timeout == 0 else args.timeout
        self.generator = self._get_generator(generator=args.generator)
        self.build_type = args.build_type
        self.asan = args.asan
        self.ubsan = args.ubsan
        self.gcov = args.gcov
        self.clang_install_path = self._get_clang_install_path(opt=args.clang)
        self.compile_dependency_check = args.compile_dependency_check

    def __str__(self):
        desc = f"\nBuild"
        desc += f"\n    Clean                   : {self.clean}"
        desc += f"\n    Timeout                 : {self.timeout}"
        desc += f"\n    CMake"
        desc += f"\n        Configure"
        desc += f"\n                  Generator : {self.generator}"
        desc += f"\n                  BuildType : {self.build_type}"
        desc += f"\n                       ASan : {self.asan}"
        desc += f"\n                      UbSan : {self.ubsan}"
        desc += f"\n                       GCov : {self.gcov}"
        desc += f"\n           ClangInstallPath : {self.clang_install_path}"
        desc += f"\n            CompileDepCheck : {self.compile_dependency_check}"
        desc += f"\n        Build"
        desc += f"\n                    Targets : {self.targets}"
        desc += f"\n                    Job Num : {self.job_num}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("--timeout", nargs="?", type=int, default=0,
                            help="build task timeout.")
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
        # 获取 Clang 安装目录
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
        def_job_num = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 48)  # 48 为缺省最大核数
        def_job_num = None if generator and generator.lower() in ["ninja", ] else def_job_num  # ninja 由其自身决定缺省核数
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_generator(generator: Optional[str]) -> Optional[str]:
        return f"\"{generator}\"" if generator else generator

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        inc_build_type = bool(ext) if ext is not None else True
        cmd = (self._cfg_require(opt="CMAKE_BUILD_TYPE", tv=self.build_type) if inc_build_type else "")
        cmd += self._cfg_require(opt="ENABLE_ASAN", ctr=self.asan)
        cmd += self._cfg_require(opt="ENABLE_UBSAN", ctr=self.ubsan)
        cmd += self._cfg_require(opt="ENABLE_GCOV", ctr=self.gcov)

        def _check_clang_toolchain(_opt: str, _b: str) -> Tuple[bool, str]:
            _p = Path(self.clang_install_path, _b)
            if _p.exists():
                return True, self._cfg_require(opt=_opt, tv=str(_p))
            logging.error("Clang Toolchain %s not exist.", _p)
            return False, ""

        def _gen_clang_cmd() -> Tuple[bool, str]:
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
    """Tests 执行相关参数
    """
    changed_file: Optional[Path] = None  # 修改文件路径
    auto_execute: bool = False  # 用例自动执行
    auto_execute_parallel: bool = False  # 用例并行执行

    def __init__(self, args):
        self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
        self.auto_execute = args.disable_auto_execute
        self.auto_execute_parallel = self.auto_execute and self.ci_model

    @property
    def ci_model(self) -> bool:
        return True if self.changed_file else False

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--changed_files", nargs="?", type=Path, default=None,
                            help="Specify the file of files changed, "
                                 "so that the corresponding test cases can be triggered incrementally.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self._cfg_require(opt="ENABLE_TESTS_EXECUTE", ctr=self.auto_execute)
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_PARALLEL", ctr=self.auto_execute_parallel)
        changed = self.changed_file and self.changed_file.exists() and self.changed_file.suffix.lower() == ".txt"
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_CHANGED_FILE", ctr=changed, tv=str(self.changed_file))
        return cmd


@dataclasses.dataclass
class TestsGoldenParam(CMakeParam):
    clean: bool = False  # 清理 Golden 标记
    path: Optional[Path] = None  # 指定 Golden 路径

    def __init__(self, args):
        self.clean = args.golden_clean
        if args.golden_path:
            # 传参且指定具体路径时, 使用指定路径, 否则具体缺省路径由 CMake 侧决定
            self.path = Path(args.golden_path).resolve()

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--golden_path", "--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific Tests golden path.", dest="golden_path")
        parser.add_argument("--golden_clean", "--golden_path_clean", "--stest_golden_path_clean",
                            action="store_true", default=False,
                            help="Clean Tests golden.", dest="golden_clean")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH_CLEAN", ctr=self.clean)
        cmd += self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH", ctr=bool(self.path), tv=str(self.path))
        return cmd


@dataclasses.dataclass
class TestsFilterParam(CMakeParam):
    cmake_option: str = ""
    enable: bool = False
    filter_str: Optional[str] = None

    def __init__(self, argv: Optional[str], opt: str = ""):
        self.cmake_option = opt
        if argv is None:
            self.enable, self.filter_str = True, "ON"  # 指定 对应参数, 但未指定内容
        elif argv == "":
            self.enable, self.filter_str = False, "OFF"  # 未指定 对应参数
        else:
            self.enable, self.filter_str = True, argv  # 指定 对应参数 且指定内容

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
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
        cmd = ""
        if self.cmake_option:
            cmd += self._cfg_require(opt=f"{self.cmake_option}", ctr=self.enable, tv=f"{self.filter_str}")
        return cmd

    def get_filter_str(self, def_filter: str) -> str:
        if not self.enable:
            return ""
        if self.filter_str not in ["ON"]:
            return self.filter_str
        if def_filter:
            return def_filter
        return self.filter_str


@dataclasses.dataclass
class STestExecuteParam(CMakeParam):
    auto_execute_device_id: str = ""
    interpreter_config: bool = False
    enable_binary_cache: bool = False
    dump_json: bool = False

    def __init__(self, args, enable_binary_cache: bool):
        devs = ["0"]
        if args.device is not None:
            devs = [str(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        self.auto_execute_device_id = ":".join(devs)
        self.dump_json = args.stest_dump_json
        self.interpreter_config = args.enable_interpreter_config
        self.enable_binary_cache = enable_binary_cache

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Device ID, default 0.")
        parser.add_argument("--stest_dump_json", action="store_true", default=False,
                            help="Dump json files.")
        parser.add_argument("--enable_interpreter_config", action="store_true", default=False,
                            help="enable STest Interpreter Config")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self._cfg_require(opt="ENABLE_STEST_EXECUTE_DEVICE_ID", tv=self.auto_execute_device_id)
        cmd += self._cfg_require(opt="ENABLE_STEST_DUMP_JSON", ctr=self.dump_json)
        cmd += self._cfg_require(opt="ENABLE_STEST_INTERPRETER_CONFIG", ctr=self.interpreter_config)
        cmd += self._cfg_require(opt="ENABLE_STEST_BINARY_CACHE", ctr=self.enable_binary_cache)
        return cmd


@dataclasses.dataclass
class STestToolsParam(CMakeParam):
    cases_csv_file: Optional[Path] = None
    intercept_flag: bool = False
    output_clean: bool = False

    prof_enable: bool = False
    prof_level: str = "l1"
    prof_warn_up_cnt: Optional[int] = None
    prof_try_cnt: Optional[int] = None
    prof_max_cnt: Optional[int] = None

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        # Tools
        parser_tools = parser.add_parser('tools', help="Tools")
        parser_tools.add_argument("--cases_csv_file", nargs=1, type=Path, default=None,
                                  help="Specify cases.csv")
        parser_tools.add_argument("--intercept", action="store_true", default=False,
                                  help="intercept, Intercept if have failed case result")
        parser_tools.add_argument("--tools_output_clean", action="store_true", default=False,
                                  help="clean, Specify clean flag, clean tools output dir")
        # Tools.Profiling
        sub_parser_prof = parser_tools.add_subparsers(dest="Tolls SubCommand")
        parser_prof = sub_parser_prof.add_parser('profiling', help="Profiling", aliases=['prof'])
        parser_prof.add_argument("-l", "--level", "--prof_level", dest="prof_level",
                                 nargs="?", type=str, default="l1", choices=["l1", "l2"],
                                 help="Specify profiling level")
        parser_prof.add_argument("-w", "--warn_up_cnt", "--prof_warn_up_cnt", dest="prof_warn_up_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling warn up cnt")
        parser_prof.add_argument("-t", "--try_cnt", "--prof_try_cnt", dest="prof_try_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling try cnt")
        parser_prof.add_argument("-m", "--max_cnt", "--prof_max_cnt", dest="prof_max_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling max cnt")
        parser_prof.set_defaults(func=SubCommandMgr.init_param_tools_profiling)

    def init_param(self, args):
        self.cases_csv_file = Path(args.cases_csv_file[0]).resolve() if args.cases_csv_file else None
        self.intercept_flag = args.intercept
        self.output_clean = args.tools_output_clean

    def init_param_profiling(self, args):
        self.prof_enable = True
        self.prof_level = args.prof_level
        self.prof_warn_up_cnt = args.prof_warn_up_cnt[0] if args.prof_warn_up_cnt else None
        self.prof_try_cnt = args.prof_try_cnt[0] if args.prof_try_cnt else None
        self.prof_max_cnt = args.prof_max_cnt[0] if args.prof_max_cnt else None

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF", ctr=self.prof_enable)

        # 当前 tools 下仅支持 prof 工具, 当其未使能时, 不需设置其他 option
        if not self.prof_enable:
            return cmd

        # 公共参数
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_CASE_FILE", ctr=bool(self.cases_csv_file),
                                 tv=str(self.cases_csv_file))
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_INTERCEPT", ctr=self.intercept_flag)
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_OUTPUT_CLEAN", ctr=self.output_clean)

        # Profiling 工具参数
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_LEVEL", tv=self.prof_level)
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT",
                                 ctr=self.prof_warn_up_cnt is not None,
                                 tv=f"{self.prof_warn_up_cnt}")
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_TRY_CNT",
                                 ctr=self.prof_try_cnt is not None,
                                 tv=f"{self.prof_try_cnt}")
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_MAX_CNT",
                                 ctr=self.prof_max_cnt is not None,
                                 tv=f"{self.prof_max_cnt}")
        return cmd


class TestsParam(CMakeParam):

    def __init__(self, args):
        self.exec: TestsExecuteParam = TestsExecuteParam(args=args)
        self.golden: TestsGoldenParam = TestsGoldenParam(args=args)
        self.utest: TestsFilterParam = TestsFilterParam(argv=args.utest, opt="ENABLE_UTEST")
        self.utest_module: TestsFilterParam = TestsFilterParam(argv=args.utest_module, opt="ENABLE_UTEST_MODULE")
        self.stest_exec: STestExecuteParam = STestExecuteParam(args=args, enable_binary_cache=self.exec.ci_model)
        self.stest_tools: STestToolsParam = STestToolsParam()
        self.stest: TestsFilterParam = TestsFilterParam(argv=args.stest, opt="ENABLE_STEST")
        self.stest_group: TestsFilterParam = TestsFilterParam(argv=args.stest_group, opt="ENABLE_STEST_GROUP")
        self.stest_distributed: TestsFilterParam = TestsFilterParam(argv=args.stest_distributed,
                                                                    opt="ENABLE_STEST_DISTRIBUTED")
        self.example: TestsFilterParam = TestsFilterParam(argv=args.example)

    def __str__(self):
        desc = ""
        if self.enable:
            desc += f"\nTests"
            desc += f"\n    Execute"
            desc += f"\n               Changed File : {self.exec.changed_file}"
            desc += f"\n                       Auto : {self.exec.auto_execute}"
            desc += f"\n                   Parallel : {self.exec.auto_execute_parallel}"
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
                if self.stest_tools.prof_enable:
                    desc += f"\n        Tools"
                    desc += f"\n              Case Csv File : {self.stest_tools.cases_csv_file}"
                    desc += f"\n             Intercept Flag : {self.stest_tools.intercept_flag}"
                    desc += f"\n               Output Clean : {self.stest_tools.output_clean}"
                    desc += f"\n        Tools Profiling"
                    desc += f"\n                     Enable : {self.stest_tools.prof_enable}"
                    desc += f"\n                      Level : {self.stest_tools.prof_level}"
                    desc += f"\n                Warn Up Cnt : {self.stest_tools.prof_warn_up_cnt}"
                    desc += f"\n                    Try Cnt : {self.stest_tools.prof_try_cnt}"
                    desc += f"\n                    Max Cnt : {self.stest_tools.prof_max_cnt}"
            if self.stest_distributed.enable:
                desc += f"\n    Stest Distributed"
                desc += f"\n                     Enable : {self.stest_distributed.enable}"
                desc += f"\n                     Filter : {self.stest_distributed.filter_str}"
            if self.example.enable:
                desc += f"\n    Example"
                desc += f"\n                     Enable : {self.example.enable}"
                desc += f"\n                     Filter : {self.example.filter_str}"
        return desc

    @property
    def enable(self) -> bool:
        return self.utest.enable or self.stest.enable or self.stest_distributed.enable or self.example.enable

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        TestsExecuteParam.reg_args(parser=parser)
        TestsGoldenParam.reg_args(parser=parser)
        TestsFilterParam.reg_args(parser=parser, ext="utest")
        TestsFilterParam.reg_args(parser=parser, ext="utest_module")
        STestExecuteParam.reg_args(parser=parser)
        STestToolsParam.reg_args(parser=ext)
        TestsFilterParam.reg_args(parser=parser, ext="stest")
        TestsFilterParam.reg_args(parser=parser, ext="stest_group")
        TestsFilterParam.reg_args(parser=parser, ext="stest_distributed")
        TestsFilterParam.reg_args(parser=parser, ext="example")

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self.utest.get_cfg_cmd()
        cmd += self.stest.get_cfg_cmd()
        cmd += self.stest_distributed.get_cfg_cmd()
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
                cmd += self.stest_tools.get_cfg_cmd()
        return cmd


@dataclasses.dataclass
class ModelParam(CMakeParam):
    prof: int = 0
    pe: int = 2
    sim: bool = False
    sim_with_onboard_aicpu: bool = False
    back_annotation_aicpu: bool = False
    back_annotation_aicore: bool = False
    replay_file_path: Optional[str] = None
    calendar: bool = False
    pvmodel: bool = False

    def __init__(self, args):
        self.prof = args.prof
        self.pe = args.pe
        self.sim = args.sim
        self.sim_with_onboard_aicpu = args.sim_with_onboard_aicpu
        self.back_annotation_aicpu = args.back_annotation_aicpu
        self.back_annotation_aicore = args.back_annotation_aicore
        self.replay_file_path = args.replay_file_path
        self.calendar = args.calendar
        self.pvmodel = args.pvmodel

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--prof", nargs="?", type=int, default=0, choices=[1, 2],
                            help="Enable workflow.")
        parser.add_argument("--pe", nargs="?", type=int, default=2, choices=[1, 2, 4, 5, 6, 7, 8],
                            help="Enable pmuEvent.")
        parser.add_argument("-s1", "--sim", action="store_true", default=False,
                            help="enable simulation")
        parser.add_argument("-s2", "--sim_with_onboard_aicpu", action="store_true", default=False,
                            help="enable simulation with onboard aicpu code")
        parser.add_argument("-b1", "--back_annotation_aicpu", action="store_true", default=False,
                            help="enable back-annotation in simulation with aicpu onboard data.")
        parser.add_argument("-b2", "--back_annotation_aicore", action="store_true", default=False,
                            help="(WIP)enable back-annotation in simulation with aicore onboard data.")
        parser.add_argument("-rf", "--replay_file_path", type=str, default=None,
                            help="Specify replay file path for back annotation.")
        parser.add_argument("-cal", "--calendar", action="store_true", default=False,
                            help="Enable calendar mode.")
        parser.add_argument("-pv", "--pvmodel", action="store_true", default=False,
                            help="Enable PVModel mode.")

    @staticmethod
    def _save_simulation_json(simulation_json, src_root: Path):
        temp_json_path = os.path.join(str(src_root), "framework/src/cost_model/simulation/scripts/tmp_simulation.json")
        os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
        with open(temp_json_path, 'w') as f:
            json.dump(simulation_json, f, indent=4)

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        return ""

    def gen_simulation_json(self, src_root: Path) -> None:
        simulation_json = {
            "global_configs": {
                "platform_configs": {},
                "simulation_configs": {}
            }
        }
        self._gen_simulation_json_sim(cfg=simulation_json)
        self._gen_simulation_json_sim_with_onboard_aicpu(cfg=simulation_json)
        self._gen_simulation_json_back_annotation_aicpu(cfg=simulation_json)
        self._gen_simulation_json_back_annotation_aicore(cfg=simulation_json)
        self._gen_simulation_json_calendar(cfg=simulation_json)
        self._gen_simulation_json_pvmodel(cfg=simulation_json)
        self._save_simulation_json(simulation_json, src_root=src_root)

    def _gen_simulation_json_sim(self, cfg: Dict[Any, Any]):
        if self.sim:
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True

    def _gen_simulation_json_sim_with_onboard_aicpu(self, cfg: Dict[Any, Any]):
        if self.sim_with_onboard_aicpu:
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True
            cfg["global_configs"]["simulation_configs"]["USE_ON_BOARD_INFO"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0"
            ]

    def _gen_simulation_json_back_annotation_aicpu(self, cfg: Dict[Any, Any]):
        if self.back_annotation_aicpu:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when back_annotation_aicpu is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
                "Model.replayAllMode=1",
                f"Model.replayFile={self.replay_file_path}"
            ]

    def _gen_simulation_json_back_annotation_aicore(self, cfg: Dict[Any, Any]):
        if self.back_annotation_aicore:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when back_annotation_aicore is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True
            cfg["global_configs"]["simulation_configs"]["USE_ON_BOARD_INFO"] = True
            cfg["global_configs"]["simulation_configs"]["json_path"] = self.replay_file_path
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0"
            ]

    def _gen_simulation_json_calendar(self, cfg: Dict[Any, Any]):
        if self.calendar:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when calendar is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
                "Model.genCalendarScheduleCpp=true",
                "Model.simulationFixedLatencyTask=true",
                f"Model.fixedLatencyTaskInfoPath={self.replay_file_path}",
                "Model.fixedLatencyTimeConvert=1",
                "Model.aicpuMachineNumber=1",
                "Model.coreMachineNumberPerAICPU=54",
                "Model.cubeMachineNumberPerAICPU=27",
                "Model.vecMachineNumberPerAICPU=27",
            ]

    def _gen_simulation_json_pvmodel(self, cfg: Dict[Any, Any]):
        if self.pvmodel:
            cfg["global_configs"]["platform_configs"]["enable_cost_model"] = True
            cfg["global_configs"]["simulation_configs"]["pv_level"] = 2
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
            ]


class BuildCtrl(CMakeParam):
    """构建过程控制.

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行.
    """
    _PYTHONPATH: str = "PYTHONPATH"

    def __init__(self, args):
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "build_out")
        self.feature: FeatureParam = FeatureParam(args=args)
        self.build: BuildParam = BuildParam(args=args)
        self.tests: TestsParam = TestsParam(args=args)
        self.model: ModelParam = ModelParam(args=args)
        self.third_party_path: Optional[Path] = Path(args.third_party_path).resolve() if args.third_party_path else None
        self.verbose: bool = args.verbose
        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError(f"Can't find cmake")
        # 表示 pip 版本是否支持传递 --config-setting 这种 pep 标准参数传递方式
        self.pip_dependence_desc: Dict[str, str] = {"pip": ">=22.1"}
        self.pip_support_config_setting = self.check_pip_dependencies(deps=self.pip_dependence_desc,
                                                                      raise_err=False, log_err=False)

    def __str__(self):
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
        desc += f"\n    Verbose                 : {self.verbose}"
        desc += f"{self.feature}"
        desc += f"{self.build}"
        desc += f"{self.tests}"
        desc += f"\n"
        return desc

    @staticmethod
    def which_cmake() -> Optional[Path]:
        """查找系统级 CMake 可执行文件路径

        排除 cmake pip 包的干扰
        """
        # 拆分 PATH 环境变量为单个目录列表（排除空目录）
        path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]

        # 遍历每个 PATH 目录，逐个调用 shutil.which 检查, 限定 shutil.which 只在当前单个目录下查找 cmake
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
    def run_build_cmd(cmd: str, update_env: Optional[Dict[str, str]] = None, check: bool = False,
                      timeout: Optional[int] = None) -> Optional[subprocess.CompletedProcess]:
        """执行具体 build 命令行

        因以下原因, 设置本函数, 而非调用原生 subprocess.run
            1. 支持多 target 构建, 各 target 构建时长共享公共 timeout 配置;
            2. UTest/STest 并行执行场景下, 执行时进程调用关系为:
                   build_ci.py(主进程) -> 进程1(CMake) -> 进程2(CMake Generator, make/ninja) -> 进程3(Python)-> 进程4(exe)
               此时若 进程1 超时, 需要触发其子/孙进程感知, 进而结束

        :param cmd: Build 命令行
        :param update_env: 环境变量(额外更新内容)
        :param check: 检查返回值
        :param timeout: 执行超时时长
        """

        def _stop_pg(_msg: str, _p: subprocess.Popen):
            """通过 SIGINT 信号通知所有子/孙进程结束, python 并行脚本内会捕获该信号进行结算处理
            """
            _pgid = os.getpgid(_p.pid)
            logging.info("%s. Send terminate event to CMake[%s]", _msg, _pgid)
            os.killpg(_pgid, signal.SIGINT)

        stdout = None
        stderr = None
        env = os.environ.copy()
        env.update(update_env if update_env else {})
        with subprocess.Popen(shlex.split(cmd), env=env, text=True, encoding='utf-8',
                              start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_pg(_msg="Timeout", _p=process)
                raise
            except KeyboardInterrupt:
                # 一般为用户主动触发, 不需再上报错误
                _stop_pg(_msg="KeyboardInterrupt", _p=process)
            except Exception:
                process.kill()
                raise
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if check and ret_code:
                raise subprocess.CalledProcessError(ret_code, process.args, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr)

    @staticmethod
    def find_match_whl(name: str, path: Path) -> Optional[Path]:
        """在指定路径下, 查找对应匹配的 whl 包文件

        :param name: 包名
        :param path: 指定路径
        :return: whl 包路径, None 表示未找到
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
        parser.add_argument("--cann_3rd_lib_path", "--third_party_path",
                            nargs="?", type=str, default="", dest="third_party_path",
                            help="Specify 3rd Libraries Path")
        parser.add_argument("--verbose", action="store_true", default=False,
                            help="verbose, enable verbose output.")

    @classmethod
    def main(cls):
        """主处理流程
        """
        parser = argparse.ArgumentParser(description=f"PyPTO Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # 子命令
        # 参数注册
        FeatureParam.reg_args(parser=parser)
        BuildParam.reg_args(parser=parser)
        TestsParam.reg_args(parser=parser, ext=sub_parser)
        ModelParam.reg_args(parser=parser)
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
            ctrl.model.gen_simulation_json(src_root=ctrl.src_root)
            ctrl.cmake_build()

    @classmethod
    def pip_uninstall(cls, name: str, path: Optional[Path] = None):
        """卸载对应 whl 包

        :param name: 包名
        :param path: 指定安装路径(可选), 如果指定对应路径, 仅会在对应路径尝试卸载
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
            ret = cls.run_build_cmd(cmd=cmd, check=True)
            ret.check_returncode()
        logging.info("Success uninstall %s package%s", name, f" from {path}" if path else "")

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

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        cmd = self._cfg_require(opt=f"PYPTO_THIRD_PARTY_PATH", ctr=bool(self.third_party_path),
                                tv=f"{self.third_party_path}")
        return cmd

    def get_cfg_update_env(self) -> Dict[str, str]:
        env = {}
        if self.third_party_path:
            env.update({"PYPTO_THIRD_PARTY_PATH": self.third_party_path})
        return env

    def pip_install(self, whl: Path, dest: Optional[Path] = None, opt: str = "",
                    update_env: Optional[Dict[str, str]] = None):
        """安装指定 whl 包

        :param whl: 包文件
        :param dest: 安装路径(可选), 未指定时会安装在默认路径
        :param opt: 额外安装参数
        :param update_env:
        """
        ts = datetime.now(tz=timezone.utc)
        edit_str = "-e " if self.feature.whl_editable else ""
        cmd = f"{sys.executable} -m pip install {edit_str}" + f"{whl} {opt}" + (" -vvv " if self.verbose else "")
        cmd += f" --target={dest}" if dest else ""
        logging.info("Begin install %s, cmd: %s", whl, cmd)
        ret = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
        ret.check_returncode()
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("Success install %s%s, Duration %s sec", whl, f" to {dest}" if dest else "", duration)

    def cmake_clean(self):
        """清理中间结果, 清理内容包括构建树, 安装树全部内容.
        """
        if self.build.clean:
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
        self.cmake_clean()
        if not self.build.clean:
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
        """CMake Configure 阶段流程.
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
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = self.run_build_cmd(cmd=cmd, update_env=update_env, check=True)
        ret.check_returncode()

    def cmake_build(self):
        """CMake Build 阶段流程.
        """
        # prof使能初始化
        update_env = {}
        if self.model.prof == 1 or self.model.prof == 2:
            update_env = wf.ini(self.build_root, self.model.prof, self.model.pe)
        if self.build.job_num:
            update_env["PYPTO_UTEST_PARALLEL_NUM"] = str(self.build.job_num)
        cmd_list = self.build.get_build_cmd_lst(cmake=self.cmake, binary_path=self.build_root)
        for i, c in enumerate(cmd_list, start=1):
            ts = datetime.now(tz=timezone.utc)
            c += " --verbose" if self.verbose else ""
            logging.info("CMake Build(%s/%s), Cmd: %s", i, len(cmd_list), c)
            try:
                ret = self.run_build_cmd(cmd=c, update_env=update_env, check=True, timeout=self.build.timeout)
            except subprocess.CalledProcessError as e:
                logging.info(f"Run cmd {c} failed, ERROR CODE: {e.returncode}")
                # 一键绘图
                if self.model.prof == 1 or self.model.prof == 2:
                    wf.work_flow_plot(self.build_root, self.model.prof, self.model.pe)
                raise
            ret.check_returncode()
            duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
            duration_str = f"{duration}/{self.build.timeout}" if self.build.timeout else f"{duration}"
            logging.info("CMake Build(%s/%s), Cmd: %s, Duration %s sec", i, len(cmd_list), c, duration_str)
            # 超时时长更新, 当指定多 target 时, 各 target 共享总超时时长
            self.build.timeout = self.build.timeout - duration if self.build.timeout else self.build.timeout
        # 一键绘图
        if self.model.prof == 1 or self.model.prof == 2:
            wf.work_flow_plot(self.build_root, self.model.prof, self.model.pe)

    def py_build(self):
        """whl 包编译处理

        支持:
            1. 正式编译, 调用 build 库触发 setuptools(bdist_wheel 命令) 进而触发 CMake 完成编译;
            2. pip编译, 调用 pip install 命令触发 setuptools(editable_wheel 命令) 进而触发 CMake 完成编译, 有两种模式:
                1. 常规安装: 适用于生产环境或代码稳定后使用, 其安装后对源码的修改不会反映到已安装的包中;
                2. 可编辑安装: 便于开发调试. 它在 site-packages 中创建指向本地的链接, 对 Python 源码的修改会即时生效, 无需重新安装;
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
            ts = datetime.now(tz=timezone.utc)
            logging.info("Begin Build whl, Cmd: %s", cmd)
            ret = self.run_build_cmd(cmd=cmd, update_env=update_env, check=True, timeout=self.build.timeout)
            ret.check_returncode()
            duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
            duration_str = f"{duration}/{self.build.timeout}" if self.build.timeout else f"{duration}"
            logging.info("Success Build whl, Cmd: %s, Duration %s sec", cmd, duration_str)

    def py_tests(self):
        if not self.tests.utest.enable and not self.tests.stest.enable and not self.tests.example.enable:
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
        # 在 Python 3.12 中，pytest-xdist 通过 os.fork() 创建子进程时会产生 DeprecationWarning。
        # 使用 -W ignore::DeprecationWarning 参数来忽略该警告。
        if self.build.job_num is not None and self.build.job_num > 0:
            n_workers = str(self.build.job_num)
        else:
            n_workers = "auto"
        self.py_tests_run_pytest(dist=dist, tests=self.tests.utest,
                                 def_filter=str(Path(self.src_root, "python/tests/ut")),
                                 ext=f"-n {n_workers} -W ignore::DeprecationWarning")

        # 设置 Device 相关参数
        dev_lst = [int(d) for d in self.tests.stest_exec.auto_execute_device_id.split(":")]
        dev_ext = " ".join(f"{d}" for d in dev_lst)
        ext_str = f"-n {len(dev_lst)} --device {dev_ext}"

        # 执行用例, STest
        self.py_tests_run_pytest(dist=dist, tests=self.tests.stest,
                                 def_filter=str(Path(self.src_root, "python/tests/st")),
                                 ext=ext_str)

        # 执行用例, Examples
        self.py_tests_run_pytest(dist=dist, tests=self.tests.example,
                                 def_filter=str(Path(self.src_root, "examples")),
                                 ext=ext_str)

    def py_tests_run_pytest(self, dist: Optional[Path], tests: TestsFilterParam, def_filter: str, ext: str = ""):
        if not tests.enable:
            return
        # filter 处理
        filter_str = tests.get_filter_str(def_filter=def_filter)
        # 执行 pytest
        self._py_tests_run_pytest(dist=dist, filter_str=filter_str, ext=ext)

    def _py_tests_run_pytest(self, dist: Optional[Path], filter_str: str, ext: str = ""):
        if not self.tests.exec.auto_execute:
            return
        # filter 处理
        filter_str = filter_str.replace(',', ' ')
        # cmd 拼接
        cmd = f"{sys.executable} -m pytest {filter_str} -v --durations=0 -s --capture=no"
        cmd += f" --rootdir={self.src_root} {ext} --forked"
        # cmd 执行
        origin_env = os.environ.copy()
        update_env = {}
        if dist:
            ori_env_python_path = origin_env.get(self._PYTHONPATH, "")
            act_env_python_path = f"{dist}:{ori_env_python_path}" if ori_env_python_path else f"{dist}"
            update_env.update({self._PYTHONPATH: act_env_python_path})
        ts = datetime.now(tz=timezone.utc)
        logging.info("pytest run, Cmd: %s", cmd)
        ret = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
        ret.check_returncode()
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("pytest run, Cmd: %s, Duration %s sec", cmd, duration)

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


class SubCommandMgr:
    @classmethod
    def init_param_tools_profiling(cls, args, ctrl: BuildCtrl):
        ctrl.tests.stest_tools.init_param(args=args)
        ctrl.tests.stest_tools.init_param_profiling(args=args)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    g_ts = datetime.now(tz=timezone.utc)
    BuildCtrl.main()
    g_duration = int((datetime.now(tz=timezone.utc) - g_ts).seconds)
    logging.info("Build[CI] Success, duration %s secs.", g_duration)

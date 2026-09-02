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
"""PyPTO project CI scenario build control entry point

This file provides a unified build entry point for PyPTO project CI scenarios,
supporting multiple build modes and configuration options.

Main features:
    - Supports regular and editable mode compilation of whl packages
    - Supports execution of UTest/STest/Examples and other test cases
    - Supports build timeout control and automatic subprocess cleanup after timeout

Usage:
    Configure build options via command-line arguments, then run the script to trigger the build process:

        python build_ci.py [options]

    Common options:
        -f/--frontend: Specify frontend type (python3/cpp)
        -b/--backend: Specify backend type (npu/cost_model)
        -t/--targets: Specify build targets
        -j/--job_num: Specify build parallelism
        --build_type: Specify build type (Debug/Release/MinSizeRel/RelWithDebInfo)
        -u/--utest: Enable UTest
        -s/--stest: Enable STest
        -c/--clean: Clean build directory and install directory

Examples:
    # Build with default configuration
    python build_ci.py

    # Build with specified frontend and backend types
    python build_ci.py -f python3 -b npu

    # Enable tests and specify parallelism
    python build_ci.py -u -s -j 8

    # Clean and rebuild
    python build_ci.py -c --build_type Debug
"""

import abc
import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
from importlib import metadata
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from packaging import requirements

try:
    from setup import _find_system_cmake, get_job_num
except ImportError:
    _find_system_cmake = None
    get_job_num = None


class CMakeParam(abc.ABC):
    """CMake parameter abstract base class

    Defines the common interface for all parameter classes that need to pass Options to CMake.
    Subclasses must implement reg_args() to register CLI arguments and get_cfg_cmd() to generate
    CMake configuration commands.
    """

    @staticmethod
    def get_system_processor() -> str:
        """Get the system processor architecture name

        Retrieves the current system processor architecture via platform.machine()
        and maps common aliases to standard names.

        :return: Standardized processor architecture name, such as x86_64 or aarch64
        :rtype: str
        """
        machine = platform.machine().lower()
        arch_map = {  # Map common architectures directly
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
        }
        return arch_map.get(machine, machine)

    @staticmethod
    @abc.abstractmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register command-line arguments

        Registers the command-line arguments supported by the current class to the argument parser.
        Subclasses must implement this method to define their respective parameter options.

        :param parser: ArgumentParser instance
        :param ext: Extension information, used for subclass-specific extensions
        :type ext: Optional[Any]
        """
        pass

    @classmethod
    def _cfg_require(cls, opt: str, ctr: bool = True, tv: str = "ON", fv: str = "OFF") -> str:
        """Get a required Option configuration for the CMake Configure stage

        Returns the corresponding CMake Option configuration string based on the value of the ctr control variable.
        This method always returns a non-empty configuration string.

        :param opt: CMake Option name, which will be reflected in the CMake -D parameter
        :type opt: str
        :param ctr: Control variable, indicating the CMake Option boolean value
        :type ctr: bool
        :param tv: Value to set when ctr is True, defaults to "ON"
        :type tv: str
        :param fv: Value to set when ctr is False, defaults to "OFF"
        :type fv: str
        :return: CMake configuration string, formatted as " -DOPT_NAME=VALUE"
        :rtype: str
        """
        return f" -D{opt}=" + (tv if ctr else fv)

    @classmethod
    def _cfg_optional(cls, opt: str, ctr: bool, v: str) -> str:
        """Get an optional Option configuration for the CMake Configure stage

        Returns the corresponding CMake Option configuration string based on the value of the ctr control variable.
        Returns an empty string when ctr is False.

        :param opt: CMake Option name, which will be reflected in the CMake -D parameter
        :type opt: str
        :param ctr: Control variable, indicating the CMake Option boolean value
        :type ctr: bool
        :param v: Value to set when the control variable is True
        :type v: str
        :return: CMake configuration string, formatted as " -DOPT_NAME=VALUE", empty string when ctr is False
        :rtype: str
        """
        return (f" -D{opt}=" + v) if ctr else ""

    @abc.abstractmethod
    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration command string based on the current parameter configuration.
        Subclasses must implement this method to define specific configuration parameters.

        :param ext: Extension information, used for subclass-specific extensions
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        pass


@dataclasses.dataclass
class FeatureParam(CMakeParam):
    """Feature control related parameters

    Manages feature options during the build process, including frontend type, backend type,
    and whl package compilation mode.
    """

    whl_name: str = "pypto"
    frontend_type: Optional[str] = None  # Frontend type, supports python3, cpp
    backend_type: Optional[str] = None  # Backend type, supports npu, cost_model
    # Numeric part of whl package py_abi_tag (e.g. 39/310), explicitly specified via --py_abi;
    whl_py3_abi: Optional[int] = None
    whl_plat_name: Optional[str] = None  # python3 whl package plat-name
    whl_isolation: bool = False  # Compile whl package in isolation mode
    whl_editable: bool = False  # Compile whl package in editable mode
    whl_break_system_packages: bool = False
    just_build_whl: bool = False  # Build whl only, without packing into run
    package_type: str = "run"  # Package type: run/rpm/deb/all

    def __init__(self, args):
        """Initialize FeatureParam instance

        Parses frontend type, backend type, and whl package compilation mode from command-line arguments.
        If the backend type is npu but ASCEND_HOME_PATH environment variable is not set,
        it automatically falls back to the cost_model backend.

        :param args: Parsed command-line arguments
        """
        self.frontend_type = "python3" if args.frontend is None else args.frontend
        self.backend_type = "npu" if args.backend is None else args.backend
        if not os.environ.get("ASCEND_HOME_PATH") and self.backend_type in ["npu"]:
            logging.warning("Environment variable ASCEND_HOME_PATH is unset/empty, falling back to cost_model backend.")
            self.backend_type = "cost_model"
        self.just_build_whl = args.just_build_whl
        self.package_type = args.pkg_type
        if self.just_build_whl:
            py_abi = args.py_abi
            plat_name = args.plat_name
        else:
            py_abi = args.py_abi if args.py_abi is not None else 37
            plat_name = args.plat_name if args.plat_name else "manylinux2014"
        self.whl_py3_abi = py_abi
        self.whl_plat_name = f"{plat_name}_{CMakeParam.get_system_processor()}" if plat_name else ""
        self.whl_isolation = args.isolation
        self.whl_editable = args.editable
        self.whl_break_system_packages = args.break_system_packages

    def __str__(self) -> str:
        """Return string representation of feature parameters

        :return: Formatted feature parameter string
        :rtype: str
        """
        desc = ""
        desc += "\nFeature"
        desc += f"\n    Frontend                : {self.frontend_type}"
        if self.frontend_type_python3:
            desc += f"\n    JustBuildWhl            : {self.just_build_whl}"
            if self.whl_py3_abi_tag:
                desc += f"\n    Py3ABI                  : {self.whl_py3_abi_tag}"
            if self.whl_plat_name:
                desc += f"\n    PlatName                : {self.whl_plat_name}"
            desc += f"\n    Isolation               : {self.whl_isolation}"
            desc += f"\n    Editable                : {self.whl_editable}"
            desc += f"\n    BreakSystemPackages     : {self.whl_break_system_packages}"
        desc += f"\n    Backend                 : {self.backend_type}"
        desc += f"\n    PackageType             : {self.package_type}"
        return desc

    @property
    def frontend_type_python3(self) -> bool:
        """Check if the frontend type is Python3

        :return: True if the frontend type is "python" or "python3"
        :rtype: bool
        """
        return self.frontend_type in ["python", "python3"]

    @property
    def whl_py3_abi_tag(self) -> Optional[str]:
        """Numeric part of whl package python abi tag, used for --py-limited-api
        parameter construction (e.g. "39" maps to cp39).

        Priority: explicitly specified via --py_abi, otherwise returns None.
        """
        if self.whl_py3_abi:
            return str(self.whl_py3_abi)
        return None

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register feature-related command-line arguments

        Registers frontend type, backend type, whl compilation mode, and other arguments to the argument parser.

        :param parser: ArgumentParser instance
        :param ext: Extension information, unused
        :type ext: Optional[Any]
        """
        parser.add_argument(
            "-f",
            "--frontend",
            nargs="?",
            type=str,
            default="python3",
            choices=["python3", "cpp"],
            help="frontend, such as python3/cpp etc.",
        )
        parser.add_argument(
            "--py_abi",
            type=int,
            default=None,
            choices=[37, 38, 39, 310, 311, 312, 313, 314],
            help="whl py abi tag numeric part, e.g. 39 for cp39(Python 3.9), 310 for cp310(Python 3.10)",
        )
        parser.add_argument(
            "--plat_name",
            nargs="?",
            type=str,
            default="",
            choices=["manylinux2014", "manylinux_2_24", "manylinux_2_28"],
            help="whl plat_name, such as manylinux2014/manylinux_2_24/manylinux_2_28 etc.",
        )
        parser.add_argument(
            "--no_isolation",
            action="store_false",
            default=True,
            dest="isolation",
            help="Disable building the project(whl) in an isolated virtual environment. "
            "Build dependencies must be installed separately when this option is used.",
        )
        parser.add_argument(
            "--editable",
            action="store_true",
            default=False,
            help="Install whl in editable mode (i.e. setuptools \"editable_wheel\")",
        )
        parser.add_argument(
            "--break_system_packages",
            action="store_true",
            default=False,
            help="Bypass system Python package protection to force global pip installation.",
        )
        parser.add_argument(
            "-b",
            "--backend",
            nargs="?",
            type=str,
            default="npu",
            choices=["npu", "cost_model"],
            help="backend, such as npu/cost_model etc.",
        )
        parser.add_argument(
            "--just_build_whl", action="store_true", default=False, help="Build whl only, without packing into run."
        )
        parser.add_argument(
            "--pkg-type",
            "--pkg_type",
            dest="pkg_type",
            type=str,
            default="run",
            choices=["run", "rpm", "deb", "all"],
            help="Specify package type (run/rpm/deb/all), default: run.",
        )

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration parameters based on frontend type and backend type.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = ""
        cmd += self._cfg_require(opt="ENABLE_FEATURE_PYTHON_FRONT_END", ctr=self.frontend_type_python3)
        cmd += self._cfg_require(opt="BUILD_WITH_CANN", ctr=self.backend_type in ["npu"])
        cmd += self._cfg_require(
            opt="ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE", ctr=self.frontend_type_python3 and self.whl_py3_abi
        )
        return cmd


@dataclasses.dataclass
class BuildParam(CMakeParam):
    """Build related parameters

    Manages build process configuration options, including CMake configuration parameters
    and build execution parameters.
    """

    # Configure
    generator: Optional[str] = None  # Generator
    build_type: Optional[str] = None  # Build type
    asan: bool = False  # Enable AddressSanitizer
    ubsan: bool = False  # Enable UndefinedBehaviorSanitizer
    gcov: bool = False  # Enable GNU Coverage
    gcov_incr: bool = False  # Enable incremental coverage GCov calculation (compatible with legacy --gcov_increment)
    cov_incr: bool = False  # Enable incremental coverage (both C++ gcov and Python coverage)
    py_cov: bool = False  # Enable Python coverage (pytest-cov)
    clang_install_path: Optional[Path] = None  # Clang installation path
    compile_dependency_check: bool = False  # Enable compile dependency relation check
    # Build
    targets: Optional[List[str]] = None  # Build targets
    job_num: Optional[int] = None  # Number of cores used during build

    def __init__(self, args):
        """Initialize BuildParam instance

        Parses build-related configuration options from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.targets = args.targets
        self.job_num = self._get_job_num(job_num=args.job_num, generator=args.generator)
        self.generator = self._get_generator(generator=args.generator)
        self.build_type = args.build_type
        self.asan = args.asan
        self.ubsan = args.ubsan
        self.gcov = args.gcov
        self.gcov_incr = args.gcov_increment
        self.cov_incr = args.cov_increment or args.gcov_increment  # cov_increment includes gcov_increment
        self.py_cov = args.py_cov
        self.clang_install_path = self._get_clang_install_path(opt=args.clang)
        self.compile_dependency_check = args.compile_dependency_check

    def __str__(self) -> str:
        """Return string representation of build parameters

        :return: Formatted build parameter string
        :rtype: str
        """
        desc = "\nBuild"
        desc += "\n    CMake"
        desc += "\n        Configure"
        desc += f"\n                  Generator : {self.generator}"
        desc += f"\n                  BuildType : {self.build_type}"
        desc += f"\n                       ASan : {self.asan}"
        desc += f"\n                      UbSan : {self.ubsan}"
        desc += f"\n                       GCov : {self.gcov}, Increment: {self.cov_incr}"
        desc += f"\n                      PyCov : {self.py_cov}, Increment: {self.cov_incr}"
        desc += f"\n           ClangInstallPath : {self.clang_install_path}"
        desc += f"\n            CompileDepCheck : {self.compile_dependency_check}"
        desc += "\n        Build"
        desc += f"\n                    Targets : {self.targets}"
        desc += f"\n                    Job Num : {self.job_num}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register build-related command-line arguments

        Registers build generator, build type, Sanitizer options, and other arguments to the argument parser.

        :param parser: ArgumentParser instance
        :param ext: Extension information, unused
        :type ext: Optional[Any]
        """
        # Configure
        parser.add_argument("--generator", nargs="?", type=str, default="", help="Specify a build system generator.")
        parser.add_argument(
            "--build_type",
            "--build-type",
            nargs="?",
            type=str,
            default="Release",
            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
            help="build type.",
        )
        parser.add_argument("--asan", action="store_true", default=False, help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False, help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument(
            "--gcov", action="store_true", default=False, help="Enable GNU Coverage Instrumentation Tool."
        )
        parser.add_argument(
            "--gcov_increment",
            action="store_true",
            default=False,
            help="Enable increment coverage calculation based on latest commit. "
            "(Deprecated alias for --cov_increment, kept for backward compatibility.)",
        )
        parser.add_argument(
            "--cov_increment",
            action="store_true",
            default=False,
            help="Enable increment coverage calculation based on latest commit (applies to both C++ and Python).",
        )
        parser.add_argument(
            "--py_cov",
            action="store_true",
            default=False,
            help="Enable Python coverage (pytest-cov). Independent from --gcov. "
            "Use with --cov_increment (or --gcov_increment) for Python increment coverage.",
        )
        parser.add_argument(
            "--clang", nargs="?", type=str, default="", help="Specify clang install path, such as /usr/bin/clang"
        )
        parser.add_argument(
            "--compile_dependency_check",
            action="store_true",
            default=False,
            help="Enable compile dependency relation check.",
        )
        # Build
        parser.add_argument(
            "-t",
            "--targets",
            nargs="?",
            type=str,
            action="append",
            help="targets, specific build targets, "
            "If you specify more than one, all targets within the specified range are built.",
        )
        parser.add_argument(
            "-j",
            "--job_num",
            nargs="?",
            type=int,
            default=-1,
            help="job num, specific job num of build. "
            "If unset, parallelism is auto-derived from CPU count and available memory "
            "(env PYPTO_BUILD_JOB_NUM overrides the default).",
        )

    @staticmethod
    def _get_clang_install_path(opt: Optional[str]) -> Optional[Path]:
        """Get Clang installation directory

        Determines the Clang installation path based on the specified clang argument or by auto-discovery.

        :param opt: Clang argument, can be None (auto-discovery), empty string (do not use Clang), or a specific path
        :type opt: Optional[str]
        :return: Clang installation directory path, returns None if Clang is not used
        :rtype: Optional[Path]
        """
        if opt is None:  # clang argument specified but no specific path given, try to find it
            cmd = "which clang"
            ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
            ret.check_returncode()
            clang_install_path = Path(ret.stdout).resolve()
        elif opt == "":  # clang argument not specified
            clang_install_path = None
        else:  # clang argument specified with a specific path
            clang_install_path = Path(opt)
        if clang_install_path is not None:
            clang_install_path = Path(clang_install_path).resolve().parent
            if not clang_install_path.exists():
                raise ValueError(f"Clang install path not exist, path={clang_install_path}")
        return clang_install_path

    @staticmethod
    def _get_job_num(job_num: Optional[int], generator: Optional[str]) -> Optional[int]:
        """Get build parallel task count

        Delegates to the cmake/scripts/_job_num shared module, which considers CPU count,
        available memory (including cgroup limits), and the PYPTO_BUILD_JOB_NUM environment
        variable to determine parallelism. For Ninja generator, Ninja decides on its own.
        Falls back to CPU-only logic if the shared module is unavailable.

        :param job_num: User-specified parallel task count
        :type job_num: Optional[int]
        :param generator: Build generator name
        :type generator: Optional[str]
        :return: Final parallel task count, None means the build tool decides automatically
        :rtype: Optional[int]
        """
        if get_job_num is not None:
            return get_job_num(job_num=job_num, generator=generator)
        # Fallback: CPU-only calculation (kept for environments where setup.py import fails)
        def_job_num = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 128)  # 128 is the default max cores
        def_job_num = (
            None
            if generator
            and generator.lower()
            in [
                "ninja",
            ]
            else def_job_num
        )  # ninja determines the default core count itself
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_generator(generator: Optional[str]) -> Optional[str]:
        """Get build generator name

        If a generator is specified, wraps the name in quotes to support generator names with spaces.

        :param generator: Build generator name
        :type generator: Optional[str]
        :return: Processed build generator name
        :rtype: Optional[str]
        """
        return f"\"{generator}\"" if generator else generator

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration command string based on build configuration parameters.
        Supports build type, Sanitizer options, coverage statistics, and Clang toolchain configuration.

        :param ext: Extension information, if True then build type is not included
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        inc_build_type = bool(ext) if ext is not None else True
        cmd = self._cfg_require(opt="CMAKE_BUILD_TYPE", tv=self.build_type) if inc_build_type else ""
        cmd += self._cfg_require(opt="ENABLE_ASAN", ctr=self.asan)
        cmd += self._cfg_require(opt="ENABLE_UBSAN", ctr=self.ubsan)
        cmd += self._cfg_require(opt="ENABLE_GCOV", ctr=self.gcov)
        cmd += self._cfg_require(opt="ENABLE_PY_COV", ctr=self.py_cov)

        def _check_clang_toolchain(_opt: str, _b: str) -> Tuple[bool, str]:
            """Check if Clang toolchain exists and generate configuration command"""
            _p = Path(self.clang_install_path, _b)
            if _p.exists():
                return True, self._cfg_require(opt=_opt, tv=str(_p))
            logging.error("Clang Toolchain %s not exist.", _p)
            return False, ""

        def _gen_clang_cmd() -> Tuple[bool, str]:
            """Generate Clang-related CMake configuration commands"""
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
        """Generate CMake build command list

        Generates the corresponding CMake build commands based on the specified build targets.

        :param cmake: CMake executable path
        :type cmake: Path
        :param binary_path: Binary build directory path
        :type binary_path: Path
        :return: CMake build command list
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
    """Test execution related parameters

    Manages test execution configuration options, including auto-execution, parallel execution,
    timeout control, and duration caching.
    """

    changed_file: Optional[Path] = None  # Changed file path
    auto_execute: bool = False  # Auto-execute test cases
    auto_execute_parallel: bool = False  # Parallel execution of test cases
    case_execute_timeout: Optional[int] = None  # Timeout duration per test case
    case_execute_cpu_rank_size: Optional[int] = None  # CPU affinity Rank Size for parallel test execution
    dump_case_duration_json: Optional[Path] = None  # Case duration cache file path
    dump_case_duration_max_num: Optional[int] = None  # Maximum number of cases to cache duration for
    dump_case_duration_min_secends: Optional[int] = None  # Minimum duration in seconds for caching

    def __init__(self, args):
        """Initialize TestsExecuteParam instance

        Parses test execution related configuration options from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
        self.auto_execute = args.disable_auto_execute
        self.auto_execute_parallel = self.auto_execute and self.ci_model
        timeout = args.case_execute_timeout
        self.case_execute_timeout = timeout if timeout and timeout > 0 else None  # Per-case execution timeout
        self.case_execute_cpu_rank_size = args.cpu_rank_size
        duration_json = args.dump_case_duration_json
        self.dump_case_duration_json = Path(duration_json).resolve() if duration_json else None
        self.dump_case_duration_max_num = args.dump_case_duration_max_num
        self.dump_case_duration_min_secends = args.dump_case_duration_min_secends

    def __str__(self) -> str:
        """Return string representation of test execution parameters

        :return: Formatted test execution parameter string
        :rtype: str
        """
        desc = "\n    Execute"
        desc += f"\n               Changed File : {self.changed_file}"
        desc += f"\n                       Auto : {self.auto_execute}"
        desc += f"\n                   Parallel : {self.auto_execute_parallel}"
        desc += f"\n                CaseTimeout : {self.case_execute_timeout}"
        desc += "\n        CaseDuration"
        desc += f"\n                       Json : {self.dump_case_duration_json}"
        desc += f"\n                     MaxNum : {self.dump_case_duration_max_num}"
        desc += f"\n                     MinSec : {self.dump_case_duration_min_secends}"
        return desc

    @property
    def ci_model(self) -> bool:
        """Check if CI mode is enabled

        :return: True if a changed file is specified (indicating CI mode)
        :rtype: bool
        """
        return True if self.changed_file else False

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register test execution related command-line arguments

        Registers incremental testing, auto-execution, timeout control, and other arguments to the argument parser.

        :param parser: ArgumentParser instance
        :param ext: Extension information, unused
        :type ext: Optional[Any]
        """
        parser.add_argument(
            "--changed_files",
            nargs="?",
            type=Path,
            default=None,
            help="Specify the file of files changed, "
            "so that the corresponding test cases can be triggered incrementally.",
        )
        parser.add_argument(
            "--disable_auto_execute",
            action="store_false",
            default=True,
            help="Disable auto execute STest/Utest with build.",
        )
        parser.add_argument("--case_execute_timeout", nargs="?", type=int, default=None, help="Case execute timeout.")
        parser.add_argument(
            "--cpu_rank_size",
            nargs="?",
            type=int,
            default=None,
            help="Specify the rank size for CPU affinity grouping.",
        )
        parser.add_argument(
            "--dump_case_duration_json",
            nargs="?",
            type=Path,
            default=None,
            help="Specify the path to the case duration json cache file.",
        )
        parser.add_argument(
            "--dump_case_duration_max_num",
            nargs="?",
            type=int,
            default=None,
            help="Maximum number of cases to dump to duration json cache.",
        )
        parser.add_argument(
            "--dump_case_duration_min_secends",
            nargs="?",
            type=int,
            default=None,
            help="Minimum duration (in seconds) for cases to dump to duration json cache.",
        )

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration parameters based on test execution configuration.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_TESTS_EXECUTE", ctr=self.auto_execute)
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_PARALLEL", ctr=self.auto_execute_parallel)
        changed = self.changed_file and self.changed_file.exists() and self.changed_file.suffix.lower() == ".txt"
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_CHANGED_FILE", ctr=changed, tv=str(self.changed_file))
        return cmd


@dataclasses.dataclass
class TestsGoldenParam(CMakeParam):
    """Golden test related parameters

    Manages system test (STest) Golden standard data related configuration.
    """

    clean: bool = False  # Clean Golden marker
    path: Optional[Path] = None  # Specify Golden path

    def __init__(self, args):
        """Initialize TestsGoldenParam instance

        Parses Golden test related configuration from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.clean = args.golden_clean
        if args.golden_path:
            # When the argument is provided with a specific path, use it;
            # otherwise the default path is determined by the CMake side
            self.path = Path(args.golden_path).resolve()

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register Golden test related command-line arguments

        :param parser: ArgumentParser instance
        :param ext: Extension information, unused
        :type ext: Optional[Any]
        """
        parser.add_argument(
            "--golden_path",
            "--stest_golden_path",
            nargs="?",
            type=str,
            default="",
            help="Specific Tests golden path.",
            dest="golden_path",
        )
        parser.add_argument(
            "--golden_clean",
            "--golden_path_clean",
            "--stest_golden_path_clean",
            action="store_true",
            default=False,
            help="Clean Tests golden.",
            dest="golden_clean",
        )

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration parameters based on Golden test configuration.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH_CLEAN", ctr=self.clean)
        cmd += self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH", ctr=bool(self.path), tv=str(self.path))
        return cmd


@dataclasses.dataclass
class TestsFilterParam(CMakeParam):
    """Test filter parameters

    Used to filter test cases by conditions, supporting multiple test types and filter modes.
    """

    cmake_option: str = ""
    enable: bool = False
    filter_str: Optional[str] = None

    def __init__(self, argv: Optional[str], opt: str = ""):
        """Initialize TestsFilterParam instance

        Determines the enable status and filter string of the filter option based on command-line argument values.

        :param argv: Command-line argument value, None means enable default filter,
                     empty string means disabled, other values mean specified filter string
        :type argv: Optional[str]
        :param opt: CMake option name
        :type opt: str
        """
        self.cmake_option = opt
        if argv is None:
            self.enable, self.filter_str = True, "ON"  # Argument specified but no content provided
        elif argv == "":
            self.enable, self.filter_str = False, "OFF"  # Argument not specified
        else:
            self.enable, self.filter_str = True, argv  # Argument specified with content provided

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register test filter related command-line arguments

        Generates the corresponding command-line argument options based on extension information.

        :param parser: ArgumentParser instance
        :param ext: Extension information, used to generate argument name and help text
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
        """Generate CMake Configure command

        Generates the corresponding CMake configuration parameters based on filter configuration.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = ""
        if self.cmake_option:
            cmd += self._cfg_require(opt=f"{self.cmake_option}", ctr=self.enable, tv=f"{self.filter_str}")
        return cmd

    def get_filter_str(self, def_filter: str) -> str:
        """Get test filter string

        Generates the final filter string based on configuration and default filter conditions.

        :param def_filter: Default filter condition
        :type def_filter: str
        :return: Filter string, returns empty string if not enabled
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
    """STest execution related parameters

    Manages system test (STest) execution configuration, including device ID, JSON export, etc.
    """

    auto_execute_device_id: str = ""
    interpreter_config: bool = False
    enable_binary_cache: bool = False
    dump_json: bool = False

    def __init__(self, args, enable_binary_cache: bool):
        """Initialize STestExecuteParam instance

        Parses STest execution related configuration from command-line arguments.

        :param args: Parsed command-line arguments
        :param enable_binary_cache: Whether to enable binary cache
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
        """Register STest execution related command-line arguments

        :param parser: ArgumentParser instance
        :param ext: Extension information, unused
        :type ext: Optional[Any]
        """
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append", help="Device ID, default 0.")
        parser.add_argument("--stest_dump_json", action="store_true", default=False, help="Dump json files.")
        parser.add_argument(
            "--enable_interpreter_config", action="store_true", default=False, help="enable STest Interpreter Config"
        )

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        Generates the corresponding CMake configuration parameters based on STest execution configuration.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = self._cfg_require(opt="ENABLE_STEST_EXECUTE_DEVICE_ID", tv=self.auto_execute_device_id)
        cmd += self._cfg_require(opt="ENABLE_STEST_DUMP_JSON", ctr=self.dump_json)
        cmd += self._cfg_require(opt="ENABLE_STEST_INTERPRETER_CONFIG", ctr=self.interpreter_config)
        cmd += self._cfg_require(opt="ENABLE_STEST_BINARY_CACHE", ctr=self.enable_binary_cache)
        return cmd


class TestsParam(CMakeParam):
    """Test parameter aggregator class

    Aggregates all test-related parameter configurations, including execution parameters,
    Golden parameters, filter parameters, etc.
    """

    def __init__(self, args):
        """Initialize TestsParam instance

        Parses and initializes all test-related parameter configurations from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.exec: TestsExecuteParam = TestsExecuteParam(args=args)
        self.golden: TestsGoldenParam = TestsGoldenParam(args=args)
        self.utest: TestsFilterParam = TestsFilterParam(argv=args.utest, opt="ENABLE_UTEST")
        self.utest_module: TestsFilterParam = TestsFilterParam(argv=args.utest_module, opt="ENABLE_UTEST_MODULE")
        self.stest_exec: STestExecuteParam = STestExecuteParam(args=args, enable_binary_cache=False)
        self.stest: TestsFilterParam = TestsFilterParam(argv=args.stest, opt="ENABLE_STEST")
        self.stest_group: TestsFilterParam = TestsFilterParam(argv=args.stest_group, opt="ENABLE_STEST_GROUP")
        self.stest_distributed: TestsFilterParam = TestsFilterParam(
            argv=args.stest_distributed, opt="ENABLE_STEST_DISTRIBUTED"
        )
        self.models: TestsFilterParam = TestsFilterParam(argv=args.models)
        self.example: TestsFilterParam = TestsFilterParam(argv=args.example)

    def __str__(self) -> str:
        """Return string representation of test parameters

        :return: Formatted test parameter string
        :rtype: str
        """
        if not self.enable:
            return ""
        desc = "\nTests"
        desc += f"{self.exec}"
        if self.utest.enable:
            desc += "\n    Utest"
            desc += f"\n                     Enable : {self.utest.enable}"
            desc += f"\n                     Filter : {self.utest.filter_str}"
        if self.stest.enable or self.stest_distributed.enable:
            desc += "\n    Golden"
            desc += f"\n                      Clean : {self.golden.clean}"
            desc += f"\n                       Path : {self.golden.path}"
            desc += "\n    Stest Execute"
            desc += f"\n                     Device : {self.stest_exec.auto_execute_device_id}"
            desc += f"\n                   DumpJson : {self.stest_exec.dump_json}"
            desc += f"\n         Interpreter Config : {self.stest_exec.interpreter_config}"
            desc += f"\n        Enable Binary Cache : {self.stest_exec.enable_binary_cache}"
        if self.stest.enable:
            desc += "\n    Stest"
            desc += f"\n                     Enable : {self.stest.enable}"
            desc += f"\n                     Filter : {self.stest.filter_str}"
            desc += f"\n                     Group  : {self.stest_group.filter_str}"
        if self.stest_distributed.enable:
            desc += "\n    Stest Distributed"
            desc += f"\n                     Enable : {self.stest_distributed.enable}"
            desc += f"\n                     Filter : {self.stest_distributed.filter_str}"
        if self.models.enable:
            desc += "\n    Models"
            desc += f"\n                     Enable : {self.models.enable}"
            desc += f"\n                     Filter : {self.models.filter_str}"
        if self.example.enable:
            desc += "\n    Example"
            desc += f"\n                     Enable : {self.example.enable}"
            desc += f"\n                     Filter : {self.example.filter_str}"
        return desc

    @property
    def enable(self) -> bool:
        """Check if any test is enabled

        :return: True if any type of test is enabled
        :rtype: bool
        """
        tests_enable = self.utest.enable or self.stest.enable or self.stest_distributed.enable
        return tests_enable or self.example.enable or self.models.enable

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """Register all test-related command-line arguments

        Registers test execution, Golden test, filter options, and other arguments to the argument parser.

        :param parser: ArgumentParser instance
        :param ext: Extension information (subcommand parser)
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
    """Build process control class

    This class contains control flags/parameters specified or parsed from the command line
    to control build process execution. It serves as the entry point and controller for the
    entire build flow, responsible for coordinating the whole build process.
    """

    _PYTHONPATH: str = "PYTHONPATH"

    def __init__(self, args):
        """Initialize BuildCtrl instance

        Parses and initializes all build-related configuration from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.clean: bool = args.clean  # Force clean Build-Tree and Install-Tree marker
        # Timeout duration
        self.origin_timeout: Optional[int] = args.timeout if args.timeout and args.timeout > 0 else None
        self.remain_timeout: Optional[int] = self.origin_timeout
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_dir_file: Path = self.src_root / "build_dir.json"
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "build_out")
        self.feature: FeatureParam = FeatureParam(args=args)
        self.build: BuildParam = BuildParam(args=args)
        self.tests: TestsParam = TestsParam(args=args)
        third_party_path = args.third_party_path or os.environ.get("PYPTO_THIRD_PARTY_PATH")
        self.third_party_path: Optional[Path] = Path(third_party_path).resolve() if third_party_path else None
        self.verbose: bool = args.verbose
        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError("Can't find cmake")
        # Indicates whether pip version supports passing --config-setting (PEP standard argument passing)
        self.pip_dependence_desc: Dict[str, str] = {"pip": ">=22.1"}
        self.pip_support_config_setting = self.check_pip_dependencies(
            deps=self.pip_dependence_desc, raise_err=False, log_err=False
        )
        # Used to unify the build timestamp for run and whl
        timestamp = ""
        self.tag_info: str = os.environ.get('tagInfo')
        if self.tag_info:
            parts = self.tag_info.split('_')
            if len(parts) >= 4:
                timestamp = '_'.join(parts[-3:-1])
        if not timestamp:
            # tagInfo format must be prefix_date_time_suffix (at least 4 segments), consistent with
            # generate_version_info.py in cmake common repo and setup.py, both using split('_')[-3:-1]
            # to extract date_time. If the external environment variable content does not match,
            # regenerate it.
            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S%f')[:-3]
            self.tag_info = f"pypto_{timestamp}_build"

    def __str__(self) -> str:
        """Return string representation of build control parameters

        :return: Formatted build control parameter string
        :rtype: str
        """
        py3_ver = sys.version_info
        pip_ver = metadata.version("pip")
        desc = ""
        desc += "\nEnviron"
        desc += f"\n    Python3                 : {sys.executable} ({py3_ver.major}.{py3_ver.minor}.{py3_ver.micro})"
        desc += f"\n    pip3                    : {pip_ver}"
        desc += "\nPath"
        desc += f"\n    Source  Dir             : {self.src_root}"
        desc += f"\n    Build   Dir             : {self.build_root}"
        desc += f"\n    Install Dir             : {self.install_root}"
        desc += f"\n    3rd     Dir             : {self.third_party_path}"
        desc += "\nFlag"
        desc += f"\n    Clean                   : {self.clean}"
        desc += f"\n    Verbose                 : {self.verbose}"
        desc += "\nOthers"
        desc += f"\n    Timeout                 : {self.origin_timeout}"
        desc += f"\n    TagInfo                 : {self.tag_info}"
        desc += f"{self.feature}"
        desc += f"{self.build}"
        desc += f"{self.tests}"
        desc += "\n"
        return desc

    @staticmethod
    def which_cmake() -> Optional[Path]:
        """Find system-level CMake executable path

        This function exists to avoid interference from the cmake pip package;
        otherwise calling cmake directly in Python would invoke the cmake pip package.
        Searches for ELF-format CMake executables by traversing PATH environment variable directories.

        :return: Absolute path to system-level cmake executable, returns None if not found
        :rtype: Optional[Path]
        """
        if _find_system_cmake:
            return _find_system_cmake()
        return None

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument(
            "-c",
            "--clean",
            action="store_true",
            default=False,
            help="clean, clean Build-Tree and Install-Tree before build.",
        )
        parser.add_argument("--timeout", nargs="?", type=int, default=None, help="Total timeout.")
        parser.add_argument(
            "--cann_3rd_lib_path",
            "--third_party_path",
            nargs="?",
            type=str,
            default="",
            dest="third_party_path",
            help="Specify 3rd Libraries Path",
        )
        parser.add_argument("--verbose", action="store_true", default=False, help="verbose, enable verbose output.")

    @staticmethod
    def _resolve_ascend_cann_package_path() -> str:
        """Resolve ASCEND_CANN_PACKAGE_PATH, priority consistent with build.sh:
        CANN_PATH > ASCEND_HOME_PATH > ASCEND_OPP_PATH > default path
        """
        for env_key in ("CANN_PATH", "ASCEND_HOME_PATH"):
            val = os.environ.get(env_key)
            if val:
                return val
        opp_path = os.environ.get("ASCEND_OPP_PATH")
        if opp_path:
            return os.path.dirname(opp_path)
        if platform.machine() in ("aarch64", "armv7l"):
            default = os.path.join(os.path.expanduser("~"), "Ascend", "cann")
        else:
            default = "/usr/local/Ascend/cann"
        return default if os.path.isdir(default) else ""

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
        # Calculate total duration
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
        """Main processing flow"""
        parser = argparse.ArgumentParser(description="PyPTO Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # Subcommands
        # Register arguments
        FeatureParam.reg_args(parser=parser)
        BuildParam.reg_args(parser=parser)
        TestsParam.reg_args(parser=parser, ext=sub_parser)
        BuildCtrl.reg_args(parser=parser)

        # Parse arguments
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        # Process flow
        if ctrl.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        # Distinguish between python3 frontend and cpp frontend
        logging.info("%s", ctrl)
        if ctrl.feature.frontend_type_python3:
            logging.info("Front-end(python3), start process")
            ctrl.py_clean()
            ctrl.py_build()
            ctrl.py_build_run()
            ctrl.py_tests()
        else:
            logging.info("Front-end(cpp), start process with CMake")
            if 'func' in args:
                args.func(args=args, ctrl=ctrl)
            ctrl.cmake_clean()
            ctrl.cmake_configure()
            ctrl.cmake_build()

    def run_build_cmd(
        self, cmd: str, update_env: Optional[Dict[str, str]] = None, check: bool = True, pg_desc: str = "CMake"
    ) -> Tuple[subprocess.CompletedProcess, str]:
        """Execute a specific build command

        This function is used instead of calling subprocess.run directly for the following reasons:
            1. Supports multi-target builds, where build duration shares a common timeout configuration;
            2. In UTest/STest parallel execution scenarios, the process invocation chain is:
                   build_ci.py(main process) -> process1(CMake) -> process2(CMake Generator, make/ninja)
                   -> process3(Python) -> process4(exe)
               If process1 times out, it needs to notify all child/grandchild processes to terminate

        This function supports timeout recalculation, which only occurs on successful execution.

        :param cmd: Build command
        :param update_env: Environment variables (additional updates)
        :param check: Check return value
        :param pg_desc: Process Group Description
        """

        def _stop_pg(_msg: str, _p: subprocess.Popen):
            """Notify all child/grandchild processes to terminate via SIGINT signal;
            the Python parallel script catches this signal for finalization processing"""
            _pgid = os.getpgid(_p.pid)
            logging.info("%s. Send terminate event to %s[%s]", _msg, pg_desc, _pgid)
            os.killpg(_pgid, signal.SIGINT)

        ts = datetime.now(tz=timezone.utc)
        stdout = None
        stderr = None
        env = os.environ.copy()
        env.update(update_env if update_env else {})
        with subprocess.Popen(
            shlex.split(cmd), env=env, text=True, encoding='utf-8', start_new_session=True
        ) as process:
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
        # Update remaining timeout
        duration = self._duration(ts=ts)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr), duration

    def get_cfg_cmd(self, ext: Optional[Any] = None) -> str:
        """Generate CMake Configure command

        BuildCtrl class does not directly generate CMake configuration commands, returns empty string.

        :param ext: Extension information, unused
        :type ext: Optional[Any]
        :return: Empty string
        :rtype: str
        """
        return ""

    def get_cfg_update_env(self) -> Dict[str, str]:
        """Get environment variables for CMake Configure stage

        Generates environment variables to be passed to CMake Configure based on configuration.

        :return: Environment variable dictionary
        :rtype: Dict[str, str]
        """
        env = {}
        if self.third_party_path:
            env.update({"PYPTO_THIRD_PARTY_PATH": str(self.third_party_path)})
        # Unify build_timestamp in run and whl via tag_info environment variable
        env.update({"tagInfo": self.tag_info})
        # Prevent direct pip install . from overwriting existing PyPTO in cann package
        # via PYPTO_ALLOW_WHL_BUILD environment variable
        env.update({"PYPTO_ALLOW_WHL_BUILD": "1"})
        return env

    def get_cmake_build_update_env(self) -> Dict[str, str]:
        """Get environment variables for CMake Build stage

        Generates environment variables to be passed to CMake Build and test execution based on configuration.

        :return: Environment variable dictionary
        :rtype: Dict[str, str]
        """
        env = {}
        if self.build.job_num:
            env["PYPTO_TESTS_PARALLEL_NUM"] = str(self.build.job_num)
        if self.build.cov_incr:
            env["PYPTO_BUILD_COV_INCREMENT"] = "True"
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

    def pip_install(
        self, whl: Path, dest: Optional[Path] = None, opt: str = "", update_env: Optional[Dict[str, str]] = None
    ):
        """Install the specified whl package

        Uses pip command to install the specified whl package, supporting custom installation path and arguments.

        :param whl: whl package file path
        :type whl: Path
        :param dest: Installation path, uses default path when not specified
        :type dest: Optional[Path]
        :param opt: Additional installation arguments
        :type opt: str
        :param update_env: Environment variables (additional updates)
        :type update_env: Optional[Dict[str, str]]
        """
        edit_str = "-e " if self.feature.whl_editable else ""
        cmd = f"{sys.executable} -m pip install {edit_str}" + f"{whl} {opt}" + (" -vvv " if self.verbose else "")
        cmd += f" --target={dest}" if dest else ""
        cmd += " --break-system-packages" if self.feature.whl_break_system_packages else ""
        logging.info("Install %s, Cmd: %s, Timeout: %s secs", whl, cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pip")
        logging.info("Install %s%s success, %s", whl, f" to {dest}" if dest else "", duration)

    def pip_uninstall(self, name: str, path: Optional[Path] = None):
        """Uninstall the specified whl package

        Depending on whether an installation path is specified, either uses pip uninstall or directly deletes files.

        :param name: Package name
        :type name: str
        :param path: Specified installation path, if specified then directly deletes files under that path
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
            cmd += " --break-system-packages" if self.feature.whl_break_system_packages else ""
            logging.info("Uninstall %s package, Cmd: %s, Timeout: %s secs", name, cmd, self.remain_timeout)
            _, _ = self.run_build_cmd(cmd=cmd, pg_desc="pip")
        logging.info("Uninstall %s package%s success", name, f" from {path}" if path else "")

    def cmake_clean(self):
        """Clean CMake build intermediate results

        Cleans build tree, install tree contents, and ast data cache. Only executes when clean flag is True.
        """
        if self.clean:
            if self.build_root.exists():
                logging.info("Clean Build-Tree(%s)", self.build_root)
                shutil.rmtree(self.build_root)
            if self.install_root.exists():
                logging.info("Clean Install-Tree(%s)", self.install_root)
                cann_install_root = self.install_root / "cann"
                if cann_install_root.exists():
                    logging.info("Restore owner write permission for Install-Tree(%s)", cann_install_root)
                    subprocess.run(["chmod", "-R", "u+w", str(cann_install_root)], check=True)
                shutil.rmtree(self.install_root)
            home_dir = os.environ.get('HOME')
            astdata_folder = os.path.join(home_dir, 'ast_data')
            if os.path.exists(astdata_folder):
                logging.info("Clean ast data cache folder(%s)", astdata_folder)
                shutil.rmtree(astdata_folder)

    def py_clean(self):
        """Clean Python frontend build intermediate results

        Cleans CMake build directory, Python cache files, output directory, etc.
        Only performs additional cleaning when clean flag is True.
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
            Path(pkg_src, "lib"),  # editable mode
            self.build_dir_file,  # Python GCov scenario
        ]
        so_glob = pkg_src.glob(pattern="*.so")
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
        # Clean online compilation runtime cache
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        runtime_cache = cache_home / "cann" / "pypto"
        if runtime_cache.exists():
            logging.info("Clean runtime cache(%s)", runtime_cache)
            shutil.rmtree(runtime_cache)

    def cmake_configure(self):
        """Execute CMake Configure stage

        Generates CMake build configuration, including setting generator, Python interpreter path,
        compilation options, etc.
        """
        # Basic configuration: CMake internally calls python3, so pass the python3 interpreter
        # to ensure consistent python3 version usage
        cmd = f"{self.cmake} -S {self.src_root} -B {self.build_root}"
        cmd += f" -G {self.build.generator}" if self.build.generator else ""
        cmd += f" -DPython3_EXECUTABLE={sys.executable}"
        cmd += f" -DASCEND_CANN_PACKAGE_PATH={self._resolve_ascend_cann_package_path()}"
        cmd += self.feature.get_cfg_cmd()
        cmd += self.build.get_cfg_cmd()
        cmd += self.tests.get_cfg_cmd()
        # Execute
        update_env = self.get_cfg_update_env()
        update_env["CCACHE_BASEDIR"] = str(self.src_root)
        logging.info("CMake Configure, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env)
        logging.info("CMake Configure success, %s", duration)

    def cmake_build(self):
        """Execute CMake Build stage

        Executes the actual compilation process based on BuildParam configuration,
        supporting multi-target builds.
        """
        update_env = self.get_cmake_build_update_env()
        update_env["CCACHE_BASEDIR"] = str(self.src_root)
        self._prepare_cann_device_artifacts(update_env=update_env)
        cmd_list = self.build.get_build_cmd_lst(cmake=self.cmake, binary_path=self.build_root)
        for i, c in enumerate(cmd_list, start=1):
            c += " --verbose" if self.verbose else ""
            logging.info("CMake Build(%s/%s), Cmd: %s, Timeout: %s secs", i, len(cmd_list), c, self.remain_timeout)
            try:
                _, duration = self.run_build_cmd(cmd=c, update_env=update_env)
            except subprocess.CalledProcessError as e:
                logging.info("CMake Build(%s/%s) failed, ERROR CODE: %s", i, len(cmd_list), e.returncode)
                raise e
            logging.info("CMake Build(%s/%s) success, %s", i, len(cmd_list), duration)

    def py_build(self):
        """whl package compilation

        Supports two compilation modes:
            1. Official build: Calls the build library to trigger setuptools (bdist_wheel command),
               which in turn triggers CMake to complete compilation
            2. pip build: Calls pip install command to trigger setuptools (editable_wheel command),
               which in turn triggers CMake to complete compilation
               pip build has two modes:
               - Regular install: Suitable for production or stable code, modifications to source
                 code after installation are not reflected in the installed package
               - Editable install: Facilitates development and debugging, creates a link in
                 site-packages pointing to the local directory, so Python source code changes
                 take effect immediately without reinstallation
        """
        update_env = self.get_cfg_update_env()
        if self._use_pip_install_mode() or self.feature.whl_editable:
            opt = " --no-compile --no-deps"
            opt += " --no-build-isolation" if not self.feature.whl_isolation else ""

            cmd_config_setting, env_config_setting = self._get_setuptools_build_ext_config_setting()
            if self.feature.whl_editable:
                update_env["PYPTO_BUILD_EXT_ARGS"] = env_config_setting
            else:
                if self.pip_support_config_setting:
                    opt += f" {cmd_config_setting}" if cmd_config_setting else ""
                else:
                    # Older pip versions lack --config-setting argument, pass via environment variable instead
                    update_env["PYPTO_BUILD_EXT_ARGS"] = env_config_setting

            # Reinstall whl package
            dist = self._get_pip_install_dist()
            self.pip_uninstall(name=self.feature.whl_name, path=dist)
            self.pip_install(whl=self.src_root, dest=dist, opt=opt, update_env=update_env)
        else:
            # Check if build package version meets requirements; this check is placed here because
            # the build-system.requires check in pyproject.toml is implemented by the build package itself,
            # so writing it in pyproject.toml cannot perform early validation
            self.check_pip_dependencies(deps={"build": ">=1.0.3"}, raise_err=True, log_err=True)
            cmd = f"{sys.executable} -m build --outdir={self.install_root}"
            cmd += " --no-isolation" if not self.feature.whl_isolation else ""
            cmd += f" {self._get_setuptools_bdist_wheel_config_setting()}"
            logging.info("Build whl, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
            _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="build")
            logging.info("Build whl success, %s", duration)

    def py_build_run(self):
        """Pack whl into run package

        Flow:
            1. Find the whl file already produced by py_build()
            2. cmake configure: in a separate binary directory (build_run), enable ENABLE_FEATURE_PACKING_WHL_INTO_RUN
            3. cpack: generate .run self-extracting installer
        """
        # Condition filter
        if self.feature.just_build_whl:
            return
        if self._use_pip_install_mode() or self.feature.whl_editable:
            logging.warning("Packing whl into run is not supported in pip-install or editable mode, skip.")
            return
        whl_file = self._find_match_whl(name=self.feature.whl_name, path=self.install_root)
        if not whl_file:
            raise RuntimeError(
                f"Can't find {self.feature.whl_name} whl file from {self.install_root}, please run py_build first."
            )
        # Pack run
        run_build_root = Path(self.build_root, "run")
        run_build_root.mkdir(parents=True, exist_ok=True)
        cmd = f"{self.cmake} -S {self.src_root} -B {run_build_root}"
        cmd += " -DENABLE_FEATURE_PACKING_WHL_INTO_RUN=ON"
        cmd += f" -DWHL_FILE_PATH={whl_file}"
        cmd += f" -DRUN_OUTPUT_DIR={self.install_root}"
        cmd += f" -DPACKAGE_TYPE={self.feature.package_type}"
        cmd += f" -DASCEND_CANN_PACKAGE_PATH={self._resolve_ascend_cann_package_path()}"
        cmd += f" -DPYPTO_THIRD_PARTY_PATH={self.third_party_path}" if self.third_party_path else ""
        update_env = self.get_cfg_update_env()
        logging.info("CMake Configure(run), Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="cmake-configure-run")
        logging.info("CMake Configure(run) success, %s", duration)

        cmd = f"{self.cmake} --build {run_build_root} --target package"
        cmd += " --verbose" if self.verbose else ""
        logging.info("CMake Build(run) package, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="cmake-build-package")
        logging.info("CMake Build(run) package success, %s", duration)

    def py_tests(self):
        """Execute Python frontend tests

        Includes unit tests (UTest), system tests (STest), model tests (Models), and example tests (Examples).
        If pip install mode is not used, the whl package will be uninstalled and reinstalled first.
        """
        tests_enable = self.tests.utest.enable or self.tests.stest.enable
        if not tests_enable and not self.tests.example.enable and not self.tests.models.enable:
            return
        dist = self._get_pip_install_dist()
        if not self._use_pip_install_mode():
            # Reinstall the corresponding whl package
            self.pip_uninstall(name=self.feature.whl_name, path=dist)  # Uninstall whl package
            whl = self._find_match_whl(name=self.feature.whl_name, path=dist)  # Find whl package
            if not whl:
                raise RuntimeError(f"Can't find {self.feature.whl_name} whl file from {dist}")
            self.pip_install(whl=whl, dest=dist, opt="--no-compile --no-deps")  # Install whl package

        # Execute test cases, UTest
        # In Python 3.12, pytest-xdist creates subprocesses via os.fork() which produces DeprecationWarning.
        # Use -W ignore::DeprecationWarning to suppress this warning.
        if self.build.job_num is not None and self.build.job_num > 0:
            n_workers = str(self.build.job_num)
        else:
            n_workers = "auto"
        self.py_tests_run_pytest(
            dist=dist,
            params=self._get_python_utest_params(),
            ext=f"-n {n_workers} -W ignore::DeprecationWarning",
        )

        # Execute test cases, Models/STest, supports mixed execution
        dev_lst = [int(d) for d in self.tests.stest_exec.auto_execute_device_id.split(":")]
        dev_ext = " ".join(f"{d}" for d in dev_lst)
        ext_str = f"-n {len(dev_lst)} --device {dev_ext}"
        self.py_tests_run_pytest(
            dist=dist, params=[(self.tests.models, "models"), (self.tests.stest, "python/tests/st")], ext=ext_str
        )
        # Execute multi-card cases, distinguished by world_size
        for cards_per_case in [4, 16]:
            if cards_per_case <= 1 or cards_per_case > len(dev_lst):
                continue
            # Grouping strategy: one worker corresponds to one group of cards
            n_workers = len(dev_lst) // cards_per_case
            needed_devices = n_workers * cards_per_case
            used_dev_lst = dev_lst[:needed_devices]
            used_dev_ext = " ".join(f"{d}" for d in used_dev_lst)
            ext_str = f'-n {n_workers} --device {used_dev_ext} --cards-per-case {cards_per_case} -m "world_size"'
            self.py_tests_run_pytest(
                dist=dist,
                params=[
                    (self.tests.models, "python/tests/st"),
                ],
                ext=ext_str,
            )

        # Execute test cases, Examples
        dev_ext_comma = ",".join(f"{d}" for d in dev_lst)
        self.py_run_examples(
            dist=dist,
            tests=self.tests.example,
            def_filter=str(Path(self.src_root, "examples")),
            dev_ext_comma=dev_ext_comma,
            n_workers=n_workers,
        )

        # Generate coverage report (if gcov or py_cov is enabled)
        if self.build.gcov or self.build.py_cov:
            self._py_generate_coverage()

    def py_tests_run_pytest(self, dist: Optional[Path], params: List[Tuple[TestsFilterParam, str]], ext: str = ""):
        """Invoke pytest to execute test cases

        Supports mixed execution of test cases across multiple paths, with configurable parallel execution.

        :param dist: Binary distribution package installation path
        :type dist: Optional[Path]
        :param params: Parameter list, supports mixed execution across multiple paths,
                       each element is (TestsFilterParam, test path)
        :type params: List[Tuple[TestsFilterParam, str]]
        :param ext: Extended command arguments
        :type ext: str
        """
        # Filter processing
        filter_str = ""
        for cur_tests, cur_filter_str in params:
            cur_filter_str = cur_tests.get_filter_str(def_filter=cur_filter_str)
            if cur_filter_str:
                filter_str += f" {cur_filter_str}"
        if not filter_str:
            return
        # Execute pytest
        self._py_tests_run_pytest(dist=dist, filter_str=filter_str, ext=ext)

    def py_run_examples(
        self,
        dist: Optional[Path],
        tests: TestsFilterParam,
        def_filter: str,
        dev_ext_comma: str = "0",
        n_workers: str = "auto",
    ):
        """Run example test cases

        Determines execution mode (NPU or SIM) based on backend_type,
        supporting device allocation and timeout control.

        :param dist: Binary distribution package installation path
        :type dist: Optional[Path]
        :param tests: Test filter parameters
        :type tests: TestsFilterParam
        :param def_filter: Default filter condition
        :type def_filter: str
        :param dev_ext_comma: Device ID list (comma-separated)
        :type dev_ext_comma: str
        :param n_workers: Number of parallel workers
        :type n_workers: str
        """
        if not tests.enable:
            return
        if not self.tests.exec.auto_execute:
            return
        # Filter processing
        filter_str = tests.get_filter_str(def_filter=def_filter).replace(',', ' ')

        # Determine execution mode based on backend_type
        update_env = self._get_py_tests_update_env(dist=dist)
        # Get case_timeout argument
        case_timeout = self.tests.exec.case_execute_timeout
        timeout_arg = f" --timeout {case_timeout}" if case_timeout and case_timeout > 0 else ""

        if self.feature.backend_type == "npu":
            # NPU mode
            cmd = f"{sys.executable} examples/validate_examples.py -t {filter_str} -d {dev_ext_comma}{timeout_arg}"
            logging.info("examples --run_mode npu, Cmd: %s", cmd)
            ret, duration = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
            ret.check_returncode()
            logging.info("examples --run_mode npu, Cmd: %s, Duration %s sec", cmd, duration)
        else:
            # SIM mode
            n_workers_val = int(n_workers) if n_workers != "auto" else 16
            cmd = f"{sys.executable} examples/validate_examples.py -t {filter_str} \
                  --run_mode sim -w {n_workers_val}{timeout_arg} --no-serial-fallback"
            logging.info("examples --run_mode sim, Cmd: %s", cmd)
            ret, duration = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
            ret.check_returncode()
            logging.info("examples --run_mode sim, Cmd: %s, Duration %s sec", cmd, duration)

    def _prepare_cann_device_artifacts(self, update_env: Dict[str, str]):
        """Build the independent device project and stage its runtime files for C++ tests."""
        if self.feature.backend_type != "npu":
            return

        target_help = subprocess.run(
            [str(self.cmake), "--build", str(self.build_root), "--target", "help"],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            env={**os.environ, **update_env},
        )
        has_device_target = target_help.returncode == 0 and any(
            "cann_device" in line.replace(":", " ").split() for line in target_help.stdout.splitlines()
        )
        if not has_device_target:
            return

        cmd = f"{self.cmake} --build {self.build_root} --target cann_device"
        cmd += f" -j {self.build.job_num}" if self.build.job_num else ""
        cmd += " --verbose" if self.verbose else ""
        logging.info("CMake Device Build, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="cmake-build-device")
        logging.info("CMake Device Build success, %s", duration)

        device_package = self.build_root / "device_build" / "device-pypto.tar.gz"
        if not device_package.exists():
            raise RuntimeError(f"Can't find device package: {device_package}")
        output_root = self.build_root / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        cmd = f"tar -zxf {shlex.quote(str(device_package))} -C {shlex.quote(str(output_root))}"
        logging.info("Extract device package for tests, Cmd: %s", cmd)
        self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="extract-device-package")

    def _find_match_whl(self, name: str, path: Path) -> Optional[Path]:
        """
        Find the matching whl package file in the specified path

        :param name: Package name
        :type name: str
        :param path: Specified path
        :type path: Path
        :return: Matching whl file path, or None if not found
        :rtype: Path | None
        """
        pytag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abitag = pytag
        if self.feature.whl_py3_abi_tag:
            pytag = f"cp{self.feature.whl_py3_abi_tag}"
            abitag = "abi3"
        tags = f"{pytag}-{abitag}"

        pattern = f"{name}-*-{tags}-*.whl"
        whl_glob = path.glob(pattern=pattern)
        whl_files = [Path(f) for f in whl_glob]
        whl_file = whl_files[0] if whl_files else None
        if whl_file:
            logging.info("Success find match %s from %s", whl_file, path)
        else:
            logging.error("Failed to find match %s whl from %s, pattern=%s", name, path, pattern)
        return whl_file

    def _py_tests_run_pytest(self, dist: Optional[Path], filter_str: str, ext: str = ""):
        if not self.tests.exec.auto_execute:
            return
        # Filter processing
        filter_str = filter_str.replace(',', ' ')
        # Build command
        cmd = f"{sys.executable} -m pytest {filter_str} -v --durations=0 -s --capture=no"
        cmd += f" --rootdir={self.src_root} {ext} --forked"
        if self.check_pip_dependencies(deps={"pytest-xdist": ">=3.8.0"}, raise_err=False, log_err=False):
            cmd += " --no-loadscope-reorder"
        # Execute command
        update_env = self._get_py_tests_update_env(dist=dist)
        # Enable Python coverage data collection (requires pytest-cov, triggered only by --py_cov)
        if self.build.py_cov and self.check_pip_dependencies(
            deps={"pytest-cov": ">=4.0.0"}, raise_err=False, log_err=False
        ):
            cov_data_file = self._get_py_cov_data_file()
            if cov_data_file is not None:
                cmd += " --cov=pypto"
                update_env["COVERAGE_FILE"] = str(cov_data_file)
        logging.info("pytest run, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pytest")
        logging.info("pytest run success, %s", duration)

    def _get_py_tests_update_env(self, dist: Optional[Path]) -> Dict[str, str]:
        update_env = {}

        if dist:
            origin_env = os.environ.copy()
            ori_env_python_path = origin_env.get(self._PYTHONPATH, "")
            act_env_python_path = f"{dist}:{ori_env_python_path}" if ori_env_python_path else f"{dist}"
            update_env.update({self._PYTHONPATH: act_env_python_path})
        update_env.update(self._py_tests_get_xsan_env())
        return update_env

    def _py_tests_get_xsan_env(self) -> Dict[str, str]:
        update_env = {}
        if not (self.build.asan or self.build.ubsan):
            return update_env
        logging.warning("ASAN/UBSAN support in WHL package scenarios is experimental - use with caution.")

        py3_ver = sys.version_info
        dir_name = f"temp.linux-{self.build.get_system_processor()}-cpython-{py3_ver.major}{py3_ver.minor}"
        xsan_config_file = Path(self.build_root, dir_name, "_pypto_xsan_config.txt")
        if not xsan_config_file.exists():
            logging.warning("XSAN config file not found: %s", xsan_config_file)
            return update_env

        with open(xsan_config_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                update_env[k] = v
        for k, v in update_env.items():
            logging.info("%s=%s", k, v)
        return update_env

    def _get_python_utest_params(self) -> List[Tuple[TestsFilterParam, str]]:
        """Generate Python UTest pytest params list based on --utest_module argument

        Maps module names specified by utest_module to subdirectory paths under python/tests/ut,
        each module corresponds to a params entry, supporting mixed execution of multiple modules.

        :return: pytest params list, each element is (TestsFilterParam, path)
        :rtype: List[Tuple[TestsFilterParam, str]]
        """
        base = "python/tests/ut"
        mod = self.tests.utest_module.filter_str
        if not mod or mod in ("ON", "OFF"):
            return [(self.tests.utest, base)]
        modules = [m.strip() for m in mod.replace(":", ",").split(",") if m.strip()]
        return [(self.tests.utest, f"{base}/{m}") for m in modules]

    def _get_py_cov_data_file(self) -> Optional[Path]:
        """Get Python coverage data file path (consistent with gen_coverage.py data_dir)"""
        if not self.build_dir_file.exists():
            return None
        with open(self.build_dir_file, 'r', encoding='utf-8') as f:
            marker = json.load(f)
        build_dir = Path(marker["cmake_binary_dir"]).resolve()
        return Path(build_dir, ".coverage")

    def _py_generate_coverage(self):
        """Generate coverage report

        This function reads the build_dir marker file generated by setup.py and calls
        gen_coverage.py to generate the coverage report.
        Supports independent or combined generation of C++ (--gcov) and Python (--py_cov) coverage.

        Flow:
            1. Read .pypto_build_dir.json -> get CMake build directory
            2. Read gcov_config.json -> get C++ filter_dirs (only present with --gcov)
            3. Call gen_coverage.py -> generate coverage report

        Notes:
            - Must be called after pytest execution (when .gcda / .coverage files are generated)
            - --gcov: Generate C++ coverage (requires CMake ENABLE_GCOV=ON compilation instrumentation)
            - --py_cov: Generate Python coverage (requires pytest-cov, pytest has collected data via --cov)
            - --cov_increment (or --gcov_increment): Incremental coverage (applies to both C++ and Python)
        """
        # 1. Read build_dir marker file
        if not self.build_dir_file.exists():
            logging.warning("Build dir marker file not found: %s, skip coverage generation", self.build_dir_file)
            return

        with open(self.build_dir_file, 'r', encoding='utf-8') as f:
            marker = json.load(f)
        build_dir = Path(marker["cmake_binary_dir"]).resolve()

        # 2. Read gcov config file (only C++ coverage needs filter_dirs; may not exist when --py_cov is used alone)
        filter_dirs = []
        config_file = build_dir / "gcov_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            filter_dirs = config.get("filter_dirs", [])
        elif self.build.gcov:
            logging.warning("GCov config file not found: %s, skip coverage generation", config_file)
            return

        # 3. Build gen_coverage.py arguments
        gen_cov_py = self.src_root / "cmake/scripts/gen_coverage.py"

        cmd = f"{sys.executable} {gen_cov_py} -s={self.src_root} -d={build_dir} "
        for filter_dir in filter_dirs:
            cmd += f" -f={filter_dir}"
        cmd += " -i" if self.build.cov_incr else ""  # Incremental coverage
        cmd += " --py_cov" if self.build.py_cov else ""  # Python coverage
        cmd += " --gcov" if self.build.gcov else ""  # Generate C++ coverage

        # 4. Execute coverage generation
        logging.info("Generate coverage, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        ret, duration = self.run_build_cmd(cmd=cmd, check=True, pg_desc="gen_coverage")
        ret.check_returncode()
        logging.info("Generate coverage success, %s", duration)

    def _tests_enable(self) -> bool:
        return self.tests.utest.enable or self.tests.stest.enable

    def _use_pip_install_mode(self) -> bool:
        return self.tests.utest.enable or self.tests.stest.enable

    def _get_pip_install_dist(self) -> Optional[Path]:
        # pip install -e scenario requires installing to the default site-packages path
        # (conflicts with --target argument logic); other scenarios install to custom directory
        return None if self._use_pip_install_mode() and self.feature.whl_editable else self.install_root

    def _get_setuptools_build_ext_config_setting(self) -> Tuple[str, str]:
        cmake_args = f"{self.feature.get_cfg_cmd()} {self.build.get_cfg_cmd(ext=False)}"
        cann_path = self._resolve_ascend_cann_package_path()
        if cann_path:
            cmake_args += f' -DASCEND_CANN_PACKAGE_PATH="{cann_path}"'
        env_setting = ""
        env_setting += f" --cmake-generator={self.build.generator}" if self.build.generator else ""
        env_setting += f" --cmake-build-type={self.build.build_type}" if self.build.build_type else ""
        env_setting += f" --cmake-options=\"{cmake_args}\"" if cmake_args else ""
        env_setting += f" --backend-type={self.feature.backend_type}"
        env_setting += " --cmake-verbose" if self.verbose else ""
        cmd_setting = ""
        if env_setting:
            cmd_setting = f" --config-setting=--build-option='build_ext {env_setting}'"
        return cmd_setting, env_setting

    def _get_setuptools_bdist_wheel_config_setting(self) -> str:
        cmd = f" bdist_wheel --plat-name={self.feature.whl_plat_name}" if self.feature.whl_plat_name else ""
        whl_py3_abi = self.feature.whl_py3_abi_tag
        if whl_py3_abi:
            cmd += f" --py-limited-api=cp{whl_py3_abi}"
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

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

"""Backend-specific compile configuration for PyPTO Pro JIT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

CCE_BACKEND = "cce"


@dataclass(frozen=True)
class KernelTarget:
    """Compiler target and its AIC/AIV participants per logical launch block.

    Core counts describe the emitted binary's ABI. Keeping them with the compiler
    target allows different architectures to use different mixed-core ratios,
    independently of the device or stream's currently available resources.
    """

    npu_arch: str
    aic_per_block: int
    aiv_per_block: int
    fat_object: bool


@dataclass(frozen=True)
class JitCompileConfig:
    """Compile settings owned by one PyPTO Pro JIT backend."""

    backend: str
    kernel_targets: Mapping[str, Mapping[str, KernelTarget]]
    memory_arch_flags: Mapping[str, str]
    arch_flags: tuple[str, ...]
    fatobj_flags: tuple[str, ...]
    common_flags: tuple[str, ...]
    print_debug_flags: tuple[str, ...]
    llvm_common_args: tuple[str, ...]
    llvm_arch_args: Mapping[str, tuple[str, ...]]
    runtime_include_dirs: tuple[str, ...]
    link_dirs: tuple[str, ...]
    link_libraries: tuple[str, ...]

    @staticmethod
    def _format_values(values: tuple[str, ...], variables: Mapping[str, str]) -> list[str]:
        try:
            return [value.format(**variables) for value in values]
        except KeyError as exc:
            raise RuntimeError(f"JIT compile config references unknown variable '{exc.args[0]}'") from exc

    def build_bisheng_flags(
        self,
        *,
        toolkit_home: str,
        arch: str,
        target: KernelTarget,
        enable_print_debug: bool,
    ) -> list[str]:
        arch = arch.strip().lower()
        variables = {
            "toolkit_home": toolkit_home,
            "mem_arch": self._resolve_memory_arch_flag(arch),
            "npu_arch": target.npu_arch,
        }
        common = self._format_values(self.common_flags, variables)
        if enable_print_debug:
            common.extend(self._format_values(self.print_debug_flags, variables))
        flags = self._format_values(self.arch_flags, variables)
        if target.fat_object:
            flags.extend(self._format_values(self.fatobj_flags, variables))
        return [*flags, *common]

    def build_llvm_args(self, arch: str) -> list[str]:
        arch = arch.strip().lower()
        arch_key = self._resolve_arch_key(arch)
        arch_args = self.llvm_arch_args.get(arch_key)
        if arch_args is None:
            raise RuntimeError(f"JIT compile config does not define llvm_arch_args.{arch_key}")
        return [*self.llvm_common_args, *arch_args]

    def runtime_include_flags(self, ascend_home_path: str) -> list[str]:
        variables = {"ascend_home": ascend_home_path}
        flags = [f"-I{include_dir}" for include_dir in self._format_values(self.runtime_include_dirs, variables)]
        import pypto_pro

        pypto_include = os.path.join(os.path.dirname(pypto_pro.__file__), "include")
        flags.append(f"-I{pypto_include}")
        return flags

    def runtime_link_args(self, ascend_home_path: str) -> list[str]:
        variables = {"ascend_home": ascend_home_path}
        link_args: list[str] = []
        for link_dir in self._format_values(self.link_dirs, variables):
            link_args.extend(["-L", link_dir])
        for library in self.link_libraries:
            link_args.append(library if library.startswith("-l") else f"-l{library}")
        return link_args

    def resolve_kernel_target(self, arch: str, *, has_cube: bool, has_vector: bool) -> KernelTarget:
        """Resolve once so compiler flags and launch geometry consume the same ABI."""
        if not (has_cube or has_vector):
            raise ValueError("Cannot compile a kernel without cube or vector code; add a target section")
        arch_key = self._resolve_arch_key(arch.strip().lower())
        arch_config = self.kernel_targets.get(arch_key)
        if arch_config is None:
            raise RuntimeError(f"JIT compile config does not define kernel_targets for arch '{arch}'")
        variant = "cube_vec" if has_cube and has_vector else "cube" if has_cube else "vec"
        target = arch_config.get(variant)
        if target is None:
            raise RuntimeError(f"JIT compile config does not define kernel_targets.{arch_key}.{variant}")
        return target

    def _resolve_memory_arch_flag(self, arch: str) -> str:
        arch_key = self._resolve_arch_key(arch)
        mem_arch = self.memory_arch_flags.get(arch_key)
        if mem_arch is None:
            raise RuntimeError(f"JIT compile config does not define memory_arch_flags for arch '{arch}'")
        return mem_arch

    @staticmethod
    def _resolve_arch_key(arch: str) -> str:
        return {"a2": "a2a3", "a3": "a2a3"}.get(arch, arch)


_DEFAULT_CCE_JIT_COMPILE_CONFIG = JitCompileConfig(
    backend=CCE_BACKEND,
    kernel_targets={
        "a2a3": {
            "cube_vec": KernelTarget("dav-c220", aic_per_block=1, aiv_per_block=2, fat_object=True),
            "cube": KernelTarget("dav-c220-cube", aic_per_block=1, aiv_per_block=0, fat_object=False),
            "vec": KernelTarget("dav-c220-vec", aic_per_block=0, aiv_per_block=1, fat_object=False),
        },
        "a5": {
            "cube_vec": KernelTarget("dav-c310", aic_per_block=1, aiv_per_block=2, fat_object=True),
            "cube": KernelTarget("dav-c310-cube", aic_per_block=1, aiv_per_block=0, fat_object=False),
            "vec": KernelTarget("dav-c310-vec", aic_per_block=0, aiv_per_block=1, fat_object=False),
        },
    },
    memory_arch_flags={
        "a2a3": "-DMEMORY_BASE",
        "a5": "-DREGISTER_BASE",
    },
    arch_flags=("--cce-aicore-arch={npu_arch}",),
    fatobj_flags=("--cce-fatobj-link",),
    common_flags=(
        "-fPIC",
        "-shared",
        "-xcce",
        "{mem_arch}",
        "-O3",
        "-std=c++17",
        "-I{toolkit_home}/include",
    ),
    print_debug_flags=(
        "--cce-enable-print",
        "-D_DEBUG",
        "-DCCEBlockMaxSize=1048576",
        "-DPTOAS_ENABLE_CCE_PRINT=1",
    ),
    llvm_common_args=(
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=false",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "--cce-auto-sync=off",
    ),
    llvm_arch_args={
        "a2a3": (
            "-O3",
            "--cce-disable-kernel-global-attr-check",
            "-Wno-parentheses-equality",
            "-Wno-unused-command-line-argument",
            "-Werror",
            "-Wno-cce-compat",
        ),
        "a5": (
            "-mllvm",
            "-tile-fusion-skip-reduceop-fusion=true",
            "-mllvm",
            "-tile-fusion-skip-legality-check=false",
            "-O3",
            "--cce-disable-kernel-global-attr-check",
            "--enable-pto-tile-fusion",
            "-Wno-parentheses-equality",
            "-Wno-unused-command-line-argument",
            "-Werror",
            "-Wno-c++20-extensions",
            "-Wno-cce-compat",
        ),
    },
    runtime_include_dirs=(
        "{ascend_home}/include",
        "{ascend_home}/pkg_inc",
        "{ascend_home}/pkg_inc/runtime",
        "{ascend_home}/pkg_inc/",
        "{ascend_home}/pkg_inc/profiling",
        "{ascend_home}/include/experiment/runtime",
        "{ascend_home}/include/experiment/msprof",
        "{ascend_home}/pkg_inc/runtime/runtime",
    ),
    link_dirs=("{ascend_home}/lib64/",),
    link_libraries=("runtime", "profapi", "ascendcl"),
)


def get_jit_compile_config(backend: str = CCE_BACKEND) -> JitCompileConfig:
    backend = backend.strip().lower()
    if backend != CCE_BACKEND:
        raise NotImplementedError(f"PyPTO Pro JIT currently only supports the CCE backend, got {backend!r}")
    return _DEFAULT_CCE_JIT_COMPILE_CONFIG

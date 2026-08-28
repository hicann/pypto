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
"""Shared utility for computing build parallelism (job_num) with memory awareness.

This module is loaded by-path from both setup.py and build_ci.py (same pattern as
_which_cmake.py) to avoid import-time dependency on an installed pypto package.
"""

import math
import multiprocessing
import os
from pathlib import Path
from typing import Optional

# Conservative estimate of peak RSS per cc1plus translation unit.
# Heavy C++ template instantiation (quantization, codegen, etc.) can reach ~2 GiB.
DEFAULT_MEM_PER_JOB_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GiB

# Environment variable for explicit override of build parallelism.
ENV_JOB_NUM: str = "PYPTO_BUILD_JOB_NUM"


def _read_cgroup_memory_limit() -> Optional[int]:
    """Read the cgroup memory limit in bytes (v2 then v1).

    :return: Memory limit in bytes, or None if unlimited / unavailable.
    :rtype: Optional[int]
    """
    # cgroup v2: /sys/fs/cgroup/memory.max (value "max" means no limit)
    v2_path = Path("/sys/fs/cgroup/memory.max")
    if v2_path.exists():
        raw = v2_path.read_text(encoding="utf-8").strip()
        if raw == "max":
            return None
        try:
            limit = int(raw)
            if limit > 0:
                return limit
        except ValueError:
            pass

    # cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    v1_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_path.exists():
        raw = v1_path.read_text(encoding="utf-8").strip()
        try:
            limit = int(raw)
            # cgroup v1 uses a very large sentinel (e.g. 2^63 - page) for "no limit"
            if 0 < limit < 2**62:
                return limit
        except ValueError:
            pass
    return None


def _read_system_memory() -> Optional[int]:
    """Read available system memory from /proc/meminfo (Linux only).

    :return: Available memory in bytes, or None if unavailable.
    :rtype: Optional[int]
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    mem_total = None
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024  # kB -> bytes
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                mem_total = int(parts[1]) * 1024  # kB -> bytes
    return mem_total


def _get_available_memory() -> Optional[int]:
    """Get available memory in bytes, preferring cgroup limits.

    :return: Available memory in bytes, or None if unavailable.
    :rtype: Optional[int]
    """
    cgroup_limit = _read_cgroup_memory_limit()
    if cgroup_limit:
        return cgroup_limit
    return _read_system_memory()


def _parse_env_int(env_var: str) -> Optional[int]:
    """Parse a positive integer from an environment variable.

    :param env_var: Environment variable name.
    :return: Parsed positive int, or None if unset / invalid.
    :rtype: Optional[int]
    """
    raw = os.environ.get(env_var)
    if not raw:
        return None
    try:
        val = int(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return None


def get_job_num(
    job_num: Optional[int] = None,
    generator: Optional[str] = None,
    max_cpu_ratio: float = 0.9,
    max_cpu: int = 128,
    mem_per_job_bytes: int = DEFAULT_MEM_PER_JOB_BYTES,
) -> Optional[int]:
    """Compute build parallelism considering CPU count and available memory.

    Priority (highest first):
        1. Explicit job_num from CLI (-j flag), if > 0
        2. PYPTO_BUILD_JOB_NUM environment variable
        3. min(cpu_based, memory_based, max_cpu) — memory-aware default
        4. None for Ninja generator (let Ninja decide)

    :param job_num: User-specified parallelism from CLI. None or <=0 means "auto".
    :type job_num: Optional[int]
    :param generator: CMake generator name. Ninja returns None (auto-decide).
    :type generator: Optional[str]
    :param max_cpu_ratio: Fraction of CPU count to use for the CPU-based default.
    :type max_cpu_ratio: float
    :param max_cpu: Hard cap on CPU-based parallelism.
    :type max_cpu: int
    :param mem_per_job_bytes: Estimated peak RSS per compilation unit (bytes).
    :type mem_per_job_bytes: int
    :return: Job count, or None for Ninja / automatic.
    :rtype: Optional[int]
    """
    # Ninja decides its own parallelism
    if generator and generator.strip().strip('"').lower() in ("ninja",):
        return None

    # Priority 1: explicit CLI flag
    if job_num and job_num > 0:
        return job_num

    # Priority 2: environment variable override
    env_val = _parse_env_int(ENV_JOB_NUM)
    if env_val:
        return env_val

    # Priority 3: memory-aware default
    cpu_based = min(int(math.ceil(float(multiprocessing.cpu_count()) * max_cpu_ratio)), max_cpu)
    available_mem = _get_available_memory()
    if available_mem and available_mem > 0:
        mem_based = max(1, available_mem // mem_per_job_bytes)
        return min(cpu_based, mem_based)
    return cpu_based

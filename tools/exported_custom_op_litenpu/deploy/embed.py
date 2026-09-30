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
"""Loader for the shipped kernel-compile snippet (called by ``PtoCustomOp::Compile``).

The SHIPPED, dev-editable ``<opp_root>/op_kernel/<stem>.py`` is the SINGLE SOURCE OF TRUTH for an op's
kernel-compile entry (``__pypto_compile(...)``) and its ``infer_shape``/``infer_dtype`` functions. At
op-compile time the C++ base resolves that file ENV-first over the ``ASCEND_CUSTOM_OPP_PATH`` entries
(``<entry>/op_kernel/<basename>``, dladdr upward-walk fallback), imports this module and calls
``load_embedded_compile_module`` with the resolved path.

The ``.py`` is imported BY PATH, so a developer can edit it in place and the change takes effect on the
next op-compile with NO ``.so`` rebuild, the memo keys on ``(stem, mtime)``, so an edited file
re-imports. Loading is STRICT: an empty or absent path is a FATAL ``FileNotFoundError``.

The snippet is shape-independent, so the module is memoized per ``(stem, mtime)``, one load per op,
reused across runtime shapes.

Loading also gates the package's recorded pypto version: the package root (the shipped ``.py``'s
``parent.parent``) must carry a ``pypto_version.info`` naming no NEWER pypto than the one compiling here.
The gate sits BELOW the path guards and ABOVE the memo, so their diagnoses fire first and a cached
module cannot skip it.
"""

import importlib.metadata
import importlib.util
import os
import re
import sys
import types
import typing
import warnings

__all__ = (
    "VERSION_INFO_BASENAME",
    "check_pypto_package_version",
    "installed_pypto_version",
    "is_newer",
    "load_embedded_compile_module",
    "parse_version_tuple",
)

# Memoize per (stem, mtime), the shipped snippet is constant for an op (it takes shapes/dtypes as
# args), so we load it once and reuse it across compiles for different runtime shapes. The mtime in the
# key means an EDITED shipped .py yields a fresh key -> re-import.
_loaded: dict = {}

# Deliberately PREFIXED: GE version-gates an ASCEND_CUSTOM_OPP_PATH entry on its plain ``version.info``
# and SILENTLY drops the whole op package when out of range. GE matches exact names, so ours is never read.
VERSION_INFO_BASENAME = "pypto_version.info"
_VERSION_INFO_KEY = "Version"

# The leading release segment only; everything after it (``rc1``, ``.dev0``, ``+g1234``, ``.post2``) is
# ignored, so a dev or local build compares equal to the release it was cut from.
_RELEASE_SEGMENT_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def installed_pypto_version() -> str:
    """Return the installed pypto distribution's version, or ``"unknown"``.

    Re-derives the installed pypto version without importing pypto: spliced into the ``.so``, this code
    may import no pypto.
    """
    try:
        return importlib.metadata.version("pypto")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse_version_tuple(raw: str) -> typing.Optional[typing.Tuple[int, ...]]:
    """Return *raw*'s leading release numbers, or ``None`` when it carries none.

    ``"1.0-beta"``/``"1.0.post2"`` -> ``(1, 0)``; ``"2!1.0"`` -> ``(2,)``; ``"unknown"`` -> ``None``. Never raises.
    """
    match = _RELEASE_SEGMENT_RE.match(raw.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(*, recorded: typing.Tuple[int, ...], local: typing.Tuple[int, ...]) -> bool:
    """Whether *recorded* is a strictly newer release than *local*, zero-padded to a common length."""
    width = max(len(recorded), len(local))
    return recorded + (0,) * (width - len(recorded)) > local + (0,) * (width - len(local))


def _read_recorded_version(info_path: str) -> str:
    """Return the ``Version=`` value from *info_path*, or ``""`` when no such line is present."""
    with open(info_path, encoding="utf-8") as handle:
        for line in handle:
            key, sep, value = line.partition("=")
            if sep and key.strip() == _VERSION_INFO_KEY:
                return value.strip()
    return ""


def _version_info_path(*, root: str, prefix: str) -> str:
    """Return *root*'s ``pypto_version.info`` path, raising when the file is absent."""
    info_path = os.path.join(root, VERSION_INFO_BASENAME)
    if not os.path.isfile(info_path):
        raise RuntimeError(
            f"{prefix}{root} has no {VERSION_INFO_BASENAME}; it was not produced by a pypto build of "
            f"this vintage. Rebuild the op package with this pypto."
        )
    return info_path


def _reject_stray_version_info(*, root: str, prefix: str, what: str) -> None:
    """Reject (install) or warn (atc) when *root* carries a plain ``version.info``."""
    # Fatal at install, where pypto owns the directory; a warning at atc, where the root is user-pointed
    # and may legitimately carry an in-range file GE accepts today.
    if os.path.isfile(os.path.join(root, "version.info")):
        stray = (
            f"{prefix}{root} contains a plain version.info. GE version-gates the "
            f"ASCEND_CUSTOM_OPP_PATH entry on that file and silently drops the op package when it is "
            f"out of range, which surfaces only as an unregistered op. Remove it; pypto records its "
            f"own version in {VERSION_INFO_BASENAME}."
        )
        if what == "install":
            raise RuntimeError(stray)
        # stacklevel=3: one frame below the public entry, so this attributes to the caller
        warnings.warn(stray, stacklevel=3)


def _recorded_version(*, info_path: str, root: str, prefix: str) -> str:
    """Raise when *info_path* cannot be read; otherwise return what it records."""
    try:
        return _read_recorded_version(info_path)
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError: without it a non-UTF-8 version file escapes
        # unclassified and reaches GE as something other than the RuntimeError every refusal here raises.
        raise RuntimeError(
            f"{prefix}{root} has a {VERSION_INFO_BASENAME} that cannot be read ({exc}). Rebuild the "
            f"op package with this pypto."
        ) from exc


def _compare_or_warn(*, recorded_raw: str, local_raw: str, root: str, prefix: str) -> None:
    """Raise when *recorded_raw* is a newer release than *local_raw*; warn when either cannot be parsed."""
    recorded = parse_version_tuple(recorded_raw)
    local = parse_version_tuple(local_raw)
    if recorded is None:
        warnings.warn(
            f"{prefix}pypto op package at {root} records {VERSION_INFO_BASENAME} "
            f"{_VERSION_INFO_KEY}={recorded_raw!r}, which cannot be compared with this pypto "
            f"({local_raw}); skipping the package-version check.",
            # stacklevel=3: one frame below the public entry, so this attributes to the caller
            stacklevel=3,
        )
        return
    if local is None:
        warnings.warn(
            f"{prefix}this pypto reports version {local_raw!r}, which cannot be compared with the "
            f"version recorded by the op package at {root} ({recorded_raw}); skipping the "
            f"package-version check.",
            # stacklevel=3: one frame below the public entry, so this attributes to the caller
            stacklevel=3,
        )
        return
    if is_newer(recorded=recorded, local=local):
        raise RuntimeError(
            f"{prefix}this op package was built with pypto {recorded_raw}, but this pypto is "
            f"{local_raw}. Rebuild the op package with this pypto, or use pypto >= {recorded_raw}."
        )


def check_pypto_package_version(opp_root, *, what: str) -> None:
    """Refuse an op package built by a NEWER pypto than the one running here.

    *what* (``"install"``/``"atc"``) selects the install-only ``[SETUP] `` prefix and stray-``version.info`` severity.
    """
    # abspath BEFORE realpath: abspath collapses ".." LEXICALLY, so <entry>/op_proto/.. normalises to
    # <entry> rather than to wherever a symlinked op_proto resolves — realpath alone follows the link
    # first and lands elsewhere. Separately, the atc site can pass "" (dirname(dirname(<.py path>)) is ""
    # for a bare relative path) and both calls turn that into the process CWD, a root nobody named.
    # Normalising gives one package one spelling however it was reached, so its repeated touches build an
    # identical warning the default filter shows once per location, with no seen-set here.
    root = os.path.realpath(os.path.abspath(str(opp_root)))
    prefix = "[SETUP] " if what == "install" else ""
    info_path = _version_info_path(root=root, prefix=prefix)
    _reject_stray_version_info(root=root, prefix=prefix, what=what)
    recorded_raw = _recorded_version(info_path=info_path, root=root, prefix=prefix)
    _compare_or_warn(recorded_raw=recorded_raw, local_raw=installed_pypto_version(), root=root, prefix=prefix)


def load_embedded_compile_module(stem: str, deploy_py_path: str) -> types.ModuleType:
    """Import the shipped ``op_kernel/<stem>.py`` and return the module.

    An empty or absent path raises ``FileNotFoundError``; a missing or newer ``pypto_version.info``, ``RuntimeError``.
    """
    if not deploy_py_path:
        raise FileNotFoundError(
            f"pypto op_kernel: no op_kernel/*.py path was resolved for stem {stem!r}; install it at "
            f"<opp_root>/op_kernel/<stem>.py, reachable via ASCEND_CUSTOM_OPP_PATH."
        )
    if not os.path.isfile(deploy_py_path):
        raise FileNotFoundError(
            f"pypto op_kernel: resolved op_kernel .py for stem {stem!r} does not exist: "
            f"{deploy_py_path!r}."
        )
    # The package root is the shipped .py's parent.parent (<opp_root>/op_kernel/<stem>.py). This call's
    # placement between the path guards and the memo is load-bearing — see the module docstring.
    check_pypto_package_version(os.path.dirname(os.path.dirname(deploy_py_path)), what="atc")
    key = (stem, os.path.getmtime(deploy_py_path))
    cached = _loaded.get(key)
    if cached is not None:
        return cached
    # Import the shipped, dev-editable .py by path (a real file on disk, so getsourcelines reads the
    # nested @jit kernel directly). Not unlinked: it is the developer's file.
    name = f"_pypto_deploy_{stem}"
    spec = importlib.util.spec_from_file_location(name, deploy_py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _loaded[key] = mod
    return mod

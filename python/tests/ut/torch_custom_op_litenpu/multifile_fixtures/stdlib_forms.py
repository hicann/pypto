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
"""Fixtures for the stdlib header-import STATEMENT forms: every import spelling whose binding must be
reproduced by the emitted snippet header (plain / aliased / bare-dotted / from-import-of-module), plus
the alias-collision residual. The helper bodies keep their chains verbatim; only the header statements
differ per form.
"""
import os
import os as _os
import xml as x
import xml.etree.ElementTree  # loads the submodule chain the ``x.etree...`` walk descends at pack time


def helper_plain_dotted_attr():
    """Module-scope plain ``import os`` + an ``os.path.*`` chain -> header stays exactly ``import os``
    (``os.path`` is a module NAMED ``posixpath``, so the submodule walk must stop at ``os``)."""
    return os.path.join("a", "b")


def helper_aliased_root():
    """Module-scope ``import os as _os`` -> header ``import os as _os``."""
    return _os.path.sep


def helper_aliased_top_deep_chain():
    """Module-scope ``import xml as x`` + a deep submodule chain -> TWO header statements:
    ``import xml.etree.ElementTree`` (loads the chain) + ``import xml as x`` (binds the alias)."""
    return x.etree.ElementTree.Comment


def helper_local_aliased_stdlib(shape):
    """Body-local ``import math as m`` -> header ``import math as m``, body chain verbatim."""
    import math as m
    return m.prod(shape)


def helper_local_dotted_stdlib():
    """Body-local bare-dotted ``import xml.etree.ElementTree`` -> the FULL dotted statement
    (``import xml`` alone would leave ``xml.etree`` unresolvable at deploy)."""
    import xml.etree.ElementTree
    return xml.etree.ElementTree.Comment


def helper_local_from_import_module():
    """Body-local ``from os import path`` (a MODULE member) -> ``import posixpath as path``."""
    from os import path
    return path.sep


def helper_collision_json():
    """Half of the alias-collision residual: binds ``j`` to json."""
    import json as j
    return j.dumps


def helper_collision_os():
    """Other half of the alias-collision residual: binds the SAME ``j`` to os."""
    import os as j
    return j.sep


def entry_plain_dotted():
    return helper_plain_dotted_attr()


def entry_aliased_root():
    return helper_aliased_root()


def entry_aliased_top_deep_chain():
    return helper_aliased_top_deep_chain()


def entry_local_aliased(shape):
    return helper_local_aliased_stdlib(shape)


def entry_local_dotted():
    return helper_local_dotted_stdlib()


def entry_local_from_import():
    return helper_local_from_import_module()


def entry_alias_collision():
    return helper_collision_json() and helper_collision_os()


def create_kernel_stdlib_alias(shape, dtype, soc_version):
    """Factory fixture for the snippet-level aliased-stdlib check: packing this factory must surface
    the helper's ``import math as m`` as a header statement of the emitted snippet."""
    n = helper_local_aliased_stdlib(shape)
    return n, dtype, soc_version

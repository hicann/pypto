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
import pypto
from pypto.experimental import set_operation_config, get_operation_config


def test_print_options():
    pypto.set_print_options(edgeitems=1,
                            precision=2,
                            threshold=3,
                            linewidth=4)


def test_pass_option():
    # int
    pypto.set_pass_options(cube_l1_reuse_mode=1)
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_l1_reuse_mode"] == 1
    # map
    pypto.set_pass_options(cube_nbuffer_setting={3: 4})
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_setting"] == {3: 4}


def test_host_option():
    pypto.set_host_options(only_codegen=True)
    host_option = pypto.get_host_options()
    assert host_option["only_codegen"] == True


def test_runtime_option():
    pypto.set_runtime_options(stitch_function_size=30000)
    runtime_option = pypto.get_runtime_options()
    assert runtime_option["stitch_function_size"] == 30000


def test_reset_option():
    pypto.set_runtime_options(stitch_function_num_initial=23)
    runtime_option = pypto.get_runtime_options()
    assert runtime_option["stitch_function_num_initial"] == 23
    pypto.set_host_options(only_codegen=True)
    host_option = pypto.get_host_options()
    assert host_option["only_codegen"] == True
    pypto.reset_options()
    runtime_option = pypto.get_runtime_options()
    host_option = pypto.get_host_options()
    assert runtime_option["stitch_function_num_initial"] == 30
    assert host_option["only_codegen"] == False


def test_option():
    pypto.set_option("profile_enable", True)
    option = pypto.get_option("profile_enable")
    assert option == True


def test_operation_option():
    set_operation_config(force_combine_axis=True)
    option = get_operation_config()
    assert option["force_combine_axis"] == True
    set_operation_config(combine_axis=True)
    option = get_operation_config()
    assert option["combine_axis"] == True


def test_global_option():
    res = pypto.get_global_config("platform.ENABLE_COST_MODEL")
    assert res == False
    pypto.set_global_config("platform.ENABLE_COST_MODEL", True)
    res = pypto.get_global_config("platform.ENABLE_COST_MODEL")
    assert res == True

    pypto.set_global_config("codegen.parallel_compile", 10)
    res = pypto.get_global_config("codegen.parallel_compile")
    assert res == 10

if __name__ == "__main__":
    test_global_option()

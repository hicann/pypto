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
import pytest


def test_pass_config():
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True) is False
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    # reset
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False)

    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction",
                                 pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction",
                                 pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    configs = pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
    assert configs.dumpGraph is True
    # reset
    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)

    with pytest.raises(TypeError, match=r"Expected boolean type, but received int"):
        pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, -2)


def test_pass_option():
    test_params = {
        "pg_skip_partition": True,
        "pg_upper_bound": 2,
        "pg_lower_bound": 3,
        "pg_parallel_lower_bound": 4,
        "sg_set_scope": 5,
        "mg_vec_parallel_lb": 6,
        "vec_nbuffer_mode": 2,
        "vec_nbuffer_setting": {1: 2},
        "cube_l1_reuse_mode": 6,
        "cube_l1_reuse_setting": {2: 3},
        "cube_nbuffer_mode": 2,
        "cube_nbuffer_setting": {-1: 2},
        "mg_copyin_upper_bound": 7
    }
    pypto.set_pass_options(**test_params)
    option = pypto.get_pass_options()
    for key, expect_valuie in test_params.items():
        assert option[key] == expect_valuie

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ast

from pypto_pro.runtime.pipeline._astutil import is_vf_function
import pytest


@pytest.mark.parametrize(
    "decorator",
    [
        "pl.vector_function",
        "vector_function",
        'pl.vector_function(mode="simd")',
        'vector_function(mode="simd")',
    ],
)
def test_is_vf_function_recognizes_simd_decorator_forms(decorator):
    function = ast.parse(f"@{decorator}\ndef helper():\n    pass\n").body[0]
    assert is_vf_function(function)

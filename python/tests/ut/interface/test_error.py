#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for pypto._error module.

This test suite verifies that error handling works correctly in PyPTO by
testing error scenarios that trigger different error types through the frontend.
"""

import pytest
import pypto
import torch

from pypto.error import ParserError


def test_varargs_error():
    """Test that variable-length arguments trigger proper error handling."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def varargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        *args):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Variable-length arguments"):
        varargs_kernel(x, out)


def test_kwargs_error():
    """Test that keyword arguments trigger proper error handling."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kwargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        **kwargs):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Keyword argument packing"):
        kwargs_kernel(x, out)


def test_error_message_contains_error_code():
    """Test that error messages contain error codes."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def varargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        *args):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    try:
        varargs_kernel(x, out)
        assert False, "Should have raised an error"
    except Exception as e:
        error_str = str(e)
        assert "ErrCode: F00005" in error_str
        assert len(error_str) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

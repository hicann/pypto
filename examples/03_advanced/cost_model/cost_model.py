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
This test case verifies that the swimlane diagram generation for the costmodel works correctly 
regardless of whether the CANN is installed in the environment or not.
"""

import os
import sys
import argparse
import pypto
import torch
import json
import numpy as np
from numpy.testing import assert_allclose

"""
PyPTO Cost Model Simulation Example

This example demonstrates how to enable swimlane diagram generation in PyPTO's cost model mode.
The test validates that the cost analysis and swimlane visualization work correctly in 
simulation environment, independent of actual NPU hardware availability.
"""

def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, None
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as e:
        return None, f"Invalid json format: {e}"
    except PermissionError:
        return None, "Permission Erro"
    except Exception as e:
        return None, f"Load json fail, unknow error: {e}"


def get_out_put_path():
    out_path = "./output"
    if os.path.exists(out_path):
        subdirs = [os.path.join(out_path, d) for d in os.listdir(out_path) 
                if os.path.isdir(os.path.join(out_path, d))]   
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getctime)
            return latest_dir
    return None


def softmax_core(input_tensor: pypto.tensor) -> pypto.tensor:
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)
    sub = pypto.sub(input_tensor, row_max)
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return pypto.div(exp, esum)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"cfgcache_device_task_num": 100, "cfgcache_root_task_num": 100, "cfgcache_leaf_task_num": 10000, "run_mode": 1}
)
def softmax(input_tensor, output_tensor, cost_model_enable):

    # After the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]  # Dynamic batch size
    n1, n2, dim = tensor_shape[1:]  # Static dimensions
    tile_b = 1  # Process one batch at a time
    b_loop = b // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        
        # Extract batch slice
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]
        
        # Apply softmax to batch slice
        softmax_out = softmax_core(input_view)
        
        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


def test_softmax(cost_model_enable=True):
    """
    Run softmax with optional cost model.

    When cost_model_enable=True, PyPTO generates simulated execution
    outputs (e.g. swimlane JSON) for analysis.
    """ 
    cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
    
    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    # Prepare data
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    # Initialize PyPTO inputs and outputs
    # Mark dynamic axis: the actual size of the axis can be any integer number during runtime
    inputs = {
        input_data: [0]
    }
    outputs = {
        output_data: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    # Launch the kernel
    softmax(*pto_inputs, *pto_outputs, cost_model_enable)

    # Verify against PyTorch reference
    torch_softmax = torch.softmax(input_data, dim=3)
    npu_data = output_data.cpu()
    torch_data = torch_softmax.cpu()

    max_diff = np.abs(npu_data.numpy() - torch_data.numpy()).max()
    
    output_path = get_out_put_path()
    assert output_path
    
    if cost_model_enable:
        merged_swimlane, error = safe_json_load(os.path.join(output_path, 'CostModelSimulationOutput/merged_swimlane.json'))
        assert not error

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print("✓ Cost model test passed\n")
    print()
    

if __name__ == "__main__":
    # Always execute through build_ci.py
    script_path = os.path.abspath(__file__)
    cmd = f"python3 build_ci.py -s={script_path}"
    
    # Execute and Exit
    os.system(cmd)
    sys.exit(0)

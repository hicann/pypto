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
import os
import math
import torch
import torch_npu
import pypto


class IndexaAddParamInfo:
    def __init__(self, axis: int, alpha, b1, s1, b2, s2):
        self.self_shape = (b1, s1)
        self.src_shape = (b2, s1)
        self.index_shape = (self.src_shape[axis],)
        self.view_shape = (max(b1, b2), s2)
        self.tile_shape = (max(b1, b2), s2)
        self.value = alpha
        self.axis = axis


def indexadd_2dim_comm_test_body(indexadd_para, test_func):
    self_shape = indexadd_para.self_shape
    src_shape = indexadd_para.src_shape
    index_shape = indexadd_para.index_shape
    view_shape = indexadd_para.view_shape
    tile_shape = indexadd_para.tile_shape
    axis = indexadd_para.axis
    value = indexadd_para.value
    pypto.runtime._device_init()

    src_tensor = pypto.tensor(src_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_SRC")
    index_tensor = pypto.tensor(index_shape, pypto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    self_tensor = pypto.tensor(self_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_SELF")
    dst_tensor = pypto.tensor(self_shape, pypto.DataType.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    with pypto.function("INDEXADD", self_tensor, src_tensor, index_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="LOOP_B0", idx_name="b_idx"):
            for s_idx in pypto.loop(s_loop_num, name="LOOP_S0", idx_name="s_idx"):
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_self = pypto.view(self_tensor, view_shape, [b_idx * view_shape[0], s_idx * view_shape[1]],
                                        valid_shape=[
                                            pypto.min(pypto.symbolic_scalar(self_shape[0]) - b_idx * view_shape[0],
                                                    pypto.symbolic_scalar(view_shape[0])),
                                            pypto.min(pypto.symbolic_scalar(self_shape[1]) - s_idx * view_shape[1],
                                                    pypto.symbolic_scalar(view_shape[1]))])
                view_src = pypto.view(src_tensor, view_shape, [b_idx * view_shape[0], s_idx * view_shape[1]],
                                        valid_shape=[
                                            pypto.min(pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                                                    pypto.symbolic_scalar(view_shape[0])),
                                            pypto.min(pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                                                    pypto.symbolic_scalar(view_shape[1]))])
                view_index = pypto.view(index_tensor, [view_shape[axis]], [b_idx * view_shape[0]],
                                        valid_shape=[
                                            pypto.min(pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                                                    pypto.symbolic_scalar(view_shape[0]))])
                tmp_dst_tensor = pypto.tensor()
                tmp_dst_tensor.move(test_func(view_self, axis, view_index, view_src, alpha=value))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)
                del view_self, view_src, view_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pypto.tensor)

    self_input = torch.rand(self_shape, dtype=torch.float32) * 200 - 100
    src_input = torch.rand(src_shape, dtype=torch.float32) * 200 - 100
    index_input = torch.randint(0, self_shape[axis], index_shape, dtype=torch.int32)
    result_tensor = torch.zeros(self_shape, dtype=torch.float32)

    pto_x1_tensor = pypto.from_torch(self_input, "x1_tensor")
    pto_x2_tensor = pypto.from_torch(src_input, "x2_tensor")
    pto_x3_tensor = pypto.from_torch(index_input, "x3_tensor")
    pto_res_tensor = pypto.from_torch(result_tensor, "res_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_x1_tensor, pto_x2_tensor, pto_x3_tensor, pto_res_tensor)

    expect = self_input.index_add(axis, index_input, src_input, alpha=value)
    assert torch.allclose(result_tensor, expect, rtol=1e-4, atol=1e-5)
    pypto.runtime._device_fini


def test_index_add_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b1 = 7
    s1 = 8
    b2 = 9
    s2 = 4
    alpha = 1.3
    indexadd_para = IndexaAddParamInfo(0, alpha, b1, s1, b2, s2)

    indexadd_2dim_comm_test_body(indexadd_para, pypto.index_add)


def test_index_add__onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b1 = 7
    s1 = 8
    b2 = 9
    s2 = 4
    alpha = 1.3
    indexadd_para = IndexaAddParamInfo(0, alpha, b1, s1, b2, s2)

    indexadd_2dim_comm_test_body(indexadd_para, pypto.index_add_)

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

"""vector op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
"""
import sys
import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import copy

g_src_root: Path = Path(Path(__file__).parent, "../../../../").resolve()
g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
if str(g_ctrl_path) not in sys.path:
    sys.path.append(str(g_ctrl_path))
from golden_register import GoldenRegister

import_path: Path = Path(g_ctrl_path, "helper").resolve()
if str(import_path) not in sys.path:
    sys.path.append(str(import_path))
from test_case_loader import TestCaseLoader
from test_case_desc import TensorDesc
from test_case_tools import parse_list_str, get_dtype_by_name, parse_dict_str, str_to_bool

bfloat16 = get_dtype_by_name("bf16", False, False)


def trans_nd_to_fractal_nz(data: np.ndarray, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == np.int8:
        m0, n0 = 16, 32
    elif data.dtype == np.float16 or data.dtype == bfloat16 or data.dtype == np.int32:
        m0, n0 = 16, 16
    else:
        m0, n0 = 16, 8
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        data = np.pad(
            data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant"
        )
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    else:
        data = np.pad(data, (batch_padding + ((0, 0), (0, padding_n))), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).transpose(*array_trans)
    return data


def dump_file(data_pool, data_path, type_str):
    np.array(data_pool).astype(get_dtype_by_name(type_str.lower())).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def load_test_cases_from_json(json_file: str) -> list:
    with open(json_file, "r") as data_file:
        json_data = json.load(data_file)
    if json_data is None:
        raise ValueError(f"Json file {json_file} is invalid.")
    if "test_cases" in json_data:
        test_cases = json_data["test_cases"]
    else:
        test_cases = [json_data]
    test_cases.sort(key=lambda x: x["case_index"])
    return test_cases


def gen_op_golden(
    op: str, golden_func, output_path: Path, case_index: int = None
) -> bool:
    def generate_golden_files(golden_func, output_path: Path, config: dict) -> bool:
        if config['operation'] in ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]:
            return generate_matmul_golden_files(golden_func, output_path, config)

        input_tensors = []
        spec_value_map = {
            "nan": np.nan,
            "inf": np.inf,
            "-inf": -np.inf,
            "max": np.finfo(np.float32).max,
            "min": np.finfo(np.float32).min,
        }
        index = 0
        for input_tensor in config["input_tensors"]:
            min = input_tensor["data_range"]["min"]
            max = input_tensor["data_range"]["max"]
            if min != max:
                assert not isinstance(min, str) and not isinstance(
                    min, str
                ), "Data range must be number when the min and max are not same."
                if op == "ScatterUpdate" and index == 1:
                    tensor = np.random.choice(
                        range(min, max), input_tensor["shape"], False
                    ).astype(get_dtype_by_name(input_tensor["dtype"]))
                else:
                    tensor = np.random.uniform(min, max, input_tensor["shape"]).astype(
                        get_dtype_by_name(input_tensor["dtype"])
                    )
            else:
                if isinstance(min, str):
                    assert (
                        min in spec_value_map.keys()
                    ), f"Data range of input tensor {input_tensor} has invalid value."
                    max = spec_value_map.get(max)
                tensor = np.full(
                    input_tensor["shape"],
                    max,
                    dtype=get_dtype_by_name(input_tensor["dtype"]),
                )
            index += 1
            input_tensors.append(tensor)

        res = golden_func(input_tensors, config)
        cube_op_list = ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]
        for input_tensor, read_input in zip(input_tensors, config["input_tensors"]):
            if (
                config.get("operation") in cube_op_list
                and read_input.get("format") == "NZ"
            ):
                input_tensor = trans_nd_to_fractal_nz(input_tensor)
            input_tensor.tofile(Path(output_path, read_input["name"] + ".bin"))

        for idx in range(len(config["output_tensors"])):
            res[idx].astype(
                get_dtype_by_name(config["output_tensors"][idx]["dtype"])
            ).tofile(Path(output_path, config["output_tensors"][idx]["name"] + ".bin"))
        return True

    case_path: Path = Path(Path(__file__).parent.parent, "test_case").resolve()
    case_file: Path = Path(case_path, op + "_st_test_cases.json").resolve()
    test_configs = load_test_cases_from_json(str(case_file))
    if len(test_configs) == 0:
        raise ValueError("Not find test cases, please check.")

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, test_config in enumerate(test_configs):
            output_path1 = Path(output_path, str(index))
            output_path1.mkdir(parents=True, exist_ok=True)
            generate_golden_files(golden_func, output_path1, test_config)
    else:
        generate_golden_files(golden_func, output_path, test_configs[case_index])
    return True


def generate_matmul_params_files(input_tensors: list, output_path: Path, config: dict):
    params = config.get("params")
    if params.get("scale_tensors"):
        scale_tensor = params["scale_tensors"]
        scale_min = scale_tensor["data_range"]["min"]
        scale_max = scale_tensor["data_range"]["max"]
        if scale_min != scale_max:
            tensor = np.random.uniform(scale_min, scale_max, scale_tensor["shape"]).astype(
                np.float32)
        else:
            tensor = np.full(
                scale_tensor["shape"],
                scale_max,
                dtype=np.float32,
            )
        tensor_data = tensor.view(np.uint32)
        mask = 0xFFFFE000
        tensor_data = tensor_data & mask
        fp32_modified = tensor_data.view(np.float32)
        tensor_data.astype(np.uint64).tofile(Path(output_path, params["scale_tensors"]["name"] + ".bin"))
        input_tensors.append(fp32_modified)

    if params.get("bias_tensors"):
        bias_tensor = params["bias_tensors"]
        bias_min = bias_tensor["data_range"]["min"]
        bias_max = bias_tensor["data_range"]["max"]
        tensor_type = get_dtype_by_name(bias_tensor["dtype"])
        if bias_tensor["dtype"] == "bf16":
            tensor_type = bfloat16
        tensor = np.random.uniform(bias_min, bias_max, bias_tensor["shape"]).astype(
            tensor_type
        )
        tensor.tofile(Path(output_path, params["bias_tensors"]["name"] + ".bin"))
        input_tensors.append(tensor)


def generate_matmul_golden_files(golden_func, output_path: Path, config: dict):
    spec_value_map = {
        "-inf": -np.inf,
        "inf": np.inf,
        "nan": np.nan,
        "min": np.finfo(np.float32).min,
        "max": np.finfo(np.float32).max,
    }
    cube_op_list = ["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"]
    index = 0
    input_tensors = []
    for input_tensor in config["input_tensors"]:
        input_min = input_tensor["data_range"]["min"]
        input_max = input_tensor["data_range"]["max"]
        assert not isinstance(input_min, str) and not isinstance(
            input_min, str
        ), "Data range must be number when the min and max are not same."
        tensor_type = get_dtype_by_name(input_tensor["dtype"])
        if input_tensor["dtype"] == "bf16":
            tensor_type = bfloat16
        tensor = np.random.uniform(input_min, input_max, input_tensor["shape"]).astype(
            tensor_type
        )
        index += 1
        input_tensors.append(tensor)

    generate_matmul_params_files(input_tensors, output_path, config)

    res = golden_func(input_tensors, config)
    for idx in range(len(config["output_tensors"])):
        tensor_type = get_dtype_by_name(config["output_tensors"][idx]["dtype"])
        if config["output_tensors"][idx]["dtype"] == "bf16":
            tensor_type = bfloat16
        res[idx].astype(tensor_type).tofile(
            Path(output_path, config["output_tensors"][idx]["name"] + ".bin")
        )

    for input_tensor, read_input in zip(input_tensors, config["input_tensors"]):
        if config.get("operation") in cube_op_list and read_input.get("format") == "NZ":
            input_tensor = trans_nd_to_fractal_nz(input_tensor)
        input_tensor.tofile(Path(output_path, read_input["name"] + ".bin"))

    params = config.get("params")
    l0c2l1_tensor = params.get("l0c2l1_tensor")
    if l0c2l1_tensor:
        # 2：l0c2l1场景res = [tensor_out, tensor_l0c2l1_data]
        assert len(res) == 2
        res[1].tofile(Path(output_path, l0c2l1_tensor["name"] + ".bin"))
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatterUpdate/ScatterUpdateOperationTest.TestScatterUpdate",
    ]
)
def gen_scatter_update_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, config):
        src = inputs[0]
        index = inputs[1]
        dst = inputs[2]
        axis = src.ndim

        # 自测，src = np.random.randint(2, 3, src.shape).astype(np.float32)
        # 自测，dst = np.random.randint(1, 2, dst.shape).astype(np.float32)
        if axis == 4:
            b = index.shape[0]
            s = index.shape[1]
            blockNum = dst.shape[0]
            blockSize = dst.shape[1]
            bs2 = blockNum * blockSize
            result = copy.copy(dst)
            for _b in range(b):
                for _s in range(s):
                    result[index[_b][_s] // blockSize][index[_b][_s] % blockSize][:] = (
                        src[_b][_s][:]
                    )
        elif axis == 2:
            b = index.shape[0]
            s = index.shape[1]
            bs2 = dst.shape[0]
            d = dst.shape[1]
            result = copy.copy(dst)

            for _b in range(b):
                for _s in range(s):
                    result[index[_b][_s]][:] = src[_b * s + _s][:]
        else:
            logging.error("axis ERROR!")

        logging.debug("src:")
        logging.debug(inputs[0])
        logging.debug("index:")
        logging.debug(inputs[1])
        logging.debug("dst:")
        logging.debug(inputs[2])
        logging.debug("result:")
        logging.debug(result)

        return [result]

    logging.info("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ScatterUpdate", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTopK/TopK4DOperationTest.test_topk",
        "TestTopK/TopK3DOperationTest.test_topk",
        "TestTopK/TopKOperationTest.test_topk",
    ]
)
def gen_topk_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    shape_list_i = []
    dim_axis = []
    k_list = []
    largest_list = []
    if "TopKOperationTest" in case_name:
        shape_list_i = [[128, 32]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    elif "TopK3DOperationTest" in case_name:
        shape_list_i = [[8, 2, 8]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    elif "TopK4DOperationTest" in case_name:
        shape_list_i = [[2, 8, 2, 8]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    if case_index is None:
        for index, arr in enumerate(shape_list_i):
            output_path = Path(str(output) + "/" + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            x_path = Path(output_path, "x.bin")
            o0_path = Path(output_path, "value.bin")
            o1_path = Path(output_path, "index.bin")
            complete = x_path.exists() and o0_path.exists()
            if complete:
                logging.debug("Case(%s), Golden complete.", case_name)
                return True
            else:
                x = torch.randn(arr)
                val, idx = x.topk(
                    k_list[index],
                    dim=dim_axis[index],
                    largest=largest_list[index],
                    sorted=True,
                )
                x.numpy().tofile(x_path)
                val.numpy().tofile(o0_path)
                idx.numpy().astype(np.int32).tofile(o1_path)
    else:
        x_path = Path(output, "x.bin")
        o0_path = Path(output, "value.bin")
        o1_path = Path(output, "index.bin")
        complete = x_path.exists() and o0_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = torch.randn(shape_list_i[case_index])
            val, idx = x.topk(
                k_list[case_index],
                dim=dim_axis[case_index],
                largest=largest_list[case_index],
                sorted=True,
            )
            x.numpy().tofile(x_path)
            val.numpy().tofile(o0_path)
            idx.numpy().astype(np.int32).tofile(o1_path)
    return True


@TestCaseLoader.reg_params_handler(ops=["Matmul", "BatchMatmul", "MatmulVerify", "BatchMatmulVerify"])
def matmul_params_func(params: dict):
    bias_params_func(params)
    fixpipe_params_func(params)
    l0c2l1_params_func(params)
    return params


def fixpipe_params_func(params: dict):
    scale_shape = [1, 1]
    scale_range = [1, 1]
    scale_dtype = "uint64"
    fixpipe_param = params.get("fixpipe_info", "")
    if fixpipe_param is None:
        params["fixpipe_info"] = ""
    if fixpipe_param != "" and fixpipe_param is not None:
        fixpipe_info = parse_dict_str(fixpipe_param)
        if "scale_value" in fixpipe_info:
            params["scale_value"] = float(fixpipe_info["scale_value"])
        if "relu_type" in fixpipe_info:
            params["relu_type"] = int(fixpipe_info["relu_type"])
        if "scale_tensor_range" in fixpipe_info and fixpipe_info["scale_tensor_range"] != "":
            scale_range = parse_list_str(fixpipe_info["scale_tensor_range"])
        if "quant_type" in fixpipe_info:
            params["quant_type"] = int(fixpipe_info["quant_type"])
        if "scale_shape" in fixpipe_info:
            scale_shape_info = parse_list_str(fixpipe_info["scale_shape"])
            if scale_shape_info[0] == 0 or len(scale_shape_info) < 2:
                scale_range = [1, 1]
            else:
                scale_shape = scale_shape_info
    # scale_tensor
    scale_tensor = TensorDesc("scale_tensor", scale_shape, scale_dtype, scale_range,
                                tensor_format="ND", need_trans=False)
    params["scale_tensors"] = scale_tensor.dump_to_json()


def bias_params_func(params: dict):
    bias_shape = [1, 1]
    bias_range = [0, 0]
    bias_dtype = "fp16"
    bias_param = params.get("bias_info", "")
    if bias_param is None:
        params["bias_info"] = ""
    if bias_param != "" and bias_param is not None:
        bias_info = parse_dict_str(bias_param)
        if "bias_range" in bias_info:
            bias_range = parse_list_str(bias_info["bias_range"])
        if "bias_dtype" in bias_info:
            bias_dtype = bias_info["bias_dtype"]
        if "bias_shape" in bias_info:
            bias_shape_info = parse_list_str(bias_info["bias_shape"])
            if bias_shape_info[0] == 0 or len(bias_shape_info) < 2:
                bias_range = [0, 0]
            else:
                bias_shape = bias_shape_info
    # bias_tensor
    bias_tensor = TensorDesc("bias_tensor", bias_shape, bias_dtype, bias_range,
                                tensor_format="ND", need_trans=False)
    params["bias_tensors"] = bias_tensor.dump_to_json()


def l0c2l1_params_func(params: dict):
    l0c2l1_info_params = params.get("l0c2l1_info", None)
    if l0c2l1_info_params is None:
        return
    l0c2l1_info = parse_dict_str(l0c2l1_info_params)
    required_keys = {"input_shape", "input_dtype", "input_format", "is_trans", "is_as_left_matrix"}
    if not required_keys.issubset(l0c2l1_info):
        raise ValueError("l0c2l1_info params is invalid, please check!")
    l0c2l1_is_as_left_matrix = str_to_bool(l0c2l1_info["is_as_left_matrix"])
    l0c2l1_is_trans = str_to_bool(l0c2l1_info.get("is_trans", False))
    l0c2l1_range = parse_list_str(l0c2l1_info.get("input_range", "[-1, 1]"))
    l0c2l1_tensor = TensorDesc("l0c2l1_tensor", parse_list_str(l0c2l1_info["input_shape"]), l0c2l1_info["input_dtype"],
                l0c2l1_range, l0c2l1_info["input_format"], need_trans=l0c2l1_is_trans)
    params["l0c2l1_tensor"] = l0c2l1_tensor.dump_to_json()
    params["l0c2l1_params"] = {"is_as_left_matrix": l0c2l1_is_as_left_matrix, "is_l0c2l1_trans": l0c2l1_is_trans}


@TestCaseLoader.reg_params_handler(ops=["Cast"])
def cast_params_func(params: dict):
    params["mode"] = int(params.get("mode", "0"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestOneHot/OneHotOperationTest.TestOneHot",
    ]
)
def gen_onehot_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        num_classes = config["params"]["num_classes"]
        return [np.eye(num_classes, dtype=np.int32)[np.array(inputs[0])]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("OneHot", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCast/CastOperationTest.TestCast",
    ]
)
def gen_cast_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        output_dtype = config.get("output_tensors")[0].get("dtype")
        dst_dtype = params.get("dst_dtype", output_dtype)
        if inputs[0].dtype == bfloat16:
            dtype_out = get_dtype_by_name(dst_dtype)
            x = inputs[0].astype(dtype_out)
        else:
            dtype_out = get_dtype_by_name(dst_dtype, True)
            if dtype_out is None:
                return [inputs[0].astype(get_dtype_by_name(dst_dtype))]
            x = torch.from_numpy(inputs[0])
            if dtype_out == torch.bfloat16:
                x = x.to(torch.float32).numpy().astype(bfloat16)
            else:
                x = x.to(dtype_out).numpy()

        return [x]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Cast", golden_func, output, case_index)


def l0c2l_golden_generate(inputs: list, config: dict):
    params = config.get("params")
    l0c2l1_tensor = params["l0c2l1_tensor"]
    l0c2l1_min = l0c2l1_tensor["data_range"]["min"]
    l0c2l1_max = l0c2l1_tensor["data_range"]["max"]
    tensor_l0c2l1 = np.random.uniform(l0c2l1_min, l0c2l1_max, l0c2l1_tensor["shape"]).astype(
        get_dtype_by_name(l0c2l1_tensor["dtype"])
    )
    tensor_l0c2l1_data = tensor_l0c2l1
    tensor_a = (inputs[0] if not params.get("transA")
        else np.swapaxes(inputs[0], inputs[0].ndim - 2, inputs[0].ndim - 1)
    )
    tensor_b = (inputs[1] if not params.get("transB")
        else np.swapaxes(inputs[1], inputs[1].ndim - 2, inputs[1].ndim - 1)
    )
    tensor_l0c2l1 = tensor_l0c2l1 if not l0c2l1_tensor["need_trans"] else \
        np.swapaxes(tensor_l0c2l1, tensor_l0c2l1.ndim - 2, tensor_l0c2l1.ndim - 1)

    tensor_tmp = torch.matmul(
            torch.from_numpy(tensor_a.astype(np.float32)).to(torch.float32),
            torch.from_numpy(tensor_b.astype(np.float32)).to(torch.float32)
        ).to(torch.float32)
    if params.get("relu_type") == 1:
        tensor_tmp = F.relu(tensor_tmp)
    if params.get("scale_value"):
        mask = 0xFFFFE000
        scale_data = np.float32([params.get("scale_value")]).view(np.uint32)
        uint32_num = scale_data & mask
        fp32_scale_modified = uint32_num.view(np.float32)[0]
        tensor_tmp = tensor_tmp * fp32_scale_modified
    if params.get("quant_type") is not None and params.get("quant_type") == 2:
        # quant type中no quant为0, pertensor为1, perchannel为2.
        tensor_tmp = tensor_tmp * inputs[2]
    tensor_tmp = tensor_tmp.numpy().astype(get_dtype_by_name(l0c2l1_tensor["dtype"]))
    l0c2l1_is_left_matrix = str_to_bool(params["l0c2l1_params"]["is_as_left_matrix"])
    if l0c2l1_is_left_matrix:
        tensor_out = torch.matmul(
            torch.from_numpy(tensor_l0c2l1.astype(np.float32)).to(torch.float32),
            torch.from_numpy(tensor_tmp.astype(np.float32)).to(torch.float32)
        ).to(torch.float32).numpy().astype(get_dtype_by_name(params["outDtype"]))
    else:
        tensor_out = torch.matmul(
            torch.from_numpy(tensor_tmp.astype(np.float32)).to(torch.float32),
            torch.from_numpy(tensor_l0c2l1.astype(np.float32)).to(torch.float32)
        ).to(torch.float32).numpy().astype(get_dtype_by_name(params["outDtype"]))

    if l0c2l1_tensor["format"] == "NZ":
        tensor_l0c2l1_data = trans_nd_to_fractal_nz(tensor_l0c2l1_data)

    if params.get("isCMatrixNz"):
        tensor_out = trans_nd_to_fractal_nz(tensor_out, True)

    return [tensor_out, tensor_l0c2l1_data]


def matmul_golden_func(inputs: list, config: dict):
    params = config.get("params")
    if params.get("l0c2l1_tensor"):
        return l0c2l_golden_generate(inputs, config)
    tensor_a = (
        inputs[0]
        if not params.get("transA")
        else np.swapaxes(inputs[0], inputs[0].ndim - 2, inputs[0].ndim - 1)
    )
    tensor_b = (
        inputs[1]
        if not params.get("transB")
        else np.swapaxes(inputs[1], inputs[1].ndim - 2, inputs[1].ndim - 1)
    )
    assert params.get("outDtype") in ("fp32", "fp16", "bf16", "int32")
    if params.get("outDtype") in ("fp32", "fp16", "bf16"):
        tensor_c = torch.matmul(
            torch.from_numpy(tensor_a.astype(np.float32)).to(torch.float32),
            torch.from_numpy(tensor_b.astype(np.float32)).to(torch.float32)
        ).to(torch.float32)
        if params.get("bias_info") is not None and params.get("bias_info") != "":
            tensor_c = tensor_c + torch.from_numpy(inputs[3].astype(np.float32)).to(torch.float32)
    else:
        tensor_c = torch.matmul(
            torch.from_numpy(tensor_a.astype(np.int32)).to(torch.int32),
            torch.from_numpy(tensor_b.astype(np.int32)).to(torch.int32)
        ).to(torch.int32)
        if params.get("bias_info") is not None and params.get("bias_info") != "":
            tensor_c = tensor_c + torch.from_numpy(inputs[3].astype(np.int32)).to(torch.int32)

    if params.get("relu_type") == 1:
        tensor_c = F.relu(tensor_c)
    if params.get("scale_value"):
        tensor_c = tensor_c * params.get("scale_value")
    if params.get("quant_type") is not None and params.get("quant_type") == 2:
        # quant type中no quant为0, pertensor为1, perchannel为2.
        tensor_c = tensor_c * inputs[2]
    tensor_c = tensor_c.numpy()
    tensor_type = get_dtype_by_name(params.get("outDtype"))
    if params.get("outDtype") == "bf16":
        tensor_type = bfloat16
    tensor_c = tensor_c.astype(tensor_type)

    if params.get("isCMatrixNz"):
        tensor_c = trans_nd_to_fractal_nz(tensor_c, True)

    return [tensor_c]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExpand/ExpandOperationTest.TestExpand",
    ]
)
def gen_expand_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        output_dtype = config.get("output_tensors")[0].get("dtype")
        output_shape = config.get("output_tensors")[0].get("shape")
        dst_dtype = params.get("dst_dtype", output_dtype)
        dst_shape = params.get("dst_shape", output_shape)
        if inputs[0].dtype == bfloat16:
            dtype_out = get_dtype_by_name(dst_dtype)
            x = inputs[0].astype(dtype_out)
        else:
            dtype_out = get_dtype_by_name(dst_dtype, True)
            if dtype_out is None:
                x = inputs[0].astype(get_dtype_by_name(dst_dtype))
            else:
                x = torch.from_numpy(inputs[0])
                if dtype_out == torch.bfloat16:
                    x = x.to(torch.float32).numpy().astype(bfloat16)
                else:
                    x = x.to(dtype_out).numpy()

        x = np.broadcast_to(x, np.array(dst_shape))
        x = np.copy(x)
        return [x]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Expand", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMatmul/MatmulOperationTest.TestMatmul",
    ], 
    version=0,
    timeout=0
)
def gen_matmul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Matmul", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMatmulVerify/MatmulVerifyOperationTest.TestMatmulVerify",
    ]
)
def gen_matmulverify_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("MatmulVerify", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBatchMatmul/BatchMatmulOperationTest.TestBatchMatmul",
    ]
)
def gen_batchmatmul_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BatchMatmul", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestBatchMatmulVerify/BatchMatmulVerifyOperationTest.TestBatchMatmulVerify",
    ]
)
def gen_batchmatmulverify_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("BatchMatmulVerify", matmul_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExp/ExpOperationTest.TestExp",
    ]
)
def gen_exp_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.exp(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Exp", golden_func, output, case_index)

@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAbs/AbsOperationTest.TestAbs",
    ]
)
def gen_abs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        return [np.abs(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Abs", golden_func, output, case_index)

@GoldenRegister.reg_golden_func(
    case_names=[
        "TestNeg/NegOperationTest.TestNeg",
    ]
)
def gen_neg_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.negative(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Neg", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLog/LogOperationTest.TestLog",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        base = _config["params"]["base"]
        input_dtype = inputs[0].dtype
        if input_dtype == np.float16:
            inputs[0].astype(np.float32)
        if base == "e":
            output = [np.log(inputs[0])]
        elif base == "2":
            output = [np.log2(inputs[0])]
        elif base == "10":
            output = [np.log10(inputs[0])]
        if input_dtype == np.float16:
            output = [output[0].astype(np.float16)]
        return output

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Log", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestPows/PowsOperationTest.TestPows",
    ]
)
def gen_log_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, _config: dict):
        params = _config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [np.power(inputs[0], params["scalar"])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Pows", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRsqrt/RsqrtOperationTest.TestRsqrt",
    ]
)
def gen_rsqrt_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [1 / np.sqrt(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Rsqrt", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSqrt/SqrtOperationTest.TestSqrt",
    ]
)
def gen_sqrt_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [np.sqrt(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sqrt", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdd/AddOperationTest.TestAdd",
    ]
)
def gen_add_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] + inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Add", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLogicalNot/LogicalNotOperationTest.TestLogicalNot",
    ]
)
def gen_logical_not_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x = safe_tensor_conversion(inputs[0])
        x = torch.logical_not(x)
        return [np.array(x)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("LogicalNot", golden_func, output, case_index)

@GoldenRegister.reg_golden_func(
    case_names=[
        "TestLogicalAnd/LogicalAndOperationTest.TestLogicalAnd",
    ]
)
def gen_logical_and_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        x0 = torch.tensor(inputs[0])
        x1 = torch.tensor(inputs[1])
        y = torch.logical_and(x0, x1)
        return [y.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("LogicalAnd", golden_func, output, case_index)



@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSub/SubOperationTest.TestSub",
    ]
)
def gen_sub_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] - inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sub", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMul/MulOperationTest.TestMul",
    ]
)
def gen_mul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] * inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Mul", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Maximum", "Minimum"])
def maximum_minimum_parameter(params: dict):
    params["scalar"] = 0 if not params.get("scalar") else params["scalar"]
    params["scalar_type"] = "" if not params.get("scalar_type") else params["scalar_type"]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMaximum/MaximumOperationTest.TestMaximum",
    ]
)
def gen_maximum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs: list, config: dict):
        is_element_mode = len(inputs) <= 1
        if is_element_mode:
            params = config["params"]
            x = inputs[0]
            scalar_type = params.get("scalar_type", "fp32")
            scalar = get_dtype_by_name(scalar_type)(params["scalar"])
            y = np.where(x < scalar, scalar, x)
            return [y]
        return [np.maximum(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Maximum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMinimum/MinimumOperationTest.TestMinimum",
    ]
)
def gen_minimum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs: list, config: dict):
        is_element_mode = len(inputs) <= 1
        if is_element_mode:
            params = config["params"]
            x = inputs[0]
            scalar_type = params.get("scalar_type", "fp32")
            scalar = get_dtype_by_name(scalar_type)(params["scalar"])
            y = np.where(x > scalar, scalar, x)
            return [y]
        return [np.minimum(inputs[0], inputs[1])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Minimum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDiv/DivOperationTest.TestDiv",
    ]
)
def gen_div_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, _config: dict):
        return [inputs[0] / inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Div", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdds/AddsOperationTest.TestAdds",
    ]
)
def gen_adds_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] + params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Adds", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMuls/MulsOperationTest.TestMuls",
    ]
)
def gen_muls_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] * params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Muls", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDivs/DivsOperationTest.TestDivs",
    ]
)
def gen_divs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] / params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Divs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestFull/FullOperationTest.TestFull",
    ]
)
def gen_vector_dup_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [np.full(inputs[0].shape, params["scalar"], inputs[0].dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Full", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSubs/SubsOperationTest.TestSubs",
    ]
)
def gen_subs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        params["scalar_type"] = params.get("scalar_type", "fp32")
        params["scalar"] = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] - params["scalar"]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Subs", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Sum", "Amax", "Amin"])
def params_dims_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["keepDim"] = params.get("keepDim", True)
    assert isinstance(params["keepDim"], bool), "keepDim must be bool"
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSum/SumOperationTest.TestSum",
    ]
)
def gen_reduce_sum_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.sum(axis=dims[0], keepdims=keepdim)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sum", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAmax/AmaxOperationTest.TestAmax",
    ]
)
def gen_reduce_max_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = inputs[0]
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [x.max(axis=dims[0], keepdims=keepdim)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Amax", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAmin/AminOperationTest.TestAmin",
    ]
)
def gen_reduce_min_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        dims = params["dims"]
        keepdim = params.get("keepDim", True)
        return [inputs[0].min(axis=dims[0], keepdims=keepdim)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Amin", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Transpose"])
def transpose_params_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["first_dim"] = int(params.pop("first_dim"))
    params["second_dim"] = int(params.pop("second_dim"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTranspose/TransposeOperationTest.TestTranspose",
    ]
)
def gen_transpose_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        return [np.transpose(inputs[0], axes=tuple(params["dims"]))]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Transpose", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestWhere/WhereOperationTest.TestWhere",
    ]
)
def gen_where_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        condition_np = inputs[0]
        params = config.get("params")
        flag = params["flag"]
        x_scalar = params["x_scalar"]
        y_scalar = params["y_scalar"]

        def castCondition(cond: np.ndarray) -> torch.Tensor:
            if cond.dtype == np.bool_:
                return torch.from_numpy(cond).bool()
            elif cond.dtype == np.uint8:
                orig_shape = cond.shape
                N = int(np.prod(orig_shape[:-1]))
                D = orig_shape[-1]
                reshaped = cond.reshape(N, D)
                bits = ((reshaped[:, :, None] >> np.arange(8)) & 1).astype(bool)
                expanded = bits.reshape(N, D * 8)
                new_shape = list(orig_shape[:-1]) + [D * 8]
                return torch.from_numpy(expanded.reshape(new_shape)).bool()
            else:
                raise TypeError(f"Unsupported condition dtype: {cond.dtype}")
        condition = castCondition(condition_np)

        x = safe_tensor_conversion(inputs[1])
        y = safe_tensor_conversion(inputs[2])

        if flag == 0:
            res = torch.where(condition, x, y)
        elif flag == 1:
            res = torch.where(condition, x, y_scalar)
        elif flag == 2:
            res = torch.where(condition, x_scalar, y)
        elif flag == 3:
            res = torch.where(condition, x_scalar, y_scalar)
        else:
            raise ValueError(f"Invalid flag value: {flag}")

        if res.dtype == torch.bfloat16:
            res = res.to(torch.float32).numpy()
        else:
            res = res.numpy()
        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Where", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["TopK"])
def topk_params_func(params: dict):
    params["dims"] = parse_list_str(params.get("dims"))
    params["count"] = parse_list_str(params.get("count"))
    params["islargest"] = [bool(x) for x in parse_list_str(params.get("islargest"))]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTopK/TopKOperationTest.TestTopK",
    ]
)
def gen_topk_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        x = torch.from_numpy(inputs[0])
        dims = params["dims"]
        count = params["count"]
        islargest = params["islargest"]
        val, idx = x.topk(count[0], dim=dims[0], largest=islargest[0], sorted=True)
        return [val.numpy(), idx.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("TopK", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestRange/RangeOperationTest.TestRange",
    ]
)
def gen_numpy_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        start = params["start"]
        end = params["end"]
        step = params["step"]
        output_tensors_type = config["output_tensors"][0]["dtype"]
        inputdata_type = get_dtype_by_name(output_tensors_type)
        if isinstance(start, float) or isinstance(end, float) or isinstance(step, float):
            if inputdata_type == bfloat16:
                result = torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float32)
                return [result.numpy().astype(bfloat16)]
            elif inputdata_type == np.float16:
                return [torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float16).numpy()]
            else:
                return [torch.arange(np.float32(start), np.float32(end), np.float32(step), dtype=torch.float32).numpy()]
        return [torch.arange(start, end, step).numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Range", golden_func, output, case_index)


class GatherError(Exception):
    pass


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGather/GatherOperationTest.TestGather",
    ]
)
def gen_gather_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs: list, config: dict):

        def _normalize_axis(axis, ndim):
            """把 axis 转成 [0, ndim) 区间"""
            if axis < 0:
                axis += ndim
            if not (0 <= axis < ndim):
                raise GatherError(f"axis={axis} 越界，ndim={ndim}")
            return axis

        def np_gather(params, indices, axis=0, batch_dims=0, *, validate_indices=True):
            """
            用 NumPy 模拟 TensorFlow 的 tf.gather。
            当 batch_dims==0 时直接走 np.take；当 batch_dims>0 时手动做 batch 切片再拼接。
            """
            p = np.asarray(params)
            idx = np.asarray(indices)

            # ---- 基础校验 ----
            if not np.issubdtype(idx.dtype, np.integer):
                raise GatherError("indices 必须是整数类型")
            if p.ndim == 0:
                raise GatherError("params 必须至少 1 维")

            axis = _normalize_axis(axis, p.ndim)

            # ---- batch_dims 归一化 ----
            if batch_dims < 0:
                batch_dims += min(p.ndim, idx.ndim)
            if not (0 <= batch_dims <= axis):
                raise GatherError("batch_dims 必须满足 0<=batch_dims<=axis")

            # ---- batch_dims == 0：直接 np.take ----
            if batch_dims == 0:
                if validate_indices:
                    size = p.shape[axis]
                    if idx.size > 0:
                        low, high = idx.min(), idx.max()
                        if low < -size or high >= size:
                            raise IndexError("indices 越界")
                mode = "raise" if validate_indices else "wrap"
                return np.take(p, idx, axis=axis, mode=mode)

            # ---- batch_dims > 0：逐 batch 切片 ----
            if idx.ndim < batch_dims:
                raise GatherError("indices 的秩必须 >= batch_dims")

            # 前 batch_dims 维必须完全对齐，否则按 TF 语义报错
            if p.shape[:batch_dims] != idx.shape[:batch_dims]:
                raise GatherError("params 与 indices 的前 batch_dims 维不一致")

            # 先做一次轴搬运，把 axis 移到 batch_dims 后面，方便后面统一处理
            move_to = batch_dims  # 目标位置
            if axis != move_to:
                p = np.moveaxis(p, axis, move_to)
                # 注意：搬完后真正的 轴 就是 move_to
                axis = move_to

            # 结果形状模板
            remain_dim = batch_dims + 1
            res_shape = (
                p.shape[:batch_dims]  # 保留的 batch 维
                + idx.shape[batch_dims:]  # indices 带来的新维
                + p.shape[remain_dim:]  # 剩下的数据维
            )

            # 提前分配结果
            res = np.empty(res_shape, dtype=p.dtype)

            # 遍历所有 batch 切片
            for b in np.ndindex(p.shape[:batch_dims]):
                # 取出当前 batch 的 params 切片，形状 (...,)  1D 轴
                p_slice = p[b]  # shape: [D_axis] + p.shape[batch_dims+1:]
                # 取出当前 batch 的 indices 切片
                idx_slice = idx[b]  # shape: idx.shape[batch_dims:]

                if validate_indices:
                    size = p_slice.shape[0]
                    if idx_slice.size > 0:
                        low, high = idx_slice.min(), idx_slice.max()
                        if low < -size or high >= size:
                            raise IndexError("indices 越界")

                mode = "raise" if validate_indices else "wrap"
                gathered = np.take(p_slice, idx_slice, axis=0, mode=mode)
                # 写回结果
                res[b] = gathered

            # 最后把 axis 搬回原始位置
            if move_to != axis:
                res = np.moveaxis(res, move_to, axis)
            return res

        params = config.get("params")
        axis = params["axis"]
        res = np_gather(inputs[0], inputs[1], axis)
        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Gather", golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["GatherElement", "Concat", "Gather"])
def params_axis_func(params: dict):
    params["axis"] = int(params.get("axis"))
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestGatherElement/GatherElementOperationTest.TestGatherElement",
    ]
)
def gen_gatherelement_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs: list, config: dict):
        params = config.get("params")
        axis = params["axis"]
        if inputs[1].dtype == np.int32:
            indices = torch.from_numpy(inputs[1]).long()
        else:
            indices = torch.from_numpy(inputs[1])
        if inputs[0].dtype == bfloat16:
            src = torch.from_numpy(inputs[0].astype(np.float32))
            res = src.gather(axis, indices).numpy().astype(bfloat16)
        else:
            src = torch.from_numpy(inputs[0])
            res = src.gather(axis, indices).numpy()

        return [res]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("GatherElement", golden_func, output, case_index)


def cumsum_golden_func(inputs: list, config: dict):
    params = config.get("params")
    axis = params["axis"]
    inputTensor = torch.from_numpy(inputs[0])
    res = torch.cumsum(inputTensor, axis)
    
    return [res.numpy()]

@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCumSum/CumSumOperationTest.TestCumSum",
    ]
)
def gen_cumsum_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("CumSum", cumsum_golden_func, output, case_index)


def from_numpy(array: np.array):
    is_bfloat16 = array.dtype == bfloat16
    if is_bfloat16:
        result = torch.tensor(array.astype(np.float32), dtype=torch.bfloat16)
    else:
        result = torch.tensor(array)
    return result


def to_numpy(tensor: torch.Tensor):
    is_bfloat16 = tensor.dtype == torch.bfloat16
    if is_bfloat16:
        result = tensor.to(torch.float32).numpy().astype(bfloat16)
    else:
        result = tensor.numpy()
    return result


def indexadd_golden_func(inputs: list, config: dict):
    params = config.get("params")
    axis = params["axis"]
    self = from_numpy(inputs[0])
    source = from_numpy(inputs[1])
    indices = from_numpy(inputs[2])
    try:
        alp = float(params["alpha"])
    except (KeyError, ValueError, TypeError):
        alp = 1
    res = self.index_add(axis, indices, source, alpha=alp)
    
    return [to_numpy(res)]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestIndexAdd/IndexAddOperationTest.TestIndexAdd",
    ]
)
def gen_indexadd_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("IndexAdd", indexadd_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestIndexAdd_/IndexAdd_OperationTest.TestIndexAdd_",
    ]
)
def gen_indexadd__op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("IndexAdd_", indexadd_golden_func, output, case_index)


@TestCaseLoader.reg_params_handler(ops=["Scatter", "Scatter_", "ScatterTensor", "Scatter_Tensor"])
def params_axis_reduce_func(params: dict):
    params["axis"] = int(params.get("axis"))
    params["reduce"] = "" if params["reduce"] is None else params["reduce"]
    return params

def scatter_golden_func(inputs, config: dict):
    params = config.get("params")
    axis = params["axis"]
    reduceop = params["reduce"]
    scalar = params["src"]
    indices = torch.from_numpy(inputs[1])

    # 和pypto测试golden生成保持一致，fp16和bf16先转换为fp32在进行计算，否则golden精度会不如实际npu计算结果，导致比对失败
    if inputs[0].dtype == bfloat16 or inputs[0].dtype == np.float16:
        src = torch.from_numpy(inputs[0].astype(np.float32))
        if len(reduceop) == 0 or reduceop == "None":
            res = src.scatter(axis, indices, scalar).numpy().astype(inputs[0].dtype)
        else:
            res = (
                src.scatter(axis, indices, scalar, reduce=reduceop)
                .numpy()
                .astype(inputs[0].dtype)
            )
    else:
        src = torch.from_numpy(inputs[0])
        if len(reduceop) == 0 or reduceop == "None":
            res = src.scatter(axis, indices, scalar).numpy()
        else:
            res = src.scatter(axis, indices, scalar, reduce=reduceop).numpy()

    return [res]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatter/ScatterOperationTest.TestScatter",
    ]
)
def gen_scatter_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Scatter", scatter_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatter_/Scatter_OperationTest.TestScatter_",
    ]
)
def gen_scatter__op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Scatter_", scatter_golden_func, output, case_index)


def scatter_tensor_golden_func(inputs, config: dict):
    params = config.get("params")
    axis = params["axis"]
    reduceop = params["reduce"]
    indices = torch.from_numpy(inputs[1])

    if inputs[0].dtype == bfloat16:
        dst = torch.from_numpy(inputs[0].astype(np.float32))
        src = torch.from_numpy(inputs[2].astype(np.float32))
        if len(reduceop) == 0 or reduceop == "None":
            res = dst.scatter(axis, indices, src).numpy().astype(bfloat16)
        else:
            res = dst.scatter(axis, indices, src, reduce=reduceop).numpy().astype(bfloat16)
    else:
        dst = torch.from_numpy(inputs[0])
        src = torch.from_numpy(inputs[2])
        if len(reduceop) == 0 or reduceop == "None":
            res = dst.scatter(axis, indices, src).numpy()
        else:
            res = dst.scatter(axis, indices, src, reduce=reduceop).numpy()

    return [res]


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatterTensor/ScatterTensorOperationTest.TestScatterTensor",
    ]
)
def gen_scatter_tensor_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("ScatterTensor", scatter_tensor_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestScatter_Tensor/Scatter_TensorOperationTest.TestScatter_Tensor",
    ]
)
def gen_scatter__tensor_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Scatter_Tensor", scatter_tensor_golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestConcat/ConcatOperationTest.TestConcat",
    ]
)
def gen_concat_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs, config: dict):
        params = config.get("params")
        axis = params["axis"]
        output_tensors_type = config["output_tensors"][0]["dtype"]
        inputdata_type = get_dtype_by_name(output_tensors_type)
        if inputdata_type == bfloat16:
            inputs_tensors = [
                torch.as_tensor(x.astype(np.float32)).to(torch.bfloat16) for x in inputs
            ]
        else:
            inputs_tensors = [torch.as_tensor(x) for x in inputs]
        res = torch.cat(inputs_tensors, dim=axis)
        if inputdata_type == bfloat16:
            res = res.to(torch.float32).numpy().astype(bfloat16)
            return [res]
        return [res.numpy()]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Concat", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCompare/CompareOperationTest.TestCompare",
        "TestCompareBitMode/CompareOperationTest.TestCompareBitMode",
        "TestCmps/CmpsOperationTest.TestCmps",
        "TestCmpsBitMode/CmpsOperationTest.TestCmpsBitMode"
    ]
)
def gen_compare_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs: list, config: dict):
        params = config.get("params", {})
        operation = params["compare_op"]
        mode = params.get("mode", "bool")
        input1 = safe_tensor_conversion(inputs[0])
        if len(inputs) > 1:
            input2 = safe_tensor_conversion(inputs[1])
        elif "scalar" in params:
            input2 = params["scalar"]
        else:
            input2 = params.get("other", 0)
        cmp_operations = {
            "eq": torch.eq,
            "ne": torch.ne,
            "lt": torch.lt,
            "le": torch.le,
            "gt": torch.gt,
            "ge": torch.ge,
        }
        result = cmp_operations[operation](input1, input2)
        if mode == "bit":
            bool_np = result.numpy()

            last_dim = bool_np.shape[-1] if bool_np.shape else 0
            if last_dim % 8 != 0:
                raise ValueError(
                    f"Last dimension {last_dim} must be divisible by 8 in BIT mode"
                )
            new_shape = list(bool_np.shape[:-1]) + [last_dim // 8, 8]
            bool_reshaped = bool_np.reshape(new_shape)
            bitmask = np.packbits(bool_reshaped, axis=-1, bitorder="little")
            bitmask = bitmask.reshape(bool_np.shape[:-1] + (last_dim // 8,))
            result = torch.tensor(bitmask, dtype=torch.uint8)
        else:
            result = result.to(torch.bool)
        return [result.numpy()]

    op_type = "Compare" if "Compare" in case_name else "Cmps"
    logging.debug("Case(%s), %s Golden creating...", case_name, op_type)
    return gen_op_golden(op_type, golden_func, output, case_index)


def as_float(value):
    """将数字、inf、nan、-inf 等字符串转换为浮点数"""
    if isinstance(value, str):
        value = value.lower()
        if not value:
            return None
    value = float(value)
    return value


def safe_tensor_conversion(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == np.dtype('bfloat16'):
        return torch.tensor(arr.astype(np.float32), dtype=torch.bfloat16)
    else:
        return torch.tensor(arr)


def element_mode(inputs, params, is_bfloat16):
    test_type = int(params["test_type"])
    min_dtype_str, max_dtype_str = params.get("min_dtype", ""), params.get(
        "max_dtype", ""
    )
    if min_dtype_str and test_type in [-1, 0, 1, 2, 7]:
        min_dtype = get_dtype_by_name(min_dtype_str, is_torch=True)
        min_value = as_float(params.get("min_value"))
        if min_value is not None:
            min_ = torch.tensor(min_value, dtype=min_dtype)
        else:
            min_ = None
    else:
        min_ = None
    if max_dtype_str and test_type in [-1, 0, 1, 2, 8]:
        max_dtype = get_dtype_by_name(max_dtype_str, is_torch=True)
        max_value = as_float(params.get("max_value"))
        if max_value is not None:
            max_ = torch.tensor(max_value, dtype=max_dtype)
        else:
            max_ = None
    else:
        max_ = None
    return min_, max_


def tensor_mode(inputs, params, is_bfloat16):
    test_type = int(params["test_type"])

    if test_type in [-1, 0, 1, 2, 5]:
        min_ = (
            torch.tensor(inputs[1].astype(np.float32), dtype=torch.bfloat16)
            if is_bfloat16
            else torch.tensor(inputs[1])
        )
    else:
        min_ = None
    if test_type in [-1, 0, 1, 2, 4]:
        max_ = (
            torch.tensor(inputs[2].astype(np.float32), dtype=torch.bfloat16)
            if is_bfloat16
            else torch.tensor(inputs[2])
        )
    else:
        max_ = None
    return min_, max_


@TestCaseLoader.reg_params_handler(ops=["Clip"])
def clip_parameter(params: dict):
    params["max_value"] = "" if not params.get("max_value") else params["max_value"]
    params["min_value"] = "" if not params.get("min_value") else params["min_value"]
    params["max_dtype"] = "" if not params.get("min_dtype") else params["min_dtype"]
    params["min_dtype"] = "" if not params.get("min_dtype") else params["min_dtype"]
    return params


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestClip/ClipOperationTest.TestClip",
    ]
)
def gen_clip_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:

    def golden_func(inputs: list, config: dict):
        params = config.get("params", {})
        is_bfloat16 = inputs[0].dtype == bfloat16
        x = (
            torch.tensor(inputs[0].astype(np.float32), dtype=torch.bfloat16)
            if is_bfloat16
            else torch.tensor(inputs[0])
        )
        min_dtype_str, max_dtype_str = params.get("min_dtype", ""), params.get(
            "max_dtype", ""
        )
        is_element = min_dtype_str or max_dtype_str
        if is_element:
            min_, max_ = element_mode(inputs, params, is_bfloat16)
        else:
            min_, max_ = tensor_mode(inputs, params, is_bfloat16)
        # Torch 不支持两者均为 None
        if min_ is None and max_ is None:
            min_ = float("-inf")
        result = torch.clip(x, min_, max_)
        return [
            (
                result.to(torch.float32).numpy().astype(bfloat16)
                if is_bfloat16
                else result.numpy()
            )
        ]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Clip", golden_func, output, case_index)


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestAdd/AddOperationTest.TestAdd",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_add_op_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file torch_tensor_converter.h
 * \brief Convert PyTorch tensors to device/on-board data (DeviceTensorData).
 */

#pragma once

#include "pybind_common.h"
#include "machine/runtime/device_launcher_binding.h"
#include "interface/inner/dlpack_dtype.h"
#include "tilefwk/data_type.h"

#include <cstdint>
#include <vector>
namespace pypto {

bool ParseDlpackCapsule(py::object &cap, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                       int &deviceId, npu::tile_fwk::DataType &dtypeOut);

bool TryParseDlpack(py::object &torchTensor, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                    int &deviceId, npu::tile_fwk::DataType &dtypeOut,
                    py::object toDlpack = py::none());

class TorchTensorConverter {
public:
    static void Convert(py::sequence &tensors, py::sequence &tensor_defs,
        std::vector<npu::tile_fwk::dynamic::DeviceTensorData> &tensors_data,
        std::vector<int> &device_ids);
};

int ValidateAndGetDeviceId(std::vector<int> &deviceIds);
size_t ValidateInputs(py::sequence &tensors, py::sequence &tensorDefs);

}  // namespace pypto

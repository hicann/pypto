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
 * \file torch_tensor_converter.cpp
 * \brief Implementation of PyTorch tensor to DeviceTensorData conversion.
 */

#include "bindings/torch_tensor_converter.h"
#include "tilefwk/error.h"

#include <stdexcept>
#include <string>

using namespace npu::tile_fwk;
namespace {
    py::object GetTorchToDlpack() {
        try {
            return py::module::import("torch").attr("_C").attr("_to_dlpack");
        } catch (...) {
            return py::none();
        }
    }
}

namespace pypto {
bool ParseDlpackCapsule(py::object &cap, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                        int &deviceId, npu::tile_fwk::DataType &dtypeOut) {
    if (cap.is_none()) return false;
    void *ptr = PyCapsule_GetPointer(cap.ptr(), "dltensor");
    if (!ptr) {
        PyErr_Clear();
        return false;
    }
    DLManagedTensor *tensor = static_cast<DLManagedTensor *>(ptr);
    DLManagedTensor::DLTensor &t = tensor->dl_tensor;

    dataPtr = reinterpret_cast<uintptr_t>(static_cast<char *>(t.data) + t.byte_offset);

    int32_t ndim = t.ndim;
    shape.clear();
    shape.reserve(ndim);
    for (int32_t i = 0; i < ndim; ++i) {
        shape.push_back(t.shape[i]);
    }

    deviceId = t.device_id;
    DlpackDtypeToDataType(t.dtype.code, t.dtype.bits, t.dtype.lanes, &dtypeOut);
    return true;
}

bool TryParseDlpack(py::object &torchTensor, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                    int &deviceId, npu::tile_fwk::DataType &dtypeOut, py::object toDlpack) {
    if (toDlpack.is_none()) toDlpack = GetTorchToDlpack();
    if (toDlpack.is_none()) return false;
    py::object cap;
    try {
        cap = toDlpack(torchTensor);
    } catch (...) {
        PyErr_Clear();
        return false;
    }
    return ParseDlpackCapsule(cap, dataPtr, shape, deviceId, dtypeOut);
}

void TorchTensorConverter::Convert(py::sequence &tensors, py::sequence &tensor_defs,
    std::vector<npu::tile_fwk::dynamic::DeviceTensorData> &tensors_data,
    std::vector<int> &device_ids) {
    using namespace npu::tile_fwk;
    using namespace npu::tile_fwk::dynamic;

    const size_t n = static_cast<size_t>(py::len(tensors));
    tensors_data.reserve(n);
    device_ids.reserve(n);

    py::object toDlpack = GetTorchToDlpack();

    for (size_t i = 0; i < n; i++) {
        py::object torchTensor = tensors[py::int_(i)];
        py::object tensorDef = tensor_defs[py::int_(i)];
        std::vector<int64_t> shape;
        uintptr_t dataPtr = 0;
        int deviceId = -1;
        DataType dtype = DataType::DT_BOTTOM;

        if (!TryParseDlpack(torchTensor, dataPtr, shape, deviceId, dtype, toDlpack)) {
            try {
                dataPtr = static_cast<uintptr_t>(py::cast<int64_t>(torchTensor.attr("data_ptr")()));
                for (auto dim : torchTensor.attr("shape")) {
                    shape.push_back(py::cast<int64_t>(dim));
                }
            } catch (...) {
                PyErr_Clear();
                throw std::runtime_error("Input tensor is not a valid torch tensor type");
            }
        }
        if (dtype == DataType::DT_BOTTOM) {
            auto base = py::getattr(tensorDef, "_base", py::none());
            if (!base.is_none() && py::isinstance<Tensor>(base)) {
                dtype = base.cast<Tensor &>().GetDataType();
            } else {
                dtype = tensorDef.attr("dtype").cast<DataType>();
            }
        }

        // Get format from torch tensor, it takes 10~15 us per tensor
        // If you get format from tensorDef, you need to mark format in the kernel parameters
        TileOpFormat format = TileOpFormat::TILEOP_ND;
        std::string device_type = py::cast<std::string>(torchTensor.attr("device").attr("type"));
        if (device_type == "npu") {
            py::module torch_npu = py::module::import("torch_npu");
            int npu_format = py::cast<int>(torch_npu.attr("get_npu_format")(torchTensor));
            if (npu_format == 29) {
                format = TileOpFormat::TILEOP_NZ;
            }
        }
        tensors_data.emplace_back(dtype, dataPtr, shape, format);

        device_ids.push_back(deviceId);
    }
}

int ValidateAndGetDeviceId(std::vector<int> &deviceIds) {
    int deviceId = -1;
    for (size_t i = 0; i < deviceIds.size(); i++) {
        if (deviceIds[i] < 0) continue;

        if (deviceId < 0) {
            deviceId = deviceIds[i];
        } else if (deviceIds[i] != deviceId) {
            throw std::runtime_error(
                "Tensor at index " + std::to_string(i) +
                " is on device " + std::to_string(deviceIds[i]) +
                ", expected device " + std::to_string(deviceId));
        }
    }

    if (deviceId < 0) {
        throw std::runtime_error(
            "Unable to determine device ID: ensure all tensors support DLPack");
    }
    return deviceId;
}

size_t ValidateInputs(py::sequence &tensors, py::sequence &tensorDefs) {
    size_t n = static_cast<size_t>(py::len(tensors));
    CHECK(n == static_cast<size_t>(py::len(tensorDefs)))
        << "Input length mismatch: tensors(" << n << ") vs tensor_defs(" << py::len(tensorDefs) << ")";
    CHECK(n != 0) << "Empty tensor list";
    return n;
}

}  // namespace pypto

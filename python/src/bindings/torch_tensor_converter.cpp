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

void ParseTensorData(py::object &torchTensor, py::object &tensorDef, py::object &toDlpack,
                     uintptr_t &dataPtr, std::vector<int64_t> &shape, DataType &dtype) {
    if (!pypto::TryParseDlpack(torchTensor, dataPtr, shape, dtype, toDlpack)) {
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
}

}  // namespace

namespace pypto {
bool ParseDlpackCapsule(py::object &cap, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                        npu::tile_fwk::DataType &dtypeOut) {
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

    DlpackDtypeToDataType(t.dtype.code, t.dtype.bits, t.dtype.lanes, &dtypeOut);
    return true;
}

bool TryParseDlpack(py::object &torchTensor, uintptr_t &dataPtr, std::vector<int64_t> &shape,
                    npu::tile_fwk::DataType &dtypeOut, py::object toDlpack) {
    if (toDlpack.is_none()) toDlpack = GetTorchToDlpack();
    if (toDlpack.is_none()) return false;
    py::object cap;
    try {
        cap = toDlpack(torchTensor);
    } catch (...) {
        PyErr_Clear();
        return false;
    }
    return ParseDlpackCapsule(cap, dataPtr, shape, dtypeOut);
}

int TorchTensorConverter::Convert(py::sequence &tensors, py::sequence &tensor_defs,
    std::vector<npu::tile_fwk::dynamic::DeviceTensorData> &tensors_data) {
    const size_t n = static_cast<size_t>(py::len(tensors));
    tensors_data.reserve(n);

    py::object toDlpack = GetTorchToDlpack();
    py::object device = py::none();
    
    py::module torch_npu;
    for (size_t i = 0; i < n; i++) {
        py::object torchTensor = tensors[py::int_(i)];
        py::object tensorDef = tensor_defs[py::int_(i)];
        std::vector<int64_t> shape;
        uintptr_t dataPtr = 0;
        DataType dtype = DataType::DT_BOTTOM;

        ParseTensorData(torchTensor, tensorDef, toDlpack, dataPtr, shape, dtype);

        TileOpFormat format = TileOpFormat::TILEOP_ND;
        py::object tensorDevice = torchTensor.attr("device");

        if (tensorDef.attr("explict_format").cast<bool>()) {
            format = tensorDef.attr("format").cast<TileOpFormat>();
        } else {
            std::string device_type = py::cast<std::string>(tensorDevice.attr("type"));
            if (device_type == "npu") {
                if (torch_npu.ptr() == nullptr) {
                    torch_npu = py::module::import("torch_npu");
                }
                int npu_format = py::cast<int>(torch_npu.attr("get_npu_format")(torchTensor));
                if (npu_format == 29) {
                    format = TileOpFormat::TILEOP_NZ;
                }
            }
        }
        
        tensors_data.emplace_back(dtype, dataPtr, shape, format);

        if (device.is_none()) {
            device = tensorDevice;
        } else if (!device.equal(tensorDevice)) {
            throw std::runtime_error("All input tensors must be on the same device");
        }
    }
    if (py::getattr(device, "type").cast<std::string>() != "npu") {
        throw std::runtime_error("Not npu device");
    }
    return py::getattr(device, "index").cast<int>();
}

size_t ValidateInputs(py::sequence &tensors, py::sequence &tensorDefs) {
    size_t n = static_cast<size_t>(py::len(tensors));
    CHECK(n == static_cast<size_t>(py::len(tensorDefs)))
        << "Input length mismatch: tensors(" << n << ") vs tensor_defs(" << py::len(tensorDefs) << ")";
    CHECK(n != 0) << "Empty tensor list";
    return n;
}

}  // namespace pypto

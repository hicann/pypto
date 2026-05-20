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
#include "interface/utils/error.h"

#include <stdexcept>
#include <string>

using namespace npu::tile_fwk;
namespace {
py::object GetTorchToDlpack()
{
    try {
        return py::module::import("torch").attr("_C").attr("_to_dlpack");
    } catch (...) {
        return py::none();
    }
}

void ParseTensorData(
    py::object& torchTensor, py::object& tensorDef, py::object& toDlpack, uintptr_t& dataPtr,
    std::vector<int64_t>& shape, DataType& dtype)
{
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
    if (dtype == DataType::DT_BOTTOM || (!tensorDef.is_none() && !tensorDef.attr("explicit_dtype").is_none())) {
        auto base = py::getattr(tensorDef, "_base", py::none());
        if (!base.is_none() && py::isinstance<Tensor>(base)) {
            dtype = base.cast<Tensor&>().GetDataType();
        } else {
            dtype = tensorDef.attr("dtype").cast<DataType>();
        }
    }
}

TileOpFormat ResolveFormat(py::object& tensorDef, Tensor& baseTensor, py::object& torchTensor, py::module& torch_npu)
{
    if (!tensorDef.attr("explicit_format").is_none()) {
        return baseTensor.Format();
    }
    std::string device_type = py::cast<std::string>(torchTensor.attr("device").attr("type"));
    if (device_type == "npu") {
        if (torch_npu.ptr() == nullptr) {
            torch_npu = py::module::import("torch_npu");
        }
        int npu_format = py::cast<int>(torch_npu.attr("get_npu_format")(torchTensor));
        if (npu_format == 29) {
            return TileOpFormat::TILEOP_NZ;
        }
    }
    return TileOpFormat::TILEOP_ND;
}

py::object ConvertSingleTensor(
    py::object torchTensor, py::object tensorDef, py::object toDlpack, py::module& torch_npu,
    npu::tile_fwk::dynamic::DeviceTensorData& out)
{
    std::vector<int64_t> shape;
    uintptr_t dataPtr = 0;
    DataType dtype = DataType::DT_BOTTOM;

    ParseTensorData(torchTensor, tensorDef, toDlpack, dataPtr, shape, dtype);
    if (dtype == DataType::DT_FP4_E1M2 || dtype == DataType::DT_FP4_E2M1) {
        shape.back() *= 2;
    }

    auto base = py::getattr(tensorDef, "_base", py::none());
    FE_ASSERT(FeError::INVALID_TYPE, py::isinstance<Tensor>(base))
        << "the '_base' attribute must be a Tensor type";
    auto& t = base.cast<Tensor&>();

    TileOpFormat format = ResolveFormat(tensorDef, t, torchTensor, torch_npu);
        
    out = npu::tile_fwk::dynamic::DeviceTensorData(dtype, dataPtr, shape, format);

    return torchTensor.attr("device");
}

int ValidateDeviceAndReturnIndex(py::object& device)
{
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
        if (py::getattr(device, "type").cast<std::string>() != "cpu") {
            throw std::runtime_error("Not cpu device");
        }
        return 0;
    }
    if (py::getattr(device, "type").cast<std::string>() != "npu") {
        throw std::runtime_error("Not npu device");
    }
    return py::getattr(device, "index").cast<int>();
}

} // namespace

namespace pypto {
bool ParseDlpackCapsule(
    py::object& cap, uintptr_t& dataPtr, std::vector<int64_t>& shape, npu::tile_fwk::DataType& dtypeOut)
{
    if (cap.is_none())
        return false;
    void* ptr = PyCapsule_GetPointer(cap.ptr(), "dltensor");
    if (!ptr) {
        PyErr_Clear();
        return false;
    }
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr);
    DLManagedTensor::DLTensor& t = tensor->dl_tensor;

    dataPtr = reinterpret_cast<uintptr_t>(static_cast<char*>(t.data) + t.byte_offset);

    int32_t ndim = t.ndim;
    shape.clear();
    shape.reserve(ndim);
    for (int32_t i = 0; i < ndim; ++i) {
        shape.push_back(t.shape[i]);
    }

    DlpackDtypeToDataType(t.dtype.code, t.dtype.bits, t.dtype.lanes, &dtypeOut);
    return true;
}

bool TryParseDlpack(
    py::object& torchTensor, uintptr_t& dataPtr, std::vector<int64_t>& shape, npu::tile_fwk::DataType& dtypeOut,
    py::object toDlpack)
{
    if (toDlpack.is_none())
        toDlpack = GetTorchToDlpack();
    if (toDlpack.is_none())
        return false;
    py::object cap;
    try {
        cap = toDlpack(torchTensor);
    } catch (...) {
        PyErr_Clear();
        return false;
    }
    return ParseDlpackCapsule(cap, dataPtr, shape, dtypeOut);
}

int TorchTensorConverter::Convert(
    py::sequence& tensors, py::sequence& tensor_defs,
    std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors_data)
{
    const size_t n = static_cast<size_t>(py::len(tensors));
    tensors_data.reserve(n);

    py::object toDlpack = GetTorchToDlpack();
    py::module torch_npu;
    py::object device = py::none();

    for (size_t i = 0; i < n; i++) {
        tensors_data.emplace_back();
        py::object tensorDevice = ConvertSingleTensor(
            tensors[py::int_(i)], tensor_defs[py::int_(i)], toDlpack, torch_npu, tensors_data.back());
        if (device.is_none()) {
            device = tensorDevice;
        } else if (!device.equal(tensorDevice)) {
            throw std::runtime_error("All input tensors must be on the same device");
        }
    }
    return ValidateDeviceAndReturnIndex(device);
}

size_t ValidateInputs(py::sequence& tensors, py::sequence& tensorDefs)
{
    size_t n = static_cast<size_t>(py::len(tensors));
    CHECK(FeError::INVALID_VAL, n == static_cast<size_t>(py::len(tensorDefs)))
        << "Input length mismatch: tensors(" << n << ") vs tensor_defs(" << py::len(tensorDefs) << ")";
    CHECK(FeError::INVALID_VAL, n != 0) << "Empty tensor list";
    return n;
}

} // namespace pypto

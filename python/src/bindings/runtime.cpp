/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file runtime.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/host/perf_analysis.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

void CopyToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyDevToHost(devTensor, hostTensor);
}

void CopyToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyHostToDev(devTensor, hostTensor);
}

void SetVerifyData(const std::vector<DeviceTensorData> &inputs,
                   const std::vector<DeviceTensorData> &outputs,
                   const std::vector<DeviceTensorData> &goldens) {
    ProgramData::GetInstance().Reset();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(
            inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(
            outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
    for (size_t i = 0; i < goldens.size(); i++) {
        if (goldens[i].GetAddr() == 0) {
            ProgramData::GetInstance().AppendGolden(nullptr);
        } else {
            auto rawData = RawTensorData::CreateTensor(
            goldens[i].GetDataType(), goldens[i].GetShape(), (uint8_t *)goldens[i].GetAddr());
            ProgramData::GetInstance().AppendGolden(rawData);
        }
    }
}

static std::string ValidateFunctionAndIO(Function *func, const std::vector<DeviceTensorData> &inputs,
                                   const std::vector<DeviceTensorData> &outputs) {
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }

    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }

    auto inputSize = attr->startArgsInputLogicalTensorList.size();
    auto outputSize = attr->startArgsOutputLogicalTensorList.size();
    if (inputSize != inputs.size() || outputSize != outputs.size()) {
        return "mismatch input/output";
    }
    return "";
}

static void InitializeInputOutputData(const std::vector<DeviceTensorData> &inputs,
                               const std::vector<DeviceTensorData> &outputs) {
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
}

std::string DeviceRunOnceDataFromHost(
    const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs) {
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    ProgramData::GetInstance().Reset();
    Function *func = Program::GetInstance().GetLastFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    InitializeInputOutputData(inputs, outputs);

    DevControlFlowCache* hostCache = nullptr;
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        EmulationLauncher::BuildControlFlowCache(func, inputs, outputs, &hostCache, config);
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1 && EmulationLauncher::EmulationRunOnce(func, hostCache) != 0) {
        return "emulation run failed";
    }

    if (DeviceRunOnce(func, reinterpret_cast<uint8_t*>(hostCache)) != 0) {
        return "device run failed";
    }

    if (hostCache) {
        free(hostCache);
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        auto output = ProgramData::GetInstance().GetOutputData(i);
        StringUtils::DataCopy(outputs[i].GetAddr(), output->GetDataSize(), output->data(), output->GetDataSize());
    }

    if (HasInplaceArgs(Program::GetInstance().GetLastFunction()) || outputs.size() == 0) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = ProgramData::GetInstance().GetInputData(i);
            StringUtils::DataCopy(inputs[i].GetAddr(), input->GetDataSize(), input->data(), input->GetDataSize());
        }
    }
    return "";
}

std::string OperatorDeviceRunOnceDataFromDevice([[maybe_unused]] py::int_ pythonOperatorPython,
    [[maybe_unused]] const std::vector<DeviceTensorData> &inputs, [[maybe_unused]] const std::vector<DeviceTensorData> &outputs,
    [[maybe_unused]] py::int_ incomingStreamPython, [[maybe_unused]] py::int_ workspaceData,
    [[maybe_unused]] py::int_ devCtrlCache) {

    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::RunDevice);

#ifdef BUILD_WITH_CANN
    auto opAddr = static_cast<uintptr_t>(pythonOperatorPython);
    if (opAddr == 0) {
        return "invalid operator";
    }

    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    Function *func = op->GetFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputs, outputs, config) != 0) {
            return "emulation run failed";
        }
    }

    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicoreStream = incomingStream;
    auto aicpuStream = DeviceGetAicpuStream();
    auto workspaceDataAddr = static_cast<uintptr_t>(workspaceData);
    auto ctrlCache = static_cast<uintptr_t>(devCtrlCache);
    int rc = ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputs, outputs,
        aicpuStream, aicoreStream, false, reinterpret_cast<uint8_t*>(ctrlCache),
        DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr));
    if (rc < 0) {
        return "device run failed";
    }
#endif

    HOST_PERF_EVT_END(EventPhase::RunDevice);
    return "";
}

uint64_t GetWorkSpaceSize(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputs,
    const std::vector<DeviceTensorData> &outputs) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (op) {
        return op->GetWorkSpaceSize(inputs, outputs);
    }
    return 0;
}

std::string OperatorDeviceSynchronize(py::int_ incomingStreamPython) {
    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicpuStream = incomingStream;
    auto aicoreStream = DeviceGetAicoreStream();
    int rc = DeviceSynchronize(aicpuStream, aicoreStream);
    if (rc < 0) {
        return "device sync failed";
    }
    return "";
}

void DeviceInit() {
    DeviceLauncherInit();
}

void DeviceFini() {
    DeviceLauncherFini();
}

uintptr_t OperatorBegin() {
    ExportedOperator *op = ExportedOperatorBegin();
    auto opAddr = reinterpret_cast<uintptr_t>(op);
    return opAddr;
}

std::string OperatorEnd(uintptr_t opAddr) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    ExportedOperatorEnd(op);
    return "";
}

int64_t BuildCache(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList, [[maybe_unused]] bool isCapturing) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        uint8_t* ctrlCache = op->FindCtrlFlowCache(inputList, outputList);
        if (ctrlCache == nullptr) {
            HOST_PERF_EVT_BEGIN(EventPhase::BuildCtrlFlowCache);
            DevControlFlowCache* hostCache = nullptr;
            if (EmulationLauncher::BuildControlFlowCache(op->GetFunction(),
                inputList, outputList, &hostCache, config) != 0) {
                return 0;
            }

#ifdef BUILD_WITH_CANN
            if (isCapturing) {
                ChangeCaptureModeRelax();
            }

            if (hostCache) {
                ctrlCache = CopyHostToDev(reinterpret_cast<uint8_t*>(hostCache),
                    reinterpret_cast<DevControlFlowCache*>(hostCache)->allCacheSize);
                free(hostCache);
            }

            if (isCapturing) {
                ChangeCaptureModeGlobal();
            }
#else
            ctrlCache = reinterpret_cast<uint8_t*>(hostCache);
#endif

            if (ctrlCache) {
                op->InsertCtrlFlowCache(inputList, outputList, ctrlCache);
            }
            HOST_PERF_EVT_END(EventPhase::BuildCtrlFlowCache);
        }

        return ctrlCache == nullptr ? 0 : reinterpret_cast<int64_t>(ctrlCache);
    }

    return 0;
}

void BindRuntime(py::module &m) {
    m.def("DeviceInit", &DeviceInit);
    m.def("DeviceFini", &DeviceFini);
    m.def("DeviceRunOnceDataFromHost", &DeviceRunOnceDataFromHost);
    m.def("OperatorDeviceRunOnceDataFromDevice", &OperatorDeviceRunOnceDataFromDevice);
    m.def("OperatorDeviceSynchronize", &OperatorDeviceSynchronize);
    m.def("GetWorkSpaceSize", &GetWorkSpaceSize);
    m.def("OperatorBegin", OperatorBegin);
    m.def("OperatorEnd", OperatorEnd);
    m.def("SetVerifyData", &SetVerifyData);
    m.def("BuildCache", BuildCache);
    m.def("CopyToHost", &CopyToHost);
    m.def("CopyToDev", &CopyToDev);

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(py::init<DataType, uintptr_t, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);
}
} // namespace pypto

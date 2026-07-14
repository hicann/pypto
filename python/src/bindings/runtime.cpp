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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "adapter/api/acl_define.h"
#include "adapter/api/runtime_define.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/utils/op_info_manager.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_stage_scope.h"
#include "interface/function/rebuildable_attribute.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/launcher/eslmodel_launcher.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/runtime/launcher/launcher_router.h"
#include "machine/host/perf_analysis.h"
#include "bindings/torch_tensor_converter.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

std::string ValidateDynamicFunctionAndIO(
    Function* func, const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
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

static bool IsUint8GoldenAndHf8InOut(const DeviceTensorData& inOutTensor, const DeviceTensorData& goldenTensor)
{
    return inOutTensor.GetDataType() == DT_HF8 && goldenTensor.GetDataType() == DT_UINT8;
}

static void ValidateVerifyOutputAndGolden(
    const std::vector<DeviceTensorData>& inOutTensors, const std::vector<DeviceTensorData>& goldens)
{
    auto ShapeToString = [](const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    };

    if (inOutTensors.size() != goldens.size()) {
        return;
    }

    for (size_t i = 0; i < inOutTensors.size(); i++) {
        bool outputIsNone = inOutTensors[i].GetAddr() == nullptr;
        bool goldenIsNone = goldens[i].GetAddr() == nullptr;
        if (outputIsNone || goldenIsNone) {
            continue;
        }

        ASSERT(
            VerifyResultScene::VERIFY_RESULT_DTYPE_DIFF, inOutTensors[i].GetDataType() == goldens[i].GetDataType() ||
                                                             IsUint8GoldenAndHf8InOut(inOutTensors[i], goldens[i]))
            << "dtype mismatch at index " << i << ", output dtype: " << DataType2String(inOutTensors[i].GetDataType())
            << ", golden dtype: " << DataType2String(goldens[i].GetDataType());

        auto& outputShape = inOutTensors[i].GetShape();
        auto& goldenShape = goldens[i].GetShape();
        ASSERT(VerifyResultScene::VERIFY_RESULT_SHAPE_DIFF, outputShape.size() == goldenShape.size())
            << "shape rank mismatch at golden index " << i << ", output rank: " << outputShape.size()
            << ", golden rank: " << goldenShape.size();
        for (size_t dim = 0; dim < outputShape.size(); dim++) {
            ASSERT(VerifyResultScene::VERIFY_RESULT_SHAPE_DIFF, outputShape[dim] == goldenShape[dim])
                << "shape mismatch at golden index " << i << ", dim " << dim
                << ", output shape: " << ShapeToString(outputShape) << ", golden shape: " << ShapeToString(goldenShape);
        }
    }
}

void CopyToHost(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    CopyDevToHost(devTensor, hostTensor);
}

void CopyToDev(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    CopyHostToDev(devTensor, hostTensor);
}

void SetVerifyData(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs,
    const std::vector<DeviceTensorData>& goldens)
{
    auto ToLogicalShape = [](DataType dtype, const std::vector<int64_t>& shape) -> std::vector<int64_t> {
        auto logical = shape;
        if ((dtype == DT_FP4_E2M1 || dtype == DT_FP4_E1M2) && !logical.empty()) {
            logical.back() *= 2;
        }
        return logical;
    };

    std::vector<DeviceTensorData> inOutTensors;
    inOutTensors.reserve(inputs.size() + outputs.size());
    inOutTensors.insert(inOutTensors.end(), inputs.begin(), inputs.end());
    inOutTensors.insert(inOutTensors.end(), outputs.begin(), outputs.end());
    ValidateVerifyOutputAndGolden(inOutTensors, goldens);

    ProgramData::GetInstance().Reset();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto logicalShape = ToLogicalShape(inputs[i].GetDataType(), inputs[i].GetShape());
        auto rawData =
            RawTensorData::CreateTensor(inputs[i].GetDataType(), logicalShape, (uint8_t*)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto logicalShape = ToLogicalShape(outputs[i].GetDataType(), outputs[i].GetShape());
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), logicalShape);
        ProgramData::GetInstance().AppendOutput(rawData);
    }
    for (size_t i = 0; i < goldens.size(); i++) {
        if (goldens[i].GetAddr() == 0) {
            ProgramData::GetInstance().AppendGolden(nullptr);
        } else {
            auto goldenType = goldens[i].GetDataType();
            if (i < inOutTensors.size() && IsUint8GoldenAndHf8InOut(inOutTensors[i], goldens[i])) {
                goldenType = DT_HF8;
            }
            auto logicalShape = ToLogicalShape(goldenType, goldens[i].GetShape());
            auto rawData = RawTensorData::CreateTensor(goldenType, logicalShape, (uint8_t*)goldens[i].GetAddr());
            ProgramData::GetInstance().AppendGolden(rawData);
        }
    }
}

static std::string ValidateFunctionAndIO(
    Function* func, const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    return ValidateDynamicFunctionAndIO(func, inputs, outputs);
}

static ExportedOperator* GetValidatedOperator(uintptr_t opAddr)
{
    if (opAddr == 0) {
        return nullptr;
    }

    if (opAddr % alignof(ExportedOperator) != 0) {
        MACHINE_LOGE(DevCommonErr::PARAM_INVALID, "Invalid operator address alignment");
        return nullptr;
    }
    return reinterpret_cast<ExportedOperator*>(opAddr);
}

static void InitializeInputOutputData(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData =
            RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t*)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
}

std::string DeviceRunOnceDataFromHost(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    ProgramData::GetInstance().Reset();
    Function* func = Program::GetInstance().GetLastFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    InitializeInputOutputData(inputs, outputs);

    DevControlFlowCache* hostCache = nullptr;
    EmulationMemoryUtils memUtils;
    DeviceLauncherConfig config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EmulationLauncher::BuildControlFlowCache(func, memUtils, inputs, outputs, &hostCache, config);

    auto launchMode = LauncherRouter::ResolveCurrent();
    if (launchMode == LaunchMode::EMULATION && EmulationLauncher::EmulationRunOnce(func, hostCache, config) != 0) {
        return "emulation run failed";
    }

    if (launchMode == LaunchMode::AICORE_MODEL &&
        AicoreModelLauncher::AicoreModelRunOnce(func, hostCache, config) != 0) {
        return "aicore model run failed";
    }

    if (launchMode != LaunchMode::AICORE_MODEL) {
        if (DeviceRunOnce(func, reinterpret_cast<uint8_t*>(hostCache), config) != 0) {
            return "device run failed";
        }
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

std::string OperatorDeviceRunOnceDataFromDevice(
    [[maybe_unused]] py::int_ pythonOperatorPython, [[maybe_unused]] const std::vector<DeviceTensorData>& inputs,
    [[maybe_unused]] const std::vector<DeviceTensorData>& outputs, [[maybe_unused]] py::int_ incomingStreamPython,
    [[maybe_unused]] py::int_ workspaceData, [[maybe_unused]] py::int_ devCtrlCache)
{
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
    ExportedOperator* op = GetValidatedOperator(opAddr);
    if (op == nullptr) {
        return "invalid operator";
    }
    Function* func = op->GetFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }
    auto launchMode = LauncherRouter::ResolveCurrent();
    if (launchMode == LaunchMode::EMULATION) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputs, outputs, config, nullptr) != 0) {
            return "emulation run failed";
        }
    }
    if (launchMode == LaunchMode::AICORE_MODEL) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (AicoreModelLauncher::AicoreModelLaunchDeviceTensorData(func, inputs, outputs) != 0) {
            return "aicore model run failed";
        }
        HOST_PERF_EVT_END(EventPhase::RunDevice);
        return "";
    }

    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicoreStream = incomingStream;
    auto workspaceDataAddr = static_cast<uintptr_t>(workspaceData);
    auto ctrlCache = static_cast<uintptr_t>(devCtrlCache);
    if (launchMode != LaunchMode::AICORE_MODEL) {
        int rc = ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
            op, inputs, outputs, aicoreStream, false, reinterpret_cast<uint8_t*>(ctrlCache),
            DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr));
        if (rc < 0) {
            return "device run failed";
        }
    }
#endif

    HOST_PERF_EVT_END(EventPhase::RunDevice);
    return "";
}

uint64_t GetWorkSpaceSize(
    uintptr_t opAddr, const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    ExportedOperator* op = GetValidatedOperator(opAddr);
    if (op) {
        return op->GetWorkSpaceSize(inputs, outputs);
    }
    return 0;
}

std::string OperatorDeviceSynchronize(py::int_ incomingStreamPython)
{
    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicpuStream = incomingStream;
    int rc = DeviceSynchronize(aicpuStream);
    if (rc < 0) {
        return "device sync failed";
    }
    return "";
}

void DeviceInit() { DeviceLauncherInit(); }

void DeviceFini() { DeviceLauncherFini(); }

uintptr_t OperatorBegin()
{
    ExportedOperator* op = ExportedOperatorBegin();
    auto opAddr = reinterpret_cast<uintptr_t>(op);
    return opAddr;
}

std::string OperatorEnd(uintptr_t opAddr)
{
    ExportedOperator* op = GetValidatedOperator(opAddr);
    if (op == nullptr) {
        return "invalid operator";
    }
    ExportedOperatorEnd(op);
    return "";
}

int64_t BuildCache(
    uintptr_t opAddr, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    [[maybe_unused]] bool isCapturing)
{
    ExportedOperator* op = GetValidatedOperator(opAddr);
    if (op == nullptr) {
        return -1;
    }
    DeviceLauncherConfig config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    uint8_t* ctrlCache = op->FindCtrlFlowCache(inputList, outputList);
    EmulationMemoryUtils memUtils;
    if (ctrlCache == nullptr) {
        HOST_PERF_EVT_BEGIN(EventPhase::BuildCtrlFlowCache);
        DevControlFlowCache* hostCache = nullptr;
        if (EmulationLauncher::BuildControlFlowCache(
                op->GetFunction(), memUtils, inputList, outputList, &hostCache, config) != 0) {
            return 0;
        }

#ifdef BUILD_WITH_CANN
        if (isCapturing) {
            ChangeCaptureModeRelax();
        }

        if (hostCache) {
            ctrlCache = CopyHostToDev(
                reinterpret_cast<uint8_t*>(hostCache),
                reinterpret_cast<DevControlFlowCache*>(hostCache)->usedCacheSize);
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

#ifdef BUILD_WITH_CANN
#define ENABLE_VERBOSE_LOG 0

class KernelModule {
public:
    KernelModule(py::object& module)
    {
        InitCachedArgs();
        InitConfigOptions(module);
    }

    ~KernelModule()
    {
        for (auto& k : kernels) {
            delete k;
        }
    }

    bool IsCompileStageAllComplete() { return compileStageAllComplete; }

    KernelBinary* GetKernelBinary(std::vector<DeviceTensorData>& tensors)
    {
        for (auto& k : kernels) {
            if (k->CheckArgs(tensors)) {
                return k;
            }
        }
        return nullptr;
    }

    uint8_t* FindCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors)
    {
        if (kernel->DisableHostCtrlFlowCacheBuild()) {
            COMPILER_LOGI("Skip host control flow cache build due to RUNTIME_FUNCKEY_CACHESTOP.");
            return nullptr;
        }
        auto devCache = kernel->FindCtrlFlowCache(tensors, true);
        if (devCache == nullptr) {
            std::vector<std::vector<int64_t>> shape;
            if (DeviceLauncher::IsCaptureMode()) { // KernelModule
                AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
                devCache = kernel->BuildControlFlowCache(tensors, true);
            } else {
                AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
                devCache = kernel->BuildControlFlowCache(tensors, true);
            }
        }
#if ENABLE_VERBOSE_LOG
        std::stringstream ss;
        for (auto& t : tensors) {
            for (auto& s : t.GetShape()) {
                ss << s << " ";
            }
        }
        COMPILER_LOGI("find ctrlflow cache: %p shape %s", devCache, ss.str().c_str());
#endif
        return devCache;
    }

    KernelBinary* Compile(py::object& module, py::sequence& torch_tensors, py::sequence& tensor_defs)
    {
        COMPILER_LOGI("New frontend compile from torch begin once.");
        // Prepare stage starts here and ends at Program::UpdateCompileTask() for NEW
        // "Prepare" 在Initialize中设置
        MonitorManager::Instance().Initialize(
            compileMonitorEnable, intervalSec, timeoutSec, totalTimeoutSec, compileMonitorPassDetailEnable);
        auto compile = py::getattr(module, "compile");
        compile(torch_tensors, tensor_defs);
        return RegisterLastCompiledKernel();
    }

    KernelBinary* RegisterLastCompiledKernel()
    {
        auto func = Program::GetInstance().GetLastFunction();
        auto attr = func->GetDyndevAttribute();
        if (attr->devProgBinary.empty() || attr->kernelBinary.empty()) {
            return nullptr;
        }
        auto kernel = new KernelBinary(Program::GetInstance().GetFunctionSharedPtr(func));
        kernels.push_back(kernel);
        return kernel;
    }

    int64_t GetWorkspaceSize(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors)
    {
        return kernel->GetWorkspaceSize(tensors);
    }

    void SetTensorData(const std::vector<DeviceTensorData>& tensors)
    {
        Function* func = Program::GetInstance().GetLastFunction();
        if (func == nullptr) {
            return;
        }
        size_t inputSize = func->inCasts_.size();
        size_t outputSize = func->outCasts_.size();
        if (tensors.size() != (inputSize + outputSize)) {
            return;
        }
        if (ProgramData::GetInstance().GetInputDataList().empty()) {
            for (size_t i = 0; i < inputSize; i++) {
                RawTensorDataPtr rawDataPtr =
                    std::make_shared<RawTensorData>(tensors.at(i).GetDataType(), tensors.at(i).GetShape());
                ProgramData::GetInstance().AppendInput(rawDataPtr);
            }
        }
        if (ProgramData::GetInstance().GetOutputDataList().empty()) {
            for (size_t i = inputSize; i < tensors.size(); i++) {
                RawTensorDataPtr rawDataPtr =
                    std::make_shared<RawTensorData>(tensors.at(i).GetDataType(), tensors.at(i).GetShape());
                ProgramData::GetInstance().AppendOutput(rawDataPtr);
            }
        }
    }

    void Launch(
        KernelBinary* kernel, AclRtStream aicoreStream, std::vector<DeviceTensorData>& tensors, uint8_t* ctrlFlowCache,
        int64_t* workspace)
    {
        SetTensorData(tensors);
        auto [args, argsSize] = kernel->BuildKernelArgs(tensors);
        rtAicpuArgs.args = args;
        rtAicpuArgs.argsSize = argsSize;

        args->kArgs.ctrlFlowCache = (int64_t*)ctrlFlowCache;
        args->kArgs.workspace = workspace;
        args->kArgs.parameter.globalRound = ++sequence;
        args->kArgs.maxDynamicAssembleOutcastMem = kernel->GetMaxDynamicAssembleOutcastMem();
        args->kArgs.maxDynamicCellMatchTableMem = kernel->GetMaxDynamicCellMatchTableMem();
        args->kArgs.runtimeDynamicCellMatchAddr = kernel->GetRuntimeDynamicCellMatchAddr();
        args->kArgs.runtimeDynamicCellMatchCapacity = kernel->GetRuntimeDynamicCellMatchCapacity();
        args->kArgs.schedSyncMode = kernel->GetSyncMode();
        auto isCaptureMode = DeviceLauncher::IsCaptureMode();  // KernelModule
        bool debugEnable = !isCaptureMode && isDebugMode;

#if ENABLE_VERBOSE_LOG
        COMPILER_LOGI("Sequence %ld workspace %p cfgcache %p", sequence.load(), workspace, ctrlFlowCache);
#endif
        int ret = DeviceLauncher::LaunchSyncTask(aicoreStream, isCaptureMode, launchEarlyMode);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "launch pre sync failed: " << ret;

        DeviceLauncher::SetDevPerfAddr(debugEnable, isCaptureMode);
        if (!isCaptureMode) {
           args->kArgs.toSubMachineConfig = kernel->GetMachineConfig();
        }
        ret = DeviceLauncher::LaunchAicpuKernel(rtAicpuArgs, debugEnable, kernel->GetFunction(), tensors);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "launch aicpu failed: " << ret;

        kernelArgs[5] = args->kArgs.cfgdata; // 5 is cfgdata
        // copy tensor to aicore args
        kernelArgs[0] = const_cast<char*>(kernel->GetKernelname().c_str());
        kernelArgs[4] = (int64_t*)(args + 1);
        kernelArgs[6] = (DevTensorData*)((int64_t*)(args + 1) + 2);
        ret = DeviceLauncher::LaunchAicoreKernel(
            aicoreStream, kernel->GetKernelBin(), rtAicoreArgs, rtTaskCfg, debugEnable, kernel->GetFunction());
        MACHINE_ASSERT(ret == RT_SUCCESS) << "launch aicore failed: " << ret;
    }

    DevControlFlowCache* GetHostCtrlFlowCache(
        KernelBinary* kernel, std::vector<DeviceTensorData>& tensors, uint8_t* devCache,
        std::vector<uint8_t>& hostCache)
    {
        DevControlFlowCache* ctrlCache = FindHostCtrlFlowCache(tensors, hostCache);
        if (ctrlCache == nullptr && devCache != nullptr) {
            auto devProg =
                reinterpret_cast<DevAscendProgram*>(kernel->GetFunction()->GetDyndevAttribute()->devProgBinary.data());
            size_t ctrlCacheSize = devProg->ctrlFlowCacheSize;
            std::vector<uint8_t> hostCacheVec;
            hostCacheVec.resize(ctrlCacheSize);
            AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
            RuntimeMemcpy(
                hostCacheVec.data(), ctrlCacheSize, devCache, ctrlCacheSize, RtMemcpyKind::DEVICE_TO_HOST);
            AddHostCtrlFlowCache(tensors, std::move(hostCacheVec));
            ctrlCache = FindHostCtrlFlowCache(tensors, hostCache);
        }
        return ctrlCache;
    }

    void EmulationLaunch(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors, uint8_t* devCache)
    {
        if (launchMode_ == LaunchMode::DEVICE_RT) {
            return;
        }
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        std::vector<uint8_t> hostCache;
        DevControlFlowCache* ctrlCache = GetHostCtrlFlowCache(kernel, tensors, devCache, hostCache);
        int ret = 0;
        if (launchMode_ == LaunchMode::EMULATION) {
            ret = EmulationLauncher::EmulationLaunchDeviceTensorData(
                kernel->GetFunction(), tensors, {}, config, ctrlCache);
            MACHINE_ASSERT(ret == RT_SUCCESS) << "emulation run failed: " << ret;
        } else if (launchMode_ == LaunchMode::AICORE_MODEL) {
            ret = AicoreModelLauncher::AicoreModelLaunchDeviceTensorData(
                kernel->GetFunction(), tensors, {}, config, ctrlCache);
            MACHINE_ASSERT(ret == RT_SUCCESS) << "aicore model run failed: " << ret;
        }
    }

    bool IsAicoreModelMode() const { return launchMode_ == LaunchMode::AICORE_MODEL; }

    void EslModelLaunch(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors)
    {
        DeviceLauncherConfig config;
        ProgramData::GetInstance().Reset();
        InitializeInputOutputData(tensors, {});
        int ret = EslModelLauncher::EslModelRunOnce(kernel->GetKernelBin(), config);
        for (size_t i = 0; i < tensors.size(); i++) {
            auto input = ProgramData::GetInstance().GetInputData(i);
            StringUtils::DataCopy(tensors[i].GetAddr(), input->GetDataSize(), input->data(), input->GetDataSize());
        }
        SIM_ASSERT(ret == RT_SUCCESS) << "EslModelLaunch run failed: " << ret;
    }

    void EslModelLiteLaunch(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors)
    {
        int ret = EslModelLauncher::EslModelLiteRunOnce(kernel->GetFunction(), tensors);
        SIM_ASSERT(ret == RT_SUCCESS) << "EslModelLiteLaunch run failed: " << ret;
    }

private:
    void InitCachedArgs()
    {
        memset_s(&rtAicpuArgs, sizeof(RtAicpuArgsEx), 0, sizeof(RtAicpuArgsEx));
        rtAicpuArgs.kernelNameAddrOffset = offsetof(AiCpuArgs, kernelName);
        rtAicpuArgs.soNameAddrOffset = offsetof(AiCpuArgs, soName);
        rtAicpuArgs.hostInputInfoNum = 1;
        hostInfo.addrOffset = offsetof(AiCpuArgs, kArgs.inputs);
        hostInfo.dataOffset = sizeof(AiCpuArgs);
        rtAicpuArgs.hostInputInfoPtr = &hostInfo;
        rtAicpuArgs.timeout = AICPU_EXECUTE_TIMEOUT;
        memset_s(&rtAicoreArgs, sizeof(RtArgsEx), 0, sizeof(RtArgsEx));
        kernelArgs.resize(0x7, nullptr); // see aicore.ascpp
        rtAicoreArgs.args = kernelArgs.data();
        rtAicoreArgs.argsSize = kernelArgs.size() * sizeof(void*);

        memset_s(&rtTaskCfg, sizeof(RtTaskCfgInfo), 0, sizeof(RtTaskCfgInfo));
        rtTaskCfg.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
    }

    void InitConfigOptions(py::object& module)
    {
        if (!module.attr("_runtime_options").is_none()) {
            auto rutimeOptions = module.attr("_runtime_options").cast<py::dict>();
            if (rutimeOptions.contains("launch_early_mode")) {
                launchEarlyMode = rutimeOptions["launch_early_mode"].cast<int>();
            }
        }

        if (!module.attr("_debug_options").is_none()) {
            auto debugOptions = module.attr("_debug_options").cast<py::dict>();
            if (debugOptions.contains("runtime_debug_mode")) {
                auto debugMode = debugOptions["runtime_debug_mode"].cast<int64_t>();
                launchMode_ = LauncherRouter::ResolveByDebugMode(debugMode);
                isDebugMode = (debugMode == CFG_DEBUG_ALL);
            }
        }

        if (!module.attr("_host_options").is_none()) {
            auto host_options = module.attr("_host_options").cast<py::dict>();
            if (host_options.contains("compile_stage")) {
                auto stage = host_options["compile_stage"];
                int64_t stageValue =
                    py::hasattr(stage, "value") ? stage.attr("value").cast<int64_t>() : stage.cast<int64_t>();
                compileStageAllComplete = (stageValue == CS_ALL_COMPLETE);
            }
            if (host_options.contains("compile_monitor_enable")) {
                int compileMonitorMode = host_options["compile_monitor_enable"].cast<int>();
                compileMonitorEnable = compileMonitorMode > 0;
                compileMonitorPassDetailEnable = compileMonitorMode == 0x2;
            }
            if (host_options.contains("compile_monitor_print_interval")) {
                intervalSec = host_options["compile_monitor_print_interval"].cast<int>();
            }
            if (host_options.contains("compile_timeout_stage")) {
                timeoutSec = static_cast<double>(host_options["compile_timeout_stage"].cast<int>());
            }
            if (host_options.contains("compile_timeout")) {
                totalTimeoutSec = host_options["compile_timeout"].cast<int>();
            }
        }

        if (launchEarlyMode < 0) {
            launchEarlyMode = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ? 2 : 0;
        }
    }

    DevControlFlowCache* FindHostCtrlFlowCache(std::vector<DeviceTensorData>& tensors, std::vector<uint8_t>& hostCache)
    {
        int64_t hash = ControlFlowCache::Hash(tensors);
        for (auto& cache : hostCtrlFlowCaches) {
            if (cache.hash == hash) {
                hostCache = cache.hostCache; // Copy new backup
                return reinterpret_cast<DevControlFlowCache*>(hostCache.data());
            }
        }
        return nullptr;
    }

    void AddHostCtrlFlowCache(std::vector<DeviceTensorData>& tensors, std::vector<uint8_t>&& hostCache)
    {
        hostCtrlFlowCaches.emplace_back(tensors, std::move(hostCache));
    }

private:
    struct HostControlFlowCache {
        int64_t hash;
        std::vector<uint8_t> hostCache;

        HostControlFlowCache(std::vector<DeviceTensorData>& datas, std::vector<uint8_t>&& hcache)
            : hostCache(std::move(hcache))
        {
            hash = ControlFlowCache::Hash(datas);
        }
    };

    bool isDebugMode{false};
    bool compileStageAllComplete{true};
    LaunchMode launchMode_{LaunchMode::DEVICE_RT};
    bool compileMonitorEnable{false};
    bool compileMonitorPassDetailEnable{false};
    int intervalSec{60};
    double timeoutSec{static_cast<double>(config::GetHostOption<int>(TIMEOUT_SEC))};
    int totalTimeoutSec{600};
    int launchEarlyMode{-1};

    RtHostInputInfo hostInfo;
    RtAicpuArgsEx rtAicpuArgs;

    RtArgsEx rtAicoreArgs;
    RtTaskCfgInfo rtTaskCfg;
    std::vector<void*> kernelArgs;
    std::vector<KernelBinary*> kernels;
    std::vector<HostControlFlowCache> hostCtrlFlowCaches;

    static std::atomic<int64_t> sequence;
};
using KernelModulePtr = std::shared_ptr<KernelModule>;

std::atomic<int64_t> KernelModule::sequence(0);

class KernelLauncher {
private:
    py::object& module;
    py::sequence& torchTensors;
    py::sequence& tensorDefs;
    AclRtStream aicoreStream;
    std::vector<DeviceTensorData>& tensors;
    KernelModulePtr kmodule;
    AclMdlRI rtModel;

    std::optional<ConfigManagerNg::JitScopeGuard> jitScopeGuard;

public:
    KernelLauncher(
        py::object& m, int64_t stream, py::sequence& torch_tensors, py::sequence& tensor_defs,
        std::vector<DeviceTensorData>& tensors_ref, int devId)
        : module(m),
          torchTensors(torch_tensors),
          tensorDefs(tensor_defs),
          aicoreStream((AclRtStream)stream),
          tensors(tensors_ref)
    {
        ValidateRuntimeDevice(devId);
        kmodule = py::getattr(module, "kmodule").cast<KernelModulePtr>();
        DeviceLauncher::SaveStream(aicoreStream);
        DeviceLauncher::GetCaptureInfo(aicoreStream, rtModel);
    }

    void Execute()
    {
        HOST_PERF_TRACE_START();
        HOST_PERF_EVT_BEGIN(EventPhase::LaunchKernel);

        auto kbinary = CompileIfNeeded();
        HOST_PERF_TRACE(TracePhase::LaunchGetKernel);
        if (!kbinary || !kmodule->IsCompileStageAllComplete()) {
            HOST_PERF_EVT_END(EventPhase::LaunchKernel);
            return;
        }

        DoLaunch(kbinary);
        HOST_PERF_EVT_END(EventPhase::LaunchKernel);
    }

private:
    KernelBinary* CompileIfNeeded()
    {
        HOST_PERF_TRACE(TracePhase::LaunchInit);
        auto kbinary = kmodule->GetKernelBinary(tensors);
        if (kbinary)
            return kbinary;

        jitScopeGuard.emplace("jit_scope", std::map<std::string, std::any>{});
        Program::GetInstance().Reset();
        AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
#if ENABLE_VERBOSE_LOG
        COMPILER_LOGI("compile kernel");
#endif

        return kmodule->Compile(module, torchTensors, tensorDefs);
    }

    void DoLaunch(KernelBinary* kbinary)
    {
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
            if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
                kmodule->EslModelLiteLaunch(kbinary, tensors);
            } else {
                kmodule->EslModelLaunch(kbinary, tensors);
            }
            return;
        }

        DeviceLauncher::CheckAscendDriverVersionOnboard();

        int64_t* wsAddr = nullptr;
        int64_t wsSize = kmodule->GetWorkspaceSize(kbinary, tensors);
        if (wsSize) {
            auto pyalloc = py::getattr(module, "alloc");
            wsAddr = (int64_t*)pyalloc(wsSize).cast<int64_t>();
        }
#if ENABLE_VERBOSE_LOG
        COMPILER_LOGI("alloc workspace %ld", wsSize);
#endif
        HOST_PERF_TRACE(TracePhase::LaunchAllocWorkSpace);

        DeviceLauncher::AddAicpuStream(DeviceLauncher::IsCaptureMode(), rtModel);
        HOST_PERF_TRACE(TracePhase::LaunchAttachStream);

        uint8_t* ctrlFlowCache = kmodule->FindCtrlFlowCache(kbinary, tensors);
        HOST_PERF_TRACE(TracePhase::FindCtrlFlowCache);

        kmodule->EmulationLaunch(kbinary, tensors, ctrlFlowCache);
        if (kmodule->IsAicoreModelMode()) {
            return;
        }
        kmodule->Launch(kbinary, aicoreStream, tensors, ctrlFlowCache, wsAddr);
        HOST_PERF_TRACE(TracePhase::Launch);
        DeviceLauncher::DumpIOTensorsWithCann(aicoreStream, tensors, kbinary->GetFunction()->GetRawName());
    }
};

void LaunchKernelTorch(py::object& module, int64_t stream, py::sequence& torchTensors, py::sequence& tensorDefs)
{
    ValidateInputs(torchTensors, tensorDefs);

    std::vector<DeviceTensorData> tensors;
    int devId = TorchTensorConverter::Convert(torchTensors, tensorDefs, tensors);
    KernelLauncher(module, stream, torchTensors, tensorDefs, tensors, devId).Execute();
}
#else
void LaunchKernelTorch(py::object&, int64_t, py::sequence&, py::sequence&) {}
class KernelModule {
public:
    KernelModule(py::object&) {}
};
using KernelModulePtr = std::shared_ptr<KernelModule>;
#endif

void BindRuntime(py::module_& m)
{
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
    m.def("LaunchKernelTorch", &LaunchKernelTorch);
    m.def("GetCompilerMonitorTotalElapsed", []() { return MonitorManager::Instance().GetTotalElapsed(); });

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(
            py::init<DataType, uintptr_t, const std::vector<int64_t>&>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);

    py::class_<KernelModule, KernelModulePtr>(m, "KernelModule").def(py::init<py::object&>());
}
} // namespace pypto

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
 * \file load_aicpu_op.cpp
 * \brief
 */

#include "machine/runtime/runner/load_aicpu_op.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <limits.h>
#include "tilefwk/pypto_fwk_log.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/utils/machine_utils.h"
#include "tilefwk/error_code.h"
using Json = nlohmann::json;

namespace {
const std::string ControlFlowLaunchKernelName = "batchLoadsoFrombuf";
const std::string ControlFlowKernelSoName = "libcontrol_flow.so";
const std::string BuiltInKernelInitName = "DynPyptoKernelServerInit";
const std::string BuiltInKernelRunName = "DynPyptoKernelServer";
const std::string BuiltInKernelNullName = "DynPyptoKernelServerNull";

const std::string BuiltInSoName = "libtilefwk_backend_server.so";
const std::string KfcKernerLib = "KFCKernel";
const std::string AicpuKernerLib = "AICPUKernel";
constexpr int BuiltInOpNum = 3;
std::string BuiltInFunName[BuiltInOpNum] = {"PyptoInit", "PyptoRun", "PyptoNull"};
} // namespace

namespace npu::tile_fwk {

constexpr int DUMP_LEVEL_FOUR = 4;

void LoadAicpuOp::GenBuiltInOpInfo()
{
#ifdef BUILD_WITH_NEW_CANN
    std::string jsonPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(jsonPath);
    Json builtInOp;
    AicpuOpConfig pyptoInit;
    pyptoInit.functionName = BuiltInKernelInitName;
    pyptoInit.kernelSo = BuiltInSoName;
    pyptoInit.opKernelLib = KfcKernerLib;
    pyptoInit.opType = BuiltInFunName[0];

    AicpuOpConfig pyptoRun = pyptoInit;
    pyptoRun.opType = BuiltInFunName[1];
    pyptoRun.functionName = BuiltInKernelRunName;

    AicpuOpConfig pyptoNull = pyptoInit;
    pyptoNull.opType = BuiltInFunName[2];
    pyptoNull.opKernelLib = AicpuKernerLib;
    pyptoNull.functionName = BuiltInKernelNullName;

    GenAicpuOpInfoJson(builtInOp, {pyptoInit, pyptoRun, pyptoNull});
    builtInOp.dump(DUMP_LEVEL_FOUR);
    builtInOpJsonPath_ = jsonPath + "/pypto_op_info.json";
    if (!SaveFile(builtInOpJsonPath_, builtInOp.dump(DUMP_LEVEL_FOUR))) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Contrust custom op json failed");
        return;
    }
#endif
}

void LoadAicpuOp::CustomAiCpuSoLoad()
{
#ifdef BUILD_WITH_NEW_CANN
    RtLoadBinaryConfig optionCfg;
    auto loadBinOptions = std::make_unique<RtLoadBinaryOption>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RtLoadBinaryOptionType::CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 1;
    optionCfg.numOpt = 1;
    std::string customOpJsonPath = OpInfoManager::GetInstance().GetCustomOpJsonPath();
    if (RealPath(customOpJsonPath).empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Custom op json path is empty");
        return;
    }
    customBinHandle_ = OpInfoManager::GetInstance().GetControlBinHandle(customOpJsonPath);
    if (customBinHandle_ != nullptr) {
        return;
    }
    auto ret = RuntimeBinaryLoadFromFile(customOpJsonPath.c_str(), &optionCfg,
                                         reinterpret_cast<void**>(&customBinHandle_));
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_LOAD_FAILED, "Load aicpu json failed ret is %d", ret);
    }
    OpInfoManager::GetInstance().SetControlBinHandle(customBinHandle_);
#endif
}

int LoadAicpuOp::AicpuKernelLaunch([[maybe_unused]] void* funcHandle, [[maybe_unused]] const RtStream& stream,
                                   [[maybe_unused]] DeviceKernelArgs* kArgs, [[maybe_unused]] const uint32_t& blockDim)
{
#ifdef BUILD_WITH_NEW_CANN
    RtFuncHandle aicpuFuncHandle = static_cast<RtFuncHandle>(funcHandle);
    RtAicpuArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = kArgs;
    rtArgs.argsSize = sizeof(DeviceKernelArgs);

    RtCpuKernelArgs argInfo;
    memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
    argInfo.baseArgs = rtArgs;
    RtKernelLaunchCfg kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<RtLaunchKernelAttr>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();
    return RuntimeLaunchCpuKernel(aicpuFuncHandle, blockDim, stream, &kernelLaunchCfg, &argInfo);
#else
    return 0;
#endif
}

int LoadAicpuOp::LaunchCustomOp([[maybe_unused]] RtStream stream, [[maybe_unused]] DeviceKernelArgs* kArgs,
                                [[maybe_unused]] std::string& OpType) const
{
#ifdef BUILD_WITH_NEW_CANN
    ASSERT(DevCommonErr::PARAM_INVALID, customBinHandle_ != nullptr) << "customBinHandle cannot be null";
    RtFuncHandle custFuncHandle;
    auto ret = RuntimeFuncGetByName(customBinHandle_, OpType.c_str(), &custFuncHandle);
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "Get OpType[%s] funcHandle failed ret[%d]", OpType.c_str(), ret);
        return ret;
    }
    return AicpuKernelLaunch(custFuncHandle, stream, kArgs, 1);
#else
    return 0;
#endif
}

int LoadAicpuOp::GetBuiltInOpBinHandle()
{
#ifdef BUILD_WITH_NEW_CANN
    if (RealPath(builtInOpJsonPath_).empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "JsonPath is empty");
        return -1;
    }
    RtLoadBinaryConfig optionCfg;
    auto loadBinOptions = std::make_unique<RtLoadBinaryOption>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RtLoadBinaryOptionType::CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 0;
    optionCfg.numOpt = 1;
    void* binHandle;
    auto ret = RuntimeBinaryLoadFromFile(builtInOpJsonPath_.c_str(), &optionCfg, reinterpret_cast<void**>(&binHandle));
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_LOAD_FAILED, "Get built in bin handle failed");
        return -1;
    }

    for (int i = 0; i < BuiltInOpNum; i++) {
        RtFuncHandle funcHandle;
        ret = RuntimeFuncGetByName(binHandle, BuiltInFunName[i].c_str(), &funcHandle);
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "Get BuiltIn FuncName[%s] funcHandle failed ret[%d]",
                         BuiltInFunName[i].c_str(), ret);
            return ret;
        }
        builtInFuncMap_[BuiltInFunName[i]] = funcHandle;
    }
#endif
    return 0;
}

int LoadAicpuOp::LaunchBuiltInOp([[maybe_unused]] RtStream stream, [[maybe_unused]] DeviceKernelArgs* kArgs,
                                 [[maybe_unused]] const int& aicpuNum,
                                 [[maybe_unused]] const std::string& funcName) const
{
#ifdef BUILD_WITH_NEW_CANN
    RtFuncHandle funcHandle;
    auto it = builtInFuncMap_.find(funcName);
    if (it != builtInFuncMap_.end()) {
        funcHandle = it->second;
    } else {
        MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "The func name[%s] is invalid", funcName.c_str());
        return -1;
    }
    return AicpuKernelLaunch(funcHandle, stream, kArgs, aicpuNum);
#else
    return 0;
#endif
}

int LoadAicpuOp::LaunchPyptoNullOp(RtStream stream, DeviceKernelArgs* kArgs, const int& aicpuNum)
{
    if (!isPyptoNullLaunched_) {
        auto ret = LaunchBuiltInOp(stream, kArgs, aicpuNum, "PyptoNull");
        if (ret != 0) {
            MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "launch built null failed");
            return ret;
        }
        isPyptoNullLaunched_ = true;
    }
    return 0;
}
} // namespace npu::tile_fwk

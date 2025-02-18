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

#include <nlohmann/json.hpp>
#include <fstream>
#include <limits.h>
#ifdef BUILD_WITH_CANN
#include "load_aicpu_op.h"
#include "runtime/mem.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "runtime.h"
#include "machine/utils/machine_utils.h"
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
}

namespace npu::tile_fwk {

constexpr int DUMP_LEVEL_FOUR = 4;

void LoadAicpuOp::GenBuiltInOpInfo(const std::string &jsonPath) {
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
    if (!DumpFile(builtInOp.dump(DUMP_LEVEL_FOUR), builtInOpJsonPath_)) {
        ALOG_ERROR_F("Contrust custom op json failed");
        return;
    }
    return;
}

void LoadAicpuOp::CustomAiCpuSoLoad() {
#ifdef BUILD_WITH_NEW_CANN
    rtLoadBinaryConfig_t optionCfg;
    auto loadBinOptions = std::make_unique<rtLoadBinaryOption_t>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 1;
    optionCfg.numOpt = 1;
    std::string customOpJsonPath = OpInfoManager::GetInstance().GetCustomOpJsonPath();
    if (RealPath(customOpJsonPath).empty()) {
        ALOG_ERROR_F("Custom op json path is empty");
        return;
    }
    customBinHandle_ = OpInfoManager::GetInstance().GetControlBinHandle(customOpJsonPath);
    if (customBinHandle_ != nullptr) {
        return;
    }
    auto ret = rtsBinaryLoadFromFile(customOpJsonPath.c_str(), &optionCfg, reinterpret_cast<void**>(&customBinHandle_));
    if (ret != 0) {
        ALOG_ERROR_F("Load aicpu json failed ret is %d", ret);
    }
    OpInfoManager::GetInstance().SetControlBinHandle(customBinHandle_);
#endif
}

int LoadAicpuOp::LaunchCustomOp([[maybe_unused]]rtStream_t stream, [[maybe_unused]]AstKernelArgs *kArgs, [[maybe_unused]]std::string &OpType) {
#ifdef BUILD_WITH_NEW_CANN
    ASSERT(customBinHandle_ != nullptr) << "customBinHandle cannot be null";
    rtFuncHandle custFuncHandle;
    auto ret = rtsFuncGetByName(customBinHandle_, OpType.c_str(), &custFuncHandle);
    if (ret != 0) {
        ALOG_ERROR_F("Get OpType[%s] funcHandle failed ret[%d]", OpType.c_str(), ret);
        return ret;
    }
    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = kArgs;
    rtArgs.argsSize = sizeof(AstKernelArgs);

    rtCpuKernelArgs_t argInfo;
    memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
    argInfo.baseArgs = rtArgs;
    rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();
    return rtsLaunchCpuKernel(custFuncHandle, 1, stream, &kernelLaunchCfg, &argInfo);
#else
    return 0;
#endif
}

int LoadAicpuOp::GetBuiltInOpBinHandle() {
#ifdef BUILD_WITH_NEW_CANN
    if (RealPath(builtInOpJsonPath_).empty()) {
        ALOG_ERROR_F("JsonPath is empty");
        return -1;
    }
    rtLoadBinaryConfig_t optionCfg;
    auto loadBinOptions = std::make_unique<rtLoadBinaryOption_t>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 0;
    optionCfg.numOpt = 1;
    void *binHandle;
    auto ret = rtsBinaryLoadFromFile(builtInOpJsonPath_.c_str(), &optionCfg, reinterpret_cast<void**>(&binHandle));
    if (ret != 0) {
        ALOG_ERROR_F("Get built in bin handle failed");
        return -1;
    }

    for (int i = 0; i < BuiltInOpNum; i++) {
        rtFuncHandle funcHandle;
        ret = rtsFuncGetByName(binHandle, BuiltInFunName[i].c_str(), &funcHandle);
        if (ret != 0) {
            ALOG_ERROR_F("Get BuiltIn FuncName[%s] funcHandle failed ret[%d]", BuiltInFunName[i].c_str(), ret);
            return ret;
        }
        builtInFuncMap_[BuiltInFunName[i]] = funcHandle;
    }
#endif
    return 0;
}

int LoadAicpuOp::LaunchBuiltInOp([[maybe_unused]]rtStream_t stream, [[maybe_unused]]AstKernelArgs *kArgs, [[maybe_unused]]const int &aicpuNum,
                                 [[maybe_unused]]const std::string &funcName) {
#ifdef BUILD_WITH_NEW_CANN
    rtFuncHandle funcHandle;
    auto it = builtInFuncMap_.find(funcName);
    if (it != builtInFuncMap_.end()) {
        funcHandle = it->second;
    } else {
        ALOG_ERROR_F("The func name[%s] is invalid", funcName.c_str());
        return -1;
    }
    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = kArgs;
    rtArgs.argsSize = sizeof(AstKernelArgs);

    rtCpuKernelArgs_t argInfo;
    memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
    argInfo.baseArgs = rtArgs;
    rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();
    return rtsLaunchCpuKernel(funcHandle, aicpuNum, stream, &kernelLaunchCfg, &argInfo);
#else
    return 0;
#endif
}
}// namespace
#endif
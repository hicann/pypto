/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_exception_dump.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "device_exception_dump.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "interface/program/program.h"
#include "runtime_utils.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk::dynamic {
constexpr int32_t MAX_AICPU_ARG_NUM = 7;

void GetTensorInfo(uint32_t inputSize, DevTensorData* tensorData, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    auto func = Program::GetInstance().GetLastFunction();
    if (func == nullptr) {
        MACHINE_LOGW("Function is nullptr not support to dump exception info");
        return;
    }
    auto dynAttr = func->GetDyndevAttribute();
    if (dynAttr == nullptr) {
        MACHINE_LOGW("dynAttr is nullptr not support to dump exception info");
        return;
    }
    auto& disableL2List = dynAttr->disableL2List;
    auto l2Offset = GetRuntimeL2Offset();
    if (inputSize > MAX_TENSOR_NUM) {
        inputSize = MAX_TENSOR_NUM;
        MACHINE_LOGW("Current funciton input is larger than %d", MAX_TENSOR_NUM);
    }
    for (uint32_t i = 0; i < inputSize; i++) {
        if (tensorData[i].address == 0) {
            MACHINE_LOGW("GetTensorInfo tensorData[%u].address is nullptr", i);
            continue;
        }
        exceptionDumpInfo->tensorInfo[i].tensorAddr = reinterpret_cast<int64_t*>(tensorData[i].address);
        // 双页tensor地址，需要进行还原
        if (disableL2List.size() && disableL2List[i]) {
            exceptionDumpInfo->tensorInfo[i].tensorAddr -= l2Offset;
        }      
        exceptionDumpInfo->tensorInfo[i].dataType = DataType2CannType(static_cast<DataType>(tensorData[i].dataType));
        exceptionDumpInfo->tensorInfo[i].tensorSize = 1;
        for (int shapeIdx = 0; shapeIdx < tensorData[i].shape.dimSize; shapeIdx++) {
            exceptionDumpInfo->tensorInfo[i].shape.emplace_back(tensorData[i].shape.dim[shapeIdx]);
            exceptionDumpInfo->tensorInfo[i].tensorSize *= tensorData[i].shape.dim[shapeIdx];
        }
        exceptionDumpInfo->tensorInfo[i].tensorSize *= BitsOf(static_cast<DataType>(tensorData[i].dataType)) / 8;
    }
    exceptionDumpInfo->extraTensorNum = inputSize;
}

int32_t GetAicoreExceptionDumpInfo(std::vector<void*> kernelArg, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    int64_t* tensor = static_cast<int64_t*>(kernelArg[4]);
    uint32_t tensorSize = tensor[0] - tensor[1];
    MACHINE_LOGD("GetAicoreExceptionDumpInfo: tensorSize=%u, outputTensorSize:[%ld]", tensorSize, tensor[1]);
    auto tensorData = (DevTensorData*)kernelArg[6];
    GetTensorInfo(tensorSize, tensorData, exceptionDumpInfo);
    return 0;
}

int32_t GetDeviceExceptionDumpInfo(RtAicoreExDetailInfo& aicoreExceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    auto kernelArgAddr = aicoreExceptionInfo.exceptionArgs.argAddr;
    auto argsSize = aicoreExceptionInfo.exceptionArgs.argsize;
    MACHINE_LOGD("GetDeviceExceptionDumpInfo: kernelArgAddr=%p, argsSize=%u", kernelArgAddr, argsSize);

    if (kernelArgAddr == nullptr) {
        MACHINE_LOGW("GetDeviceExceptionDumpInfo failed: kernelArgAddr is nullptr");
        return 1;
    }

    auto aicoreArgsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    // Check it maybe pto
    if (argsSize != aicoreArgsize) {
        MACHINE_LOGI("GetDeviceExceptionDumpInfo failed: argsize not from pto info");
        return 0;
    }

    // memcpy D2H
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    int rc = RuntimeMemcpy(kernelArg.data(), argsSize, kernelArgAddr, argsSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (rc != 0) {
        MACHINE_LOGW("GetDeviceExceptionDumpInfo D2H memcpy failed: ret=%d", rc);
        return rc;
    }
    // kernel launch kernalArg 0: kernelName; 4 inputSize; 6 tensorData
    char* kernelName = static_cast<char*>(kernelArg[0]);
    // only support handle pto exception info
    if (kernelName != nullptr && strncmp(kernelName, "PyPTO", 5) != 0) {
        MACHINE_LOGI("Current exception info not PyPTO, which kernelName is[%s]", kernelName);
        return 0;
    }
    exceptionDumpInfo->argAddr = kernelArgAddr;
    exceptionDumpInfo->argssize = argsSize;
    auto ret = strcpy_s(exceptionDumpInfo->kernelName, MAX_KERNEL_BUF_LEN, kernelName);
    if (ret != 0) {
        MACHINE_LOGW("Mem cpy KernelName failed");
    }
    ret = strcpy_s(exceptionDumpInfo->kernelDisplayName, MAX_KERNEL_BUF_LEN, kernelName);
    if (ret != 0) {
        MACHINE_LOGW("Mem cpy kernelDisplayName failed");
    }
    return GetAicoreExceptionDumpInfo(kernelArg, exceptionDumpInfo);
}

int32_t DeviceExceptionDumpCallBack(RtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    RtExceptionRegInfo exceptionRegInfo = {0, nullptr};
    auto ret = RuntimeGeExceptionRegInfo(exceptionInfo, &exceptionRegInfo);
    if (ret == 0 && exceptionRegInfo.errRegInfo != nullptr) {
        exceptionDumpInfo->coreId = exceptionRegInfo.errRegInfo->coreId;
        exceptionDumpInfo->coreType = exceptionRegInfo.errRegInfo->coreType;
        MACHINE_LOGD(
            "Current exception from %s coreId: %u",
            exceptionDumpInfo->coreType == RtCoreType::RT_CORE_TYPE_AIC ? "AIC" : "AIV", exceptionDumpInfo->coreId);
    } else {
        MACHINE_LOGW("Cannot Get ExceptionRegInfo, which CoreType coreId would not support");
    }
    auto expandInfo = exceptionInfo->expandInfo;
    MACHINE_LOGD("DeviceExceptionDumpCallBack: expandInfo.type=%d", static_cast<int>(expandInfo.type));
    if (expandInfo.type == RtExceptionExpandType::AICORE) {
        return GetDeviceExceptionDumpInfo(expandInfo.u.aicoreInfo, &exceptionDumpInfo[0]);
    }
    return 0;
}

int32_t ExceptionDumpCallBack(
    AclRtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo, uint32_t exceptionDumpSize,
    uint32_t* exceptionDumpRealSize, AdxExceptionDumpMode* mode)
{
    MACHINE_LOGI("ExceptionDumpCallBack enter: exceptionInfo");
    if (exceptionInfo == nullptr || exceptionDumpInfo == nullptr || exceptionDumpRealSize == nullptr ||
        mode == nullptr) {
        MACHINE_LOGW(
            "DeviceExceptionDumpCallBack failed: the input params is invalid [%p, %p, %p, %p]", (void*)exceptionInfo,
            (void*)exceptionDumpInfo, (void*)exceptionDumpRealSize, (void*)(mode));
        return 1;
    }
    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE;
    *exceptionDumpRealSize = 1;
    (void)exceptionDumpSize;
    return DeviceExceptionDumpCallBack(exceptionInfo, exceptionDumpInfo);
}

int32_t AdumpRegExceptionDump() { return AdumpRegExceptionDumpCallBack(ExceptionDumpCallBack); }
} // namespace npu::tile_fwk::dynamic
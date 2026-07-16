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
 * \file test_apapter_api.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "adapter/api/acl_api.h"
#include "adapter/api/adump_api.h"
#include "adapter/api/hal_api.h"
#include "adapter/api/hcomm_api.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/runtime_api.h"
#include "adapter/api/runtime_capture_context.h"
#include "adapter/manager/adapter_manager.h"

namespace npu::tile_fwk {
class TestAdapterApi : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestAdapterApi, test_acl_api)
{
    EXPECT_EQ(AclInit(nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclFinalize(), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtMemcpy(nullptr, 0, nullptr, 0, AclRtMemcpyKind::HOST_TO_HOST), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtSetDevice(0), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtResetDevice(0), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtCreateEvent(nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtRecordEvent(nullptr, nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtCreateEventExWithFlag(nullptr, 0), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtStreamWaitEvent(nullptr, nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtGetStreamResLimit(nullptr, AclRtDevResLimitType::CUBE_CORE, nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtGetStreamAttribute(nullptr, AclRtStreamAttr::FAILURE_MODE, nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtCacheLastTaskOpInfo(nullptr, 0), ACLRT_SUCCESS);
    EXPECT_EQ(AclRtSetExceptionInfoCallback(nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclMdlRICaptureGetInfo(nullptr, nullptr, nullptr), ACLRT_SUCCESS);
    EXPECT_EQ(AclMdlRICaptureThreadExchangeMode(nullptr), ACLRT_SUCCESS);
    char versionBuf[1] = {'x'};
    EXPECT_EQ(AclSysGetVersionStr("pkg", versionBuf), ACLRT_SUCCESS);
    EXPECT_EQ(versionBuf[0], '\0');
}

TEST_F(TestAdapterApi, test_adump_api)
{
    EXPECT_EQ(AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR), 0);
    std::vector<AdxTensorInfoV2> tensors;
    EXPECT_EQ(AdxDumpDumpTensorV2("op_name", "op_type", tensors, nullptr), 0);
}

TEST_F(TestAdapterApi, test_hal_api)
{
    EXPECT_EQ(HalMemCtl(0, nullptr, 0, nullptr, nullptr), HAL_ERROR_NONE);
    EXPECT_EQ(HalResMap(0, nullptr, nullptr, nullptr), HAL_ERROR_NONE);
    EXPECT_EQ(HalGetDeviceInfoByBuff(0, 0, 0, nullptr, nullptr), HAL_ERROR_NONE);
}

TEST_F(TestAdapterApi, test_hccl_api)
{
    EXPECT_EQ(HcommGetCommName(nullptr, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetL0TopoTypeEx(nullptr, nullptr, 0), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetCommHandleByGroup(nullptr, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetRootInfo(nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommCommDestroy(nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommCommInitRootInfo(0, nullptr, 0, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr), HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api)
{
    EXPECT_EQ(MspfSysCycleTime(), 0);
    EXPECT_EQ(MspfGetHashId(nullptr, 0), 0);
    EXPECT_EQ(MspfReportApi(0, nullptr), 0);
    EXPECT_EQ(MspfReportCompactInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MspfReportAdditionalInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MspfRegisterCallback(0, nullptr), 0);
}

TEST_F(TestAdapterApi, test_runtime_api)
{
    EXPECT_EQ(RuntimeMalloc(nullptr, 0, 0, 0), RT_SUCCESS);
    EXPECT_EQ(RuntimeMemset(nullptr, 0, 0, 0), RT_SUCCESS);
    EXPECT_EQ(RuntimeMemcpyDirect(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST), RT_SUCCESS);

    RuntimeCaptureContext::SetCaptureMode(true);
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, true);
    EXPECT_EQ(RuntimeMemcpyDirect(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST), RT_SUCCESS);
    EXPECT_EQ(RuntimeMemcpyDirectAsync(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST, nullptr), RT_SUCCESS);
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    RuntimeCaptureContext::SetCaptureMode(false);
    EXPECT_EQ(RuntimeFree(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeSetDevice(0), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetDevice(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetSocSpec(nullptr, nullptr, nullptr, 0), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetSocVersion(nullptr, 0), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetAiCpuCount(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetL2CacheOffset(0, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeGetLogicDevIdByUserDevId(0, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeFuncGetByName(nullptr, nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeBinaryLoadFromFile(nullptr, nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeStreamCreate(nullptr, 0), RT_SUCCESS);
    EXPECT_EQ(RuntimeStreamDestroy(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeStreamAddToModel(nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeStreamSynchronize(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeDevBinaryUnRegister(nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeRegisterAllKernel(nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeKernelLaunchWithHandleV2(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeLaunchCpuKernel(nullptr, 0, nullptr, nullptr, nullptr), RT_SUCCESS);
    EXPECT_EQ(RuntimeAicpuKernelLaunchExWithArgs(0, nullptr, 0, nullptr, nullptr, nullptr, 0), RT_SUCCESS);
}

TEST_F(TestAdapterApi, test_acl_adapter)
{
    bool hasCann = std::getenv("ASCEND_HOME_PATH") != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Finalize) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMemcpy) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtResetDevice) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEvent) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtRecordEvent) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEventExWithFlag) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtStreamWaitEvent) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamResLimit) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamAttribute) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCacheLastTaskOpInfo) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetExceptionInfoCallback) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureGetInfo) != nullptr, hasCann);
    EXPECT_EQ(
        AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureThreadExchangeMode) != nullptr,
        hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::SysGetVersionStr) != nullptr, hasCann);
}

TEST_F(TestAdapterApi, test_adump_adapter)
{
    bool hasCann = std::getenv("ASCEND_HOME_PATH") != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::GetDumpSwitch) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpTensorV2) != nullptr, hasCann);
}

TEST_F(TestAdapterApi, test_hal_adapter)
{
    bool hasCann = std::getenv("ASCEND_HOME_PATH") != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::MemCtl) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::GetDeviceInfoByBuff) != nullptr, hasCann);
}

TEST_F(TestAdapterApi, test_msprof_adapter)
{
    bool hasCann = std::getenv("ASCEND_HOME_PATH") != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::SysCycleTime) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::GetHashId) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportApi) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportCompactInfo) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportAdditionalInfo) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::RegisterCallback) != nullptr,
              hasCann);
}

TEST_F(TestAdapterApi, test_runtime_adapter)
{
    bool hasCann = std::getenv("ASCEND_HOME_PATH") != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Malloc) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopyAsync) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Free) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::SetDevice) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetDevice) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocSpec) != nullptr, hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocVersion) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetAiCpuCount) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetL2CacheOffset) != nullptr,
              hasCann);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetLogicDevIdByUserDevId) != nullptr,
        hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::FuncGetByName) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::BinaryLoadFromFile) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamCreate) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamDestroy) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamAddToModel) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamSynchronize) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryUnRegister) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::RegisterAllKernel) != nullptr,
              hasCann);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::LaunchCpuKernel) != nullptr,
              hasCann);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunchWithHandleV2) != nullptr,
        hasCann);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::AicpuKernelLaunchExWithArgs) != nullptr,
        hasCann);
}
} // namespace npu::tile_fwk

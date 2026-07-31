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
#include "tilefwk/error.h"

namespace npu::tile_fwk {
// Check whether CANN shared libraries are actually loaded (not just whether ASCEND_HOME_PATH is set).
static bool HasCannLoaded() { return AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init) != nullptr; }

// Always call the API (to cover both CANN and stub code paths), but only assert the return value in stub mode.
// In CANN mode the real CANN functions may return non-success error codes for invalid (nullptr) arguments.
#define CHECK_API_RET(call, expected)     \
    do {                                  \
        auto _apiRet = (call);            \
        if (!hasCann) {                   \
            EXPECT_EQ(_apiRet, expected); \
        }                                 \
    } while (0)

class TestAdapterApi : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override
    {
        RuntimeCaptureContext::SetCaptureMode(false);
        RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    }
};

TEST_F(TestAdapterApi, test_acl_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(AclInit(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclFinalize(), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtMemcpy(nullptr, 0, nullptr, 0, AclRtMemcpyKind::HOST_TO_HOST), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSetDevice(0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtResetDevice(0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateEvent(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtRecordEvent(nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateEventExWithFlag(nullptr, 0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtStreamWaitEvent(nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtGetStreamResLimit(nullptr, AclRtDevResLimitType::CUBE_CORE, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtGetStreamAttribute(nullptr, AclRtStreamAttr::FAILURE_MODE, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCacheLastTaskOpInfo(nullptr, 0), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSetExceptionInfoCallback(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtMalloc(nullptr, 0, AclRtMemMallocPolicy::HUGE_FIRST), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtFree(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtCreateStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtSynchronizeStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclRtDestroyStream(nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclMdlRICaptureGetInfo(nullptr, nullptr, nullptr), ACLRT_SUCCESS);
    CHECK_API_RET(AclMdlRICaptureThreadExchangeMode(nullptr), ACLRT_SUCCESS);
    char versionBuf[1] = {'x'};
    CHECK_API_RET(AclSysGetVersionStr("pkg", versionBuf), ACLRT_SUCCESS);
    if (!hasCann) {
        EXPECT_EQ(versionBuf[0], '\0');
    }
}

TEST_F(TestAdapterApi, test_adump_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR), 0);
    CHECK_API_RET(AdumpRegExceptionDumpCallBack(nullptr), 0);
    std::vector<AdxTensorInfoV2> emptyTensors;
    CHECK_API_RET(AdxDumpDumpTensorV2("op_name", "op_type", emptyTensors, nullptr), 0);

    AdxTensorInfoV2 tensorInfo{};
    tensorInfo.type = AdxTensorType::INPUT;
    tensorInfo.tensorSize = 128;
    tensorInfo.format = 0;
    tensorInfo.dataType = 1;
    tensorInfo.tensorAddr = nullptr;
    tensorInfo.addrType = AdxAddressType::TRADITIONAL;
    tensorInfo.placement = 0;
    tensorInfo.argsOffSet = 0;
    tensorInfo.shape = {1, 128};
    tensorInfo.originShape = {1, 128};
    std::vector<AdxTensorInfoV2> tensors = {tensorInfo};
    CHECK_API_RET(AdxDumpDumpTensorV2("op_name", "op_type", tensors, nullptr), 0);
}

TEST_F(TestAdapterApi, test_hal_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(HalMemCtl(0, nullptr, 0, nullptr, nullptr), HAL_ERROR_NONE);
    CHECK_API_RET(HalResMap(0, nullptr, nullptr, nullptr), HAL_ERROR_NONE);
    CHECK_API_RET(HalGetDeviceInfoByBuff(0, 0, 0, nullptr, nullptr), HAL_ERROR_NONE);
}

TEST_F(TestAdapterApi, test_hccl_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(HcommGetCommName(nullptr, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetL0TopoTypeEx(nullptr, nullptr, 0), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetCommHandleByGroup(nullptr, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommGetRootInfo(nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommCommDestroy(nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommCommInitRootInfo(0, nullptr, 0, nullptr), HCOMM_SUCCESS);
    CHECK_API_RET(HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr), HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(MspfSysCycleTime(), 0);
    CHECK_API_RET(MspfGetHashId(nullptr, 0), 0);
    CHECK_API_RET(MspfReportApi(0, nullptr), 0);
    CHECK_API_RET(MspfReportCompactInfo(0, nullptr, 0), 0);
    CHECK_API_RET(MspfReportAdditionalInfo(0, nullptr, 0), 0);
    CHECK_API_RET(MspfRegisterCallback(0, nullptr), 0);
}

TEST_F(TestAdapterApi, test_runtime_api)
{
    bool hasCann = HasCannLoaded();
    CHECK_API_RET(RuntimeMalloc(nullptr, 0, 0, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeMemset(nullptr, 0, 0, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeMemcpyDirect(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST), RT_SUCCESS);

    RuntimeCaptureContext::SetCaptureMode(true);
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, true);
    CHECK_API_RET(RuntimeMemcpyDirect(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST), RT_SUCCESS);
    CHECK_API_RET(RuntimeMemcpyDirectAsync(nullptr, 0, nullptr, 0, RtMemcpyKind::HOST_TO_HOST, nullptr), RT_SUCCESS);
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    RuntimeCaptureContext::SetCaptureMode(false);
    CHECK_API_RET(RuntimeFree(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeSetDevice(0), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetDevice(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetSocSpec(nullptr, nullptr, nullptr, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetSocVersion(nullptr, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetAiCpuCount(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetL2CacheOffset(0, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeGetLogicDevIdByUserDevId(0, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeFuncGetByName(nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeBinaryLoadFromFile(nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeStreamCreate(nullptr, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeStreamDestroy(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeStreamAddToModel(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeStreamSynchronize(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeDevBinaryUnRegister(nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeRegisterAllKernel(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeDevBinaryRegister(nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeFunctionRegister(nullptr, nullptr, nullptr, nullptr, 0), RT_SUCCESS);
    CHECK_API_RET(RuntimeKernelLaunch(nullptr, 0, nullptr, 0, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeKernelLaunchWithHandleV2(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeLaunchCpuKernel(nullptr, 0, nullptr, nullptr, nullptr), RT_SUCCESS);
    CHECK_API_RET(RuntimeAicpuKernelLaunchExWithArgs(0, nullptr, 0, nullptr, nullptr, nullptr, 0), RT_SUCCESS);
    RtExceptionRegInfo regInfo{};
    CHECK_API_RET(RuntimeGeExceptionRegInfo(nullptr, &regInfo), RT_SUCCESS);
}

TEST_F(TestAdapterApi, test_acl_adapter)
{
    bool loaded = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Finalize) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMemcpy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtResetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtRecordEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEventExWithFlag) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtStreamWaitEvent) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamResLimit) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamAttribute) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCacheLastTaskOpInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetExceptionInfoCallback) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureGetInfo) != nullptr, loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureThreadExchangeMode) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::SysGetVersionStr) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMalloc) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtFree) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateStream) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSynchronizeStream) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtDestroyStream) != nullptr, loaded);
}

TEST_F(TestAdapterApi, test_adump_adapter)
{
    bool loaded = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::GetDumpSwitch) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpTensorV2) != nullptr, loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpFailTaskExceptionCallBack) != nullptr,
        loaded);
}

TEST_F(TestAdapterApi, test_hal_adapter)
{
    bool loaded = AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::MemCtl) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetHalAdapter().GetFunction(HalFunc::GetDeviceInfoByBuff) != nullptr, loaded);
}

TEST_F(TestAdapterApi, test_msprof_adapter)
{
    bool loaded = AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::SysCycleTime) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::GetHashId) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportApi) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportCompactInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::ReportAdditionalInfo) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetMsprofAdapter().GetFunction(MsprofFunc::RegisterCallback) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_hccl_adapter)
{
    bool loaded = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommName) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetL0TopoTypeEx) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommHandleByGroup) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetRootInfo) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommInitRootInfo) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommDestroy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::AllocComResourceByTiling) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_runtime_adapter)
{
    bool loaded = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Malloc) != nullptr;
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::rtMemset) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopy) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopyAsync) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Free) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::SetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetDevice) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocSpec) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocVersion) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetAiCpuCount) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetL2CacheOffset) != nullptr,
              loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetLogicDevIdByUserDevId) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::FuncGetByName) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::BinaryLoadFromFile) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamCreate) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamDestroy) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamAddToModel) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamSynchronize) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryUnRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::RegisterAllKernel) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::LaunchCpuKernel) != nullptr,
              loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunchWithHandleV2) != nullptr,
        loaded);
    EXPECT_EQ(
        AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::AicpuKernelLaunchExWithArgs) != nullptr,
        loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::FunctionRegister) != nullptr,
              loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunch) != nullptr, loaded);
    EXPECT_EQ(AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetExceptionRegInfo) != nullptr,
              loaded);
}

TEST_F(TestAdapterApi, test_runtime_capture_context)
{
    RuntimeCaptureContext::SetCaptureMode(true);
    EXPECT_TRUE(RuntimeCaptureContext::IsCaptureMode());

    AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, true);
    EXPECT_TRUE(RuntimeCaptureContext::QueryThreadCaptureMode(mode));
    EXPECT_EQ(mode, AclMdlRICaptureMode::RELAXED);

    RuntimeCaptureContext::SetTestThreadCaptureMode(AclMdlRICaptureMode::RELAXED, false);
    EXPECT_TRUE(RuntimeCaptureContext::QueryThreadCaptureMode(mode));

    RuntimeCaptureContext::SetCaptureMode(false);
    EXPECT_FALSE(RuntimeCaptureContext::IsCaptureMode());
}

TEST_F(TestAdapterApi, test_error_message)
{
    ErrorMessage msg;
    std::vector<int> emptyVec;
    msg << emptyVec;
    EXPECT_EQ(msg.Message(), "[]");

    ErrorMessage msg2;
    std::vector<int> singleVec = {42};
    msg2 << singleVec;
    EXPECT_EQ(msg2.Message(), "[42]");

    ErrorMessage msg3;
    std::vector<std::string> multiVec = {"a", "b", "c"};
    msg3 << multiVec;
    EXPECT_EQ(msg3.Message(), "[a, b, c]");
}

} // namespace npu::tile_fwk

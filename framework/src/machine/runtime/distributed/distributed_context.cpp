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
 * \file distributed_context.cpp
 * \brief
 */

#include <memory>
#include "securec.h"
#include "distributed_context.h"
#include "interface/utils/common.h"
#include "machine/runtime/runtime.h"
#include "machine/device/dynamic/device_utils.h"
#include "tilefwk/tilefwk_log.h"

#ifdef BUILD_WITH_CANN
#include "hcom.h"
#include "acl/acl.h"
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* MC2Tiling, void** commContext);
#endif

constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_MESH = 0b1u;
constexpr uint32_t WINDATA_INDEX = 0;
constexpr uint32_t WINSTATUS_INDEX = 1;
constexpr uint32_t WINDEBUG_INDEX = 2;
std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> g_context; //key: groupname; value: deviceCommContext,hostCommContext

namespace npu::tile_fwk::dynamic {
template<typename T>
void DistributedContext::FillCommCtxAttr(TileOp::CommContext *ctxHost, T *hcclParamhost) {
    (void)ctxHost;
    (void)hcclParamhost;
    return;
}

template<>
void DistributedContext::FillCommCtxAttr<npu::tile_fwk::HcclCombinOpParam>(TileOp::CommContext *ctxHost, npu::tile_fwk::HcclCombinOpParam *hcclParamhost) {
    ctxHost->rankId = hcclParamhost->rankId;
    ctxHost->rankNum = hcclParamhost->rankNum;
    ctxHost->statusIndex = hcclParamhost->rankNum;
    ctxHost->debugIndex = hcclParamhost->rankNum * 2;
    ctxHost->winDataSize = hcclParamhost->winSize;
    ctxHost->winStatusSize = hcclParamhost->winExpSize;
    ctxHost->winDebugSize = hcclParamhost->winSize;
    ctxHost->totalWinNum = hcclParamhost->rankNum * WIN_TYPE_NUM;
    return;
}

template<>
void DistributedContext::FillCommCtxAttr<npu::tile_fwk::HcclOpResParamHead>(TileOp::CommContext *ctxHost, npu::tile_fwk::HcclOpResParamHead *hcclParamhost) {
    ctxHost->rankId = hcclParamhost->localUsrRankId;
    ctxHost->rankNum = hcclParamhost->rankSize;
    ctxHost->statusIndex = hcclParamhost->rankSize;
    ctxHost->debugIndex = hcclParamhost->rankSize * 2;
    ctxHost->winDataSize = hcclParamhost->winSize;
    ctxHost->winStatusSize = hcclParamhost->winSize;
    ctxHost->winDebugSize = hcclParamhost->winExpSize;
    ctxHost->totalWinNum = hcclParamhost->rankSize * WIN_TYPE_NUM;
    return;
}

template<typename T>
void DistributedContext::FillCommCtxWinArr(int i, TileOp::CommContext *ctxHost, T *hcclParamhost) {
    (void)i;
    (void)ctxHost;
    (void)hcclParamhost;
    return;
}

template<>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclCombinOpParam>(int i, TileOp::CommContext *ctxHost, npu::tile_fwk::HcclCombinOpParam *hcclParamhost) {
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn[i];
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsExp[i];
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsOut[i];
    return;
}

template<>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclOpResParamHead>(int i, TileOp::CommContext *ctxHost, npu::tile_fwk::HcclOpResParamHead *hcclParamhost) {
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsIn;
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsExp;
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->localWindowsOut;
    return;
}

template<>
void DistributedContext::FillCommCtxWinArr<npu::tile_fwk::HcclRankRelationResV2>(int i, TileOp::CommContext *ctxHost, npu::tile_fwk::HcclRankRelationResV2 *hcclParamhost) {
    ctxHost->winAddr[i + (WINDATA_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsIn;
    ctxHost->winAddr[i + (WINSTATUS_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsExp;
    ctxHost->winAddr[i + (WINDEBUG_INDEX * ctxHost->rankNum)] = hcclParamhost->windowsOut;
    return;
}

template<ResType T>
uint64_t DistributedContext::AllocCommContext([[maybe_unused]] const uint64_t ctxAddr, [[maybe_unused]]const std::string &groupName) {
    return 0;
}

template<>
uint64_t DistributedContext::AllocCommContext<ResType::MESH>([[maybe_unused]] const uint64_t ctxAddr, [[maybe_unused]]const std::string &groupName) {
#ifdef BUILD_WITH_CANN
    npu::tile_fwk::HcclCombinOpParam *hcclParamDevice = (npu::tile_fwk::HcclCombinOpParam *)ctxAddr;
    npu::tile_fwk::HcclCombinOpParam *hcclParamhost = nullptr;
    hcclParamhost = (npu::tile_fwk::HcclCombinOpParam *)machine::GetRuntimeHostAgent()->AllocHostAddr(sizeof(npu::tile_fwk::HcclCombinOpParam));
    ASSERT(hcclParamhost != nullptr) << "hcclParamhost malloc failed";
    size_t offsetRankId = offsetof(npu::tile_fwk::HcclCombinOpParam, rankId);
    size_t offsetHcomId = offsetof(npu::tile_fwk::HcclCombinOpParam, hcomId);
    size_t offsetWinExpSize = offsetof(npu::tile_fwk::HcclCombinOpParam, winExpSize);
    size_t offsetMultiServerFlag = offsetof(npu::tile_fwk::HcclCombinOpParam, multiServerFlag);
    aclrtMemcpy(&(hcclParamhost->rankId), offsetHcomId - offsetRankId, &(hcclParamDevice->rankId),
                offsetHcomId - offsetRankId, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&(hcclParamhost->winExpSize), offsetMultiServerFlag - offsetWinExpSize,
                &(hcclParamDevice->winExpSize), offsetMultiServerFlag - offsetWinExpSize,
                ACL_MEMCPY_DEVICE_TO_HOST);

    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM;
    TileOp::CommContext *ctxHost = 
            (TileOp::CommContext *)machine::GetRuntimeHostAgent()->AllocHostAddr(commCtxSize);
    ASSERT(ctxHost != nullptr) << "ctxHost malloc failed";
    FillCommCtxAttr<npu::tile_fwk::HcclCombinOpParam>(ctxHost, hcclParamhost);
    for (uint32_t i = 0; i < ctxHost->rankNum; i++) {
        FillCommCtxWinArr<npu::tile_fwk::HcclCombinOpParam>(i, ctxHost, hcclParamhost);
    }
    TileOp::CommContext *ctxDevice = nullptr;
    machine::GetRA()->AllocDevAddr((uint8_t **)&ctxDevice, commCtxSize);
    ASSERT(ctxDevice != nullptr)<< "ctxDevice malloc failed";
    aclrtMemcpy(ctxDevice, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM,
                ctxHost, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM,
                ACL_MEMCPY_HOST_TO_DEVICE);
    g_context[groupName].first = (uint64_t)ctxDevice;
    g_context[groupName].second = (uint64_t)ctxHost;
    return (uint64_t)ctxDevice;
#endif
    return 0;
}

template<>
uint64_t DistributedContext::AllocCommContext<ResType::RING>([[maybe_unused]] const uint64_t ctxAddr, [[maybe_unused]]const std::string &groupName) {
#ifdef BUILD_WITH_CANN
    npu::tile_fwk::HcclOpResParam *hcclParam = (npu::tile_fwk::HcclOpResParam *)ctxAddr;
    npu::tile_fwk::HcclOpResParamHead *hcclParamhost = 
            (npu::tile_fwk::HcclOpResParamHead *)machine::GetRuntimeHostAgent()->AllocHostAddr(sizeof(npu::tile_fwk::HcclOpResParamHead));
    ASSERT(hcclParamhost != nullptr) << "hcclParamhost malloc failed";
    size_t offsetLocalUsrRankId = offsetof(npu::tile_fwk::HcclOpResParam, localUsrRankId);
    size_t offsetRWinStart = offsetof(npu::tile_fwk::HcclOpResParam, rWinStart);
    aclrtMemcpy(&(hcclParamhost->localUsrRankId), offsetRWinStart - offsetLocalUsrRankId,
                &(hcclParam->localUsrRankId), offsetRWinStart - offsetLocalUsrRankId,
                ACL_MEMCPY_DEVICE_TO_HOST);

    size_t remoteResSize = hcclParamhost->rankSize * sizeof(npu::tile_fwk::RemoteResPtr);
    npu::tile_fwk::RemoteResPtr *remoteResPtr = 
            (npu::tile_fwk::RemoteResPtr *)machine::GetRuntimeHostAgent()->AllocHostAddr(remoteResSize);
    ASSERT(remoteResPtr != nullptr) << "remoteResPtr malloc failed";
    aclrtMemcpy(remoteResPtr, remoteResSize,
                &(hcclParam->remoteRes), remoteResSize,
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM;
    TileOp::CommContext *ctxHost = (TileOp::CommContext *)machine::GetRuntimeHostAgent()->AllocHostAddr(commCtxSize);
    ASSERT(ctxHost != nullptr) << "ctxHost malloc failed";
    FillCommCtxAttr<npu::tile_fwk::HcclOpResParamHead>(ctxHost, hcclParamhost);
    for (uint64_t i = 0; i < hcclParamhost->rankSize; i++) {
        if (i == hcclParamhost->localUsrRankId) {
            FillCommCtxWinArr<npu::tile_fwk::HcclOpResParamHead>(i, ctxHost, hcclParamhost);
            continue;
        }
        uint64_t remoteResDevicePtr;
        aclrtMemcpy(&remoteResDevicePtr, sizeof(uint64_t), 
                    &(remoteResPtr[i].nextDevicePtr), sizeof(uint64_t), 
                    ACL_MEMCPY_DEVICE_TO_HOST); // 设备二级指针值拷贝到主机
        npu::tile_fwk::HcclRankRelationResV2 remoteParam;
        aclrtMemcpy(&remoteParam, sizeof(npu::tile_fwk::HcclRankRelationResV2), 
                (void *)remoteResDevicePtr, sizeof(npu::tile_fwk::HcclRankRelationResV2), ACL_MEMCPY_DEVICE_TO_HOST);
        FillCommCtxWinArr<npu::tile_fwk::HcclRankRelationResV2>(i, ctxHost, &remoteParam);
    }
    TileOp::CommContext *ctxDevice = nullptr;
    machine::GetRA()->AllocDevAddr((uint8_t **)&ctxDevice, commCtxSize);
    ASSERT(ctxDevice != nullptr) << "ctxDevice malloc failed";
    aclrtMemcpy(ctxDevice, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM, 
                ctxHost, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM,
                ACL_MEMCPY_HOST_TO_DEVICE);
    g_context[groupName].first = (uint64_t)ctxDevice;
    g_context[groupName].second = (uint64_t)ctxHost;
    return (uint64_t)ctxDevice;
#endif
    return 0;
}

std::vector<uint64_t> DistributedContext::GetCommContext([[maybe_unused]] const std::vector<std::string> &groupNames) {
#ifdef BUILD_WITH_CANN
    if (groupNames.size() == 0) {
        return {};
    }
    CommTopo topoRet;
    const char *group = groupNames[0].c_str();
    ASSERT(HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE) == HCCL_SUCCESS) << "Get hcom topo failed";
    uint32_t topoType = static_cast<uint32_t>(topoRet);
    std::shared_ptr<TilingStructBase> tilingStruct;
    if (topoType == COMM_MESH) {
        tilingStruct = std::make_shared<TilingStruct>();
    } else {
        tilingStruct = std::make_shared<TilingStructV2>();
    }
    std::vector<uint64_t> commContext(groupNames.size(), 0);
    CHECK(groupNames.size() <= DIST_COMM_GROUP_NUM) <<"Commgroup size is not be supported";
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        if (g_context.find(groupName) != g_context.end()) { // 检查context缓存
            commContext[groupIndex] = g_context[groupName].first;
            continue;
        }
        HcclComm commHandle = nullptr;
        ASSERT(HcomGetCommHandleByGroup(groupName.c_str(), &commHandle) == 0) << "Get hcom handle failed";
        tilingStruct->MakeMc2TilingStruct(groupName);
        auto ret = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(), tilingStruct->GetMc2CommConfig(),
            reinterpret_cast<void **>(&commContext[groupIndex]));
        ASSERT((ret == 0) && (commContext[groupIndex] != 0UL)) << "Hccl alloc resource failed";
        DISTRIBUTED_LOGI("groupIndex=%u, groupName=%s, commContext=%lu", groupIndex, groupName.c_str(),
            commContext[groupIndex]);
        if (topoType == COMM_MESH) {
            commContext[groupIndex] = AllocCommContext<ResType::MESH>(commContext[groupIndex], groupName);
        } else {
            commContext[groupIndex] = AllocCommContext<ResType::RING>(commContext[groupIndex], groupName);
        }
    }
    return commContext;
#endif
    return {};
}

std::vector<uint64_t> DistributedContext::GetCommContextToHost([[maybe_unused]] const std::vector<std::string> &groupNames) {
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> devAddrs = GetCommContext(groupNames);
    std::vector<uint64_t> hostContext;
    CHECK(groupNames.size() <= DIST_COMM_GROUP_NUM) << "Commgroup size is not be supported";
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {
        std::string groupName = groupNames[groupIndex];
        hostContext.push_back((uint64_t)g_context[groupName].second);
    }
    return hostContext;
#endif
    return {};
}
} // namespace npu::tile_fwk::dynamic
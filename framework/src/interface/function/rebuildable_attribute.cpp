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
 * \file rebuildable_attribute.cpp
 * \brief
 */

#include "rebuildable_attribute.h"

#include "tilefwk/pypto_fwk_log.h"
#include "interface/utils/common.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

void RebuildableAttributeBase::Rebuild(Function *func)
{
    (void)func;
    FE_LOGE(FeError::EINTERNAL, "Reset is not implemented!");
}

void RebuildableAttributeBase::Reset(void *data)
{
    (void)data;
    FE_LOGE(FeError::EINTERNAL, "Reset is not implemented!");
}

EntryRegistrarGroup &RebuildableAttributeManager::GetRegistrarGroup() {
    static EntryRegistrarGroup group;
    return group;
}

RebuildableAttributeManager &RebuildableAttributeManager::GetInstance()
{
    static RebuildableAttributeManager instance;
    return instance;
}

void RebuildableAttributeManager::InitAttrsForFunc(Function *func)
{
    RebuildableAttrInitContext ctx{this, func};
    GetRegistrarGroup().Init(&ctx);
}

#define ComputeSize(name) \
    constexpr uint64_t ALIGNMENT_32K = 32 * 1024; \
    uint64_t tensorRootInnerSpill = desc.maxRootInnerSpilledMem; \
    uint64_t tensorDevTaskInnerExclusiveOutcast = desc.maxRootTotalExclusiveOutcastMem; \
    uint64_t tensorMaxOutcast = std::max(desc.maxStaticOutcastMem, maxDynamicAssembleOutcastMem); \
    uint64_t tensorBoundaryAndInnerTemporalOutcastNum = desc.devTaskBoundaryOutcastNum + desc.devTaskInnerTemporalOutcastNum; \
    uint64_t tensorPerOutcast = tensorMaxOutcast * tensorBoundaryAndInnerTemporalOutcastNum; \
    uint64_t tensorTotal = \
        tensorRootInnerSpill + tensorDevTaskInnerExclusiveOutcast + tensorPerOutcast; \
    uint64_t tensorTotalAlloc = AlignUp(tensorTotal, ALIGNMENT_32K) * desc.config.parallelism; \
    uint64_t leafSpill = desc.platform.aicoreCount * desc.maxLeafPerCoreSpilledMem; \
    uint64_t name = tensorTotalAlloc + leafSpill + debugSize;

uint64_t RebuildableWorkspaceDesc::GetSizeForCheckOnly(
    uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const
{
    ComputeSize(workspaceSize);
    return workspaceSize;
}

std::string RebuildableWorkspaceDesc::PrettyDumpSize(
    uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const
{
    std::ostringstream oss;
    oss << "Config:\n"
        << "  " << std::setw(30) << std::left << "innerSpilledRecyclePeriod:"
                << std::setw(10) << std::right << desc.config.innerSpilledRecyclePeriod << "\n"
        << "  " << std::setw(30) << std::left << "unrollStitchCount:"
                << std::setw(10) << std::right << desc.config.unrollStitchCount << "\n"
        << "  " << std::setw(30) << std::left << "actualStitchCount:"
                << std::setw(10) << std::right << desc.config.actualStitchCount << "\n"
        << "  " << std::setw(30) << std::left << "parallelism:"
                << std::setw(10) << std::right << desc.config.parallelism << "\n";
    oss << "Root:\n";
    for (auto &rootFuncDesc : desc.rootFuncDescList) {
        oss << "  " << "name: " << rootFuncDesc.devFuncName << "\n";
        oss << "    " << "unroll:" << std::setw(3) << rootFuncDesc.unroll
            << " innerSpilledRaw:" << std::setw(10) << rootFuncDesc.rootInnerSpilledRawMem
            << " innerSpilled:" << std::setw(10) << rootFuncDesc.rootInnerSpilledMem
            << " leafSpilled:" << std::setw(7) << rootFuncDesc.leafPerCoreSpilledMem
            << " totalOutcastRaw:" << std::setw(10) << rootFuncDesc.rootTotalExclusiveOutcastRawMem
            << " totalOutcast:" << std::setw(10) << rootFuncDesc.rootTotalExclusiveOutcastMem
            << " staticOutcast:" << std::setw(10) << rootFuncDesc.rootMaxExclusiveOutcastMem << "\n";
    }

    ComputeSize(workspaceSize);

    auto print = [&](int indent, const std::string &name, uint64_t value) {
        oss << std::setw(indent * 2) << " "
            << std::setw(50 - indent * 2) << std::left << name
            << std::setw(10) << std::right << value << "\n";
    };
    print(1, "workspace:", workspaceSize);
    print(2, "tensorAlloc:", tensorTotalAlloc);
    print(3, "tensor:", tensorTotal);
    print(4, "rootInnerSpilled:", tensorRootInnerSpill);
    print(4, "devTaskInnerOutcast:", tensorDevTaskInnerExclusiveOutcast);
    print(4, "devTaskBoundaryAndInnerOutcast:", tensorPerOutcast);
    print(5, "maxOutcast:", tensorMaxOutcast);
    print(6, "staticOutcast:", desc.maxStaticOutcastMem);
    print(6, "dynamicOutcast:", maxDynamicAssembleOutcastMem);
    print(5, "devTaskBoundaryAndInnerOutcastCount:", tensorBoundaryAndInnerTemporalOutcastNum);
    print(6, "devTaskBoundaryOutcastCount:", desc.devTaskBoundaryOutcastNum);
    print(6, "devTaskInnerOutcastCount:", desc.devTaskInnerTemporalOutcastNum);
    print(3, "parallel:", desc.config.parallelism);
    print(2, "leafSpill:", leafSpill);
    print(3, "aicoreCount:", desc.platform.aicoreCount);
    print(3, "maxLeafPerCoreSpilled:", desc.maxLeafPerCoreSpilledMem);
    print(2, "debug:", debugSize);
    return oss.str();
}

RBUILDABLE_ATTRIBUTE_REGISTER(RebuildableWorkspaceDesc);

}

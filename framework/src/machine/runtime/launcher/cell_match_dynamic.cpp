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
 * \file cell_match_dynamic.cpp
 * \brief Per-launch evaluation and refresh of dynamic CellMatchTableDesc.stride on host.
 */

#include "machine/runtime/launcher/cell_match_dynamic.h"

#include <vector>

#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::dynamic {

namespace {

bool TryBuildDynamicCellMatchDesc(
    const DyndevFunctionAttribute::DynamicCellMatchLaunchMeta& launchMeta, Evaluator& eval,
    DevCellMatchTableDesc& patchedDesc)
{
    patchedDesc.SetCellShape(launchMeta.cellShape);
    const int dim = patchedDesc.GetDimensionSize();
    if (launchMeta.candidateRawDims.empty() || dim > DEV_SHAPE_DIM_MAX) {
        return false;
    }

    bool consistent = true;
    int64_t refStride[DEV_SHAPE_DIM_MAX]{0};
    for (size_t c = 0; c < launchMeta.candidateRawDims.size(); ++c) {
        int64_t currentStride[DEV_SHAPE_DIM_MAX]{0};
        for (int d = dim - 1; d >= 0; --d) {
            auto expr = launchMeta.candidateRawDims[c][d];
            int64_t tensorDim = eval.Evaluate(expr);
            int64_t cellDim = std::max<int64_t>(patchedDesc.GetCellShape(d), 1);
            int64_t tile = (tensorDim + cellDim - 1) / cellDim;
            ASSERT(tile > 0) << "Invalid tile for dynamic cell match slot=" << launchMeta.slotIndex << ", dim=" << d;
            currentStride[d] = tile;
        }
        if (c == 0) {
            for (int d = 0; d < dim; ++d) {
                refStride[d] = currentStride[d];
            }
            continue;
        }
        for (int d = 0; d < dim; ++d) {
            if (refStride[d] != currentStride[d]) {
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            break;
        }
    }

    if (!consistent) {
        return false;
    }

    std::vector<int> strideShape(dim);
    for (int d = 0; d < dim; ++d) {
        strideShape[d] = static_cast<int>(refStride[d]);
    }
    patchedDesc.SetStrideShape(strideShape);
    return true;
}

} // namespace

std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(
    const DyndevFunctionAttribute& dynAttr, Evaluator& eval)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    if (dynAttr.dynamicCellMatchLaunchMetaList.empty()) {
        return patches;
    }

    for (const auto& launchMeta : dynAttr.dynamicCellMatchLaunchMetaList) {
        DevCellMatchTableDesc patchedDesc;
        bool ready = TryBuildDynamicCellMatchDesc(launchMeta, eval, patchedDesc);
        if (!ready) {
            ASSERT(false) << "dynamic cell match launch prepare failed, slot=" << launchMeta.slotIndex;
        }
        DevDynamicCellMatchStridePatch patch;
        patch.descOffset = launchMeta.descOffset;
        patch.stride = patchedDesc.stride;
        patches.push_back(patch);
    }
    return patches;
}

void PatchHostDynamicCellMatchTableDesc(
    DevAscendProgram* hostDevProg, const std::vector<DevDynamicCellMatchStridePatch>& patches)
{
    if (hostDevProg == nullptr || patches.empty()) {
        return;
    }
    auto* cfgBytes = reinterpret_cast<uint8_t*>(hostDevProg);
    for (const auto& patch : patches) {
        auto* dstDesc = reinterpret_cast<DevCellMatchTableDesc*>(cfgBytes + patch.descOffset);
        dstDesc->stride = patch.stride;
    }
}

void WriteDynamicCellMatchStridePatchesToLaunchArgs(
    int64_t* launchInputs, const std::vector<DevDynamicCellMatchStridePatch>& patches)
{
    if (launchInputs == nullptr) {
        return;
    }
    const uint64_t inputCount = static_cast<uint64_t>(launchInputs[0]);
    const uint64_t outputCount = static_cast<uint64_t>(launchInputs[1]);
    auto* patchCountPtr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(launchInputs + TENSOR_INFO_OFFSET) + inputCount + outputCount);
    *patchCountPtr = patches.size();
    auto* patchArr = reinterpret_cast<DevDynamicCellMatchStridePatch*>(patchCountPtr + 1);
    for (size_t i = 0; i < patches.size(); ++i) {
        patchArr[i] = patches[i];
    }
}

static bool IsOutputTensorStitchSlot(const DevAscendProgram* hostDevProg, int slotIndex)
{
    if (hostDevProg == nullptr || slotIndex < 0) {
        return false;
    }
    const auto& partialUpdateList = hostDevProg->partialUpdateList;
    if (static_cast<size_t>(slotIndex) >= partialUpdateList.size() || partialUpdateList.Data() == nullptr) {
        return false;
    }
    const auto* partialUpdates = reinterpret_cast<const DevAscendProgramPartialUpdate*>(
        reinterpret_cast<const uint8_t*>(hostDevProg) +
        reinterpret_cast<uintptr_t>(partialUpdateList.Data()));
    return partialUpdates[slotIndex].isOutputTensorStitchSlot;
}

static bool HasAnyOutputTensorStitchSlot(const DevAscendProgram* hostDevProg)
{
    if (hostDevProg == nullptr) {
        return false;
    }
    for (size_t i = 0; i < hostDevProg->partialUpdateList.size(); ++i) {
        if (IsOutputTensorStitchSlot(hostDevProg, static_cast<int>(i))) {
            return true;
        }
    }
    return false;
}

void ValidateDynamicCellMatchTableMemBudget(
    const DyndevFunctionAttribute& dynAttr, DevAscendProgram* hostDevProg)
{
    if (hostDevProg == nullptr ||
        (dynAttr.constructAssembleNeedAllocRuntimeSlots.empty() && !HasAnyOutputTensorStitchSlot(hostDevProg))) {
        return;
    }
    const auto* cfgBytes = reinterpret_cast<const uint8_t*>(hostDevProg);
    for (const auto& launchMeta : dynAttr.dynamicCellMatchLaunchMetaList) {
        if (dynAttr.constructAssembleNeedAllocRuntimeSlots.count(launchMeta.slotIndex) == 0 &&
            !IsOutputTensorStitchSlot(hostDevProg, launchMeta.slotIndex)) {
            continue;
        }
        const auto* desc = reinterpret_cast<const DevCellMatchTableDesc*>(cfgBytes + launchMeta.descOffset);
        const uint64_t cellMatchStride0 = desc->stride.dimStride[0];
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, cellMatchStride0 < static_cast<uint64_t>(MAX_CELLMATCHSSTRIDE))
            << " Dynamic cell match slot=" << launchMeta.slotIndex
            << " stitch results in excessive memory consumption,"
            << "Please appropriately configure the view shape and tile shape, and ensure aligned with the input shape.";
    }
}

void RefillDynamicMemBudgets(
    DevAscendProgram* hostDevProg, DyndevFunctionAttribute& dynAttr, Evaluator& eval)
{
    if (hostDevProg == nullptr) {
        return;
    }
    if (dynAttr.maxDynamicAssembleOutcastMem.IsValid()) {
        hostDevProg->memBudget.tensor.maxDynamicAssembleOutcastMem =
            eval.Evaluate(dynAttr.maxDynamicAssembleOutcastMem);
    }
    if (!dynAttr.maxDynamicCellMatchTableMem.IsValid()) {
        return;
    }
    hostDevProg->memBudget.metadata.maxDynamicCellMatchTableMem =
        eval.Evaluate(dynAttr.maxDynamicCellMatchTableMem);
    uint64_t totalDynamicCellMatchSlotNum = hostDevProg->memBudget.metadata.dynamicCellMatchSlotNum;
    hostDevProg->memBudget.metadata.dynamicCellMatch =
        totalDynamicCellMatchSlotNum * hostDevProg->memBudget.metadata.maxDynamicCellMatchTableMem;
}

std::vector<DevDynamicCellMatchStridePatch> PrepareHostDynamicCellMatchForLaunch(
    DyndevFunctionAttribute& dynAttr, Evaluator& eval, DevAscendProgram* hostDevProg)
{
    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    PatchHostDynamicCellMatchTableDesc(hostDevProg, patches);
    RefillDynamicMemBudgets(hostDevProg, dynAttr, eval);
    if (dynAttr.maxDynamicCellMatchTableMem.IsValid()) {
        ValidateDynamicCellMatchTableMemBudget(dynAttr, hostDevProg);
    }
    return patches;
}

} // namespace npu::tile_fwk::dynamic

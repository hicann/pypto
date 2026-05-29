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
 * \file dump_device_topo.cpp
 * \brief 
 */
#include "machine/device/dynamic/context/dump_device_topo.h"

#ifndef __DEVICE__
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#endif

namespace npu::tile_fwk::dynamic::topo_dump {

#ifndef __DEVICE__
namespace {

constexpr const char* DeviceVerifyDumpSubDir = "dep_verify_dump";

const std::string& DeviceDepVerifyDumpDir()
{
    static const std::string dir = []() {
        std::string d = config::LogTopFolder() + "/" + DeviceVerifyDumpSubDir;
        (void)CreateDir(d);
        return d;
    }();
    return dir;
}

void OpenIfPathChanged(std::ofstream& ofs, std::string& lastPath,
                       const std::string& path, const char* header)
{
    if (path == lastPath) {
        return;
    }
    if (ofs.is_open()) {
        ofs.close();
    }
    lastPath = path;
    ofs.open(path);
    if (ofs.is_open()) {
        ofs << header << '\n';
    }
}

inline bool DumpEnabled()
{
    static const bool cached =
        (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_VERIFY);
    return cached;
}

struct CellMatchHandleCollect {
    static inline void Process(int index, std::vector<int>* cellIdxListOut)
    {
        cellIdxListOut->push_back(index);
    }
};

bool CollectCellIdxForUse(
    DevAscendFunction* devFunc, const DevAscendFunctionCallOperandUse& use,
    const uint64_t* runtimeExpressionList, bool isIOperand,
    const DevCellMatchTableDesc& cellMatchTableDesc, std::vector<int>* cellIdxListOut)
{
    uint64_t offset[DEV_SHAPE_DIM_MAX];
    uint64_t validShape[DEV_SHAPE_DIM_MAX];
    bool paramConcrete = GetTensorOffsetAndValidShape<false>(
        devFunc, offset, validShape, runtimeExpressionList, cellMatchTableDesc, cellMatchTableDesc.GetDimensionSize(),
        use.operationIdx, use.operandIdx, isIOperand);
    if (paramConcrete) {
        CellMatchHandle<CellMatchHandleCollect>(offset, validShape, cellMatchTableDesc, cellIdxListOut);
    }
    return paramConcrete;
}

void AppendSlotAccessRow(
    uint32_t seqNo, int slotIdx, uint32_t funcIdx, uint32_t opIdx,
    char accessType, const std::vector<int>& cellIdxList, bool allConcrete)
{
    static std::mutex mu;
    static std::ofstream ofs;
    static std::string lastPath;
    std::lock_guard<std::mutex> lock(mu);
    OpenIfPathChanged(ofs, lastPath, DeviceDepVerifyDumpDir() + "/dyn_slot_access.csv",
                      "seqNo,slotIdx,funcIdx,opIdx,taskId,accessType,cellIdxList,allConcrete");
    if (!ofs.is_open()) {
        return;
    }
    const uint32_t taskId = (funcIdx << 16) | (opIdx & 0xffffU);
    ofs << seqNo << ',' << slotIdx << ',' << funcIdx << ',' << opIdx << ','
        << taskId << ',' << accessType << ",[";
    for (size_t i = 0; i < cellIdxList.size(); ++i) {
        if (i != 0) {
            ofs << ';';
        }
        ofs << cellIdxList[i];
    }
    ofs << "]," << (allConcrete ? 1 : 0) << '\n';
}

} // namespace

void DumpProducerCellAccess(
    uint32_t devTaskId, int slotIdx, uint32_t devNextIdx,
    DevAscendFunction& devRootSrc, DevAscendFunctionOutcast& outcast,
    const DeviceExecuteSlot& slot, const uint64_t* expressionList)
{
    if (!DumpEnabled()) {
        return;
    }
    const DevCellMatchTableDesc& desc = slot.isPartialUpdateStitch
        ? slot.partialUpdate->cellMatchTableDesc
        : outcast.cellMatchTableDesc;
    const DevAscendFunctionCallOperandUse* useList = nullptr;
    size_t useSize = 0;
    if (slot.isPartialUpdateStitch && outcast.producerList.size() == 0) {
        useList = &devRootSrc.At(outcast.stitchPolicyFullCoverProducerList, 0);
        useSize = outcast.stitchPolicyFullCoverProducerList.size();
    } else {
        useList = &devRootSrc.At(outcast.producerList, 0);
        useSize = outcast.producerList.size();
    }
    for (size_t i = 0; i < useSize; ++i) {
        std::vector<int> cellIdxList;
        bool allConcrete =
            CollectCellIdxForUse(&devRootSrc, useList[i], expressionList, false, desc, &cellIdxList);
        AppendSlotAccessRow(
            devTaskId, slotIdx, devNextIdx, static_cast<uint32_t>(useList[i].operationIdx),
            'W', cellIdxList, allConcrete);
    }
}

void DumpConsumerCellAccess(
    uint32_t devTaskId, int slotIdx, uint32_t devNextIdx,
    DevAscendFunction& nextSrc, const DevAscendFunctionCallOperandUse& consumer,
    const DevCellMatchTableDesc& cellMatchTableDesc, const uint64_t* expressionList)
{
    if (!DumpEnabled()) {
        return;
    }
    std::vector<int> cellIdxList;
    bool allConcrete =
        CollectCellIdxForUse(&nextSrc, consumer, expressionList, true, cellMatchTableDesc, &cellIdxList);
    AppendSlotAccessRow(
        devTaskId, slotIdx, devNextIdx, static_cast<uint32_t>(consumer.operationIdx),
        'R', cellIdxList, allConcrete);
}

void DumpStitchEdge(
    const DevAscendFunctionDupped& producerDup, const DevAscendFunctionDupped& consumerDup,
    size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
    DeviceStitchContext::StitchKind stitchKind, int slotIdx)
{
    if (!DumpEnabled()) {
        return;
    }
    static std::mutex mu;
    static std::ofstream ofs;
    static std::string lastPath;
    std::lock_guard<std::mutex> lock(mu);
    OpenIfPathChanged(ofs, lastPath, DeviceDepVerifyDumpDir() + "/dyn_stitch_edges.csv",
                      "stitchKind,slotIdx,"
                      "producerFuncKey,producerFuncIdx,producerOpIdx,producerTaskId,"
                      "consumerFuncKey,consumerFuncIdx,consumerOpIdx,consumerTaskId");
    if (!ofs.is_open()) {
        return;
    }

    auto* prodSrc = producerDup.GetSource();
    auto* consSrc = consumerDup.GetSource();
    int producerFuncIdx = prodSrc->GetFuncidx();

    ofs << DeviceStitchContext::GetStitchKindName(stitchKind) << ',' << slotIdx << ','
        << prodSrc->funcKey << ',' << producerFuncIdx << ',' << producerOperationIdx << ','
        << MakeTaskID(producerFuncIdx, producerOperationIdx) << ','
        << consSrc->funcKey << ',' << consumerIdx << ',' << consumerOperationIdx << ','
        << MakeTaskID(consumerIdx, consumerOperationIdx) << '\n';
}

#else // __DEVICE__

void DumpProducerCellAccess(
    uint32_t, int, uint32_t,
    DevAscendFunction&, DevAscendFunctionOutcast&,
    const DeviceExecuteSlot&, const uint64_t*)
{}

void DumpConsumerCellAccess(
    uint32_t, int, uint32_t,
    DevAscendFunction&, const DevAscendFunctionCallOperandUse&,
    const DevCellMatchTableDesc&, const uint64_t*)
{}

void DumpStitchEdge(
    const DevAscendFunctionDupped&, const DevAscendFunctionDupped&,
    size_t, size_t, size_t, DeviceStitchContext::StitchKind, int)
{}

#endif // __DEVICE__

} // namespace npu::tile_fwk::dynamic::topo_dump

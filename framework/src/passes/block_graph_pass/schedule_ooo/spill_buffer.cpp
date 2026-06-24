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
 * \file spill_buffer.cpp
 * \brief
 */

#include "ooo_scheduler.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext &ctx)
{
    auto reqMemType = localBufferMap_[opReqMemIdsMap[allocOp][0]]->memType;
    auto reqSize = localBufferMap_[opReqMemIdsMap[allocOp][0]]->size;
    std::vector<int> spillGroup = SelectSpillBuffers(allocOp);
    if (spillGroup.empty()) {
        // 选不出可spill的，报错
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    ctx.spillMemIds = spillGroup;
    for (auto &memId : spillGroup) {
        if (SpillBuffer(memId, allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill tensor[%d] for %s failed!",
                memId, GetOpInfo(allocOp).c_str());
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
    }

    if (RearrangeBuffer(allocOp, reqMemType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed for %s.", GetFormatBacktrace(*allocOp).c_str());
    }
    if (!HasEnoughBuffer(allocOp, reqMemType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed!");
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

std::vector<std::vector<int>> OoOScheduler::GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill)
{
    // 薄壳: 委托给 BufferPool::GetSpillGroup, 与 GetDualSpillGroup 形成对称的 scheduler 层入口。
    // 决策分叉 (单池 vs 双池) 留在 SelectSpillBuffers, 此处不做额外逻辑。
    return pool.GetSpillGroup(sizeNeedSpill);
}

std::vector<std::vector<int>> OoOScheduler::GetDualSpillGroup(
    BufferPool& poolA, BufferPool& poolB, size_t sizeNeedSpill)
{
    // 双池嵌套滑窗:
    //   外层在 poolA 上找候选窗 [iA, jA), 内层在 poolB 上找候选窗 [iB, jB);
    //   两侧"freed segment 的起点 startAddr"必须一致 (== 表示 dualdst 双池在同一地址段
    //   都能腾出足够空间) 才输出 combined memIds。
    // poolB 的 startAddrB 随 iB 单调递增, 所以内层在 startAddrB > startAddrA 时直接 break。
    std::vector<std::vector<int>> result;
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();

    size_t iA = 0;
    while (iA < bufsA.size()) {
        size_t startAddrA = poolA.ObtainStartAddr(iA, bufsA);
        if ((poolA.GetMemSize() - startAddrA) < sizeNeedSpill) {
            break;
        }
        size_t jA = poolA.UpdateIdx(iA, sizeNeedSpill, startAddrA, bufsA);
        if (iA == jA) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect idx for poolA allocatedBufs.");
            return result;
        }

        size_t iB = 0;
        while (iB < bufsB.size()) {
            size_t startAddrB = poolB.ObtainStartAddr(iB, bufsB);
            if (startAddrB != startAddrA) {
                // startAddrB 在 iB 上单调递增: 没追上就推进, 已超过就停止本轮
                if (startAddrB < startAddrA) { iB++; continue; }
                break;
            }
            if ((poolB.GetMemSize() - startAddrB) < sizeNeedSpill) {
                break;
            }
            size_t jB = poolB.UpdateIdx(iB, sizeNeedSpill, startAddrB, bufsB);
            if (iB == jB) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect idx for poolB allocatedBufs.");
                return result;
            }
            std::vector<int> combined;
            combined.reserve((jA - iA) + (jB - iB));
            for (size_t k = iA; k < jA; k++) combined.push_back(std::get<0>(bufsA[k]));
            for (size_t k = iB; k < jB; k++) combined.push_back(std::get<0>(bufsB[k]));
            result.push_back(std::move(combined));
            iB++;
        }
        iA++;
    }
    return result;
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp)
{
    // dualdst / 单池两条路径只在"挑哪组"上分叉, "spill 执行"段 (GenBufferSpill 内的
    // SpillBuffer 循环 + RearrangeBuffer + HasEnoughBuffer 兜底) 完全共用。
    LocalBufferPtr allocBuffer = localBufferMap_[opReqMemIdsMap[allocOp][0]];
    std::vector<int> spillGroup;           // spill-all 兜底
    std::vector<std::vector<int>> canSpillGroups;

    if (IsDualDstAlloc(allocOp)) {
        DualDstAllocCtx ctx;
        if (ResolveDualDstAllocCtx(allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "DualDst spill select: ResolveDualDstAllocCtx failed.");
            return {};
        }
        auto& poolA = bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
        auto& poolB = bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];
        // 兜底"spill all": 两池 memId 全部, 不去重 (memId 跨 core 全局唯一)
        auto sortedA = poolA.GetAddrSortedBufs();
        auto sortedB = poolB.GetAddrSortedBufs();
        spillGroup.reserve(sortedA.size() + sortedB.size());
        spillGroup.insert(spillGroup.end(), sortedA.begin(), sortedA.end());
        spillGroup.insert(spillGroup.end(), sortedB.begin(), sortedB.end());
        canSpillGroups = GetDualSpillGroup(poolA, poolB, ctx.bufA->size);
    } else {
        auto coreType = opCoreLocationMap[allocOp];
        auto& pool = bufferManagerMap[coreType][allocBuffer->memType];
        spillGroup = pool.GetAddrSortedBufs();
        canSpillGroups = GetSpillGroup(pool, allocBuffer->size);
    }

    if (canSpillGroups.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return spillGroup;
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto &group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed, begin spill all tensor.");
            return spillGroup;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return spillGroup;
    }
    spillGroup = canSpillGroups[groupSel];
    return spillGroup;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache) {
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillOp, allocOp)) {
            // 存在非法memId时将该group排除
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = GetBufNextUseTime(memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

bool OoOScheduler::IsBelongSpillBlackList(Operation* spillOp, Operation* op) {
    std::set<Operation*> filterLtags;
    FindFilterLtags(op, filterLtags);
    if (opIsAllocMap[spillOp] || filterLtags.count(spillOp) != 0 || !CheckMachineAndL1(spillOp, op)) {
        return true;
    }
    return false;
}

void OoOScheduler::FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags) {
    auto dstOpList = depManager_.GetSuccessors(allocOp);
    for (auto dstOp : dstOpList) {
        if (COPY_IN_OPS.find(dstOp->GetOpcode()) == COPY_IN_OPS.end()) {
            for (auto &inOp : depManager_.GetPredecessors(dstOp)) {
                filterLtags.insert(inOp);
            }
            continue;
        }
        for (auto &dstOpId : depManager_.GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto &inOp : depManager_.GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
}

bool OoOScheduler::CheckMachineAndL1(Operation* spillOp, Operation* allocOp) {
    if (!spillOp->GetInputOperand(0) &&
        allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        APASS_LOG_WARN_F(Elements::Tensor, "CheckMachineAndL1: spillOp %s has no inputOperand.", GetOpInfo(spillOp).c_str());
        return false;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
        allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos &&
        spillOp->GetOpcodeStr().find("RESHAPE") == std::string::npos &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return false;
    }
    return true;
}

Status OoOScheduler::SpillBuffer(int memId, Operation* spillAllocOp, SpillContext &ctx) {
    Operation* spillOp = GetSpillOp(memId);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
        return FAILED;
    }
    if (opIsAllocMap[spillOp] || !CheckMachineAndL1(spillOp, spillAllocOp)) {
        return SUCCESS;
    }
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Find %s spill tensor[%d] failed.", GetOpInfo(spillOp).c_str(), memId);
        return FAILED;
    }
    SingleSpillCreatedOps created;
    if (HandleSpillMode(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Spill %s tensor[%d] failed.", GetOpInfo(spillOp).c_str(), memId);
        return FAILED;
    }
    // Emit after HandleSpillMode so created ops are populated.
    NotifySpill(spillTensor, memId, spillAllocOp, created);
    if (bufferManagerMap[opCoreLocationMap[spillAllocOp]][localBufferMap_[memId]->memType].Free(memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", memId);
        return FAILED;
    }
    tensorOccupyMap.erase(memId);
    return SUCCESS;
}

Status OoOScheduler::HandleSpillMode(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Begin spill %s, Tensor[%d]", GetOpInfo(spillOp).c_str(), memId);
    // 场景1: 若spill的op为copyin, 无需搬出, 重新搬入即可
    if (spillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        if (SpillBufferFromDDR(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillBufferFromDDR failed!");
            return FAILED;
        }
    } else if (localBufferMap_[memId]->memType == MemoryType::MEM_L1 &&
        Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        // 场景2: 3510平台L1发生spill
        if (SpillL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillL1BufferFor3510 failed!");
            return FAILED;
        }
    } else if (IsMultiProducerTensor(spillTensor)) {
        // 场景3: spill的tensor存在多消费者
        if (SpillMultiProducerBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiProducerBuffer failed!");
            return FAILED;
        }
    } else if (localBufferMap_[memId]->memType == MemoryType::MEM_L0C) {
        // 场景4: spill的tensor为L0C内存单元
        if (SpillL0CBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillL0CBuffer failed!");
            return FAILED;
        }
    } else {
        // // 场景5: 通用spill
        if (SpillGeneralBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillGeneralBuffer failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

// GMTensor --> spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillBufferFromDDR(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillBufferFromDDR begin.");
    LogicalTensorPtr gmTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CloneCopyinOp(spillOp, gmTensor, localTensor);

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {localTensor->memoryrange.memId}},
        {copyinOp, {localTensor->memoryrange.memId}}
    };

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, nullptr);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillBufferFromDDR end.");
    return SUCCESS;
}

// spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralBuffer begin.");
    if (GetActualSpillForNd2nz(spillOp, spillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpillForNd2nz failed.");
        return FAILED;
    }

    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor,
        OpImmediate::Specified(gmTensor->GetOffset()));
    Operation *allocOp = CreateAllocOp(localTensor);
    Operation *copyinOp = CreateCopyinOp(gmTensor, localTensor, OpImmediate::Specified(gmTensor->GetOffset()));

    if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemId, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {localTensor->memoryrange.memId}},
        {copyinOp, {localTensor->memoryrange.memId}}
    };

    if (UpdateScheduleStatus(opMemidMap, spillMemId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralBuffer end.");
    return SUCCESS;
}

Status OoOScheduler::SpillMultiProducerBufferFor3510(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBufferFor3510 begin.");
    Operation* actualTriggerOp = nullptr;
    LogicalTensorPtr actualTriggerTensor = nullptr;
    if (GetActualSpill(spillOp, actualTriggerOp, actualTriggerTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, actualTriggerTensor, spillMemid);
    LogicalTensorPtr l1Tensor = CreateLocalTensor(spillTensor);
    if (CopyoutParticalBuffer(spillTensor, gmTensor, ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CopyoutParticalBuffer failed.");
        return FAILED;
    }
    Operation *allocOp = CreateAllocOp(l1Tensor);
    Operation *copyinOp = CreateCopyinOp(gmTensor, l1Tensor, OpImmediate::Specified(gmTensor->GetOffset()));

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {l1Tensor->memoryrange.memId}},
        {copyinOp, {l1Tensor->memoryrange.memId}}
    };

    if (!HasUnexecutedProducer(spillTensor)) {
        if (UpdateScheduleStatus(opMemidMap, spillMemid, spillAllocOp, l1Tensor, spillOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
            return FAILED;
        }
    } else {
        if (UpdateNeedDeleteScheduleStatus(opMemidMap, spillMemid, spillAllocOp, spillTensor, spillOp, ctx)) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateNeedDeleteScheduleStatus failed.");
            return FAILED;
        }
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBufferFor3510 end.");
    return SUCCESS;
}

Status OoOScheduler::SpillL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    if (IsMultiProducerTensor(spillTensor)) {
        if (SpillMultiProducerBufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiProducerBufferFor3510 failed.");
            return FAILED;
        }
    } else if (spillOp->GetOpcode() != Opcode::OP_RESHAPE) {
        if (spillOp->GetIOperands().size() == 1) {
            SpillGeneralL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created);
        } else {
            return FAILED;
        }
    } else {
        Operation* actualSpillOp = nullptr;
        for (auto &preOp : depManager_.GetPredecessors(spillOp)) {
            if (!opIsAllocMap[preOp]) {
                actualSpillOp = preOp;
            }
        }
        if (actualSpillOp == nullptr || actualSpillOp->GetIOperands().size() != 1) {
            return FAILED;
        }
        if (actualSpillOp->GetOpcode() == Opcode::OP_COPY_IN) {
            SpillReshapeFromDDRFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx, created);
        } else {
            SpillReshapeL1BufferFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx, created);
        }
    }
    return SUCCESS;
}

// actualSpillTensor(L0C/UB) --> spillOp --> spillTensor(L1)
Status OoOScheduler::SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralL1BufferFor3510 begin.");
    Operation* actualOp = nullptr;
    LogicalTensorPtr actualSpillTensor = nullptr;
    if (GetActualSpill(spillOp, actualOp, actualSpillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    // workspaceOffset 在 gmtensor 创建时被更新，copyout 属性需要使用当前的 workspaceOffset
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, actualSpillTensor, memId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = 
        CreateCopyoutOp(spillOp, actualSpillTensor, gmTensor, OpImmediate::Specified(gmTensor->GetOffset()));
    Operation* allocOp = CreateAllocOp(localTensor);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(spillOp->GetOpAttribute());
 	if (attr == nullptr) {
 	    APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", GetOpInfo(spillOp).c_str());
 	    return FAILED;
 	}
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, attr->GetFromOffset());

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, actualOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {localTensor->memoryrange.memId}},
        {copyinOp, {localTensor->memoryrange.memId}}
    };

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralL1BufferFor3510 end.");
    return SUCCESS;
}

// actualSpillTensor(DDR) --> actualSpillOp(copyin) --> preSpillTensor --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeFromDDRFor3510(int memId, Operation* actualSpillOp, Operation* spillOp,
    LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeFromDDRFor3510 begin.");
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr ddrTensor = actualSpillOp->GetInputOperand(0);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr copyinTensor = 
        CreateParticalTensor(preSpillTensor, reshapeTensor, preSpillTensor, preSpillTensor->GetOffset());

    Operation* allocOp = CreateAllocOp(copyinTensor);
    Operation* copyinOp = CloneCopyinOp(actualSpillOp, ddrTensor, copyinTensor);
    Operation* reshapeOp = CreateReshapeOp(copyinTensor, reshapeTensor);

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {reshapeTensor->memoryrange.memId}},
        {copyinOp, {reshapeTensor->memoryrange.memId}},
        {reshapeOp, {reshapeTensor->memoryrange.memId, reshapeTensor->memoryrange.memId}}
    };

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, reshapeTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, nullptr);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeFromDDRFor3510 end.");
    return SUCCESS;
}

// actualSpillTensor(L0C) --> actualSpillOp --> preSpillTensor(L1) --> spillOp(reshape) --> spillTensor(L1)
// actualSpillTensor(UB) --> UB_COPY_ND2NZ(actual op) --> UB --> UB_COPY_L1(actualSpillOp) --> preSpillTensor(L1) --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeL1BufferFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp,
    LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeL1BufferFor3510 begin.");
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    Operation* actualOp = nullptr;
    LogicalTensorPtr actualSpillTensor = nullptr;
    if (GetActualSpill(actualSpillOp, actualOp, actualSpillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, actualSpillTensor, spillMemId);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr l1Tensor = CreateParticalTensor(preSpillTensor, reshapeTensor, preSpillTensor, preSpillTensor->GetOffset());

    Operation* copyoutOp = CreateCopyoutOp(actualSpillOp, actualOp->GetInputOperand(0), gmTensor, OpImmediate::Specified(gmTensor->GetOffset()));

    Operation* allocOp = CreateAllocOp(l1Tensor);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(actualSpillOp->GetOpAttribute());
 	if (attr == nullptr) {
 	    APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", GetOpInfo(actualSpillOp).c_str());
 	    return FAILED;
 	}
    // copyin shape属性来源于哪个Tensor
    Operation* copyinOp = CreateCopyinOp(gmTensor, l1Tensor, attr->GetFromOffset());
    Operation* reshapeOp = CreateReshapeOp(l1Tensor, reshapeTensor);

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, actualSpillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {l1Tensor->memoryrange.memId}},
        {copyinOp, {l1Tensor->memoryrange.memId}},
        {reshapeOp, {l1Tensor->memoryrange.memId, l1Tensor->memoryrange.memId}}
    };

    if (UpdateScheduleStatus(opMemidMap, spillMemId, spillAllocOp, reshapeTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeL1BufferFor3510 end.");
    return SUCCESS;
}

Status OoOScheduler::SpillL0CBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillL0CBuffer begin.");
    std::vector<Operation*> consumers;
    CollectL0CConsumers(spillTensor, consumers);
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemId);

    Operation *copyoutOp = 
        CreateCopyoutOp(spillOp, spillTensor, gmTensor, OpImmediate::Specified(gmTensor->GetOffset()));
    if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemId, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    for (auto *consumer : consumers) {
        auto consumerOOperand = consumer->GetOutputOperand(0);
        if (consumerOOperand == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: consumer %s has no output operand.",
                GetOpInfo(consumer).c_str());
            return FAILED;
        }
        Operation *copyinOp = 
            CreateCopyinOp(gmTensor, consumerOOperand, OpImmediate::Specified(gmTensor->GetOffset()), true);
        // 更新op的调度信息
        UpdateOpScheduleInfo(copyinOp, {consumerOOperand->memoryrange.memId}, spillAllocOp);
        // 生成的op插入排序队列
        std::replace(orderedOps.begin(), orderedOps.end(), consumer, copyinOp);
        APASS_LOG_DEBUG_F(Elements::Operation, "L0C spill: replace %s with %s.",
            GetOpInfo(consumer).c_str(), GetOpInfo(copyinOp).c_str());
        consumer->SetAsDeleted();
        EraseSchedulerSideMaps(consumer);
    }
    depManager_.InitDependencies(orderedOps, false);
    bufRefCount_[spillMemId] = 0;
    function_.EraseOperations(false, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    // L0C spill: consumers get their own copyin (no new alloc/copyin op recorded here).
    created.Record(copyoutOp, nullptr, nullptr, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillL0CBuffer end.");
    return SUCCESS;
}

// tensor(UB/L1)*n--> spillOp(Assemble/L0C_COPY_L1)*n --> spillTensor(UB/L1)
Status OoOScheduler::SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
    Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBuffer begin.");
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemid);
    LogicalTensorPtr assembleTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor, OpImmediate::Specified(gmTensor->GetOffset()));

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, spillTensor, spillMemid, spillOp) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, assembleTensor, spillMemid) != SUCCESS) {
        return FAILED;
    }

    for (auto &op : spillTensor->GetProducers()) {
        if (op->GetOpcode() != Opcode::OP_ASSEMBLE) continue;
        for (auto &producer : op->ProducerOps()) {
            if (opIsAllocMap[producer]) producer->UpdateOutputOperand(0, spillTensor);
        }
    }
    Operation* allocOp = CreateAllocOp(assembleTensor);
    UpdateOpScheduleInfo(allocOp, {assembleTensor->memoryrange.memId}, spillAllocOp);
    if (InsertOps({{allocOp, {assembleTensor->memoryrange.memId}}}, spillAllocOp, spillMemid) != SUCCESS) {
        return FAILED;
    }
    std::vector<Operation*> replaceOps;
    for (auto &op : spillTensor->GetProducers()) {
        if (opIsAllocMap[op]) {
            continue;
        }
        if (opIsRetiredMap[op]) {
            CreateParticalBuffer(spillMemid, op, assembleTensor, copyoutOp, spillAllocOp);
        } else {
            replaceOps.push_back(op);
        }
    }
    for (auto &op : replaceOps) {
        op->ReplaceOutput(assembleTensor, spillTensor);
    }

    if (UpdateRemainMemid(spillMemid, assembleTensor->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, nullptr, gmTensor);
    return SUCCESS;
}

Status OoOScheduler::CopyoutParticalBuffer(LogicalTensorPtr spillTensor, LogicalTensorPtr gmTensor, SpillContext &ctx)
{
    for (auto &op : spillTensor->GetProducers()) {
        if (opIsAllocMap[op]) {
            continue;
        }
        Operation* actualOp = nullptr;
        LogicalTensorPtr actualTensor = nullptr;
        if (GetActualSpill(op, actualOp, actualTensor) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
            return FAILED;
        }
        auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", GetOpInfo(op).c_str());
            return FAILED;
        }
        Operation *copyoutOp = CreateCopyoutOp(op, actualTensor, gmTensor, attr->GetToOffset());
        if (UpdateCopyoutScheduleInfo(
                copyoutOp, actualTensor, actualTensor->memoryrange.memId, actualOp, opIsRetiredMap[op]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
            return FAILED;
        }
        if (!opIsRetiredMap[op]) {
            ctx.newNotRetiredCopyOutSize++;
        } else {
            ctx.newCopyoutOps.push_back(copyoutOp);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleTensor,
    Operation* copyoutOp, Operation* spillAllocOp) 
{
    LogicalTensorPtr gmTensor = copyoutOp->GetOutputOperand(0);
    LogicalTensorPtr spillTensor = copyoutOp->GetInputOperand(0);

    std::vector<int64_t> toOffset;
    std::vector<SymbolicScalar> toDynOffset;
    std::vector<SymbolicScalar> fromDynValidShape;
    if (GetPartialWriteReplayAttr(producerOp, toOffset, toDynOffset, fromDynValidShape) != SUCCESS) {
 	    return FAILED;
 	}

    LogicalTensorPtr copyinTensor = CreateParticalTensor(gmTensor, assembleTensor, producerOp->GetInputOperand(0), toOffset);
    Operation* copyinOp = CreateCopyinOp(gmTensor, copyinTensor, OpImmediate::Specified(toOffset));
    Operation* assembleOp = CreateAssembleOp(copyinTensor, assembleTensor, toOffset, toDynOffset, fromDynValidShape);

    int64_t isNZ = 0;
 	producerOp->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
 	copyinOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    assembleOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    UpdateOpScheduleInfo(copyinOp, {assembleTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(assembleOp, {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId}, spillAllocOp);
    if (InsertOps({{copyinOp, {assembleTensor->memoryrange.memId}},
        {assembleOp, {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId}}},
        spillAllocOp, spillMemid) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    return SUCCESS;
}

LogicalTensorPtr OoOScheduler::CreateLocalTensor(LogicalTensorPtr spillTensor) {
    LogicalTensorPtr localTensor = irBuilder_.CreateTensorVar(
        spillTensor->Datatype(), spillTensor->GetShape(), std::vector<SymbolicScalar>{}, spillTensor->Format());
    localTensor->SetMemoryTypeToBe(spillTensor->GetMemoryTypeOriginal());
    localTensor->SetMemoryTypeOriginal(spillTensor->GetMemoryTypeOriginal());
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    int rawMagic = localTensor->GetRawTensor()->GetRawMagic();
    localTensor->memoryrange.memId = rawMagic;
    localBufferMap_[rawMagic] =
        std::make_shared<LocalBuffer>(rawMagic, localTensor->tensor->GetRawDataSize(), localTensor->GetMemoryTypeOriginal());
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create local tensor[%d].", localTensor->memoryrange.memId);
    return localTensor;
}

const std::vector<int64_t>& OoOScheduler::GetLargerShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2)
{
    for (size_t i = 0; i < shape1.size(); i++) {
        if (shape1[i] > shape2[i]) {
            return shape1;
        }
    }
    return shape2;
}

LogicalTensorPtr OoOScheduler::CreateGMTensor(LogicalTensorPtr spillTensor, LogicalTensorPtr actualSpillTensor, int spillMemId) {
    std::shared_ptr<RawTensor> gmRawTensor =
        std::make_shared<RawTensor>(spillTensor->Datatype(),
        GetLargerShape(spillTensor->tensor->rawshape, actualSpillTensor->tensor->rawshape),
        TileOpFormat::TILEOP_ND, "WorkspaceGm");
    LogicalTensorPtr gmTensor = irBuilder_.CreateTensorVar(
        gmRawTensor, spillTensor->GetOffset(), actualSpillTensor->GetShape(), std::vector<SymbolicScalar>{});
    gmTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceOffset);
    gmTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    gmTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    gmTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    gmTensor->tensor->rawshape = GetLargerShape(spillTensor->tensor->rawshape, actualSpillTensor->tensor->rawshape);
    if (localBufferMap_.find(spillMemId) == localBufferMap_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap_.", spillMemId);
        return nullptr;
    }
    gmTensor->memoryrange =
        TileRange(workspaceOffset, workspaceOffset + gmRawTensor->GetRawDataSize(), workspaceMemId++);
    workspaceOffset += gmRawTensor->GetRawDataSize();
    EmitInitDDRBuffer(gmTensor, DDRBufferKind::SPILL_TEMP);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create gm tensor[%d].", gmTensor->memoryrange.memId);
    return gmTensor;
}

LogicalTensorPtr OoOScheduler::CreateParticalTensor(
    LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand, LogicalTensorPtr spillTensor,
    std::vector<int64_t> toOffset)
{
    LogicalTensorPtr particalTensor = irBuilder_.CreateTensorVar(
        iOperand->Datatype(), iOperand->GetShape(), std::vector<SymbolicScalar>{}, iOperand->Format());
    particalTensor->SetMemoryTypeToBe(oriOperand->GetMemoryTypeToBe());
    particalTensor->SetMemoryTypeOriginal(oriOperand->GetMemoryTypeOriginal());
    particalTensor->tensor = oriOperand->tensor;
    particalTensor->memoryrange.memId = oriOperand->memoryrange.memId;
    particalTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    particalTensor->offset = toOffset;
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create partical tensor[%d].", particalTensor->memoryrange.memId);
    return particalTensor;
}

Operation* OoOScheduler::CreateAllocOp(LogicalTensorPtr oOperand) {
    Opcode opcode =
        oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    Operation& allocOp = irBuilder_.CreateTensorOpStmt(function_, opcode, {}, {oOperand});
    allocOp.UpdateLatency(1);
    tensorAllocMap[oOperand->memoryrange.memId] = &allocOp;
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", GetOpInfo(&allocOp).c_str());
    return &allocOp;
}

Operation* OoOScheduler::CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& copyinOp = spillOp->CloneOperation(function_, {iOperand}, {oOperand});
    copyinOp.SetIOpAtt(0, spillOp->GetIOpAttrOffset(0));
    copyinOp.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
    copyinOp.inParamLocation_ = spillOp->inParamLocation_;
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Clone %s", GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* OoOScheduler::CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
    std::vector<OpImmediate> offset, bool isND2NZ)
{
    Operation& copyinOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    copyinOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offset, // 搬运GM上的偏移
        oOperand->GetMemoryTypeOriginal(),
        OpImmediate::Specified(oOperand->GetShape()), // 搬运数据量
        OpImmediate::Specified(oOperand->tensor->GetDynRawShape()))); // 暂未使用
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    bool isCube = true;
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyinOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 || isND2NZ) {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        } else {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* OoOScheduler::CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
    std::vector<OpImmediate> offset) {
    Operation &copyoutOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_OUT, {iOperand}, {oOperand});
    copyoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        iOperand->GetMemoryTypeOriginal(),
        offset, OpImmediate::Specified(iOperand->GetShape()),
        OpImmediate::Specified(iOperand->GetRawTensor()->GetDynRawShape())));
    if (spillOp->HasAttribute(OpAttributeKey::scaleValue)) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        spillOp->GetAttr(OpAttributeKey::scaleValue, scaleValue);
        copyoutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    }
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyoutOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        copyoutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
    } else if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
        iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        copyoutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
    }
    copyoutOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", GetOpInfo(&copyoutOp).c_str());
    return &copyoutOp;
}

Operation* OoOScheduler::CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& reshapeOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_RESHAPE, {iOperand}, {oOperand});
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    reshapeOp.SetAttribute(OpAttributeKey::isCube, isCube);
    reshapeOp.UpdateLatency(0);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", GetOpInfo(&reshapeOp).c_str());
    return &reshapeOp;
}

Operation* OoOScheduler::CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
    std::vector<int64_t> toOffset, std::vector<SymbolicScalar> toDynOffset,
    std::vector<SymbolicScalar> fromDynValidShape)
{
    Operation& assembleOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    assembleOp.UpdateLatency(1);
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(iOperand->GetMemoryTypeOriginal(),
        toOffset, toDynOffset, fromDynValidShape));
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    assembleOp.SetAttribute(OpAttributeKey::isCube, isCube);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", GetOpInfo(&assembleOp).c_str());
    return &assembleOp;
}

LogicalTensorPtr OoOScheduler::GetSpillTensor(Operation* spillOp, int spillMemId) {
    int spillTensorIdx = GetOOperandIdx(spillOp, spillMemId);
    return spillOp->GetOutputOperand(spillTensorIdx);
}

Status OoOScheduler::GetActualSpillForNd2nz(Operation* &spillOp, LogicalTensorPtr &spillTensor) {
    if (spillOp->GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
        for (auto producer : spillOp->ProducerOps()) {
            if (opIsAllocMap[producer]) continue;
            spillTensor = spillOp->GetInputOperand(0);
            spillOp = producer;
            if (spillTensor == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Get %s spill tensor failed.", GetOpInfo(spillOp).c_str());
                return FAILED;
            }
        }
        if (spillOp->GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot spill %s.", GetOpInfo(spillOp).c_str());
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill UB_COPY_ND2NZ producer %s", GetOpInfo(spillOp).c_str());
    }
    return SUCCESS;
}

// 场景1 L0C(actualTensor)->op(actual op)->spillTensor
// 场景2 UB(actualTensor)->UB_COPY_ND2NZ(actual op)->UB->op(spill op)->spillTensor
Status OoOScheduler::GetActualSpill(Operation* op, Operation* &actualOp, LogicalTensorPtr &actualTensor) {
    auto iOperand = op->GetInputOperand(0);
 	if (iOperand == nullptr) {
 	    APASS_LOG_ERROR_F(Elements::Operation,
 	        "SmallShape spill: producer %s has null input.", GetOpInfo(op).c_str());
 	    return FAILED;
 	}
 	auto iOperandMemType = iOperand->GetMemoryTypeOriginal();
 	if (iOperandMemType != MemoryType::MEM_UB && iOperandMemType != MemoryType::MEM_L0C) {
 	    APASS_LOG_ERROR_F(Elements::Operation,
 	        "SmallShape spill: producer %s input memType is %s, expect UB/L0C.",
 	        GetOpInfo(op).c_str(), MemoryTypeToString(iOperandMemType).c_str());
 	    return FAILED;
 	}
 	actualOp = op;
 	actualTensor = iOperand;
 	if (iOperandMemType == MemoryType::MEM_UB) {
 	    // 强制两层：producer -> UB_COPY_ND2NZ -> UB1
 	    Operation* prevOp = nullptr;
 	    for (auto &preOp : depManager_.GetPredecessors(op)) {
 	        if (!opIsAllocMap[preOp]) {
 	            prevOp = preOp;
 	        }
 	    }
 	    if (prevOp == nullptr || prevOp->GetOpcode() != Opcode::OP_UB_COPY_ND2NZ) {
 	        APASS_LOG_ERROR_F(Elements::Operation,
 	            "SmallShape spill: UB-producer %s does not have UB_COPY_ND2NZ predecessor.",
 	            GetOpInfo(op).c_str());
 	        return FAILED;
 	    }
 	    actualOp = prevOp;
 	    actualTensor = prevOp->GetInputOperand(0);
 	}
 	if (actualTensor == nullptr) {
 	    APASS_LOG_ERROR_F(Elements::Operation,
 	        "SmallShape spill: actualTensor is null for producer %s.", GetOpInfo(op).c_str());
 	    return FAILED;
 	}
 	return SUCCESS;
}

void OoOScheduler::CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*> &consumers)
{
    // L0C CopyOut/DDR consumer 默认已 retired，spill 只重连未 retired 的 UB/L1 consumer。
    for (auto* consumer : spillTensor->GetConsumers()) {
        if (consumer == nullptr || opIsRetiredMap[consumer]) {
            continue;
        }
        auto output = consumer->GetOutputOperand(0);
        if (output == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s without output operand.",
                GetOpInfo(consumer).c_str());
            continue;
        }
        auto outMem = output->GetMemoryTypeOriginal();
        if (outMem == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (outMem != MemoryType::MEM_UB && outMem != MemoryType::MEM_L1) {
            APASS_LOG_WARN_F(Elements::Operation,
                "L0C spill: skip consumer %s with output memType %s.",
                GetOpInfo(consumer).c_str(), MemoryTypeToString(outMem).c_str());
            continue;
        }
        consumers.push_back(consumer);
    }
    std::sort(consumers.begin(), consumers.end(), [this](Operation* a, Operation* b) {
        return opExecOrderMap[a] < opExecOrderMap[b];
    });
}

void OoOScheduler::EraseSchedulerSideMaps(Operation* op)
{
    auto it = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (it != orderedOps.end()) {
        orderedOps.erase(it);  // 只删除第一个2
    }
    opExecOrderMap.erase(op);
    opPipeTypeMap.erase(op);
    opIsAllocMap.erase(op);
    opIsRetiredMap.erase(op);
    opCoreLocationMap.erase(op);
    opViewOpsMap.erase(op);
    opReqMemIdsMap.erase(op);
    inOutOperandsCache_.erase(op);
    depManager_.RemoveSuccessorOp(op);
    depManager_.RemovePredecessorOp(op);
}

Status OoOScheduler::UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId, 
    Operation* spillOp, bool isRetired) 
{
    opReqMemIdsMap[op] = {spillMemId};
    opIsRetiredMap[op] = isRetired;
    opIsAllocMap[op] = false;
    opPipeTypeMap[op] = RescheduleUtils::GetOpPipeType(op);
    depManager_.RegisterOp(op);
    Operation* allocOp = tensorAllocMap[spillTensor->memoryrange.memId];
    opCoreLocationMap[op] = opCoreLocationMap[allocOp];
    UpdateOpInternalSubgraphID(*op, allocOp);
    int bufNextUseTime = opExecOrderMap[spillOp];
    for (auto succOp : depManager_.GetSuccessors(spillOp)) {
        if (!opIsRetiredMap[succOp]) continue;
        if (succOp == op) continue;
        if (succOp->GetOpcodeStr().find("COPY_OUT") != std::string::npos) {
            bufNextUseTime = std::max(bufNextUseTime, opExecOrderMap[succOp]);
        }
    }
    opExecOrderMap[op] = bufNextUseTime + 1;
    InsertOrdered(op);
    return SUCCESS;
}

void OoOScheduler::UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp) {
    opPipeTypeMap[op] = RescheduleUtils::GetOpPipeType(op);
    opIsAllocMap[op] = op->GetOpcodeStr().find("ALLOC") != std::string::npos;
    opIsRetiredMap[op] = false;
    opReqMemIdsMap[op] = memIds;
    depManager_.RegisterOp(op);
    opCoreLocationMap[op] = opCoreLocationMap[AllocOp];
    UpdateOpInternalSubgraphID(*op, AllocOp);
    numTotalIssues++;
}

Status OoOScheduler::UpdateScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
    Operation* spillAllocOp, LogicalTensorPtr localTensor, Operation* spillOp) {
    // 更新op的调度信息
    for (auto &[op, memid] : opMemidMap) {
        UpdateOpScheduleInfo(op, memid, spillAllocOp);
    }

    // 生成的op插入排序队列
    if (InsertOps(opMemidMap, spillAllocOp, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    // 修改图上op与tensor链接关系
    if (UpdateSpillOpDepend(spillOp, localTensor, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateSpillOpDepend failed.");
        return FAILED;
    }
    // 更新tensor的剩余引用次数
    if (UpdateRemainMemid(memId, localTensor->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    // 更新依赖关系
    depManager_.InitDependencies(orderedOps, false);
    return SUCCESS;
}

Status OoOScheduler::UpdateNeedDeleteScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
    Operation* spillAllocOp, LogicalTensorPtr spillTensor, Operation* spillOp, SpillContext &ctx) {
    // 更新op的调度信息, 其中更新 numTotalIssues
    for (auto &[op, memid] : opMemidMap) {
        UpdateOpScheduleInfo(op, memid, spillAllocOp);
    }

    // 生成的op插入排序队列
    if (InsertOps(opMemidMap, spillAllocOp, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }

    if (UpdateSmallShapeDependAndBuf(opMemidMap, memId, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateSmallShapeDependAndBuf failed");
        return FAILED;
    }
    if (RemoveSmallShapeSpillResources(memId, spillTensor, ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RemoveSmallShapeSpillResources failed");
        return FAILED;
    }
    // 更新依赖关系
    depManager_.InitDependencies(orderedOps, false);
    return SUCCESS;
}

Status OoOScheduler::InsertOps(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
    Operation* spillAllocOp, int memId)
{
    int bufNextUseTime = GetBufNextUseTime(memId);
    if (bufNextUseTime == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use time failed.", memId);
        return FAILED;
    }
    bufNextUseTime = 
        bufNextUseTime <= opExecOrderMap[spillAllocOp] ? opExecOrderMap[spillAllocOp] + 1 : bufNextUseTime;
    for (auto &op : opMemidMap) {
        opExecOrderMap[op.first] = bufNextUseTime++;
        InsertOrdered(op.first);
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId)
{
    auto& successors = depManager_.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
                UpdateOperationInput(succOp, spillOp, newTensor);
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr newTensor) {
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        for (auto &inOp : targetOp->GetIOperands()[index]->GetProducers()) {
            if (IsViewOp(*inOp)) {
                Operation* op = SkipViewChain(inOp, true);
                UpdateTensorInputForView(*op, spillOp, newTensor);
            } else if (inOp == spillOp) {
                targetOp->UpdateInputOperand(index, newTensor);
            }
        }
    }
}

void OoOScheduler::UpdateTensorInputForView(Operation& op, Operation* spillOp, LogicalTensorPtr tensor) {
    bool hit = false;
    for (auto it : op.GetInputOperand(0)->GetProducers()) {
        if (it == spillOp) {
            hit = true;
            op.UpdateInputOperand(0, tensor);
            break;
        }
    }
    if (!hit) return;
    // 向后刷该View链路上的MemId
    for (Operation* p = &op; p != nullptr && IsViewOp(*p); ) {
        p->GetOutputOperand(0)->memoryrange.memId = tensor->memoryrange.memId;
        auto consumers = p->GetOutputOperand(0)->GetConsumers();
        if (consumers.empty()) break;
        p = *consumers.begin();
    }
}

void OoOScheduler::UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp) {
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

void OoOScheduler::ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId)
{
    std::vector<Operation*> viewConsumers;
    for (auto* consumer : startTensor->GetConsumers()) {
        if (IsViewOp(*consumer)) {
            viewConsumers.push_back(consumer);
        }
    }

    while (!viewConsumers.empty()) {
        Operation* viewOp = viewConsumers.back();
        viewConsumers.pop_back();
        auto viewOutTensor = viewOp->GetOutputOperand(0);
        if (viewOutTensor == nullptr) {
            continue;
        }
        if (viewOutTensor->memoryrange.memId == oldMemId) {
            viewOutTensor->memoryrange.memId = newMemId;
        }
        for (auto* consumer : viewOutTensor->GetConsumers()) {
            if (IsViewOp(*consumer)) {
                viewConsumers.push_back(consumer);
            }
        }
    }
}

void OoOScheduler::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId) {
    auto& reqMemIds = opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (memId == oldMemId || memId == newMemId) {
            bufRefCount_[newMemId]++;
        }
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto &outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
            ReplaceViewOpChainMemId(outTensor, oldMemId, newMemId);
        }
    }
}

Status OoOScheduler::UpdateRemainMemid(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = 0;
    bufRefCount_[oldMemId] = 0;
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        ReplaceTensorMemId(op, oldMemId, newMemId);
    }
    return SUCCESS;
}

void OoOScheduler::InsertOrdered(Operation* insertOp) {
    int execOrder = opExecOrderMap[insertOp];
    auto it = orderedOps.begin();
    for (; it != orderedOps.end(); it++) {
        if (opExecOrderMap[*it] >= execOrder) {
            break;
        }
    }
    auto insertPos = orderedOps.insert(it, insertOp);
    // 更新后续元素的execOrder
    for (auto adjustIt = insertPos + 1; adjustIt != orderedOps.end(); adjustIt++) {
        if (opExecOrderMap[*adjustIt] >= execOrder) {
            opExecOrderMap[*adjustIt]++;
        }
    }
}

int64_t OoOScheduler::CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType)
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t linearOffset = 0;
    int64_t stride = 1;
    // 从最低维到最高维计算
    for (size_t i = shape.size(); i > 0; --i) {
        linearOffset += offset[i - 1] * stride;
        if (i > 0) {
            stride *= shape[i - 1];
        }
    }
    return linearOffset * BytesOf(dataType);
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType) {
    return !bufferManagerMap[opCoreLocationMap[allocOp]][memType].IsFull(localBufferMap_[opReqMemIdsMap[allocOp][0]], false);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType) {
    std::vector<int> memIds = bufferManagerMap[opCoreLocationMap[allocOp]][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] lastest write op.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    std::vector<BufferAddrChange> changes;
    auto status =
        bufferManagerMap[opCoreLocationMap[allocOp]][memType].CompactBufferSlices(localBufferMap_, changes);
    if (status == SUCCESS) {
        NotifyBufferRearrange(allocOp, memType, std::move(changes));
    }
    return status;
}

Operation* OoOScheduler::GetSpillOp(int memId) {
    if (tensorOccupyMap.count(memId)) {
        return tensorOccupyMap[memId];
    }
    return nullptr;
}

int OoOScheduler::GetBufNextUseTime(int curMemId)
{
    for (size_t i = 0; i < orderedOps.size(); i++) {
        auto &op = orderedOps[i];
        if (opIsRetiredMap[op]) continue;
        auto &reqMemids = GetOpMemIds(op);
        if (std::find(reqMemids.begin(), reqMemids.end(), curMemId) != reqMemids.end()) {
            for (auto pre : depManager_.GetPredecessors(op)) {
                if (opIsRetiredMap[pre]) continue;
                if (opIsAllocMap[pre]) {
                    return opExecOrderMap[pre];
                }
            }
            return opExecOrderMap[op];
        }
    }
    return -1;
}

bool OoOScheduler::IsMultiProducerTensor(LogicalTensorPtr tensor)
{
    int producerCount = 0;
    for (auto &producer : tensor->GetProducers()) {
        if (producer->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            producerCount++;
        }
    }
    return producerCount > 1 ? true : false;
}

Status OoOScheduler::GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t> &toOffset,
 	     std::vector<SymbolicScalar> &toDynOffset, std::vector<SymbolicScalar> &fromDynValidShape) const
{
    if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto attr = std::static_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid AssembleOpAttribute.");
            return FAILED;
        }
        toOffset = attr->GetToOffset();
        toDynOffset = attr->GetToDynOffset();
        fromDynValidShape = attr->GetFromDynValidShape();
        return SUCCESS;
    } else if (producerOp->GetOpcode() == Opcode::OP_L0C_TO_L1 ||
        producerOp->GetOpcode() == Opcode::OP_L0C_COPY_UB) {
        auto attr = std::static_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid CopyOpAttribute.");
            return FAILED;
        }
        auto iOperand = producerOp->GetInputOperand(0);
        for (const auto &offsetImm : attr->GetToOffset()) {
            if (!offsetImm.IsSpecified() || !offsetImm.GetSpecifiedValue().ConcreteValid()) {
                APASS_LOG_ERROR_F(Elements::Operation, "L0C_TO_L1 replay only supports static concrete toOffset.");
                return FAILED;
            }
            toOffset.push_back(static_cast<int64_t>(offsetImm.GetSpecifiedValue()));
        }
        fromDynValidShape = iOperand->GetDynValidShape();
        if (fromDynValidShape.empty() && !attr->GetToDynValidShape().empty()) {
            fromDynValidShape = OpImmediate::ToSpecified(attr->GetToDynValidShape());
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "Unsupported producer opcode in SpillParticalBuffer.");
    return FAILED;
}

bool OoOScheduler::HasUnexecutedProducer(LogicalTensorPtr spillTensor)
{
    if (spillTensor == nullptr) {
        return false;
    }
    for (auto& producerOp : spillTensor->GetProducers()) {
        if (!opIsRetiredMap[producerOp]) {
            return true;
        }
    }
    return false;
}

// 辅助函数：更新 successor 的依赖关系
void OoOScheduler::UpdateSuccessorDependencies(
    Operation* succOp, Operation* spillOp, Operation* reloadCopyin, int spillMemId, int reloadMemId)
{
    auto& reqMemIds = GetOpMemIds(succOp);
    if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
        depManager_.InsertSuccessor(reloadCopyin, succOp);
        // 更新memId
        std::replace(reqMemIds.begin(), reqMemIds.end(), spillMemId, reloadMemId);
        for (auto& outTensor : succOp->GetOOperands()) {
            if (outTensor->memoryrange.memId == spillMemId) {
                outTensor->memoryrange.memId =  reloadMemId;
                ReplaceViewOpChainMemId(outTensor, spillMemId, reloadMemId);
            }
        }
        depManager_.RemovePredecessor(succOp, spillOp);
        depManager_.InsertPredecessor(succOp, reloadCopyin);
        UpdateOperationInput(succOp, spillOp, reloadCopyin->GetOutputOperand(0));
    }
}

// 辅助函数：更新 alloc 的依赖关系
void OoOScheduler::UpdatePredecessorAllocDependencies(Operation* succOp, Operation* reloadAlloc, int spillMemId)
{
    // spillTensor->reshape->tensor 情况 reshape 后 tensor 的 alloc 更新
    auto predecessors = depManager_.GetPredecessors(succOp);
    for (auto predOp : predecessors) {
        if (opIsAllocMap[predOp]) {
            auto& predReqMemIds = GetOpMemIds(predOp);
            if (std::find(predReqMemIds.begin(), predReqMemIds.end(), spillMemId) != predReqMemIds.end()) {
                depManager_.RemovePredecessor(succOp, predOp);
                depManager_.InsertPredecessor(succOp, reloadAlloc);
            }
        }
    }
}

Status OoOScheduler::UpdateSmallShapeDependAndBuf(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
    int spillMemId, Operation* spillOp)
{
    if (opMemidMap.size() != TWO_ISSUE) {
        APASS_LOG_ERROR_F(Elements::Tensor, "The number of elements in opMemidMap is invalid: %zu.", opMemidMap.size());
        return FAILED;
    }
    Operation* reloadAlloc = opMemidMap[0].first;
    Operation* reloadCopyin = opMemidMap[1].first;
    // 更新依赖
    if (bufRefCount_.find(spillMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", spillMemId);
        return FAILED;
    }
    int reloadMemId = GetOpMemIds(reloadAlloc)[0];
    bufRefCount_[reloadMemId] = TWO_ISSUE;

    auto& successors = depManager_.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (opIsRetiredMap[succOp]) {
            continue;
        }
        bufRefCount_[reloadMemId]++;
        UpdateSuccessorDependencies(succOp, spillOp, reloadCopyin, spillMemId, reloadMemId);
        UpdatePredecessorAllocDependencies(succOp, reloadAlloc, spillMemId);
    }
    return SUCCESS;
}

// 辅助函数：收集 UB 场景的 ops 和 tensors
void OoOScheduler::CollectUBSceneOpsAndTensors(
    Operation* producerOp, std::vector<Operation*>& opsToDelete, std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    opsToDelete.push_back(producerOp);
    auto ubTensor2 = producerOp->GetInputOperand(0);
    if (ubTensor2 == nullptr) return;
    for (auto* op : ubTensor2->GetProducers()) {
        if (op != nullptr && op->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos) {
            if (depManager_.GetSuccessors(op).size() > 1) {
               return; 
            }
        }
    }
    tensorsToDelete.push_back(ubTensor2);
    // 找 UB_COPY_ND2NZ
    for (auto* op : ubTensor2->GetProducers()) {
        if (op != nullptr && (opIsAllocMap[op] ||
            op->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos)) {
            opsToDelete.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "UB scene: collect %s[%d]",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

// 收集需要删除的 ops 和 tensors（两种固定场景）
void OoOScheduler::CollectProducerChainForDeletion(
    LogicalTensorPtr spillTensor, std::vector<Operation*>& opsToDelete, std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    tensorsToDelete.push_back(spillTensor); // spill_tensor 总是要删除

    // 获取 spill_tensor 的直接 producer
    for (auto* producerOp : spillTensor->GetProducers()) {
        if (producerOp == nullptr) {
            continue;
        }
        bool isUBCopyL1 = producerOp->GetOpcode() == Opcode::OP_UB_COPY_L1;
        if (isUBCopyL1) {
            // UB场景：删除 UB_COPY_L1 + UB_ALLOC + UB tensor2 + UB_COPY_ND2NZ
            CollectUBSceneOpsAndTensors(producerOp, opsToDelete, tensorsToDelete);
            APASS_LOG_DEBUG_F(Elements::Operation, "UB scene: collect UB tensor and op");
        } else {
            // 删除Alloc 以及 L0C场景：只删除 L0C_L1
            opsToDelete.push_back(producerOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "collect L0C_COPY_L1/L1_ALLOC only");
        }
    }
}

// 清理收集到的所有 op
size_t OoOScheduler::CleanupCollectedOperations(const std::vector<Operation*>& opsToDelete)
{
    size_t deleteNum = 0;
    for (auto* op : opsToDelete) {
        if (op == nullptr) {
            continue;
        }
        // 查找 op 在 orderedOps 中的位置
        auto it = std::find(orderedOps.begin(), orderedOps.end(), op);
        if (it != orderedOps.end()) {
            size_t opIndex = std::distance(orderedOps.begin(), it);
            if (opIsRetiredMap[op]) {
                deleteNum++;
            }
            // 获取删除 op 的 order
            int deletedOrder = opExecOrderMap[op];

            // 从 orderedOps 删除
            auto nextIt = orderedOps.erase(it);

            // 调整后续 ops 的 opExecOrderMap（--），补空缺
            for (auto adjustIt = nextIt; adjustIt != orderedOps.end(); adjustIt++) {
                if (opExecOrderMap.count(*adjustIt) > 0 && opExecOrderMap[*adjustIt] > deletedOrder) {
                    opExecOrderMap[*adjustIt]--;
                }
            }

            APASS_LOG_DEBUG_F(
                Elements::Operation, "Deleted op %s at index %zu (order %d).", GetOpInfo(op).c_str(), opIndex,
                deletedOrder);
        }
        EraseSchedulerSideMaps(op);

        auto predecessors = depManager_.GetPredecessors(op);
        auto successors = depManager_.GetSuccessors(op);
        for (auto* pred : predecessors) {
            depManager_.RemoveSuccessor(pred, op);
        }
        for (auto* succ : successors) {
            depManager_.RemovePredecessor(succ, op);
        }
        auto newOpsIt = std::find(newOperations_.begin(), newOperations_.end(), op);
        if (newOpsIt != newOperations_.end()) {
            newOperations_.erase(newOpsIt);
            APASS_LOG_DEBUG_F(Elements::Operation, "Removed op %s from newOperations_.", GetOpInfo(op).c_str());
        }
        op->SetAsDeleted();
        APASS_LOG_DEBUG_F(Elements::Operation, "Marked op %s as deleted.", GetOpInfo(op).c_str());
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Delete pcidx num: %zu", deleteNum);
    return deleteNum;
}

// 清理收集到的所有 tensor 的相关信息
void OoOScheduler::CleanupCollectedTensors(
    const std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    for (size_t i = 0; i < tensorsToDelete.size(); i++) {
        auto& tensor = tensorsToDelete[i];
        int memId = tensor->memoryrange.memId;

        tensorAllocMap.erase(memId);
        bufRefCount_.erase(memId);

        APASS_LOG_DEBUG_F(
            Elements::Tensor, "Cleaned tensor[%d] scheduler.", memId);
    }
}

// 手动清理孤立的 tensor
void OoOScheduler::EraseOrphanedTensors(
    const std::vector<LogicalTensorPtr>& tensorsToDelete, const std::vector<Operation*>& opsToDelete)
{
    for (auto& tensor : tensorsToDelete) {
        // 清理 tensor 的 producers 关系（使用 RemoveProducer 接口）
        for (auto* op : opsToDelete) {
            tensor->RemoveProducer(op);
        }

        // 清理 tensor 的 consumers 关系（使用 RemoveConsumer 接口）
        for (auto* op : opsToDelete) {
            tensor->RemoveConsumer(op);
        }
    }
}

// 主函数：删除 spill_tensor 及其 producer chain
Status OoOScheduler::RemoveSmallShapeSpillResources(int spillMemId, LogicalTensorPtr spillTensor, SpillContext &ctx)
{
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "spillTensor is null.");
        return FAILED;
    }

    // 1. 收集需要删除的 ops 和 tensors（根据场景类型）
    std::vector<Operation*> opsToDelete;
    std::vector<LogicalTensorPtr> tensorsToDelete;
    CollectProducerChainForDeletion(spillTensor, opsToDelete, tensorsToDelete);
    APASS_LOG_DEBUG_F(
        Elements::Operation, "Collected %zu ops and %zu tensors.", opsToDelete.size(), tensorsToDelete.size());
    for (auto deleteOp : opsToDelete) {
        if (opIsAllocMap[deleteOp]) {
            ctx.deleteAllocOps.push_back({deleteOp, 
                deleteOp->GetOutputOperand(0)->GetMemoryTypeOriginal(), opCoreLocationMap[deleteOp]});
        }
    }
    // 2. 清理 ops, 并记录其中已执行 op 的数量
    auto deleteNum = CleanupCollectedOperations(opsToDelete);
    if (deleteNum > opsToDelete.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Delete number greater than totalDeleteNumber");
        return FAILED;
    }
    // 3. 清理 tensors
    CleanupCollectedTensors(tensorsToDelete);

    // 4. 调用 Function 统一清理机制
    function_.EraseOperations(false, true);

    // 5. 手动清理孤立 tensor
    EraseOrphanedTensors(tensorsToDelete, opsToDelete);

    // 6. 更新统计信息
    ctx.deleteRetiredOpSize = deleteNum;
    ctx.deleteNotRetiredOpSize = static_cast<int>(opsToDelete.size() - deleteNum);
    APASS_LOG_DEBUG_F(
        Elements::Operation, "Deleted spill tensor[%d] and %zu ops (%zu tensors).", spillMemId, opsToDelete.size(),
        tensorsToDelete.size());

    return SUCCESS;
}
} // namespace npu::tile_fwk

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
 * \file insert_sync.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync.h"
#include <thread>
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertSync"

namespace npu {
namespace tile_fwk {
Status RangeSearchTree::ProcessTreeNode(
    const Interval &interval, IntervalTreeNode *currPtr, std::vector<IntervalTreeNode *> &intervalStack) {
    int start = currPtr->interval.start;
    if (interval.start < start) {
        if (currPtr->left != nullptr) {
            intervalStack.push_back(currPtr->left);
            return SUCCESS;
        }
        currPtr->left = new IntervalTreeNode(interval);
        if (currPtr->left == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "New created left tree node is nullptr, ProcessTreeNode failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (currPtr->right != nullptr) {
        intervalStack.push_back(currPtr->right);
        return SUCCESS;
    }
    currPtr->right = new IntervalTreeNode(interval);
    if (currPtr->right == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "New created right tree node is nullptr, ProcessTreeNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status RangeSearchTree::InsertInterval(const Interval &interval) {
    std::vector<IntervalTreeNode*> intervalStack;
    if (treeRoot == nullptr) {
        treeRoot = new IntervalTreeNode(interval);
        if (treeRoot == nullptr) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "TreeRoot is nullptr, InsertInterval failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (ProcessTreeNode(interval, currPtr, intervalStack) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "InsertInterval failed at function ProcessTreeNode.");
            return FAILED; 
        }
        if (currPtr->max < interval.end) {
            currPtr->max = interval.end;
        }
    }
    return SUCCESS;
}

void RangeSearchTree::OverlapSearch(const Interval &interval, std::set<int> &result) {
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);

    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        if (interval.start <= currPtr->interval.end && interval.end >= currPtr->interval.start) {
            result.insert(currPtr->interval.idx);
        }
        if (currPtr->left != nullptr && currPtr->left->max >= interval.start) {
            intervalStack.push_back(currPtr->left);
        }
        intervalStack.push_back(currPtr->right);
    }
}

void RangeSearchTree::FreeTree() {
    std::vector<IntervalTreeNode *> intervalStack;
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        intervalStack.push_back(currPtr->left);
        intervalStack.push_back(currPtr->right);
        delete currPtr;
    }
}

void RangeSearchTree::Insert(int left, int right, int idx) {
    Interval interval(left, right, idx);
    InsertInterval(interval);
}

std::set<int> RangeSearchTree::GetCovered(int left, int right) {
    Interval givenInterval(left, right, 0);
    std::set<int> overlappingIdx;
    OverlapSearch(givenInterval, overlappingIdx);
    return overlappingIdx;
}

void DataDependencySearcher::CheckWAWSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType currMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(currMemoryType) > 0) {
            TileRange rg = currMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = wawSearchTree_[currMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckRAWSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
        auto tensor = opWait->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1 && writeDdrMemMap.count(readDDRmemId) > 0) {
            std::set<int> found = writeDdrMemMap[readDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (rawSearchTree_.count(readMemoryType) > 0) {
            TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = rawSearchTree_[readMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckWARSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1 && readDdrMemMap.count(writeDDRmemId) > 0) {
            std::set<int> found = readDdrMemMap[writeDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (warSearchTree_.count(writeMemoryType) > 0) {
            TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = warSearchTree_[writeMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

std::set<int> DataDependencySearcher::Find(Operation *opWait) {
    std::set<int> res;

    std::string opStr = opWait->GetOpcodeStr();
    // check WAW
    CheckWAWSearchTree(opWait, res);
    // check RAW
    CheckRAWSearchTree(opWait, res);
    // check WAR
    CheckWARSearchTree(opWait, res);
    return res;
}

void DataDependencySearcher::InsertWAWSearchTree(const Operation *opSet, int idx) {
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType prevMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(prevMemoryType) == 0) {
            wawSearchTree_[prevMemoryType] = RangeSearchTree();
        }
        TileRange rg = prevMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        wawSearchTree_[prevMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertRAWSearchTree(const Operation *opSet, int idx) {
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1) {
            if (writeDdrMemMap.count(writeDDRmemId) == 0) {
                writeDdrMemMap[writeDDRmemId] = std::set<int>{};
            }
            writeDdrMemMap[writeDDRmemId].insert(idx);
        }
        if (rawSearchTree_.count(writeMemoryType) == 0) {
            rawSearchTree_[writeMemoryType] = RangeSearchTree();
        }
        TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        rawSearchTree_[writeMemoryType].Insert(rg.start, rg.end,idx);
    }
}

void DataDependencySearcher::InsertWARSearchTree(const Operation *opSet, int idx) {
    for (size_t inIdx = 0; inIdx < opSet->GetIOperands().size(); inIdx++) {
        auto tensor = opSet->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1) {
            if (readDdrMemMap.count(readDDRmemId) == 0) {
                readDdrMemMap[readDDRmemId] = std::set<int>{};
            }
            readDdrMemMap[readDDRmemId].insert(idx);
        }
        if (warSearchTree_.count(readMemoryType) == 0) {
            warSearchTree_[readMemoryType] = RangeSearchTree();
        }
        TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        warSearchTree_[readMemoryType].Insert(rg.start, rg.end,idx);
    }
}

void DataDependencySearcher::Insert(const Operation *opSet, int idx) {
    InsertWAWSearchTree(opSet, idx);
    InsertRAWSearchTree(opSet, idx);
    InsertWARSearchTree(opSet, idx);
}

void PipeSync::BuildTensorRangeMap(Operation *op) {
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
    if (opcfg.coreType_ != CoreType::AIV) {
        return;
    }
    bool isAIV1 = op->GetAIVCore() == AIVCore::AIV1;
    auto inTensors = op->GetIOperands();
    auto outTensors = op->GetOOperands();
    LogicalTensors inOutTensors;
    inOutTensors.reserve(inTensors.size() + outTensors.size());
    inOutTensors.insert(inOutTensors.end(), inTensors.begin(), inTensors.end());
    inOutTensors.insert(inOutTensors.end(), outTensors.begin(), outTensors.end());
    for (const auto &tensor : inOutTensors) {
        if (ubTensorRangeMap.count(tensor->GetMagic()) || tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            continue;
        }
        TileRange tensorRange;
        if (!isAIV1) {
            tensorRange.start = tensor->memoryrange.start;
            tensorRange.end = tensor->memoryrange.end;
        } else {
            // 将AIV1中的tensor地址映射到的区间
            size_t curUBSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
            tensorRange.start = tensor->memoryrange.start + curUBSize;
            tensorRange.end = tensor->memoryrange.end + curUBSize;
        }
        ubTensorRangeMap.emplace(std::make_pair(tensor->GetMagic(), tensorRange));
    }
}

Status PipeSync::InsertSync(Function &function, std::vector<Operation *> &syncedOpLog) {
    std::vector<IndexOp> synced;
    std::vector<Operation *> opLogPtr(function.Operations(false).DuplicatedOpList());
    oriOpList_ = opLogPtr;
    uint64_t idxInput = 0;
    for (const auto &op : opLogPtr) {
        BuildTensorRangeMap(op);
        APASS_LOG_DEBUG_F(Elements::Operation, "Input operation %d %d: %s.", idxInput, op->GetOpMagic(), op->GetOpcodeStr().c_str());
        idxInput++;
    }
    if (PipeDispatch(opLogPtr, synced) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function PipeDispatch.");
        return FAILED; 
    }

    if (IssueOp(function, opLogPtr, synced) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function IssueOp.");
        return FAILED; 
    }

    std::sort(synced.begin(), synced.end(), [](const IndexOp &a, const IndexOp &b) {
        return a.first < b.first;
    });

    for (auto &log : synced) {
        syncedOpLog.push_back(&log.second.get());
    }
    return SUCCESS;
}

std::string PipeSync::DepOp::DumpDepOp(std::vector<Operation *> opLog) {
    std::stringstream ss;
    ss << "idx: " << idx << " opmagic: " << opLog[idx]->GetOpMagic() << ", ";
    if (!opLog.empty()) {
        ss << opLog[idx]->GetOpcodeStr() << ", ";
    }
    ss << "setPipe: {";
    for (auto i : setPipe) {
        if (opLog.empty()) {
            ss << opLog[i]->GetOpMagic() << ", ";
            continue;
        }
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}, waitPipe: {";
    for (auto i : waitPipe) {
        if (opLog.empty()) {
            ss << opLog[i]->GetOpMagic() << ", ";
            continue;
        }
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::IssueQueue::DumpIssueQueue(std::vector<Operation *> opLogPtr) {
    std::stringstream ss;
    ss << "pipe type: " << GetPipeTypeDict().Find(selfPipeCore.pipe) << ", Op in this pipe: {";
    for (auto op : ops) {
        ss << opLogPtr[op]->GetOpMagic() << " " << opLogPtr[op]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::PipeDepInfo::DumpPipeDepInfo() {
    std::stringstream ss;
    ss << "    wait idx: " << waitIdx << "\n";
    ss << "    setPipes:" << "\n";
    for (auto pair : setPipes) {
        ss << "        pipetype: " << GetPipeTypeDict().Find(pair.first.pipe) << "  opidx: " << pair.second << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpLatestPipeDepMap() {
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current pipe type: " << GetPipeTypeDict().Find(pair.first.pipe) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const{
    switch (seq) {
        case PipeSeq::AIC_MTE2: return "AIC_MTE2";
        case PipeSeq::AIC_MTE1: return "AIC_MTE1";
        case PipeSeq::AIC_M: return "AIC_M";
        case PipeSeq::AIC_FIX: return "AIC_FIX";
        case PipeSeq::AIV_MTE2: return "AIV_MTE2";
        case PipeSeq::AIV_V: return "AIV_V";
        case PipeSeq::AIV_MTE3: return "AIV_MTE3";
        case PipeSeq::AIC_MTE3: return "AIC_MTE3";
        case PipeSeq::AIV_S: return "AIV_S";
        case PipeSeq::AIC_S: return "AIC_S";
        case PipeSeq::PIPE_END: return "PIPE_END";
        default: return "ILLEGAL";
    }
}

std::map<PipeSync::PipeCoreReal, PipeSeq, PipeSync::PipeCoreRealCompare> PipeSync::pipe2Seq = {
    {{PIPE_MTE2, CoreType::AIC}, PipeSeq::AIC_MTE2},
    {{PIPE_MTE1, CoreType::AIC}, PipeSeq::AIC_MTE1},
    {   {PIPE_M, CoreType::AIC},    PipeSeq::AIC_M},
    { {PIPE_FIX, CoreType::AIC},  PipeSeq::AIC_FIX},
    {{PIPE_MTE2, CoreType::AIV}, PipeSeq::AIV_MTE2},
    {   {PIPE_V, CoreType::AIV},    PipeSeq::AIV_V},
    {{PIPE_MTE3, CoreType::AIV}, PipeSeq::AIV_MTE3},
    {{PIPE_MTE3, CoreType::AIC}, PipeSeq::AIC_MTE3},
    {   {PIPE_S, CoreType::AIV},    PipeSeq::AIV_S},
    {   {PIPE_S, CoreType::AIC},    PipeSeq::AIC_S},
};

std::map<PipeSeq, PipeSync::PipeCoreReal> PipeSync::seq2pipe = {
    {PipeSeq::AIC_MTE2, {PIPE_MTE2, CoreType::AIC}},
    {PipeSeq::AIC_MTE1, {PIPE_MTE1, CoreType::AIC}},
    {   PipeSeq::AIC_M,    {PIPE_M, CoreType::AIC}},
    { PipeSeq::AIC_FIX,  {PIPE_FIX, CoreType::AIC}},
    {PipeSeq::AIV_MTE2, {PIPE_MTE2, CoreType::AIV}},
    {   PipeSeq::AIV_V,    {PIPE_V, CoreType::AIV}},
    {PipeSeq::AIV_MTE3, {PIPE_MTE3, CoreType::AIV}},
    {PipeSeq::AIC_MTE3, {PIPE_MTE3, CoreType::AIC}},
    {   PipeSeq::AIV_S,    {PIPE_S, CoreType::AIV}},
    {   PipeSeq::AIC_S,    {PIPE_S, CoreType::AIC}},
};

PipeSeq PipeSync::GetPipeSeq(PipeSync::PipeCoreReal pipe) {
    return pipe2Seq.at(pipe);
}

PipeSync::PipeCoreReal PipeSync::GetPipeFromSeq(PipeSeq seq) {
    return seq2pipe.at(seq);
}

Status PipeSync::AdjustReshapeCfg(TileOpCfg &opcfg, Operation &op) {
    if (op.GetIOperands().size() < 1 || op.GetOOperands().size() < 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d RESHAPE op operands size is 0, AdjustOpCfg failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        op.GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyInCfg(TileOpCfg &opcfg, Operation &op) {
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYIN op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto dstMemType = attr->GetCopyInAttr().second;
    if (dstMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIC;
        return SUCCESS;
    } 
    if (dstMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIV;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyOutCfg(TileOpCfg &opcfg, Operation &op) {
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYOUT op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto srcMemType = attr->GetCopyOutAttr().first;
    if (srcMemType == MemoryType::MEM_L0C) {
        opcfg.pipeIdStart_ = PipeType::PIPE_FIX;
        opcfg.pipeIdEnd_ = PipeType::PIPE_FIX;
        opcfg.coreType_ = CoreType::AIC;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIV;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIC;
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpCfg(TileOpCfg &opcfg, Operation &op) {
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        if (AdjustReshapeCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustReshapeCfg failed.");
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (AdjustCopyInCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyInCfg failed.");
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        if (AdjustCopyOutCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyOutCfg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::PipeDispatch(const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
    DataDependencySearcher dataDependencySearcher;
    dataDependencySearcher.ubTensorRangeMap = ubTensorRangeMap;
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        if (opLogPtr[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) { 
            APASS_LOG_ERROR_F(Elements::Operation, "%d ALLOC op should not appear in InsertSync, PipeDispatch failed.%s", opLogPtr[i]->GetOpMagic(), GetFormatBacktrace(opLogPtr[i]).c_str());
            return FAILED;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        if (AdjustOpCfg(opcfg, *opLogPtr[i]) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Operation, "PipeDispatch failed at function AdjustOpCfg.");
            return FAILED; 
        }
        DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_});
        DepOp &opRef = depOps_.emplace_back(op);
        FindDep(opRef, opLogPtr, i, dataDependencySearcher);
        EnqueueOp(opRef, opLogPtr, syncedOpLog);
    }
    return SUCCESS;
}

void PipeSync::InitIssueQueue() {
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp &op, const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
   if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_NOP || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_HUB) {
        syncedOpLog.emplace_back(std::make_pair(op.idx * SEQUENCE_IDX, std::ref(*opLogPtr[op.idx])));
        return;
    }
    PipeCoreReal opPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
    auto &issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCore))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreReal opPipeCoreEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
        PipePair pp{opPipeCore, opPipeCoreEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
    orderedOpList_.emplace(op.idx);
}

void PipeSync::RemoveOpDep(DepOp &setOp, DepOp &waitOp) const {
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;
    std::vector<size_t> newSetDep;
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            continue;
        }
        newSetDep.emplace_back(ele);
    }
    setOp.setPipe = newSetDep;

    std::vector<size_t> newWaitDep;
    for (auto ele : waitOp.waitPipe) {
        if (ele == setOpIdx) {
            continue;
        }
        newWaitDep.emplace_back(ele);
    }
    waitOp.waitPipe = newWaitDep;
}

Status PipeSync::AddOpDep(DepOp &setOp, DepOp &waitOp) {
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    size_t depWaitIdx = static_cast<size_t>(-1);
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) { 
            APASS_LOG_ERROR_F(Elements::Operation, "This dependency should not exist, AddOpDep failed.");
            return FAILED; 
        }
        PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
        PipeCoreReal waitOpPipeCore(depOps_[waitOpIdx].selfPipeCore.pipeStart, depOps_[waitOpIdx].selfPipeCore.core);
        if (elePipeCore == waitOpPipeCore) {
            if (ele  <= waitOpIdx) { 
                APASS_LOG_ERROR_F(Elements::Operation, "New waitidx should less than old, AddOpDep failed.");
                return FAILED; 
            }
            depWaitIdx = ele;
            break;
        }
    }
    //同一种pipecore，不需要存记录依赖关系
    if (depWaitIdx != static_cast<size_t>(-1)) {
        RemoveOpDep(setOp, depOps_[depWaitIdx]);
    }
    setOp.setPipe.emplace_back(waitOpIdx);
    waitOp.waitPipe.emplace_back(setOpIdx);
    return SUCCESS;
}

Status PipeSync::AdjustOpDep(DepOp &op, size_t waitOpIdx, IssueQueue &issueQ, bool &failedFlag) {
    //op为靠前的， waitOp为靠后的
    auto &waitOp = depOps_[waitOpIdx];

    if (issueQ.currOp + 1 == issueQ.ops.size()) {
        failedFlag = true;
        return SUCCESS;
    }

    auto &nextOpIdx = issueQ.ops[issueQ.currOp + 1];
    auto &nextOp = depOps_[nextOpIdx];
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustOpDep failed at function AddOpDep.");
        return FAILED; 
    }
    return SUCCESS;
}

Status PipeSync::HandleEventID(DepOp &op, IssueQueue &issueQ, IssueNum &issuenum, bool &deadlock, bool &res) {
    bool eventIdOk = true;
    bool failedFlag{false};
    for (auto ele : op.setPipe) {
        if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
            continue;
        }
        PipeCoreReal currPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
        PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
        PipePair pp{currPipeCore, elePipeCore};
        std::pair<CoreTypeDetail, CoreTypeDetail> setWaitCoreType;
        issuenum.maxIssueNum.emplace(pp, GetFreeEventIdQueue(pp, op.idx, ele, setWaitCoreType).size());
        issuenum.currIssueNum.emplace(pp, 0);

        if (issuenum.currIssueNum[pp] >= issuenum.maxIssueNum[pp]) {
            if (!deadlock) {
                eventIdOk = false;
                break;
            }
            // eventID deadlock, adjust op dependency to release eventID.
            if (AdjustOpDep(op, ele, issueQ, failedFlag) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "HandleEventID failed at function AdjustOpDep.");
                return FAILED;
            }
            if (failedFlag) {
                break;
            }
        }
    }
    if (failedFlag) {
        eventIdOk = false;
        deadlock = true;
    } else {
        deadlock = false;
    }
    if (!eventIdOk) {
        res = false;
        return SUCCESS;
    }
    res = true;
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp &op) {
    // current op will be issued only when all of the waitop are issued
    for (const auto &waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(IssueQueue &issueQ, std::vector<size_t> &poped, bool &deadlock) {
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto &op = depOps_[issueQ.ops[issueQ.currOp]];
        if (op.idx != orderedOpList_.front()) {
            break;
        }
        if (op.issued) {
            APASS_LOG_ERROR_F(Elements::Operation, "Try to issue a op which is already issued, PopFromQueue failed.");
            return FAILED;
        }
        if (!CheckIssuedOp(op)) {
            break;
        }
        bool res = false;
        if (HandleEventID(op, issueQ, issuenum, deadlock, res) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PopFromQueue failed at function HandleEventID.");
            return FAILED;
        }
        if (!res) {
            break;
        }
        op.issued = true;
        poped.emplace_back(op.idx);
        orderedOpList_.pop();
        for (auto ele : op.setPipe) {
            PipeCoreReal currPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
            PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
            auto pp = PipePair{currPipeCore, elePipeCore};
            issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
        }
        issueQ.currOp++;
    }
    return SUCCESS;
}

Status PipeSync::InjectWaitFlag(Function &function, size_t idx, std::vector<IndexOp> &syncedOpLog) {
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    // serch the waitpipe of current op
    uint64_t waitIdx = idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    for (const auto &ele : depOps_[idx].waitPipe) {
        PipeCore setPipe = depOps_[ele].selfPipeCore;
        PipeCoreReal setPipeReal(setPipe.pipeEnd, setPipe.core);
        PipeCoreReal currPipeReal(currPipe.pipeStart, currPipe.core);
        int eventId = setWaitPairMap_[{ele, idx}];
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_DST, {input}, {output});
        bool res = GenSyncOp(setPipeReal, currPipeReal, eventId, false, syncOp);
        if (!res) {
            syncOp.SetAsDeleted();
            continue;
        }
        // insert wait_flag
        syncedOpLog.emplace_back(std::make_pair(++waitIdx, std::ref(syncOp)));
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d",
            syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(), GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_);
        std::pair<CoreTypeDetail, CoreTypeDetail> setWaitCoreType;
        GetFreeEventIdQueue({setPipeReal, currPipeReal}, ele, idx, setWaitCoreType).push_back(eventId);
        if (setPipeReal.core != currPipeReal.core) {
            crossCoreFreeEventId_[{setWaitCoreType.second, setWaitCoreType.first}].push_back(eventId);
        }
    }
    return SUCCESS;
}

Status PipeSync::InjectSetFlag(Function &function, size_t idx, std::vector<IndexOp> &syncedOpLog) {
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    uint64_t setIdx = idx * SEQUENCE_IDX;
    for (const auto &ele : depOps_[idx].setPipe) {
        PipeCore waitPipe = depOps_[ele].selfPipeCore;
        PipeCoreReal waitPipeReal(waitPipe.pipeStart, waitPipe.core);
        PipeCoreReal currPipeReal(currPipe.pipeEnd, currPipe.core);
        int eventId{0};
        if (GetEventId({currPipeReal, waitPipeReal}, idx, ele, eventId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InjectSetFlag failed at function GetEventId.");
            return FAILED;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_SRC, {input}, {output});
        bool res = GenSyncOp(currPipeReal, waitPipeReal, eventId, true, syncOp);
        if (res) {
            // insert set_flag
            syncedOpLog.emplace_back(std::make_pair(++setIdx, std::ref(syncOp)));
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d",
                syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(), GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_);
            setWaitPairMap_[{idx, ele}] = eventId;
            continue;
        }
        syncOp.SetAsDeleted();
        setWaitPairMap_[{idx, ele}] = eventId;
    }
    return SUCCESS;
}

Status PipeSync::InjectSync(Function &function, std::vector<Operation *> opLogPtr, size_t idx, std::vector<IndexOp> &syncedOpLog) {
    // check idx range
    if (idx > std::numeric_limits<uint64_t>::max() / SEQUENCE_IDX) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation index is out of range, InjectSync failed.");
        return FAILED;
    }

    // insert wait_flag
    InjectWaitFlag(function, idx, syncedOpLog);
    
    // insert current operation
    syncedOpLog.emplace_back(std::make_pair(idx * SEQUENCE_IDX, std::ref(*opLogPtr[idx])));
    depOps_[idx].issued = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s", opLogPtr[idx]->GetOpMagic(), opLogPtr[idx]->GetOpcodeStr().c_str());

    // insert set_flag
    if (InjectSetFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectSetFlag.");
        return FAILED;
    }
    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePair &pp) {
    PipePair ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_NUM;
    }
    return EVENT_ID7;
}

Status PipeSync::ProcessDeadLock(uint64_t &eventIdDeadlockEnterTimes, bool &eventIdDeadlock, std::vector<IndexOp> &syncedOpLog) {
    eventIdDeadlockEnterTimes++;
    // eventID deadlock
    if (eventIdDeadlockEnterTimes > DEADLOCK_TIME_THRESHOLD) {
        eventIdDeadlock = true;
    }
    if (RelaxFakeDataDep(syncedOpLog) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessDeadLock failed at function RelaxFakeDataDep.");
        return FAILED; 
    }
    if (eventIdDeadlockEnterTimes >= EVENTID_DEADLOCK_ENTER_TIME) { 
        APASS_LOG_ERROR_F(Elements::Operation, "Unbreakable deadlock detected, ProcessDeadLock failed.");
        return FAILED; 
    }
    return SUCCESS;
}

Status PipeSync::IssueOpPipeSeq(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, bool &eventIdDeadlock, size_t &issued) {
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function PopFromQueue.");
            return FAILED;
        }
        issued += issuedOps.size();
        for (auto idx : issuedOps) {
            if (InjectSync(function, opLogPtr, idx, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function InjectSync.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::IssueSyncOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, size_t &totalIssued, size_t &allIssued) {
    bool eventIdDeadlock = false;
    uint64_t eventIdDeadlockEnterTimes = 0;
    while (totalIssued < allIssued) {
        size_t issued = 0;
        if (IssueOpPipeSeq(function, opLogPtr, syncedOpLog, eventIdDeadlock, issued) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function IssueOpPipeSeq.");
            return FAILED;
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issued == 0) {
            if (ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function ProcessDeadLock.");
                return FAILED;
            }
            continue;
        }
        eventIdDeadlock = false;
        eventIdDeadlockEnterTimes = 0;
    }
    return SUCCESS;
}

Status PipeSync::IssueOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        APASS_LOG_DEBUG_F(Elements::Operation, "Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(), issueState_[i].DumpIssueQueue(opLogPtr).c_str());
    }
    if (IssueSyncOp(function, opLogPtr, syncedOpLog, totalIssued, allIssued) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed with IssueSyncOp.");
        return FAILED;
    }
    if (totalIssued != allIssued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Issue error, IssueOp failed.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALL op issued: %zu", totalIssued);
    return SUCCESS;
}

std::vector<PipeSync::PipePair> PipeSync::dataDepPair = {
    // PIPE_MTE1只有AIC PIPE_V只有AIV PIPE_M只有AIC PIPE_FIX只有AIC
    // PIPE_MTE3->PIPE_MTE2
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE3
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE2->PIPE_V
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE2
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE3->PIPE_V
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE3
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_S->PIPE_V
    {{PIPE_S, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_S
    {{PIPE_V, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_M
    {{PIPE_S, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_S
    {{PIPE_M, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE1
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_S
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE2
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE2->PIPE_S
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_MTE3
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_MTE3->PIPE_S
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_FIX
    {{PIPE_S, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_S
    {{PIPE_FIX, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_M->PIPE_MTE1
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_M
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE2
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_M
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE3
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_M
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_FIX
    {{PIPE_M, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_M
    {{PIPE_FIX, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE2
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE1
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE3
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_MTE1
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_FIX
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE1
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE2->PIPE_FIX
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE2
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE3->PIPE_FIX
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE3
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
};

bool PipeSync::ConstructDepInfo(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i) {
    auto &log = syncedOpLog[i].second;
    if (log.get().GetOpcodeStr() != "SYNC_SRC") {
        return false;
    }
    auto setPipe = log.get().syncQueue_.pipeId_;
    auto waitPipe = log.get().syncQueue_.trigPipeId_;
    auto setCore = log.get().syncQueue_.coreType_;
    auto waitCore = log.get().syncQueue_.trigCoreType_;
    auto eventId = log.get().syncQueue_.eventId_;
    if (!(setPipe == depInfo.setp && setCore == depInfo.setc && waitPipe == depInfo.waitp && waitCore == depInfo.waitc)) {
        return false;
    }
    if (std::find(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end(), eventId) !=
        depInfo.setOpEventIdList.end()) {
        return false;
    }
    depInfo.setOpIdList.push_back(i);
    depInfo.setOpEventIdList.push_back(eventId);
    return true;
}

int PipeSync::GetSyncSrcLogIdx(std::vector<IndexOp> &syncedOpLog, int i) {
    int j = i - 1;
    for (; j >= 0; j--) {
        if (syncedOpLog[j].second.get().GetOpcodeStr().find("SYNC") == std::string::npos &&
            syncedOpLog[j].second.get().GetOpcodeStr().find("BAR") == std::string::npos) {
            break;
        }
    }
    return syncedOpLog[j].first;
}

bool PipeSync::FindDataDep(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i) {
    if (!ConstructDepInfo(depInfo, syncedOpLog, i)) {
        return false;
    }
    int syncSrcLogIdx = GetSyncSrcLogIdx(syncedOpLog, i) / SEQUENCE_IDX;
    DepOp &depOpSrc = depOps_[syncSrcLogIdx];
    for (auto syncDstLogIdx : depOpSrc.setPipe) { //setpipe中的op为该op之后的，依赖于该op的op id
        DepOp &depOpDst = depOps_[syncDstLogIdx];
        if (depOpDst.selfPipeCore.core == depInfo.waitc && depOpDst.selfPipeCore.pipeStart == depInfo.waitp) {
            depInfo.opDepList.push_back(std::make_pair(syncSrcLogIdx, syncDstLogIdx));
        }
    }
    return true;
}

bool PipeSync::FindMaxOverlap(DataDepInfo &depInfo, int &maxOverlapDepIdx) {
    int maxOverlap = -1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        if (depInfo.opDepList[idx].second < depInfo.opDepList[idx + 1].first) { //相邻的两个依赖之间没有重叠
            continue;
        }
        //max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        //max_overlap用来记录遍历到的最大overlap
        if ((depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first) > maxOverlap) {
            maxOverlapDepIdx = idx;
            maxOverlap = depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first;
        }
    }
    if (maxOverlapDepIdx == -1) { // 8个依赖中，前后依赖都没有重叠，或者重叠全部小于阈值。
        return false;
    }
    return true;
}

Status PipeSync::SynDependency(int maxOverlapDepIdx, const DataDepInfo &depInfo, const PipePair &pipePair, std::vector<IndexOp> &syncedOpLog) {
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    if (set1 >= set2 || wait1 >= wait2) {
        APASS_LOG_ERROR_F(Elements::Operation, "Dependency error, RelaxFakeDataDep failed.");
        return FAILED;
    }
    RemoveOpDep(depOps_[set1], depOps_[wait1]);
    RemoveOpDep(depOps_[set2], depOps_[wait2]);
    if (AddOpDep(depOps_[set2], depOps_[wait1]) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RelaxFakeDataDep failed at function AddOpDep.");
        return FAILED;
    }
    std::pair<CoreTypeDetail, CoreTypeDetail> setWaitCoreType;
    // 此函数中coretype一定是相同的
    GetFreeEventIdQueue(pipePair, set1, wait1, setWaitCoreType).push_back(eventId1);
    setWaitPairMap_[{set2, wait1}] = eventId2;
    //将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second.get().SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp> &syncedOpLog, const PipePair &pipePair, DataDepInfo &depInfo) {
    depInfo.setp = pipePair.first.pipe;
    depInfo.setc = pipePair.first.core;
    depInfo.waitp = pipePair.second.pipe;
    depInfo.waitc = pipePair.second.core;
    // 8个eventid全部被占用
    // 说明该set pipe内肯定有8个set op已经发射，而wait pipe内对应的8个op一个都没发射。
    // 找到这8个setop的idx，对应的event id，以及对应的waitop idx
    auto eventNum = static_cast<std::vector<std::pair<int, int>>::size_type>(GetMaxEventId(pipePair));
    for (int i = syncedOpLog.size() - 1; i >= 0; i--) {
        if (!(FindDataDep(depInfo, syncedOpLog, i))) {
            continue;
        }
        if (depInfo.setOpIdList.size() == eventNum) {
            break;
        }
    }
    // 由于从后向前寻找，得到的结果列表进行反转。
    std::reverse(depInfo.opDepList.begin(), depInfo.opDepList.end());
    std::reverse(depInfo.setOpIdList.begin(), depInfo.setOpIdList.end());
    std::reverse(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end());
    if (depInfo.opDepList.size() != eventNum || depInfo.setOpIdList.size() != eventNum || depInfo.setOpEventIdList.size() != eventNum) {
        APASS_LOG_ERROR_F(Elements::Operation, "dep size should be %d, RelaxFakeDataDep failed.", eventNum);
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp> &syncedOpLog) {
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    for (const auto &pipePair : dataDepPair) {
        if (HasFreeEventId(pipePair)) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePair, depInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDepInfo failed.");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePair, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

bool PipeSync::GenSyncOp(PipeCoreReal set, PipeCoreReal wait, int eventId, bool isSet, Operation &op) {
    if (set.core != wait.core) {
        op.SetOpCode(isSet ? Opcode::OP_CV_SYNC_SRC : Opcode::OP_CV_SYNC_DST);
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && !isSet && wait.core == CoreType::AIV) {
            set.pipe = PipeType::PIPE_V;
        }
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId};
        return true;
    }
    if (set.pipe != wait.pipe) {
        op.SetOpCode(isSet ? Opcode::OP_SYNC_SRC : Opcode::OP_SYNC_DST);
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId};
        return true;
    }
    if (isSet || set.pipe == PipeType::PIPE_S) {
        return false;
    }
    //同步相关的信息放在operation属性里
    op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId};
    if (set.core == CoreType::AIV) {
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            return false;
        }
        op.SetOpCode(Opcode::OP_BAR_V);
        return true;
    }
    op.SetOpCode(Opcode::OP_BAR_M);
    return true;
}

Status PipeSync::GetEventId(const PipePair &pp, size_t setIdx, size_t waitIdx, int &eventId) {
    if (pp.first.pipe == pp.second.pipe && pp.first.core == pp.second.core) {
        // Pipe Barrier
        eventId = -1;
        return SUCCESS;
    }

    std::pair<CoreTypeDetail, CoreTypeDetail> setWaitCoreType;
    auto &eventQ = GetFreeEventIdQueue(pp, setIdx, waitIdx, setWaitCoreType);
    if (eventQ.empty()) { 
        APASS_LOG_ERROR_F(Elements::Operation, "Eventid exhausted, GetEventId failed.");
        return FAILED; 
    }

    eventId = eventQ.front();
    eventQ.pop_front();
    if (pp.first.core != pp.second.core) {
        auto &eventQReverse = crossCoreFreeEventId_[{setWaitCoreType.second, setWaitCoreType.first}];
        eventQReverse.pop_front();
        if (eventQ.size() != eventQReverse.size()) {
            APASS_LOG_ERROR_F(Elements::Operation, "CV eventId queue size is not equal, GetEventId failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

bool PipeSync::HasFreeEventId(const PipePair &pp) {
    std::pair<CoreTypeDetail, CoreTypeDetail> setWaitCoreType;
    std::deque<int> &eventQ = GetFreeEventIdQueue(pp, SIZE_MAX, SIZE_MAX, setWaitCoreType);
    return !eventQ.empty();
}

bool PipeSync::BufOverlap(const TileRange &range1, int magic1, const TileRange &range2, int magic2) const {
    APASS_LOG_DEBUG_F(Elements::Tensor, "Range 1 [%zu ~ %zu], range 2 [%zu ~ %zu].", range1.start, range1.end, range2.start, range2.end);
    if (range1.end > range2.start && range2.end > range1.start) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor %d and tensor %d have overlap.", magic1, magic2);
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor %d and tensor %d don't have overlap.", magic1, magic2);
    return false;
}

bool PipeSync::CheckWawDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) {
    for (size_t setIdx = 0; setIdx < opSet.GetOOperands().size(); setIdx++) {
        for (size_t waitIdx = 0; waitIdx < opWait.GetOOperands().size(); waitIdx++) {
            if (opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal() != opWait.GetOOperands()[waitIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetOOperands()[setIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[waitIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic1] : opSet.GetOOperands()[setIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic2] : opWait.GetOOperands()[waitIdx]->memoryrange;
            if (BufOverlap(range1, magic1, range2, magic2)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "%d %zu %s and %d %zu %s has WAW data dependency", opSet.GetOpMagic(), 
                    k, opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckRawDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) {
    for (size_t outIdx = 0; outIdx < opSet.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opWait.GetIOperands().size(); inIdx++) {
            if (opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal() != opSet.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opWait.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opSet.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic1] : opWait.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic2] : opSet.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, magic1, range2, magic2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                APASS_LOG_DEBUG_F(Elements::Operation, "%d %zu %s and %d %zu %s has RAW data dependency", opSet.GetOpMagic(), 
                    k, opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckWarDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) {
    for (size_t outIdx = 0; outIdx < opWait.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opSet.GetIOperands().size(); inIdx++) {
            if (opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal() != opWait.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic1] : opSet.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ?
                ubTensorRangeMap[magic2] : opWait.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, magic1, range2, magic2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                APASS_LOG_DEBUG_F(Elements::Operation, "%d %zu %s and %d %zu %s has WAR data dependency", opSet.GetOpMagic(), 
                    k, opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::HasDataDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) {
    std::string opSetStr = opSet.GetOpcodeStr();
    std::string opWaitStr = opWait.GetOpcodeStr();

    // check WAW
    bool checkWaw = true;
    auto setCfg = OpcodeManager::Inst().GetTileOpCfg(opSet.GetOpcode());
    auto waitCfg = OpcodeManager::Inst().GetTileOpCfg(opWait.GetOpcode());
    if (waitCfg.pipeIdStart_ == setCfg.pipeIdStart_ && (opSetStr.find("CUBE_A_MUL") == std::string::npos || opWaitStr.find("CUBE_A_MUL") == std::string::npos)) {
        checkWaw = false;
    }
    if (checkWaw) {
        if (CheckWawDependency(opSet, opWait, k, idx)) {
            return true;
        }
    }

    // check RAW
    if (CheckRawDependency(opSet, opWait, k, idx)) {
        return true;
    }

    // check WAR
    if (CheckWarDependency(opSet, opWait, k, idx)) {
        return true;
    }

    return false;
}

void PipeSync::UpdateDep(DepOp &currOp, DepOp &prevOp) {
    PipeCoreReal currPipe(currOp.selfPipeCore.pipeStart, currOp.selfPipeCore.core);
    PipeCoreReal prevPipe(prevOp.selfPipeCore.pipeEnd, prevOp.selfPipeCore.core);
    auto &currPipeDep = latestPipeDep_[currPipe];
    currPipeDep.waitIdx = currOp.idx;

    auto currSetPipeIter = currPipeDep.setPipes.find(prevPipe);
    if (currSetPipeIter == currPipeDep.setPipes.end() || currSetPipeIter->second < prevOp.idx) {
        // no indirect dependency exist, save current dependency
        currOp.waitPipe.emplace_back(prevOp.idx);
        prevOp.setPipe.emplace_back(currOp.idx);
        currPipeDep.setPipes[prevPipe] = prevOp.idx;
        auto prevPipeDepIter = latestPipeDep_.find(prevPipe);
        auto prevWaitPipeIdx = prevPipeDepIter->second.waitIdx;
        if (prevPipeDepIter != latestPipeDep_.end() && prevWaitPipeIdx <= prevOp.idx) {
            // merge dependency
            std::map<PipeCoreReal, size_t, PipeCoreRealCompare> prevSetPipes = prevPipeDepIter->second.setPipes;
            for (auto ele : prevSetPipes) {
                PipeCoreReal prevSetPipeType = ele.first;
                size_t prevSetPipeIdx = ele.second;
                auto res = currPipeDep.setPipes.emplace(prevSetPipeType, prevSetPipeIdx);
                //isExist == isPrevSetPipeTypeExist
                bool isExist = !res.second;
                size_t &existIdx = res.first->second;
                if (isExist && existIdx < prevSetPipeIdx) {
                    // overwrite dependency
                    existIdx = prevSetPipeIdx;
                }
            }
        }
    }
}

bool PipeSync::IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation *> opLogPtr) {
    // true表示依赖关系可忽略，false表示依赖关系不可忽略
    // VIEW or ASSEMBLE data dependency can be ignored
    if (opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[curr]->GetOpcode() == Opcode::OP_ASSEMBLE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_NOP || opLogPtr[curr]->GetOpcode() == Opcode::OP_NOP ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_HUB || opLogPtr[curr]->GetOpcode() == Opcode::OP_HUB) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%d %s and %d %s dependency is ignorable because op is VIEW or ASSEMBLE or NOP",
            opLogPtr[prev]->GetOpMagic(), opLogPtr[prev]->GetOpcodeStr().c_str(), opLogPtr[curr]->GetOpMagic(), opLogPtr[curr]->GetOpcodeStr().c_str());
        return true;
    }
    return false;
}

// find depend op in opLog for 0 to idx
void PipeSync::FindDep(DepOp &op, const std::vector<Operation *> opLogPtr, size_t idx, DataDependencySearcher& dataDependencySearcher) {
    const auto currOp = opLogPtr[idx];
    APASS_LOG_DEBUG_F(Elements::Operation, "=== OP: %d %zu %s ===", currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());
    // check dependency from latest op to oldest
    auto dataDependencySet = dataDependencySearcher.Find(currOp);
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        const Operation *prevAOp = opLogPtr[k];
        DepOp &prevOp = depOps_[k];
        APASS_LOG_DEBUG_F(Elements::Operation, "Current process ops: %d %zu %s and %d %zu %s", prevAOp->GetOpMagic(), k, prevAOp->GetOpcodeStr().c_str(),
            currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());

        if (HasDataDependency(*prevAOp, *currOp, k, idx)) {
            bool ignorable = false;
            if (IgnorableIntraPipeDep(k, idx, opLogPtr)) {
                ignorable = true;
            }
            if (!ignorable) {
                UpdateDep(op, prevOp);
            }
        }
    }
    dataDependencySearcher.Insert(currOp, idx);
}

std::pair<PipeSync::CoreTypeDetail, PipeSync::CoreTypeDetail> PipeSync::GetCorePairDetail(const PipePair &pp, size_t setIdx, size_t waitIdx, bool &isAIV1) {
    CoreTypeDetail setCoreType;
    CoreTypeDetail waitCoreType;
    if (pp.first.core == CoreType::AIV) {
        AIVCore setAIV = oriOpList_[setIdx]->GetAIVCore();
        if (setAIV == AIVCore::AIV1) {
            isAIV1 = true;
        }
        setCoreType = {CoreType::AIV, setAIV};
        waitCoreType = {CoreType::AIC, AIVCore::UNSPECIFIED};
    } else {
        AIVCore waitAIV = oriOpList_[waitIdx]->GetAIVCore();
        if (waitAIV == AIVCore::AIV1) {
            isAIV1 = true;
        }
        setCoreType = {CoreType::AIC, AIVCore::UNSPECIFIED};
        waitCoreType = {CoreType::AIV, waitAIV};
    }
    return {setCoreType, waitCoreType};
}

void PipeSync::InitCVEventIdQ(bool isAIV1, CorePair corePair, CorePair corePairReverse) {
    if (!isAIV1) {
        for (int i = 0; i < CROSS_CORE_EVENT_NUM; i++) {
            crossCoreFreeEventId_[corePair].push_back(i);
            crossCoreFreeEventId_[corePairReverse].push_back(i);
        }
    } else {
        for (int i = CROSS_CORE_EVENT_NUM; i < CROSS_CORE_EVENT_NUM * NUM2; i++) {
            crossCoreFreeEventId_[corePair].push_back(i);
            crossCoreFreeEventId_[corePairReverse].push_back(i);
        }
    }
}

std::deque<int> &PipeSync::GetFreeEventIdQueue(const PipePair &pp, size_t setIdx, size_t waitIdx, std::pair<CoreTypeDetail, CoreTypeDetail> &setWaitCoreType) {
    // 若coretype不同，所有CV同步共享16个eventid
    if (pp.first.core != pp.second.core) {
        bool isAIV1{false};
        setWaitCoreType = GetCorePairDetail(pp, setIdx, waitIdx, isAIV1);
        CorePair corePair = {setWaitCoreType.first, setWaitCoreType.second};
        CorePair corePairReverse = {setWaitCoreType.second, setWaitCoreType.first};
        if (crossCoreFreeEventId_.count(corePair) == 0) {
            InitCVEventIdQ(isAIV1, corePair, corePairReverse);
        }
        return crossCoreFreeEventId_[corePair];
    }
    if (freeEventId_.count(pp) == 0) {
        for (int i = 0; i < GetMaxEventId(pp); i++) {
            freeEventId_[pp].push_back(i);
        }
    }
    return freeEventId_[pp];
}

void PipeSync::AddPhaseOp1(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog, size_t &i, size_t &prerun) {
    constexpr size_t prerunNum = 2;
    for (; i < srcLog.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(srcLog[i]->GetOpcode());
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        }
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        }
        if ((opcfg.pipeIdStart_ != PIPE_S && opcfg.pipeIdStart_ != PIPE_MTE2 &&
            srcLog[i]->GetOpcode() != Opcode::OP_RESHAPE && srcLog[i]->GetOpcode() != Opcode::OP_VEC_DUP) ||
            prerun == prerunNum) {
            break;
        }
        if (opcfg.pipeIdStart_ == PIPE_MTE2) {
            if (prerun == 0) {
                std::vector<std::shared_ptr<LogicalTensor>> input;
                std::vector<std::shared_ptr<LogicalTensor>> output;
                Operation &phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE1, {input}, {output});
                Operation *phaseOpPtr = &phaseOp;
                dstLog.emplace_back(phaseOpPtr);
            }
            prerun++;
        }
        dstLog.emplace_back(srcLog[i]);
    }
}

void PipeSync::AddPhaseOp2(Function &function, std::vector<Operation *> &dstLog, size_t &prerun) {
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE2, {input}, {output});
        Operation *phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
}

void PipeSync::PhaseKernelProcess(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog) {
    size_t prerun = 0;
    size_t i = 0;
    AddPhaseOp1(function, srcLog, dstLog, i, prerun);
    AddPhaseOp2(function, dstLog, prerun);
    for (; i < srcLog.size(); i++) {
        dstLog.emplace_back(srcLog[i]);
    }
}

Status PipeSync::ProcessView(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    auto it1 = std::find(opLogNew.begin(), opLogNew.end(), pair.second);
    auto it2 = std::find(opLogNew.begin(), opLogNew.end(), pair.first);
    if (it1 == opLogNew.end()) {
        if (it2 == opLogNew.end()) {
            opLogNew.emplace_back(pair.first);
            opLogNew.emplace_back(pair.second);
            return SUCCESS;
        }
        opLogNew.insert(it2+1, pair.second);
        return SUCCESS;
    }
    if (it2 == opLogNew.end()) {
        opLogNew.insert(it1, pair.first);
    }
    return SUCCESS;
}

Status PipeSync::ProcessAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    auto it1 = std::find(opLogNew.begin(), opLogNew.end(), pair.first);
    auto it2 = std::find(opLogNew.begin(), opLogNew.end(), pair.second);
    if (it1 == opLogNew.end()) {
        if (it2 == opLogNew.end()) {
            opLogNew.emplace_back(pair.second);
            opLogNew.emplace_back(pair.first);
            return SUCCESS;
        }
        opLogNew.insert(it2+1, pair.first);
        return SUCCESS;
    }
    if (it2 == opLogNew.end()) {
        opLogNew.insert(it1, pair.second);
    }
    return SUCCESS;
}

Status PipeSync::ProcessViewAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    if (pair.first->GetOpcode() == Opcode::OP_VIEW) {
        if (ProcessView(opLogNew, pair) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessView failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (pair.first->GetOpcode() != Opcode::OP_ASSEMBLE) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessViewAssemble failed, this op should be ASSEMBLE.");
        return FAILED;
    }
    if (ProcessAssemble(opLogNew, pair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessAssemble failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ReorderViewAssemble(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew, const std::unordered_map<Operation *, Operation *> &changeMap) {
    std::unordered_set<Operation *> toBeInsert;
    for (auto pair : changeMap) {
        toBeInsert.insert(pair.first);
        toBeInsert.insert(pair.second);
    }
    for (auto opPtr : opLog) {
        auto it = toBeInsert.find(opPtr);
        if (it == toBeInsert.end()) {
            opListNew.emplace_back(opPtr);
            continue;
        }
        for (auto pair : changeMap) {
            if (pair.second == opPtr && (ProcessViewAssemble(opListNew, pair) != SUCCESS)) {
                APASS_LOG_ERROR_F(Elements::Operation, "ReorderViewAssemble failed at function ProcessViewAssemble.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessViewOrder(Operation &op, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap) {
    auto consumers = op.ConsumerOps();
    if (consumers.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d VIEW op doesn't have consumer, ProcessViewAssembleOrder failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto minIt = opLog.end();
    for (auto &consumer : consumers) {
        auto it = std::find(opLog.begin(), opLog.end(), consumer);
        if (it == opLog.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Consumer of VIEW op: %d %s is not in the subgraph, ProcessViewAssembleOrder failed", 
                consumer->GetOpMagic(), consumer->GetOpcodeStr().c_str());
            return FAILED;
        }
        if (it < minIt) {
            minIt = it;
        }
    }
    changeMap[&op] = *minIt;
    APASS_LOG_DEBUG_F(Elements::Operation, "%d VIEW consumer: %d", op.GetOpMagic(), (*minIt)->GetOpMagic());
    return SUCCESS;
}

Status PipeSync::ProcessAssembleOrder(Operation &op, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap) {
    auto producers = op.ProducerOps();
    if (producers.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d ASSEMBLE op doesn't have producer, ProcessViewAssembleOrder failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto maxIt = opLog.begin();
    for (auto &producer : producers) {
        auto it = std::find(opLog.begin(), opLog.end(), producer);
        if (it == opLog.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Producer of ASSEMBLE op: %d %s is not in the subgraph, ProcessViewAssembleOrder failed.%s",
                producer->GetOpMagic(), producer->GetOpcodeStr().c_str(), GetFormatBacktrace(*producer).c_str());
            return FAILED;
        }
        if (it != opLog.begin() && it > maxIt) {
            maxIt = it;
        }
    }
    changeMap[&op] = *maxIt;
    APASS_LOG_DEBUG_F(Elements::Operation, "%d ASSEMBLE producer: %d", op.GetOpMagic(), (*maxIt)->GetOpMagic());
    return SUCCESS;
}

Status PipeSync::ProcessViewAssembleOrder(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew) {
    std::unordered_map<Operation *, Operation *> changeMap;
    for (auto &opPtr : opLog) {
        if (opPtr->GetOpcode() == Opcode::OP_VIEW) {
            if (ProcessViewOrder(*opPtr, opLog, changeMap)) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessViewOrder failed.");
                return FAILED;
            }
            continue;
        }
        if (opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            break;
        }
        if (ProcessAssembleOrder(*opPtr, opLog, changeMap)) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessAssembleOrder failed.");
            return FAILED;
        }
    }
    if (ReorderViewAssemble(opLog, opListNew, changeMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessViewAssembleOrder failed at function ReorderViewAssemble.");
        return FAILED;
    }
    return SUCCESS;
}

void InsertSync::InsertPipeAll(Function *subGraphFunc) {
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    std::vector<Operation*> newOpList;
    for (auto op : oriOpList) {
        newOpList.push_back(op);
        if (op->GetOpcode() == Opcode::OP_RESHAPE || op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = subGraphFunc->AddRawOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {input}, {output});
        syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1};
        newOpList.push_back(&syncOp);
    }
    subGraphFunc->ScheduleBy(newOpList, true);
}

Status InsertSync::CheckNewOpListSeq(const std::vector<Operation *> &oriOpList, const std::vector<Operation *> &opListNew) {
    if (oriOpList.size() <= 1) {
        return SUCCESS;
    }
    size_t i = 0;
    size_t j = 0;
    while (i < oriOpList.size() && j < opListNew.size()) {
        if (oriOpList[i] == opListNew[j]) {
            ++i;
            ++j;
        } else {
            ++j;
        }
    }
    if (i != oriOpList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "NewOpList sequence is not equal to OriOpList sequence, CheckNewOpListSeq failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InsertSync::GenNewOpList(Function *subGraphFunc, std::vector<Operation *> &opListNew) {
    PipeSync ps;
    std::vector<Operation *> syncedOpLogPtr;
    std::vector<Operation *> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    if (ps.InsertSync(*subGraphFunc, syncedOpLogPtr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function InsertSync.");
        return FAILED;
    }
    ps.PhaseKernelProcess(*subGraphFunc, syncedOpLogPtr, opListNew);
    subGraphFunc->EraseOperations(true, false);
    if (CheckNewOpListSeq(oriOpList, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function CheckNewOpListSeq.");
        return FAILED;
    }
    return SUCCESS;
}

Status InsertSync::InsertSyncMainLoop(Function *subGraphFunc) {
    if (enableDebug_) {
        InsertPipeAll(subGraphFunc);
        return SUCCESS;
    }
    std::vector<Operation *> opListNew;
    if (GenNewOpList(subGraphFunc, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSyncMainLoop failed at GenNewOpList.");
        return FAILED;
    }
    subGraphFunc->ScheduleBy(opListNew, true);
    APASS_LOG_DEBUG_F(Elements::Operation, "==========================================================================================");
    for (const auto &op : subGraphFunc->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("SYNC_SRC") != std::string::npos || op->GetOpcodeStr().find("SYNC_DST") != std::string::npos
            || op->GetOpcode() == Opcode::OP_BAR_V || op->GetOpcode() == Opcode::OP_BAR_M) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Output operation %d: %s, setpipe type: %s, setcore type: %s, waitpipe type: %s, waitcore type: %s, eventid: %d",
                op->GetOpMagic(), op->GetOpcodeStr().c_str(), GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(), GetCoreTypeDict().Find(op->syncQueue_.coreType_).c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(), GetCoreTypeDict().Find(op->syncQueue_.trigCoreType_).c_str(), op->syncQueue_.eventId_);
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Output operation %d: %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
    }
    return SUCCESS;
}

// regist pass
Status InsertSync::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Start InsertSync.");
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig("pass_thread_num", 1);
    uint64_t index = 0;
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto &subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(
        static_cast<unsigned>(leafFuncSize),
        hardwareConcurrency
    );
    std::vector<std::thread> workers;
    bool status{true};
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, leafFuncSize, &index, this, &status] {
            for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize; idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
                auto program = subPrograms[idx];
                APASS_LOG_DEBUG_F(Elements::Operation, "====================================Program %d ===========================================", index);
                if (InsertSyncMainLoop(program.second) != SUCCESS) {
                    status = false;
                    break;
                }
                index++;
            }
        });
    }
    // Wait for all threads to finish
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (!status) {
        if (threadNum == 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Single Thread scenario.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Multiple Threads scenario.");
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Finish InsertSync.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu

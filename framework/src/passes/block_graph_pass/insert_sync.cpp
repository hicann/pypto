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
    const Interval& interval, IntervalTreeNode* currPtr, std::vector<IntervalTreeNode*>& intervalStack)
{
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

Status RangeSearchTree::InsertInterval(const Interval& interval)
{
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
        IntervalTreeNode* currPtr = intervalStack.back();
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

void RangeSearchTree::OverlapSearch(const Interval& interval, std::set<int>& result)
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);

    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
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

void RangeSearchTree::FreeTree()
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        intervalStack.push_back(currPtr->left);
        intervalStack.push_back(currPtr->right);
        delete currPtr;
    }
}

void RangeSearchTree::Insert(int left, int right, int idx)
{
    Interval interval(left, right, idx);
    InsertInterval(interval);
}

std::set<int> RangeSearchTree::GetCovered(int left, int right)
{
    Interval givenInterval(left, right, 0);
    std::set<int> overlappingIdx;
    OverlapSearch(givenInterval, overlappingIdx);
    return overlappingIdx;
}

void DataDependencySearcher::CheckWAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType currMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(currMemoryType) > 0) {
            TileRange rg =
                currMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = wawSearchTree_[currMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckRAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
        auto tensor = opWait->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1 && writeDdrMemMap.count(readDDRmemId) > 0) {
            std::set<int> found = writeDdrMemMap[readDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (rawSearchTree_.count(readMemoryType) > 0) {
            TileRange rg =
                readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = rawSearchTree_[readMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckWARSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1 && readDdrMemMap.count(writeDDRmemId) > 0) {
            std::set<int> found = readDdrMemMap[writeDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (warSearchTree_.count(writeMemoryType) > 0) {
            TileRange rg =
                writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
            std::set<int> found = warSearchTree_[writeMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

std::set<int> DataDependencySearcher::Find(Operation* opWait)
{
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

void DataDependencySearcher::InsertWAWSearchTree(const Operation* opSet, int idx)
{
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType prevMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(prevMemoryType) == 0) {
            wawSearchTree_[prevMemoryType] = RangeSearchTree();
        }
        TileRange rg =
            prevMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        wawSearchTree_[prevMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertRAWSearchTree(const Operation* opSet, int idx)
{
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
        TileRange rg =
            writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        rawSearchTree_[writeMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertWARSearchTree(const Operation* opSet, int idx)
{
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
        TileRange rg =
            readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] : tensor->memoryrange;
        warSearchTree_[readMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::Insert(const Operation* opSet, int idx)
{
    InsertWAWSearchTree(opSet, idx);
    InsertRAWSearchTree(opSet, idx);
    InsertWARSearchTree(opSet, idx);
}

void PipeSync::BuildTensorRangeMap(Operation* op)
{
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
    for (const auto& tensor : inOutTensors) {
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

Status PipeSync::InsertSync(Function& function, std::vector<Operation*>& syncedOpLog)
{
    std::vector<IndexOp> synced;
    std::vector<Operation*> opLogPtr(function.Operations(false).DuplicatedOpList());
    oriOpList_ = opLogPtr;
    uint64_t idxInput = 0;
    for (const auto& op : opLogPtr) {
        BuildTensorRangeMap(op);
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Input operation %lu %lu: %s.", static_cast<unsigned long>(idxInput),
            static_cast<unsigned long>(op->GetOpMagic()), op->GetOpcodeStr().c_str());
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

    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });

    for (auto& log : synced) {
        syncedOpLog.push_back(&log.second.get());
    }
    return SUCCESS;
}

std::string PipeSync::DepOp::DumpDepOp(std::vector<Operation*> opLog)
{
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

std::string PipeSync::IssueQueue::DumpIssueQueue(std::vector<Operation*> opLogPtr)
{
    std::stringstream ss;
    ss << " Op in this pipe: {";
    for (auto op : ops) {
        ss << opLogPtr[op]->GetOpMagic() << " " << opLogPtr[op]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::PipeDepInfo::DumpPipeDepInfo()
{
    std::stringstream ss;
    ss << "    wait idx: " << waitIdx << "\n";
    ss << "    setPipeCoreEx:"
       << "\n";
    for (auto pair : setPipes) {
        ss << "        pipetype: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << "  aivCore: " << static_cast<int>(pair.first.aivCore)
           << "  opidx: " << pair.second << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpLatestPipeDepMap()
{
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current PipeCore: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << " aivCore: " << static_cast<int>(pair.first.aivCore) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const
{
    switch (seq) {
        case PipeSeq::AIC_MTE2:
            return "AIC_MTE2";
        case PipeSeq::AIC_MTE1:
            return "AIC_MTE1";
        case PipeSeq::AIC_M:
            return "AIC_M";
        case PipeSeq::AIC_FIX:
            return "AIC_FIX";
        case PipeSeq::AIV0_MTE2:
            return "AIV0_MTE2";
        case PipeSeq::AIV1_MTE2:
            return "AIV1_MTE2";
        case PipeSeq::AIV0_V:
            return "AIV0_V";
        case PipeSeq::AIV1_V:
            return "AIV1_V";
        case PipeSeq::AIV0_MTE3:
            return "AIV0_MTE3";
        case PipeSeq::AIV1_MTE3:
            return "AIV1_MTE3";
        case PipeSeq::AIC_MTE3:
            return "AIC_MTE3";
        case PipeSeq::AIV0_S:
            return "AIV0_S";
        case PipeSeq::AIV1_S:
            return "AIV1_S";
        case PipeSeq::AIC_S:
            return "AIC_S";
        case PipeSeq::PIPE_END:
            return "PIPE_END";
        default:
            return "ILLEGAL";
    }
}

std::map<PipeSync::PipeCoreRealEx, PipeSeq, PipeSync::PipeCoreRealExCompare> PipeSync::pipe2Seq = {
    {{PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE2},
    {{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE1},
    {{PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_M},
    {{PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_FIX},
    {{PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE3},
    {{PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_S},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE2},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE2},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_V},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_V},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE3},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE3},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_S},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_S},
};

std::map<PipeSeq, PipeSync::PipeCoreRealEx> PipeSync::seq2pipe = {
    {PipeSeq::AIC_MTE2, {PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE1, {PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_M, {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_FIX, {PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE3, {PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_S, {PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIV0_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_V, {PIPE_V, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_V, {PIPE_V, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_S, {PIPE_S, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_S, {PIPE_S, CoreType::AIV, AIVCore::AIV1}},
};

PipeSeq PipeSync::GetPipeSeq(PipeSync::PipeCoreRealEx pipe) { return pipe2Seq.at(pipe); }

PipeSync::PipeCoreRealEx PipeSync::GetPipeFromSeq(PipeSeq seq) { return seq2pipe.at(seq); }

Status PipeSync::AdjustCopyInCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "%d COPYIN op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
            GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto dstMemType = attr->GetCopyInAttr().second;
    if (dstMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
        return SUCCESS;
    }
    if (dstMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIV;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyOutCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "%d COPYOUT op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
            GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto srcMemType = attr->GetCopyOutAttr().first;
    if (srcMemType == MemoryType::MEM_L0C) {
        opcfg.pipeIdStart_ = PipeType::PIPE_FIX;
        opcfg.pipeIdEnd_ = PipeType::PIPE_FIX;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
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
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpCfg(TileOpCfg& opcfg, const Operation& op)
{
    opcfg.aivCore_ = op.GetAIVCore();
    if (opcfg.coreType_ == CoreType::AIC) {
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    if (opcfg.coreType_ == CoreType::AIV && opcfg.aivCore_ == AIVCore::UNSPECIFIED) {
        opcfg.aivCore_ = AIVCore::AIV0;
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

Status PipeSync::PipeDispatch(const std::vector<Operation*> opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    DataDependencySearcher dataDependencySearcher;
    dataDependencySearcher.ubTensorRangeMap = ubTensorRangeMap;
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        if (opLogPtr[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "%d ALLOC op should not appear in InsertSync, PipeDispatch failed.%s",
                opLogPtr[i]->GetOpMagic(), GetFormatBacktrace(opLogPtr[i]).c_str());
            return FAILED;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        if (AdjustOpCfg(opcfg, *opLogPtr[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PipeDispatch failed at function AdjustOpCfg.");
            return FAILED;
        }
        DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        DepOp& opRef = depOps_.emplace_back(op);
        FindDep(opRef, opLogPtr, i, dataDependencySearcher);
        EnqueueOp(opRef, opLogPtr, syncedOpLog);
    }
    return SUCCESS;
}

void PipeSync::InitIssueQueue()
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp& op, const std::vector<Operation*> opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_NOP || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_RESHAPE) {
        syncedOpLog.emplace_back(std::make_pair(op.idx * SEQUENCE_IDX, std::ref(*opLogPtr[op.idx])));
        return;
    }
    PipeCoreRealEx opPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    auto& issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCoreEx))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreRealEx opPipeCoreExStart(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipeCoreRealEx opPipeCoreExEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipePairEx pp{opPipeCoreExStart, opPipeCoreExEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
    orderedOpList_.emplace(op.idx);
}

void PipeSync::RemoveOpDep(DepOp& setOp, DepOp& waitOp) const
{
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

Status PipeSync::AddOpDep(DepOp& setOp, DepOp& waitOp)
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    size_t depWaitIdx = static_cast<size_t>(-1);
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            APASS_LOG_ERROR_F(Elements::Operation, "This dependency should not exist, AddOpDep failed.");
            return FAILED;
        }
        PipeCoreRealEx elePipeCoreEx(
            depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core, depOps_[ele].selfPipeCore.aivCore);
        PipeCoreRealEx waitOpPipeCoreEx(
            depOps_[waitOpIdx].selfPipeCore.pipeStart, depOps_[waitOpIdx].selfPipeCore.core,
            depOps_[waitOpIdx].selfPipeCore.aivCore);
        if (elePipeCoreEx == waitOpPipeCoreEx) {
            if (ele <= waitOpIdx) {
                APASS_LOG_ERROR_F(Elements::Operation, "New waitidx should less than old, AddOpDep failed.");
                return FAILED;
            }
            depWaitIdx = ele;
            break;
        }
    }
    // 同一种pipecore，不需要存记录依赖关系
    if (depWaitIdx != static_cast<size_t>(-1)) {
        RemoveOpDep(setOp, depOps_[depWaitIdx]);
    }
    setOp.setPipe.emplace_back(waitOpIdx);
    waitOp.waitPipe.emplace_back(setOpIdx);
    return SUCCESS;
}

Status PipeSync::AdjustOpDep(DepOp& op, size_t waitOpIdx, IssueQueue& issueQ, bool& failedFlag)
{
    // op为靠前的， waitOp为靠后的
    auto& waitOp = depOps_[waitOpIdx];

    if (issueQ.currOp + 1 == issueQ.ops.size()) {
        failedFlag = true;
        return SUCCESS;
    }

    auto& nextOpIdx = issueQ.ops[issueQ.currOp + 1];

    // next op的idx要小于waitop
    if (nextOpIdx > waitOpIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot AdjustOpDep because nextop idx > waitop idx.");
        failedFlag = true;
        return SUCCESS;
    }

    auto& nextOp = depOps_[nextOpIdx];
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustOpDep failed at function AddOpDep.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::RelaxMultipleEventIds(
    const CorePair& setwaitCoreType, size_t needEvIdNum, std::vector<IndexOp>& syncedOpLog)
{
    for (size_t i = 0; i < needEvIdNum; i++) {
        if (RelaxCvEventIdMain(syncedOpLog, setwaitCoreType) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RelaxMultipleEventIds failed at RelaxCvEventIdMain.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessCrossCoreCase(
    const PipePairEx& pp, const CorePair& setwaitCoreType, const CorePair& setwaitReverse, EventIdProcessContext& ctx)
{
    auto& issuenum = ctx.issuenum;
    issuenum.maxCvIssueNum.emplace(setwaitCoreType, GetFreeEventIdQueue(pp).size());
    issuenum.maxCvIssueNum.emplace(setwaitReverse, GetFreeEventIdQueue(pp).size());
    issuenum.currCvIssueNum.emplace(setwaitCoreType, 0);
    issuenum.currCvIssueNum.emplace(setwaitReverse, 0);
    coreIssueNumMap[setwaitCoreType]++;
    coreIssueNumMap[setwaitReverse]++;

    if (issuenum.currCvIssueNum[setwaitCoreType] + coreIssueNumMap[setwaitCoreType] >
        issuenum.maxCvIssueNum[setwaitCoreType]) {
        if (!ctx.deadlock) {
            ctx.eventIdOk = false;
            return SUCCESS;
        }
        if (AdjustOpDep(ctx.op, ctx.eleIdx, ctx.issueQ, ctx.failedFlag) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessCrossCoreCase failed at AdjustOpDep.");
            return FAILED;
        }
        if (ctx.failedFlag) {
            size_t needEvIdNum = issuenum.currCvIssueNum[setwaitCoreType] + coreIssueNumMap[setwaitCoreType] -
                                 issuenum.maxCvIssueNum[setwaitCoreType];
            if (RelaxMultipleEventIds(setwaitCoreType, needEvIdNum, ctx.syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessCrossCoreCase failed at RelaxMultipleEventIds.");
                return FAILED;
            }
            ctx.failedFlag = false;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx)
{
    ctx.issuenum.maxIssueNum.emplace(pp, GetFreeEventIdQueue(pp).size());
    ctx.issuenum.currIssueNum.emplace(pp, 0);

    if (ctx.issuenum.currIssueNum[pp] >= ctx.issuenum.maxIssueNum[pp]) {
        if (!ctx.deadlock) {
            ctx.eventIdOk = false;
            return SUCCESS;
        }
        if (AdjustOpDep(ctx.op, ctx.eleIdx, ctx.issueQ, ctx.failedFlag) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessSameCoreCase failed at AdjustOpDep.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessEventIdElement(EventIdProcessContext& ctx)
{
    auto& op = ctx.op;
    auto ele = ctx.eleIdx;

    if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
        return SUCCESS;
    }

    PipeCoreRealEx currEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    PipeCoreRealEx eleEx(
        depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core, depOps_[ele].selfPipeCore.aivCore);
    PipePairEx pp{currEx, eleEx};

    if (currEx.core != eleEx.core) {
        CorePair setwaitCoreType{
            {op.selfPipeCore.core, op.selfPipeCore.aivCore},
            {depOps_[ele].selfPipeCore.core, depOps_[ele].selfPipeCore.aivCore}};
        CorePair setwaitReverse{
            {depOps_[ele].selfPipeCore.core, depOps_[ele].selfPipeCore.aivCore},
            {op.selfPipeCore.core, op.selfPipeCore.aivCore}};
        return ProcessCrossCoreCase(pp, setwaitCoreType, setwaitReverse, ctx);
    }

    return ProcessSameCoreCase(pp, ctx);
}

Status PipeSync::HandleEventID(
    DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res, std::vector<IndexOp>& syncedOpLog)
{
    coreIssueNumMap.clear();
    bool eventIdOk = true;
    bool failedFlag = false;

    for (auto ele : op.setPipe) {
        EventIdProcessContext ctx{op, ele, issueQ, issuenum, syncedOpLog, deadlock, eventIdOk, failedFlag};
        if (ProcessEventIdElement(ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "HandleEventID failed at ProcessEventIdElement.");
            return FAILED;
        }
        eventIdOk = ctx.eventIdOk;
        failedFlag = ctx.failedFlag;
        if (!eventIdOk) {
            break;
        }
    }

    if (failedFlag) {
        deadlock = true;
        res = false;
    } else {
        deadlock = false;
        res = eventIdOk;
    }
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp& op)
{
    // current op will be issued only when all of the waitop are issued
    for (const auto& waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(
    IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock, std::vector<IndexOp>& syncedOpLog)
{
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto& op = depOps_[issueQ.ops[issueQ.currOp]];
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
        if (HandleEventID(op, issueQ, issuenum, deadlock, res, syncedOpLog) != SUCCESS) {
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
            PipeCoreRealEx currPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
            PipeCoreRealEx elePipeCoreEx(
                depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core, depOps_[ele].selfPipeCore.aivCore);
            auto pp = PipePairEx{currPipeCoreEx, elePipeCoreEx};
            issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
        }
        for (auto [corePair, issueNum] : coreIssueNumMap) {
            CorePair corePairReverse = {corePair.second, corePair.first};
            issuenum.currCvIssueNum[corePair] += issueNum;
            issuenum.currCvIssueNum[corePairReverse] += issueNum;
        }
        issueQ.currOp++;
    }
    return SUCCESS;
}

Status PipeSync::InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    // serch the waitpipe of current op
    uint64_t waitIdx = idx == 0 ? 0 : idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    for (const auto& ele : depOps_[idx].waitPipe) {
        PipeCore setPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx setPipeRealEx(setPipe.pipeEnd, setPipe.core, setPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeStart, currPipe.core, currPipe.aivCore);
        int eventId = setWaitPairMap_[{ele, idx}];
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_DST, {input}, {output});
        bool res = GenSyncOp(setPipeRealEx, currPipeRealEx, eventId, false, syncOp);
        if (!res) {
            syncOp.SetAsDeleted();
            continue;
        }
        // insert wait_flag
        syncedOpLog.emplace_back(std::make_pair(++waitIdx, std::ref(syncOp)));
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
            syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
            static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
        GetFreeEventIdQueue({setPipeRealEx, currPipeRealEx}).push_back(eventId);
        if (setPipeRealEx.core != currPipeRealEx.core) {
            GetFreeEventIdQueue({currPipeRealEx, setPipeRealEx}).push_back(eventId);
        }
        // 记录 set op 和 waitflag的对应关系
        waitOpMap.emplace(&syncOp, oriOpList_[ele]);
    }
    return SUCCESS;
}

Status PipeSync::InjectSetFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    uint64_t setIdx = idx * SEQUENCE_IDX;
    for (const auto& ele : depOps_[idx].setPipe) {
        PipeCore waitPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx waitPipeRealEx(waitPipe.pipeStart, waitPipe.core, waitPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeEnd, currPipe.core, currPipe.aivCore);
        int eventId{0};
        if (GetEventId({currPipeRealEx, waitPipeRealEx}, eventId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InjectSetFlag failed at function GetEventId.");
            return FAILED;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_SRC, {input}, {output});
        bool res = GenSyncOp(currPipeRealEx, waitPipeRealEx, eventId, true, syncOp);
        if (res) {
            // insert set_flag
            syncedOpLog.emplace_back(std::make_pair(++setIdx, std::ref(syncOp)));
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
                syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
                static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
            setWaitPairMap_[{idx, ele}] = eventId;
            // 记录wait op 和 setflag的对应关系
            setOpMap.emplace(&syncOp, oriOpList_[ele]);
            continue;
        }
        syncOp.SetAsDeleted();
        setWaitPairMap_[{idx, ele}] = eventId;
    }
    return SUCCESS;
}

Status PipeSync::InjectSync(
    Function& function, std::vector<Operation*> opLogPtr, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
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
    APASS_LOG_DEBUG_F(
        Elements::Operation, "Insert %d %s", opLogPtr[idx]->GetOpMagic(), opLogPtr[idx]->GetOpcodeStr().c_str());

    // insert set_flag
    if (InjectSetFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectSetFlag.");
        return FAILED;
    }
    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePairEx& pp)
{
    PipePairEx ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_NUM;
    }
    return EVENT_ID7;
}

Status PipeSync::ProcessDeadLock(
    uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock, std::vector<IndexOp>& syncedOpLog)
{
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

Status PipeSync::IssueOpPipeSeq(
    Function& function, std::vector<Operation*> opLogPtr, std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock,
    size_t& issued)
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock, syncedOpLog) != SUCCESS) {
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

Status PipeSync::IssueSyncOp(
    Function& function, std::vector<Operation*> opLogPtr, std::vector<IndexOp>& syncedOpLog, size_t& totalIssued,
    size_t& allIssued)
{
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

Status PipeSync::IssueOp(Function& function, std::vector<Operation*> opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(),
            issueState_[i].DumpIssueQueue(opLogPtr).c_str());
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

bool PipeSync::ConstructDepInfo(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i)
{
    auto& log = syncedOpLog[i].second;
    if (log.get().GetOpcodeStr() != "SYNC_SRC") {
        return false;
    }
    auto setPipe = log.get().syncQueue_.pipeId_;
    auto waitPipe = log.get().syncQueue_.trigPipeId_;
    auto setCore = log.get().syncQueue_.coreType_;
    auto waitCore = log.get().syncQueue_.trigCoreType_;
    auto eventId = log.get().syncQueue_.eventId_;
    auto setAivCore = log.get().syncQueue_.setAivCore_;
    auto waitAivCore = log.get().syncQueue_.waitAivCore_;
    if (!(setPipe == depInfo.setp && setCore == depInfo.setc && waitPipe == depInfo.waitp &&
          waitCore == depInfo.waitc && setAivCore == depInfo.setaivc && waitAivCore == depInfo.waitaivc)) {
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

int PipeSync::GetSyncSrcLogIdx(const std::vector<IndexOp>& syncedOpLog, int i)
{
    int j = i - 1;
    for (; j >= 0; j--) {
        if (syncedOpLog[j].second.get().GetOpcodeStr().find("SYNC") == std::string::npos &&
            syncedOpLog[j].second.get().GetOpcodeStr().find("BAR") == std::string::npos) {
            break;
        }
    }
    return syncedOpLog[j].first;
}

bool PipeSync::FindDataDep(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i)
{
    if (!ConstructDepInfo(depInfo, syncedOpLog, i)) {
        return false;
    }
    int syncSrcLogIdx = GetSyncSrcLogIdx(syncedOpLog, i) / SEQUENCE_IDX; // sync_src对应的非同步op的idx
    DepOp& depOpSrc = depOps_[syncSrcLogIdx];
    for (auto syncDstLogIdx : depOpSrc.setPipe) { // setpipe中的op为该op之后的，依赖于该op的op id
        DepOp& depOpDst = depOps_[syncDstLogIdx];
        if (depOpDst.selfPipeCore.core == depInfo.waitc && depOpDst.selfPipeCore.pipeStart == depInfo.waitp &&
            depOpDst.selfPipeCore.aivCore == depInfo.waitaivc) {
            depInfo.opDepList.push_back(std::make_pair(syncSrcLogIdx, syncDstLogIdx));
        }
    }
    return true;
}

bool PipeSync::FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx)
{
    int maxOverlap = -1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        if (depInfo.opDepList[idx].second < depInfo.opDepList[idx + 1].first) { // 相邻的两个依赖之间没有重叠
            continue;
        }
        // max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        // max_overlap用来记录遍历到的最大overlap
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

Status PipeSync::SynDependency(
    int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx, std::vector<IndexOp>& syncedOpLog)
{
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
    if (pipePairEx.first.core != pipePairEx.second.core) {
        PipePairEx pipePairExReverse = {pipePairEx.second, pipePairEx.first};
        GetFreeEventIdQueue(pipePairEx).push_back(eventId1);
        GetFreeEventIdQueue(pipePairExReverse).push_back(eventId1);
    } else {
        GetFreeEventIdQueue(pipePairEx).push_back(eventId1);
    }
    setWaitPairMap_[{set2, wait1}] = eventId2;
    // 将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second.get().SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo)
{
    depInfo.setp = pipePairEx.first.pipe;
    depInfo.setc = pipePairEx.first.core;
    depInfo.waitp = pipePairEx.second.pipe;
    depInfo.waitc = pipePairEx.second.core;
    depInfo.setaivc = pipePairEx.first.aivCore;
    depInfo.waitaivc = pipePairEx.second.aivCore;
    // 8个eventid全部被占用
    // 说明该set pipe内肯定有8个set op已经发射，而wait pipe内对应的8个op一个都没发射。
    // 找到这8个setop的idx，对应的event id，以及对应的waitop idx
    auto eventNum = static_cast<std::vector<std::pair<int, int>>::size_type>(GetMaxEventId(pipePairEx));
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
    if (depInfo.opDepList.size() != eventNum || depInfo.setOpIdList.size() != eventNum ||
        depInfo.setOpEventIdList.size() != eventNum) {
        APASS_LOG_ERROR_F(Elements::Operation, "dep size should be %zu, RelaxFakeDataDep failed.", eventNum);
        return FAILED;
    }
    return SUCCESS;
}

std::vector<PipeSync::CorePair> PipeSync::cvCorePair = {
    {{CoreType::AIV, AIVCore::AIV0}, {CoreType::AIC, AIVCore::UNSPECIFIED}},
    {{CoreType::AIV, AIVCore::AIV1}, {CoreType::AIC, AIVCore::UNSPECIFIED}},
    {{CoreType::AIC, AIVCore::UNSPECIFIED}, {CoreType::AIV, AIVCore::AIV0}},
    {{CoreType::AIC, AIVCore::UNSPECIFIED}, {CoreType::AIV, AIVCore::AIV1}},
};

// 检查该 CV_SYNC_SRC 之后是否有对应的 CV_SYNC_DST
bool PipeSync::HasCvSyncDstAfter(const std::vector<IndexOp>& syncedOpLog, int srcIdx, const Operation& srcOp) const
{
    auto waitPipe = srcOp.syncQueue_.trigPipeId_;
    auto setCore = srcOp.syncQueue_.coreType_;
    auto waitCore = srcOp.syncQueue_.trigCoreType_;
    auto eventId = srcOp.syncQueue_.eventId_;
    auto setAivCore = srcOp.syncQueue_.setAivCore_;
    auto waitAivCore = srcOp.syncQueue_.waitAivCore_;
    for (int j = syncedOpLog.size() - 1; j > srcIdx; j--) {
        auto& dstOp = syncedOpLog[j].second;
        if (dstOp.get().GetOpcodeStr() != "CV_SYNC_DST") {
            continue;
        }
        if (dstOp.get().syncQueue_.trigPipeId_ == waitPipe && dstOp.get().syncQueue_.coreType_ == setCore &&
            dstOp.get().syncQueue_.trigCoreType_ == waitCore && dstOp.get().syncQueue_.eventId_ == eventId &&
            dstOp.get().syncQueue_.setAivCore_ == setAivCore && dstOp.get().syncQueue_.waitAivCore_ == waitAivCore) {
            return true;
        }
    }
    return false;
}

// 填充 cvDepInfoMap 中的依赖信息条目
void PipeSync::FillCvDepInfoEntry(
    std::unordered_map<PipePair, DataDepInfo, PipePairHash>& cvDepInfoMap, const std::vector<IndexOp>& syncedOpLog,
    int idx, int eventId)
{
    auto& op = syncedOpLog[idx].second.get();
    auto setPipe = op.syncQueue_.pipeId_;
    auto waitPipe = op.syncQueue_.trigPipeId_;
    auto setCore = op.syncQueue_.coreType_;
    auto waitCore = op.syncQueue_.trigCoreType_;
    auto setAivCore = op.syncQueue_.setAivCore_;
    auto waitAivCore = op.syncQueue_.waitAivCore_;
    PipeCoreReal setpipecore(setPipe, setCore);
    PipeCoreReal waitpipecore(waitPipe, waitCore);
    PipePair pipePair = {setpipecore, waitpipecore};
    auto it = cvDepInfoMap.find(pipePair);
    if (it == cvDepInfoMap.end()) {
        DataDepInfo depInfo;
        depInfo.setp = setPipe;
        depInfo.setc = setCore;
        depInfo.waitp = waitPipe;
        depInfo.waitc = waitCore;
        depInfo.setaivc = setAivCore;
        depInfo.waitaivc = waitAivCore;
        cvDepInfoMap[pipePair] = depInfo;
    }
    // 填充DataDepInfo的setOpIdList setOpEventIdList opDepList信息
    auto& depInfo = cvDepInfoMap[pipePair];
    depInfo.setOpIdList.push_back(idx);
    depInfo.setOpEventIdList.push_back(eventId);
    // CV_SYNC_SRC 对应的非同步op的idx
    int cvSyncSrcLogIdx = GetSyncSrcLogIdx(syncedOpLog, idx) / SEQUENCE_IDX;
    DepOp& depOpCvSrc = depOps_[cvSyncSrcLogIdx];
    for (auto cvSyncDstLogIdx : depOpCvSrc.setPipe) { // setPipe为该op之后的依赖于该op的id
        DepOp& depOpCvDst = depOps_[cvSyncDstLogIdx];
        if (depOpCvDst.selfPipeCore.core == depInfo.waitc && depOpCvDst.selfPipeCore.pipeStart == depInfo.waitp &&
            depOpCvDst.selfPipeCore.aivCore == depInfo.waitaivc) {
            depInfo.opDepList.push_back(std::make_pair(cvSyncSrcLogIdx, cvSyncDstLogIdx));
        }
    }
}

void PipeSync::FindCvSyncSrcInfo(
    const std::vector<IndexOp>& syncedOpLog, std::vector<int>& eventIdVec, const CorePair& corePair,
    std::unordered_map<PipePair, DataDepInfo, PipePairHash>& cvDepInfoMap)
{
    for (int i = syncedOpLog.size() - 1; i >= 0; i--) {
        auto& op = syncedOpLog[i].second;
        if (op.get().GetOpcodeStr() != "CV_SYNC_SRC") {
            continue;
        }
        auto setCore = op.get().syncQueue_.coreType_;
        auto waitCore = op.get().syncQueue_.trigCoreType_;
        auto setAivCore = op.get().syncQueue_.setAivCore_;
        auto waitAivCore = op.get().syncQueue_.waitAivCore_;
        if (!(setCore == corePair.first.first && setAivCore == corePair.first.second &&
              waitCore == corePair.second.first && waitAivCore == corePair.second.second)) {
            continue;
        }
        auto eventId = op.get().syncQueue_.eventId_;
        // 保证被释放的eventid对应的cv_sync_src不被统计进去
        if (std::find(eventIdVec.begin(), eventIdVec.end(), eventId) != eventIdVec.end()) {
            continue;
        }
        if (HasCvSyncDstAfter(syncedOpLog, i, op.get())) {
            continue;
        }
        FillCvDepInfoEntry(cvDepInfoMap, syncedOpLog, i, eventId);
        eventIdVec.push_back(eventId);
        if (eventIdVec.size() == CROSS_CORE_EVENT_NUM) {
            break;
        }
    }
}

bool PipeSync::FindMaxOverlapForCV(
    PipePair& targetPp, int& maxOverlapIdx, std::unordered_map<PipePair, DataDepInfo, PipePairHash>& cvDepInfoMap)
{
    int maxOverlap = -1;
    for (auto& [pp, depinfo] : cvDepInfoMap) {
        std::reverse(depinfo.opDepList.begin(), depinfo.opDepList.end());
        std::reverse(depinfo.setOpIdList.begin(), depinfo.setOpIdList.end());
        std::reverse(depinfo.setOpEventIdList.begin(), depinfo.setOpEventIdList.end());
        if (depinfo.opDepList.size() < 2) {
            continue;
        }
        for (int i = 0; i < static_cast<int>(depinfo.opDepList.size() - 1); i++) {
            if (depinfo.opDepList[i].second < depinfo.opDepList[i + 1].first) {
                continue;
            }
            if ((depinfo.opDepList[i].second - depinfo.opDepList[i + 1].first) > maxOverlap) {
                maxOverlapIdx = i;
                targetPp = pp;
                maxOverlap = depinfo.opDepList[i].second - depinfo.opDepList[i + 1].first;
            }
        }
    }
    if (maxOverlapIdx == -1) {
        return false;
    }
    return true;
}

std::string PipeSync::DumpMergeCVInfo(
    PipePair targetPp, int maxOverlapIdx, std::unordered_map<PipePair, DataDepInfo, PipePairHash> cvDepInfoMap)
{
    std::stringstream ss;
    ss << "\nMerge PipePair: " << GetPipeTypeDict().Find(targetPp.first.pipe) << " ";
    ss << GetPipeTypeDict().Find(targetPp.second.pipe) << "\n";
    ss << "    set1: " << cvDepInfoMap[targetPp].opDepList[maxOverlapIdx].first << "  ";
    ss << "wait1: " << cvDepInfoMap[targetPp].opDepList[maxOverlapIdx].second << "\n";
    ss << "    set2: " << cvDepInfoMap[targetPp].opDepList[maxOverlapIdx + 1].first << "  ";
    ss << "wait2: " << cvDepInfoMap[targetPp].opDepList[maxOverlapIdx + 1].second << "\n";
    return ss.str();
}

std::string PipeSync::DataDepInfo::DumpDataDepInfo(
    const std::vector<IndexOp>& syncedOpLog, std::vector<Operation*>& oriOpList)
{
    std::stringstream ss;
    ss << "    CV_SYNC_SRC magic: ";
    for (auto sync : setOpIdList) {
        ss << syncedOpLog[sync].second.get().GetOpMagic() << " ";
    }
    ss << "\n    CV EventID: ";
    for (auto id : setOpEventIdList) {
        ss << id << " ";
    }
    ss << "\n    Set Wait Pair: \n";
    for (auto [setidx, waitidx] : opDepList) {
        ss << "        " << setidx << " " << oriOpList[setidx]->GetOpMagic() << " "
           << oriOpList[setidx]->GetOpcodeStr();
        ss << "  " << waitidx << " " << oriOpList[waitidx]->GetOpMagic() << " " << oriOpList[waitidx]->GetOpcodeStr()
           << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpDepInfoMap(
    const std::vector<IndexOp>& syncedOpLog, std::unordered_map<PipePair, DataDepInfo, PipePairHash>& cvDepInfoMap)
{
    std::stringstream ss;
    ss << "\nCV DepInfoMap info:\n";
    for (auto& [pp, depinfo] : cvDepInfoMap) {
        ss << "  " << GetPipeTypeDict().Find(pp.first.pipe) << " " << GetPipeTypeDict().Find(pp.second.pipe) << "\n";
        ss << depinfo.DumpDataDepInfo(syncedOpLog, oriOpList_);
    }
    return ss.str();
}

Status PipeSync::RelaxCvEventIdMain(std::vector<IndexOp>& syncedOpLog, const CorePair& corePair)
{
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "CoreType: %s AIVCore: %d -> CoreType: %s AIVCore: %d has no eventid to use, start relax cv eventid.",
        GetCoreTypeDict().Find(corePair.first.first).c_str(), static_cast<int>(corePair.first.second),
        GetCoreTypeDict().Find(corePair.second.first).c_str(), static_cast<int>(corePair.second.second));
    // 收集所有的eventid, 避免已经被释放的eventid被重复统计(从后向前找可以保证被释放的eventid不被统计进去)
    std::vector<int> eventIdVec{};
    // core 和 aivcore已经保证相同，不需要再加入此信息
    std::unordered_map<PipePair, DataDepInfo, PipePairHash> cvDepInfoMap;
    // 找出所有当前遍历的corePair类型的CV_SYNC_SRC及对应的op信息
    FindCvSyncSrcInfo(syncedOpLog, eventIdVec, corePair, cvDepInfoMap);
    APASS_LOG_DEBUG_F(Elements::Operation, "%s", DumpDepInfoMap(syncedOpLog, cvDepInfoMap).c_str());

    // 遍历所有的depinfo, 找到依赖间重叠最大的一对
    PipeCoreReal pp1(PIPE_S, CoreType::AIV);
    PipeCoreReal pp2(PIPE_S, CoreType::AIV);
    PipePair targetPp{pp1, pp2};
    int maxOverlapIdx = -1;
    if (!(FindMaxOverlapForCV(targetPp, maxOverlapIdx, cvDepInfoMap))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot find mergeable setwait pair");
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "%s", DumpMergeCVInfo(targetPp, maxOverlapIdx, cvDepInfoMap).c_str());

    // 合并依赖
    PipeCoreRealEx setpp(cvDepInfoMap[targetPp].setp, cvDepInfoMap[targetPp].setc, cvDepInfoMap[targetPp].setaivc);
    PipeCoreRealEx waitpp(cvDepInfoMap[targetPp].waitp, cvDepInfoMap[targetPp].waitc, cvDepInfoMap[targetPp].waitaivc);
    if (SynDependency(maxOverlapIdx, cvDepInfoMap[targetPp], {setpp, waitpp}, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::RelaxCvEventId(std::vector<IndexOp>& syncedOpLog)
{
    for (const auto& corePair : cvCorePair) {
        // 该corepair类型已无可用eventid
        if (!(crossCoreFreeEventId_.count(corePair) != 0 && crossCoreFreeEventId_[corePair].size() == 0)) {
            continue;
        }
        if (RelaxCvEventIdMain(syncedOpLog, corePair) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RelaxCvEventId failed at RelaxCvEventIdMain.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog)
{
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    std::vector<PipePairEx> dataDepPairEx;
    for (const auto& pipePair : dataDepPair) {
        if (pipePair.first.core == CoreType::AIC) {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
            PipePairEx ppEx = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx);
        } else {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
            PipePairEx ppEx1 = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx1);
            PipeCoreRealEx pp3(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV1);
            PipeCoreRealEx pp4(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV1);
            PipePairEx ppEx2 = {pp3, pp4};
            dataDepPairEx.emplace_back(ppEx2);
        }
    }

    for (const auto& pipePairEx : dataDepPairEx) {
        if (HasFreeEventId(pipePairEx)) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePairEx, depInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDepInfo failed.");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePairEx, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
            return FAILED;
        }
    }

    // 合并CV同步释放cv eventid
    if (RelaxCvEventId(syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RelaxCvEventId failed.");
        return FAILED;
    }

    return SUCCESS;
}

bool PipeSync::GenSyncOp(PipeCoreRealEx set, PipeCoreRealEx wait, int eventId, bool isSet, Operation& op)
{
    if (set.core != wait.core) {
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3113) {
            op.SetOpCode(isSet ? Opcode::OP_CV_SYNC_SRC : Opcode::OP_CV_SYNC_DST);
        }
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && !isSet && wait.core == CoreType::AIV) {
            set.pipe = PipeType::PIPE_V;
        }
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (set.pipe != wait.pipe) {
        op.SetOpCode(isSet ? Opcode::OP_SYNC_SRC : Opcode::OP_SYNC_DST);
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (isSet || set.pipe == PipeType::PIPE_S) {
        return false;
    }
    // 同步相关的信息放在operation属性里
    op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
    if (set.core == CoreType::AIV) {
        if ((Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ||
            (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3113)) {
            return false;
        }
        op.SetOpCode(Opcode::OP_BAR_V);
        return true;
    }
    op.SetOpCode(Opcode::OP_BAR_M);
    return true;
}

Status PipeSync::GetEventId(const PipePairEx& pp, int& eventId)
{
    if (pp.first.pipe == pp.second.pipe && pp.first.core == pp.second.core) {
        // Pipe Barrier
        eventId = -1;
        return SUCCESS;
    }

    auto& eventQ = GetFreeEventIdQueue(pp);
    if (eventQ.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Eventid exhausted, GetEventId failed.");
        return FAILED;
    }

    eventId = eventQ.front();
    eventQ.pop_front();
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3113) {
        if (pp.first.core != pp.second.core) {
            PipePairEx ppReverse = {pp.second, pp.first};
            auto& eventQReverse = GetFreeEventIdQueue(ppReverse);
            eventQReverse.pop_front();
            if (eventQ.size() != eventQReverse.size()) {
                APASS_LOG_ERROR_F(Elements::Operation, "CV eventId queue size is not equal, GetEventId failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool PipeSync::HasFreeEventId(const PipePairEx& pp)
{
    std::deque<int>& eventQ = GetFreeEventIdQueue(pp);
    return !eventQ.empty();
}

bool PipeSync::BufOverlap(const TileRange& range1, int magic1, const TileRange& range2, int magic2) const
{
    APASS_LOG_DEBUG_F(
        Elements::Tensor, "Range 1 [%zu ~ %zu], range 2 [%zu ~ %zu].", range1.start, range1.end, range2.start,
        range2.end);
    if (range1.end > range2.start && range2.end > range1.start) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor %d and tensor %d have overlap.", magic1, magic2);
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor %d and tensor %d don't have overlap.", magic1, magic2);
    return false;
}

bool PipeSync::CheckWawDependency(const Operation& opSet, const Operation& opWait, size_t k, size_t idx)
{
    for (size_t setIdx = 0; setIdx < opSet.GetOOperands().size(); setIdx++) {
        for (size_t waitIdx = 0; waitIdx < opWait.GetOOperands().size(); waitIdx++) {
            if (opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[waitIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetOOperands()[setIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[waitIdx]->GetMagic();
            TileRange range1 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] : opSet.GetOOperands()[setIdx]->memoryrange;
            TileRange range2 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] : opWait.GetOOperands()[waitIdx]->memoryrange;
            if (BufOverlap(range1, magic1, range2, magic2)) {
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "%d %zu %s and %d %zu %s has WAW data dependency", opSet.GetOpMagic(), k,
                    opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckRawDependency(const Operation& opSet, const Operation& opWait, size_t k, size_t idx)
{
    for (size_t outIdx = 0; outIdx < opSet.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opWait.GetIOperands().size(); inIdx++) {
            if (opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opSet.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opWait.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opSet.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] : opWait.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] : opSet.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, magic1, range2, magic2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "%d %zu %s and %d %zu %s has RAW data dependency", opSet.GetOpMagic(), k,
                    opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckWarDependency(const Operation& opSet, const Operation& opWait, size_t k, size_t idx)
{
    for (size_t outIdx = 0; outIdx < opWait.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opSet.GetIOperands().size(); inIdx++) {
            if (opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] : opSet.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 =
                memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] : opWait.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, magic1, range2, magic2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "%d %zu %s and %d %zu %s has WAR data dependency", opSet.GetOpMagic(), k,
                    opSet.GetOpcodeStr().c_str(), opWait.GetOpMagic(), idx, opWait.GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::HasDataDependency(const Operation& opSet, const Operation& opWait, size_t k, size_t idx)
{
    std::string opSetStr = opSet.GetOpcodeStr();
    std::string opWaitStr = opWait.GetOpcodeStr();

    // check WAW
    bool checkWaw = true;
    auto setCfg = OpcodeManager::Inst().GetTileOpCfg(opSet.GetOpcode());
    auto waitCfg = OpcodeManager::Inst().GetTileOpCfg(opWait.GetOpcode());
    AdjustOpCfg(setCfg, opSet);
    AdjustOpCfg(waitCfg, opWait);
    if (waitCfg.pipeIdStart_ == setCfg.pipeIdStart_ &&
        (opSetStr.find("CUBE_A_MUL") == std::string::npos || opWaitStr.find("CUBE_A_MUL") == std::string::npos)) {
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

void PipeSync::UpdateDep(DepOp& currOp, DepOp& prevOp)
{
    PipeCoreRealEx currPipe(currOp.selfPipeCore.pipeStart, currOp.selfPipeCore.core, currOp.selfPipeCore.aivCore);
    PipeCoreRealEx prevPipe(prevOp.selfPipeCore.pipeEnd, prevOp.selfPipeCore.core, prevOp.selfPipeCore.aivCore);
    auto& currPipeDep = latestPipeDep_[currPipe];
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
            std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> prevSetPipes = prevPipeDepIter->second.setPipes;
            for (auto [prevSetPipeType, prevSetPipeIdx] : prevSetPipes) {
                auto res = currPipeDep.setPipes.emplace(prevSetPipeType, prevSetPipeIdx);
                // isExist == isPrevSetPipeTypeExist
                bool isExist = !res.second;
                size_t& existIdx = res.first->second;
                if (isExist && existIdx < prevSetPipeIdx) {
                    // overwrite dependency
                    existIdx = prevSetPipeIdx;
                }
            }
        }
    }
}

bool PipeSync::IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation*> opLogPtr)
{
    // true表示依赖关系可忽略，false表示依赖关系不可忽略
    // VIEW or ASSEMBLE data dependency can be ignored
    if (opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW_TYPE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[curr]->GetOpcode() == Opcode::OP_ASSEMBLE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_NOP || opLogPtr[curr]->GetOpcode() == Opcode::OP_NOP ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_HUB || opLogPtr[curr]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_RESHAPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_RESHAPE) {
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "%d %s and %d %s dependency is ignorable because op is VIEW or ASSEMBLE or NOP or HUB or RESHAPE",
            opLogPtr[prev]->GetOpMagic(), opLogPtr[prev]->GetOpcodeStr().c_str(), opLogPtr[curr]->GetOpMagic(),
            opLogPtr[curr]->GetOpcodeStr().c_str());
        return true;
    }
    return false;
}

// find depend op in opLog for 0 to idx
void PipeSync::FindDep(
    DepOp& op, const std::vector<Operation*> opLogPtr, size_t idx, DataDependencySearcher& dataDependencySearcher)
{
    const auto currOp = opLogPtr[idx];
    APASS_LOG_DEBUG_F(
        Elements::Operation, "=== OP: %d %zu %s ===", currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());
    // check dependency from latest op to oldest
    auto dataDependencySet = dataDependencySearcher.Find(currOp);
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        const Operation* prevAOp = opLogPtr[k];
        DepOp& prevOp = depOps_[k];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Current process ops: %d %zu %s and %d %zu %s", prevAOp->GetOpMagic(), k,
            prevAOp->GetOpcodeStr().c_str(), currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());

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

void PipeSync::InitCVEventIdQ(CorePair corePair)
{
    CorePair corePairReverse = {corePair.second, corePair.first};
    if (corePair.first.second == AIVCore::AIV0 || corePair.second.second == AIVCore::AIV0) {
        for (int i = 0; i < CROSS_CORE_EVENT_NUM; i++) {
            crossCoreFreeEventId_[corePair].push_back(i);
            crossCoreFreeEventId_[corePairReverse].push_back(i);
        }
    } else {
        for (int i = CROSS_CORE_EVENT_NUM; i < CROSS_CORE_EVENT_NUM * 2; i++) {
            crossCoreFreeEventId_[corePair].push_back(i);
            crossCoreFreeEventId_[corePairReverse].push_back(i);
        }
    }
}

std::deque<int>& PipeSync::GetFreeEventIdQueue(const PipePairEx& pp)
{
    // 若coretype不同，所有CV同步共享16个eventid
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3113) {
        if (pp.first.core != pp.second.core) {
            CorePair corePair = {{pp.first.core, pp.first.aivCore}, {pp.second.core, pp.second.aivCore}};
            if (crossCoreFreeEventId_.count(corePair) == 0) {
                InitCVEventIdQ(corePair);
            }
            return crossCoreFreeEventId_[corePair];
        }
    }
    if (freeEventId_.count(pp) == 0) {
        for (int i = 0; i < GetMaxEventId(pp); i++) {
            freeEventId_[pp].push_back(i);
        }
    }
    return freeEventId_[pp];
}

void PipeSync::AddPhaseOp1(
    Function& function, std::vector<Operation*> srcLog, std::vector<Operation*>& dstLog, size_t& i, size_t& prerun)
{
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
                Operation& phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE1, {input}, {output});
                Operation* phaseOpPtr = &phaseOp;
                dstLog.emplace_back(phaseOpPtr);
            }
            prerun++;
        }
        dstLog.emplace_back(srcLog[i]);
    }
}

void PipeSync::AddPhaseOp2(Function& function, std::vector<Operation*>& dstLog, size_t& prerun)
{
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE2, {input}, {output});
        Operation* phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
}

void PipeSync::PhaseKernelProcess(Function& function, std::vector<Operation*> srcLog, std::vector<Operation*>& dstLog)
{
    size_t prerun = 0;
    size_t i = 0;
    AddPhaseOp1(function, srcLog, dstLog, i, prerun);
    AddPhaseOp2(function, dstLog, prerun);
    for (; i < srcLog.size(); i++) {
        dstLog.emplace_back(srcLog[i]);
    }
}

void InsertSync::InsertPipeAll(Function* subGraphFunc)
{
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    std::vector<Operation*> newOpList;
    for (auto op : oriOpList) {
        newOpList.push_back(op);
        if (op->GetOpcode() == Opcode::OP_RESHAPE || op->GetOpcode() == Opcode::OP_VIEW ||
            op->GetOpcode() == Opcode::OP_VIEW_TYPE || op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = subGraphFunc->AddRawOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {input}, {output});
        syncOp.syncQueue_ = {PipeType::PIPE_ALL,   PipeType::PIPE_ALL,  CoreType::AIV, CoreType::AIV, -1,
                             AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED};
        newOpList.push_back(&syncOp);
    }
    subGraphFunc->ScheduleBy(newOpList, true);
    subGraphFunc->oriOpList = oriOpList;
}

Status InsertSync::CheckNewOpListSeq(const std::vector<Operation*>& oriOpList, const std::vector<Operation*>& opListNew)
{
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
        APASS_LOG_ERROR_F(
            Elements::Operation, "NewOpList sequence is not equal to OriOpList sequence, CheckNewOpListSeq failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InsertSync::GenNewOpList(Function* subGraphFunc, std::vector<Operation*>& opListNew)
{
    PipeSync ps;
    std::vector<Operation*> syncedOpLogPtr;
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
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
    subGraphFunc->setOpMap = ps.setOpMap;
    subGraphFunc->waitOpMap = ps.waitOpMap;
    subGraphFunc->oriOpList = ps.GetOriOpList();
    return SUCCESS;
}

Status InsertSync::InsertSyncMainLoop(Function* subGraphFunc)
{
    if (enableDebug_) {
        InsertPipeAll(subGraphFunc);
        return SUCCESS;
    }
    std::vector<Operation*> opListNew;
    if (GenNewOpList(subGraphFunc, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSyncMainLoop failed at GenNewOpList.");
        return FAILED;
    }
    subGraphFunc->ScheduleBy(opListNew, true);
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "==========================================================================================");
    for (const auto& op : subGraphFunc->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("SYNC_SRC") != std::string::npos ||
            op->GetOpcodeStr().find("SYNC_DST") != std::string::npos || op->GetOpcode() == Opcode::OP_BAR_V ||
            op->GetOpcode() == Opcode::OP_BAR_M) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Output operation %d: %s, setpipe type: %s, setcore type: %s, waitpipe type: %s, waitcore type: %s, "
                "eventid: %d, setAIVcore type: %d, waitAIVcore type: %d",
                op->GetOpMagic(), op->GetOpcodeStr().c_str(), GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.coreType_).c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.trigCoreType_).c_str(), op->syncQueue_.eventId_,
                static_cast<int>(op->syncQueue_.setAivCore_), static_cast<int>(op->syncQueue_.waitAivCore_));
            continue;
        }
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Output operation %d: %s, AIV core type: %d", op->GetOpMagic(),
            op->GetOpcodeStr().c_str(), static_cast<int>(op->GetAIVCore()));
    }
    return SUCCESS;
}

// regist pass
Status InsertSync::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Start InsertSync.");
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
    uint64_t index = 0;
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto& subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(static_cast<unsigned>(leafFuncSize), hardwareConcurrency);
    std::vector<std::thread> workers;
    bool status{true};
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, leafFuncSize, &index, this, &status] {
            for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize;
                 idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
                auto program = subPrograms[idx];
                APASS_LOG_DEBUG_F(
                    Elements::Operation,
                    "====================================Program %zu ===========================================",
                    index);
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
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Single Thread scenario.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Multiple Threads scenario.");
        return FAILED;
    }

    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Finish InsertSync.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu

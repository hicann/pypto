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
 * \file l1_copy_reuse.cpp
 * \brief
 */

#include "l1_copy_reuse.h"

namespace npu::tile_fwk {
inline std::vector<uint64_t> GetGMInputFeature(const Operation &op) { // 提取GM tensor的特征
    auto ioperand = op.GetIOperands()[0];
    if (ioperand == nullptr) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op %s %d ioperand is nullptr.", op.GetOpcodeStr(), op.GetOpMagic());
        return {};
    }
    std::vector<uint64_t> vec = {static_cast<uint64_t>(ioperand->GetRawTensor()->GetRawMagic())};
    std::vector<OpImmediate> opImmList;
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        std::shared_ptr<ViewOpAttribute> attr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        opImmList = OpImmediate::Specified(attr->GetFromTensorOffset());
    } else if (op.GetOpcode() == Opcode::OP_CONVERT) {
        auto inputOffset = op.GetIOperands().front()->GetOffset();
        for (size_t i = 0; i < op.oOperand.front()->shape.size(); i++) {
            opImmList.push_back(OpImmediate::Specified(SymbolicScalar(inputOffset[i])));
        }
    }
    for (auto &opImm : opImmList) {
        auto offset = opImm.GetSpecifiedValue();
        if (offset.ConcreteValid()) {
            vec.push_back(offset);
            continue;
        }
        std::hash<std::string> hasher;
        auto offsetHash = hasher(opImm.Dump());
        vec.push_back(static_cast<uint64_t>(offsetHash));
    }
    for (auto &dim : op.GetOOperands()[0]->GetShape()) {
        vec.push_back(dim);
    }
    vec.push_back(static_cast<int>(op.GetOpcode()));
    return vec;
}

inline bool CanReuse(const Operation &op) {
    if (op.GetIOperands().size() != 0 && op.GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && 
        op.GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_CONVERT) {
            return true;
        }
    }
    return false;
}

// key : 需要被删除的copyin op, value: 保留的copyin op
Status L1CopyInReuseRunner::GetDuplicateOps(std::vector<Operation *> &opOriList,
                                            const std::vector<int> &opIdx) {
    std::map<std::vector<uint64_t>, int> tensor2Op;
    replacedCopyMap_.clear();
    tensormagic2Op_.clear();
    for (auto i : opIdx) {
        if (!CanReuse(*opOriList[i])) {
            continue;
        }
        auto outputMagic = opOriList[i]->GetOOperands()[0]->GetRawTensor()->GetRawMagic();
        auto feature = GetGMInputFeature(*opOriList[i]);
        if (feature.size() == 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDuplicateOps: op %s %d GetGMInputFeature failed. %s", 
                            opOriList[i]->GetOpcodeStr(), opOriList[i]->GetOpMagic(), GetFormatBacktrace(*opOriList[i]).c_str());
            return FAILED;
        }
        if (tensor2Op.find(feature) != tensor2Op.end() && tensor2Op[feature] != i) {
            replacedCopyMap_[i] = tensor2Op[feature];
            tensormagic2Op_[outputMagic] = tensor2Op[feature];
            continue;
        }
        tensor2Op[feature] = i;
    }
    return SUCCESS;
}

void L1CopyInReuseRunner::TackleOp(int i, Operation *op, 
                                   std::vector<std::vector<int>> &replacedInputs, 
                                   std::vector<std::vector<int>> &replacedOutputs) {
    if (CanReuse(*op)) {
        auto allocedL1BufId = op->GetOOperands()[0]->GetRawTensor()->GetRawMagic();
        if (tensormagic2Op_.find(allocedL1BufId) != tensormagic2Op_.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Remove useless op [%d, %s].", op->GetOpMagic(), op->GetOpcodeStr().c_str());
            op->SetAsDeleted();
        }
        return;
    }
    for (size_t k = 0; k < op->GetIOperands().size(); k++) {
        auto ioperandID = op->GetIOperands()[k]->GetRawTensor()->GetRawMagic();
        if (tensormagic2Op_.find(ioperandID) != tensormagic2Op_.end()) {
            replacedInputs.push_back({i, static_cast<int>(k), tensormagic2Op_[ioperandID], 0});
        }
    }
    // 这里需要处理控制依赖。
    for (size_t k = 0; k < op->GetOOperands().size(); k++) {
        auto ioperandID = op->GetOOperands()[k]->GetRawTensor()->GetRawMagic();
        if (tensormagic2Op_.find(ioperandID) != tensormagic2Op_.end()) {
            replacedOutputs.push_back({i, static_cast<int>(k), tensormagic2Op_[ioperandID], 0});
        }
    }
}

void GetOriList(Function &func, std::vector<Operation *> &oriList) {
    for (auto &op : func.Operations()) {
        oriList.emplace_back(&op);
    }
}

void L1CopyInReuseRunner::MergeProcessIdUpdate(Function &func, std::vector<std::vector<int>> &colorNode, int color) {
    std::vector<Operation *> oriList;
    GetOriList(func, oriList);
    int colorCount = 0;
    for (int j = 0; j < color; j++) {
        if (colorNode[j].empty()) {
            continue;
        }
        colorCount++;
        for (int i : colorNode[j]) {
            oriList[i]->UpdateSubgraphID(colorCount-1);
        }
    }
    func.SetTotalSubGraphCount(colorCount);
}

// 合并重复的L1_COPY_IN和L1_ALLOC节点
Status L1CopyInReuseRunner::MergeDupL1CopyIn(Function &func, std::vector<std::vector<int>> &colorNode, int color) {
    std::vector<Operation *> oriList;
    GetOriList(func, oriList);
    for (int j = 0; j < color; j++) {
        if (colorNode[j].empty()) {
            continue;
        }
        std::sort(colorNode[j].begin(), colorNode[j].end());
        if (GetDuplicateOps(oriList, colorNode[j]) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "MergeDupL1CopyIn: GetDuplicateOps failed.");
            return FAILED;
        }
        std::vector<std::vector<int>> replacedInputs, replacedOutputs;
        for (int i : colorNode[j]) {
            L1CopyInReuseRunner::TackleOp(i, oriList[i], replacedInputs, replacedOutputs);
        }
        // 重新连边
        for (auto &replacedInput : replacedInputs) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Relink op [%d] input [%d] to op [%d] output [%d].", 
                            oriList[replacedInput[0]]->GetOpMagic(), replacedInput[1],
                            oriList[replacedInput[2]]->GetOpMagic(), replacedInput[3]);
            FunctionUtils::RelinkOperationInput(oriList[replacedInput[0]], replacedInput[1],
                                                oriList[replacedInput[2]], replacedInput[3]);
        }
        for (auto &replacedOutput : replacedOutputs) {
            auto rewriteOp = oriList[replacedOutput[0]];
            auto copyinOp = oriList[replacedOutput[2]];
            if (!func.TensorReuse(rewriteOp->GetOOperands()[replacedOutput[1]], copyinOp->GetOOperands()[0])) {
                APASS_LOG_ERROR_F(Elements::Operation, "MergeDupL1CopyIn: TensorReuse failed!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

int L1CopyInReuseRunner::GetMaxInColor(const std::vector<int> &nodes, 
                                       const OperationsViewer &opOriList, 
                                       int curColor) {
    int maxInColor = -1;
    for (int j : nodes) {
        for (int k : inGraph[j]) {
            auto opColor = opOriList[k].GetSubgraphID();
            if (opColor != curColor) {
                maxInColor = std::max(maxInColor, opColor);
            }
        }
    }
    return maxInColor;
}

inline std::vector<int> GetCopyIn(const OperationsViewer &opOriList, 
                                  int color, 
                                  std::vector<std::vector<int>> &colorNode) {
    // 获取子图L1CopyIn数据量
    std::vector<int> colorCopyIn(color, 0);
    for (int i = 0; i < color; i++) {
        for (int j : colorNode[i]) {
            if (CanReuse(opOriList[j])) {
                int volume = BytesOf(opOriList[j].GetOOperands()[0]->Datatype());
                for (auto &k : opOriList[j].GetOOperands()[0]->GetShape()) {
                    volume *= k;
                }
                colorCopyIn[i] = colorCopyIn[i] + volume;
            }
        }
    }
    return colorCopyIn;
}

void L1CopyInReuseRunner::GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    const uint64_t mod = UINT64_MAX;
    uint64_t hash = 0;
    for (char c : op) {
        hash = (hash * p + static_cast<uint64_t>(c)) % mod;
    }
    for (int j : inGraph[idx]) {
        hash = (hash * p + (hashList[j] ^ a)) % mod;
    }
    hashList[idx] = hash;
}

void L1CopyInReuseRunner::GetColorHash(const OperationsViewer &opOriList, std::vector<uint64_t> &hashColor) {
    std::vector<uint64_t> hashTileOp(opOriList.size(), 0);
    for (size_t i = 0; i < opOriList.size(); i++) {
            GetOpHash(hashTileOp, opOriList[i].GetOpcodeStr(), i);
    }
    uint64_t a = 0x12345678;
    uint64_t p = 23;
    const uint64_t mod = UINT64_MAX;
    std::set<int32_t> mulaccGraph;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        if (CanReuse(opOriList[i])) {
            mulaccGraph.insert(opOriList[i].GetSubgraphID());
        }
        hashColor[opOriList[i].GetSubgraphID()] = (hashColor[opOriList[i].GetSubgraphID()] * p + (hashTileOp[i] ^ a)) % mod;
    }
    int order = 0;
    for (int i : mulaccGraph) {
        hashMap[hashColor[i]].push_back(i);
        if (hashMap[hashColor[i]].size() == 1) {
            hashOrder[hashColor[i]] = order;
            order++;
        }
    }
    for (auto& entry : hashMap) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, Subgraph ID: %s.", entry.first, IntVecToStr(entry.second).c_str());
    }
    for (auto& entry : hashOrder) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, Hash order: %d.", entry.first, entry.second);
    }
}

inline void HashUpdate(std::unordered_map<uint64_t, std::vector<int>> &hashMap, 
                       std::unordered_map<uint64_t, int> &hashOrder, 
                       int color, std::vector<uint64_t> hashColor) {
    // 更新子图哈希
    for (auto entry = hashMap.begin(); entry != hashMap.end();) {
        if (entry->second.empty()) {
            entry = hashMap.erase(entry);
            continue;
        }
        entry++;
    }
    hashOrder.clear();
    int order = 0;
    for (int i = 0; i < color; i++) {
        if (hashMap.find(hashColor[i]) != hashMap.end() && hashOrder.find(hashColor[i]) == hashOrder.end()) {
            hashOrder[hashColor[i]] = order;
            order++;
        }
    }
    for (auto& entry : hashMap) {
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph hash: %lu, Subgraph ID: %s.", 
                      entry.first, IntVecToStr(entry.second).c_str());
    }
    for (auto& entry : hashOrder) {
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph hash: %lu, Hash order: %d.", 
                      entry.first, entry.second);
    }
}

Status L1CopyInReuseRunner::SetNumLR(std::vector<int> &numLRList) {
    numLRList.assign(hashMap.size(), numLR);
    for (auto &entry : numLRMap) {
        int i = entry.first;
        if (i >= 0 && i < static_cast<int>(hashMap.size())) {
            for (auto& [hashcolor, order] : hashOrder) {
                if (order != i) continue;
                auto itHashMap = hashMap.find(hashcolor);
                if (itHashMap == hashMap.end()) {
                    APASS_LOG_ERROR_F(Elements::Config, "entry %ld not fount in hashMap.", hashcolor);
                    return FAILED;
                }
                if (entry.second < 0) {
                    APASS_LOG_ERROR_F(Elements::Config, "Invalid merge count for "
                                        "Subgraph hash %ld: merge count=%ld, please check.", hashcolor, entry.second);
                    return FAILED;
                }
                numLRList[i] = entry.second;
            }
            continue;
        }
        APASS_LOG_WARN_F(Elements::Config, "Invalid subgraph ID: %d in cubeL1ReuseSetting, ignored.", i);
    }
    return SUCCESS;
}

Status L1CopyInReuseRunner::L1MergeProcess(OperationsViewer &opOriList, std::vector<std::vector<int>> &colorNode,
                                           std::vector<uint64_t> &hashColor, std::vector<int> &colorCopyIn,
                                           std::map<std::vector<uint64_t>, int> &l1InputList, int &tmpColor,
                                           std::vector<int> &mergedNum, int &i) {
    for (auto opIdx : colorNode[i]) {
        if (!CanReuse(opOriList[opIdx])) {
            continue;
        }
        auto vec = GetGMInputFeature(opOriList[opIdx]);
        if (vec.size() == 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "L1MergeProcess: op %d %s GetGMInputFeature failed. %s", 
                            opOriList[i].GetOpMagic(), opOriList[i].GetOpcodeStr(), GetFormatBacktrace(opOriList[i]).c_str());
            return FAILED;
        }
        l1InputList[vec] = tmpColor;
    }  // 记录当前子图所有的L1_COPY_IN搬入的tensor特征
    if (tmpColor != i) {
        for (auto t : colorNode[i]) {
            opOriList[t].UpdateSubgraphID(tmpColor);
            colorNode[tmpColor].push_back(t);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph merge: %lu, %lu.", i, tmpColor);
        colorNode[i].clear();
        colorCopyIn[tmpColor] = colorCopyIn[tmpColor] + colorCopyIn[i];
        mergedNum[i] = 0;
        mergedNum[tmpColor] += 1;
        hashMap[hashColor[i]].erase(std::find(hashMap[hashColor[i]].begin(), hashMap[hashColor[i]].end(), i));
        hashMap[hashColor[tmpColor]].erase(std::find(hashMap[hashColor[tmpColor]].begin(), 
                                                        hashMap[hashColor[tmpColor]].end(), 
                                                        tmpColor));
        hashColor[tmpColor] += hashColor[i];
        hashColor[i] = 0;
        hashMap[hashColor[tmpColor]].push_back(tmpColor);
    }  // 合入子图
    return SUCCESS;
}

Status L1CopyInReuseRunner::Phase1(Function &func, int color, std::vector<std::vector<int>> &colorNode,
                                            std::vector<int> &colorCopyIn, std::vector<uint64_t> &hashColor) {
    // 针对matmul的L1 copy reuse进行子图合并
    auto opOriList = func.Operations();
    std::map<std::vector<uint64_t>, int> l1InputList;
    std::vector<int> numLRList;
    //CubeL1ReuseMode
    if (SetNumLR(numLRList) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Config, "Invalid configuration: %s.", "cubeL1ReuseSetting");
        return FAILED;
    } 
    std::vector<int> mergedNum(color, 1);
    for (int i = 0; i < color; i++) {
        int tmpColor = -1;
        auto maxInColor = GetMaxInColor(colorNode[i], opOriList, i);
        size_t j = 0;
        while (colorCopyIn[i] <= mgCopyInUpperBound && j < colorNode[i].size()) {
            auto opIdx = colorNode[i][j];
            if (!CanReuse(opOriList[opIdx])) {
                j++;
                continue;
            }
            auto vec = GetGMInputFeature(opOriList[opIdx]);
            if (vec.size() == 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "Phase1: op %s %d GetGMInputFeature failed. %s", 
                                opOriList[i].GetOpcodeStr(), opOriList[i].GetOpMagic(), GetFormatBacktrace(opOriList[i]).c_str());
                return FAILED;
            }
            auto copyId = l1InputList.find(vec);
            if (copyId != l1InputList.end() && copyId->second >= maxInColor && 
                colorCopyIn[copyId->second] + colorCopyIn[i] <= mgCopyInUpperBound &&
                mergedNum[copyId->second] > 0 && mergedNum[copyId->second] < numLRList[hashOrder[hashColor[i]]]) {
                tmpColor = copyId->second;
                break;
            }
            j++;
        }
        if (tmpColor == -1) {
            tmpColor = i;
        }
        if (L1MergeProcess(opOriList, colorNode, hashColor, colorCopyIn, l1InputList, tmpColor, mergedNum, i) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "L1MergeProcess failed; Please check the L1MergeProcess method. %s", GetFormatBacktrace(opOriList[i]).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status L1CopyInReuseRunner::SetNumDB(std::vector<int> &hashMergeNum) {
    if (numDBMap.size() == 0 && cubeNBufferMode == 1) {
        hashMergeNum.assign(hashMap.size(), -1);
    }
    auto numDB = numDBMap.find(-1);
    if (numDB != numDBMap.end()) {
        hashMergeNum.assign(hashMap.size(), numDB->second);
    }
    for (auto &entry : numDBMap) {
        int i = entry.first;
        if (i >= 0 && i < static_cast<int>(hashMap.size())) {
            for (auto& [hashcolor, order] : hashOrder) {
                if (order != i) continue;
                auto itHashMap = hashMap.find(hashcolor);
                if (itHashMap == hashMap.end()) {
                    APASS_LOG_ERROR_F(Elements::Config, "entry %ld not fount in hashMap.", hashcolor);
                    return FAILED;
                }
                if (entry.second < 1) {
                    APASS_LOG_ERROR_F(Elements::Config, "Invalid merge count for "
                                        "Subgraph hash %ld: merge count=%ld, please check.", hashcolor, entry.second);
                    return FAILED;
                }
                hashMergeNum[i] = entry.second;
            }
            continue;
        }
        APASS_LOG_WARN_F(Elements::Config, "Invalid subgraph ID: %d in cubeNBufferSetting, ignored.", i);
    }
    return SUCCESS;
}

inline std::vector<int> AdjustNumDBCore(int color, int numDB, int mx) {
    std::vector<int> pingColorList(color, 1);
    if (numDB == -1) {
        int coreNum = Platform::Instance().GetSoc().GetAICCoreNum();
        if (coreNum == 0) {
            APASS_LOG_WARN_F(Elements::Config, "Failed to get number of cores. CubeMergeProcess will be ignored.");
            pingColorList.assign(color, 0);
            return pingColorList;
        }
        int rm = color % (mx * coreNum);
        if (color <= (mx + 1) * coreNum) {
            coreNum += coreNum;
            rm = color;
        }
        else {
            for (int i = 0; i < color - rm; i += mx) {
                pingColorList[i] = 0;
            }
        }
        for (int i = 0; i < rm % coreNum; i++) {
            pingColorList[color - rm + i * ((rm + coreNum - 1) / coreNum)] = 0;
        }
        if (rm >= coreNum) {
            for (int i = 1; i < coreNum + 1 - rm % coreNum; i++) {
                pingColorList[color - i * (rm / coreNum)] = 0;
            }
        }
        return pingColorList;
    }
    numDB = std::min(numDB, mx);
    int numMerged = (color + numDB - 1) / numDB;
    for (int i = 0; i < numMerged; i++) {
        pingColorList[numDB * i] = 0;
    }
    return pingColorList;
}

void L1CopyInReuseRunner::CubeMergeProcess(std::vector<std::vector<int>> &colorNode, OperationsViewer &opOriList,
                                           std::vector<int> &hashMergeNum, std::vector<int> &colorCopyIn) {
    for (auto &entry : hashMap) {
        uint64_t colorHashValue = entry.first;
        std::vector<int> &colorValues = entry.second;
        int sz = colorCopyIn[colorValues[0]];
        if (sz > mgCopyInUpperBound) {
            continue;
        }
        int pingColor = -1;
        int mxMerge = mgCopyInUpperBound / sz;
        std::vector<int> pingColorList = AdjustNumDBCore(colorValues.size(), hashMergeNum[hashOrder[colorHashValue]], mxMerge);
        for (size_t i = 0; i < colorValues.size(); i++) {
            if (pingColorList[i] == 0) {
                pingColor = colorValues[i];
                continue;
            }
            int pongColor = colorValues[i];
            for (auto opIdxMergedDB : colorNode[pongColor]) {
                opOriList[opIdxMergedDB].UpdateSubgraphID(pingColor);
                colorNode[pingColor].push_back(opIdxMergedDB);
            }
            APASS_LOG_INFO_F(Elements::Operation, "Subgraph merge: %lu, %lu.", pingColor, pongColor);
            colorNode[pongColor].clear();
        }
    }
}

Status L1CopyInReuseRunner::Run(Function &func, int color, std::vector<std::vector<int>> &colorNode) {
    auto opOriList = func.Operations();
    std::vector<uint64_t> hashColor(color, 0);
    GetColorHash(opOriList, hashColor);   // 计算子图哈希，识别同构子图
    auto colorCopyIn = GetCopyIn(opOriList, color, colorNode);   // 记录各子图的大小
    mgCopyInUpperBound = func.paramConfigs_.sgMgCopyInUpperBound;
    numLR = func.paramConfigs_.l1ReuseNum;
    numLRMap = func.paramConfigs_.cubeL1ReuseSetting;
    numDBMap = func.paramConfigs_.cubeNBufferSetting;    // 合并阈值参数设置
    cubeNBufferMode = func.paramConfigs_.cubeNBufferMode;
    APASS_LOG_INFO_F(Elements::Operation, "Param Setting numLR %d, mgCopyInUpperBound %d.", numLR, mgCopyInUpperBound);
    if (numLR != 0 || numLRMap.size() != 0) {
        if (Phase1(func, color, colorNode, colorCopyIn, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Function, "Phase1 failed; Please check the Phase1 method.");
            return FAILED;
        }
        if (MergeDupL1CopyIn(func, colorNode, color) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Function, "Run: MergeDupL1CopyIn failed.");
            return FAILED;
        }
        HashUpdate(hashMap, hashOrder, color, hashColor);
    }
    std::vector<int> hashMergeNum(hashMap.size(), 1);  
    //NBuffer参数设置
    if (SetNumDB(hashMergeNum) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Config, "Invalid configuration: %s.", "cubeNBufferSetting");
        return FAILED;
    } 
    CubeMergeProcess(colorNode, opOriList, hashMergeNum, colorCopyIn);
    MergeProcessIdUpdate(func, colorNode, color);
    for (auto &op : func.Operations()) {
        if (static_cast<size_t>(op.GetSubgraphID()) > func.GetTotalSubGraphCount()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Run: op SubGraph ID %d out of range. %s", op.GetSubgraphID(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    RemoveUselessViews(func); //删除节点
    func.EraseOperations(true);
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

void L1CopyInReuseRunner::RemoveUselessViews(Function &func) const {
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && op.GetIOperands().size() == 1 && op.GetOOperands().size() == 1) {
            auto input = op.GetIOperands()[0];
            auto output = op.GetOOperands()[0];
            if (func.IsFromInCast(input) || func.IsFromOutCast(output)) {
                continue;
            }
            auto iOperandMem = input->GetMemoryTypeOriginal();
            auto oOperandMem = output->GetMemoryTypeOriginal();
            if (iOperandMem == MemoryType::MEM_DEVICE_DDR && oOperandMem == MemoryType::MEM_DEVICE_DDR) {
                bool hasNoConsumer{true};
                for (auto consumer : output->GetConsumers()) {
                    if (!(consumer->IsDeleted()) && consumer->BelongTo() == &func) {
                        hasNoConsumer = false;
                    }
                }
                if (hasNoConsumer) {
                    op.SetAsDeleted();
                }
            }
        }
    }
}

Status L1CopyInReuseMerge::InitColorNode(Function &func, std::vector<std::vector<int>> &colorNode) const {
    int colorMax{0};
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (CanReuse(opOriList[i])) {
            auto feature = GetGMInputFeature(opOriList[i]);
            if (feature.size() == 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "Get Feature FAILED. %s", GetFormatBacktrace(opOriList[i]).c_str());
                return FAILED;
            }
            APASS_LOG_INFO_F(Elements::Operation, "Op %d feature: %s.", i, IntVecToStr(feature).c_str());
        }
        auto opColor = opOriList[i].GetSubgraphID();
        if (opColor > colorMax) {
            colorMax = opColor;
        }
    }
    int color = colorMax + 1;
    colorNode.resize(color);
    for (size_t i = 0; i < opOriList.size(); i++) {
        auto opColor = opOriList[i].GetSubgraphID();
        colorNode[opColor].push_back(i);
    }
    return SUCCESS;
}

Status L1CopyInReuseMerge::CheckOpListValid(Function &func) const {
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetIOperands().size() != 0 && opOriList[i].GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && 
        opOriList[i].GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            if (opOriList[i].GetOpcode() == Opcode::OP_VIEW || opOriList[i].GetOpcode() == Opcode::OP_CONVERT) {
                //符合预期且合法
                continue;
            } else if (opOriList[i].GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                //预期之外，先放行，安排计划评审修复
                continue;
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "Unexpected operation %s. "
                    "Please check if the operation is within the expected range", opOriList[i].Dump().c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status L1CopyInReuseMerge::L1CopyInReuse(Function &func) const {
    auto numLR = func.paramConfigs_.l1ReuseNum;
    auto numLRMap = func.paramConfigs_.cubeL1ReuseSetting;
    auto numDBMap = func.paramConfigs_.cubeNBufferSetting;
    auto cubeNBufferMode = func.paramConfigs_.cubeNBufferMode;
    APASS_LOG_INFO_F(Elements::Config, "L1 Reuse Setting: %d", numLR);
    if (numLR < 0) {
        APASS_LOG_ERROR_F(Elements::Config, 
                            "Invalid parameters: l1ReuseNum must be >= 0. "
                            "(got l1ReuseNum=%d)", numLR);
        return FAILED;
    }
    if (numLR == 0 && cubeNBufferMode == 0 && numLRMap.size() == 0 && numDBMap.size() == 0) {
        APASS_LOG_INFO_F(Elements::Config, "Init Param default.");
        return SUCCESS;
    }
    std::vector<std::vector<int>> colorNode;
    if (CheckOpListValid(func) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckOpListValid failed; Please check the operation is valid.");
        return FAILED;
    }
    if (InitColorNode(func, colorNode) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "InitColorNode failed; Please check the InitColorNode method.");
        return FAILED;
    }

    std::vector<Operation *> opList;
    for (auto &op : func.Operations()) {
        opList.emplace_back(&op);
    }
    auto inOutGraph = RescheduleUtils::GetInOutGraphs(opList, func.GetFuncMagic());
    auto &inGraph = inOutGraph[0];
    L1CopyInReuseRunner runner(inGraph);
    if (runner.Run(func, colorNode.size(), colorNode) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "L1CopyInReuse: Run failed.");
        return FAILED;
    }
    return SUCCESS;
}

void L1CopyInReuseMerge::DoHealthCheckAfter(Function &function, const std::string &folderPath) {
    APASS_LOG_INFO_F(Elements::Function, "After L1CopyInReuseMerge, Health Report: TileGraph START.");
    std::string fileName = GetDumpFilePrefix(function);
    HealthCheckTileGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Function, "After L1CopyInReuseMerge, Health Report: TileGraph END.");
}
}  // namespace npu::tile_fwk

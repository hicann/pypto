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
 * \file n_buffer_merge.cpp
 * \brief
 */

#include "n_buffer_merge.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "interface/utils/log.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_log/pass_log.h"
#include <climits>

#define MODULE_NAME "NBufferMerge"

namespace npu::tile_fwk {

void NBufferMerge::GetOpHash(std::vector<uint64_t> &hashList, const std::string op, size_t idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    const uint64_t mod = 0xFFFFFFFFFFFFF;
    uint64_t hash = 0;
    for (char c : op) {
        hash = (hash * p + static_cast<uint64_t>(c)) % mod;
    }
    for (int j : inGraph_[idx]) {
        hash = (hash * p + (hashList[j] ^ a)) % mod;
    }
    hashList[idx] = hash;
}

void NBufferMerge::GetOpHashReverse(std::vector<uint64_t> &hashList, const std::string op, int idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    const uint64_t mod = 0xFFFFFFFFFFFFF;
    uint64_t hash = 0;
    for (char c : op) {
        hash = (hash * p + static_cast<uint64_t>(c)) % mod;
    }
    for (int j : outGraph_[idx]) {
        hash = (hash * p + (hashList[j] ^ a)) % mod;
    }
    hashList[idx] = hash;
}

void UpdateOpColor(OperationsViewer &opOriList, 
                   int &color, 
                   std::vector<int> &colorCycles,
                   std::vector<std::vector<int>> &colorNode) {
    std::vector<int> oriColor2NewColor(color);
    int colorCount = 0;
    for (int i = 0; i < color; i++) {
        if (colorCycles[i] != 0) {
            oriColor2NewColor[i] = colorCount;
            colorCount++;
        }
        colorCycles[i] = 0;
        colorNode[i].clear();
    }
    color = colorCount;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        opOriList[i].UpdateSubgraphID(oriColor2NewColor[opOriList[i].GetSubgraphID()]);
    }
}

Status NBufferMerge::ColorTopo(int &color1, 
                                   std::vector<std::vector<int>> &inputColor, 
                                   std::vector<std::vector<int>> &outputColor, 
                                   OperationsViewer &opOriList) {
    std::vector<int> colorQueue(color1);
    std::vector<int> colorInDegree(color1);
    int colorQueueHead = 0;
    int colorQueueTail = 0;
    for (int i = 0; i < color1; i++) {
        colorInDegree[i] = inputColor[i].size();
        if (colorInDegree[i] == 0) {
            // 找到入度为0的color作为queue的起始点
            colorQueue[colorQueueTail++] = i;
        }
    }
    // 从入度为0的点开始，不断解依赖，如果没有成环，那么遍历完所有color，不应该存在入度不为0的color
    while (colorQueueHead < colorQueueTail) {
        int i = colorQueue[colorQueueHead++];
        for (int j : outputColor[i]) {
            colorInDegree[j]--;
            if (colorInDegree[j] == 0) {
                 colorQueue[colorQueueTail++] = j;
            }
        }
    }
    // 这里1.0中遍历了前color个节点，这里修改成遍历所有的color
    for (int i = 0; i < color1; i++) {
        if (colorInDegree[i] != 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Color [%d] has cycle in graph; Please check and adjust the merge method.", i);
            return FAILED;
        }
    }
    std::vector<int> colorQueueReverse(color1);
    for (int i = 0; i < color1; i++) {
        colorQueueReverse[colorQueue[i]] = i;
    }
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        opOriList[i].UpdateSubgraphID(colorQueueReverse[opOriList[i].GetSubgraphID()]);
    }
    return SUCCESS;
}

Status NBufferMerge::CheckAndFixColorOrder(OperationsViewer &opOriList, 
                                            int &color1, 
                                            std::vector<int> &colorCycles1,
                                            std::vector<std::vector<int>> &colorNode1) {
    UpdateOpColor(opOriList, color1, colorCycles1, colorNode1);
    // 颜色拓扑排序
    std::vector<std::vector<int>> inputColor(color1);
    std::vector<std::vector<int>> outputColor(color1);
    for (size_t i = 0; i < opOriList.size(); i++) {
        for (int j : outGraph_[i]) {
            if (opOriList[i].GetSubgraphID() < 0 || opOriList[j].GetSubgraphID() < 0) {
                continue;
            }
            if (opOriList[i].GetSubgraphID() != opOriList[j].GetSubgraphID()) {
                outputColor[opOriList[i].GetSubgraphID()].push_back(opOriList[j].GetSubgraphID());
                inputColor[opOriList[j].GetSubgraphID()].push_back(opOriList[i].GetSubgraphID());
            }
        }
    }
    if (ColorTopo(color1, inputColor, outputColor, opOriList) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "ColorTopo failed; Please check the ColorTopo method.");
        return FAILED;
    }
    // 重新统计colorNode等
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorCycles1[opOriList[i].GetSubgraphID()] += opOriList[i].GetLatency();
        colorNode1[opOriList[i].GetSubgraphID()].push_back(i);
    }
    return SUCCESS;
}

void NBufferMerge::InitParam(OperationsViewer &opOriList) {
    std::vector<std::mutex> subgraphMtx(color_);
    std::vector<std::mutex> inColorMtx(color_);
    std::vector<std::mutex> outColorMtx(color_);
    ParallelTool::Instance().Parallel_for(0, opOriList.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for (int i = st; i < et; i++) {
            // 过滤FromInCast节点和NOP节点
            if (opOriList[i].GetSubgraphID() < 0) {
                continue;
            }
            int subgraphId = opOriList[i].GetSubgraphID();
            {
                std::unique_lock lock(subgraphMtx.at(subgraphId));
                colorCycles_[subgraphId] += opOriList[i].GetLatency();
                colorNode_[subgraphId].push_back(i);
            }
            for (const auto inputNode : opOriList[i].ProducerOps()) {
                auto parentColor = inputNode->GetSubgraphID();
                auto currentColor = opOriList[i].GetSubgraphID();
                if (parentColor != -1 && parentColor != currentColor) {
                    {
                        std::unique_lock lock(inColorMtx[currentColor]);
                        inColor_[currentColor].push_back(parentColor);
                    }
                    {
                        std::unique_lock lock(outColorMtx[parentColor]);
                        outColor_[parentColor].push_back(currentColor);
                    }
                }
            }
        }
    });
}

Status NBufferMerge::Init(Function &func) {
    size_t colorMax{0U};
    std::set<int> colorSet;
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorSet.insert(opOriList[i].GetSubgraphID());
        if (opOriList[i].GetSubgraphID() > static_cast<int>(colorMax)) {
            colorMax = opOriList[i].GetSubgraphID();
        }
    }
    if (colorSet.size() == 0) {
        APASS_LOG_INFO_F(Elements::Operation, "Color size is 0, skip nbuffer merge.");
        return SUCCESS;
    }
    if (colorSet.size() != colorMax + 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Colors are not continously numbered from 0, func magic : %d; Please check whether the subgraph IDs are correct.", func.GetFuncMagic());
        return FAILED;
    }
    color_ = colorMax + 1;
    colorNode_.resize(color_);
    colorCycles_.resize(color_, 0);
    inColor_.resize(color_);
    outColor_.resize(color_);
    InitParam(opOriList);
    std::vector<Operation *> opList;
    for (auto &op : func.Operations()) {
        opList.emplace_back(&op);
    }
    auto inOutGraph = RescheduleUtils::GetInOutGraphs(opList, func.GetFuncMagic());
    inGraph_ = inOutGraph[0];
    outGraph_ = inOutGraph[1];
    APASS_LOG_INFO_F(Elements::Operation, "Before Nbuffer merge.");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

std::map<uint64_t, size_t> NBufferMerge::GetIsoColorMergeNum(const std::map<uint64_t, std::vector<int>> &hashMap) const {
    std::map<uint64_t, size_t> hashCoreNum;
    for (const auto& entry : hashMap) {
        if (entry.first == 0 || entry.second.empty()) {
            continue;
        }
        if (hashCoreNum.find(entry.first) == hashCoreNum.end()) {
            hashCoreNum[entry.first] = mgVecParallelLb;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, size %zu, core num: %zu.", 
                    entry.first, entry.second.size(), hashCoreNum[entry.first]);
        if (entry.second.size() <= hashCoreNum[entry.first]) {
            hashCoreNum[entry.first] = 1U;
            continue;
        }
        auto initNum = (entry.second.size() + hashCoreNum[entry.first] - 1) / hashCoreNum[entry.first];
        auto usedCore = (entry.second.size() + initNum - 1) / initNum;
        while ((usedCore < hashCoreNum[entry.first]) && (initNum > 1)) {
            initNum--;
            usedCore = (entry.second.size() + initNum - 1) / initNum;
        }
        hashCoreNum[entry.first] = initNum;
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, merge num: %zu.", entry.first, hashCoreNum[entry.first]);
    }
    return hashCoreNum;
}

void NBufferMerge::GetColorHash(const OperationsViewer &opOriList, 
                                 std::vector<uint64_t> &hashColor,
                                 std::map<uint64_t, std::vector<int>> &hashMap) {
    std::vector<uint64_t> hashTileOp(opOriList.size(), 0);
    for (size_t i = 0; i < opOriList.size(); i++) {
        GetOpHash(hashTileOp, opOriList[i].GetOpcodeStr(), i);
    }
    uint64_t a = 0x12345678;
    uint64_t p = 23;
    const uint64_t mod = 0xFFFFFFFFFFFFF;
    std::set<int32_t> mulaccGraph;
    std::unordered_map<int, int> reshapeCount;
    std::unordered_map<int, int> subgraphOpCount;
    for (size_t i = 0; i < opOriList.size(); i++) {
        int subGraphID = opOriList[i].GetSubgraphID();
        if (subGraphID < 0) {
            continue;
        }
        // 单独的reshape不用合并
        subgraphOpCount[subGraphID]++;
        if (opOriList[i].GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount[subGraphID]++;
        }
        if (OpcodeManager::Inst().GetCoreType(opOriList[i].GetOpcode()) == OpCoreType::AIC) {
            mulaccGraph.insert(subGraphID);
            continue;
        }
        hashColor[subGraphID] = (hashColor[subGraphID] * p + (hashTileOp[i] ^ a)) % mod;
    }
    for (auto &[id, count] : reshapeCount) {
        if (count == subgraphOpCount[id]) {
            hashColor[id] = 0;
        }
    }
    for (auto subgraphId : mulaccGraph) {
        hashColor[subgraphId] = 0;
    }
    int order = 0;
    for (int i = 0; i < color_; i++) {
        hashMap[hashColor[i]].push_back(i);
        if (hashMap[hashColor[i]].size() == 1) {
            hashOrder[hashColor[i]] = order;
            order++;
        }
    }
}

inline int GetCopyIn(const OperationsViewer &opOriList, std::vector<int> &colorNode) {
    // 获取子图CopyIn数据量
    int colorCopyIn = 0;
    for (int j : colorNode) {
        if (opOriList[j].GetOpcode() == Opcode::OP_COPY_IN) { //getopcode == opcode::OP_COPY_IN
            int volume = BytesOf(opOriList[j].GetOOperands()[0]->Datatype());
            std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(opOriList[j].GetOpAttribute());
            if (attr == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "CopyOpAttribute is nullptr, origin op magic : %d; Please check whether the source OpAttribute attribute can be convert to CopyOpAttribute.%s", opOriList[j].GetOpMagic(), GetFormatBacktrace(opOriList[j]).c_str());
                return -1;
            }
            auto shape = attr->GetSpecifiedShape(1);
            for (int k : shape) {
                volume *= k;
            }
            colorCopyIn = colorCopyIn + volume;
        }
    }
    return colorCopyIn;
}

std::vector<std::vector<int>> NBufferMerge::SortColorWithInput(std::vector<int> &colorValues) const {
    std::map<int, std::vector<int>> inColorToOutColor;
    int inCount = -1;
    for (auto color : colorValues) {
        if (inColor_[color].empty()) {
            inColorToOutColor[inCount--].push_back(color);
            continue;
        }
        for (auto inColor : inColor_[color]) {
            inColorToOutColor[inColor].push_back(color);
        }
    }
    std::map<int, std::vector<int>> outColorToInColor;
    int outCount = -1;
    for (auto color : colorValues) {
        if (outColor_[color].empty()) {
            outColorToInColor[outCount--].push_back(color);
            continue;
        }
        for (auto outColor : outColor_[color]) {
            outColorToInColor[outColor].push_back(color);
        }
    }
    std::vector<std::vector<int>> res;
    std::map<int, std::vector<int>> colorWithSameInOut = 
        (inColorToOutColor.size() <= outColorToInColor.size()) ? inColorToOutColor : outColorToInColor;
    std::set<int> visitedColorSet;
    for (auto &entry : colorWithSameInOut) {
        std::vector<int> sortedColor;
        for (auto subgraphColor : entry.second) {
            if (visitedColorSet.count(subgraphColor) == 0) {
                visitedColorSet.insert(subgraphColor);
                sortedColor.push_back(subgraphColor);
            }
        }
        if (!sortedColor.empty()) {
            res.push_back(sortedColor);
        }
    }
    return res;
}

void NBufferMerge::MergePingPong(std::vector<std::vector<int>> &sortedColors, 
                                     const OperationsViewer &opOriList, 
                                     std::vector<uint64_t> &hashColor, 
                                     size_t &numDBmerge) {
    int pingColor = -1;
    for (auto &input2Color : sortedColors) {
        APASS_LOG_DEBUG_F(Elements::Operation, "NBuffer %d Number of subgraphs %d SubGraphIDs %s", numDBmerge, input2Color.size(), IntVecToStr(input2Color).c_str());
        if (vecNBuffermode == autoMulityInOutMerge ||  vecNBuffermode == manualMulityInOutMerge) {
            std::sort(input2Color.begin(), input2Color.end(), [&](int x, int y) { return dfsColorOrder_[x] < dfsColorOrder_[y]; });
        }
        for (size_t i = 0; i < input2Color.size(); i++) {
            if (numDBmerge == 0) {
                continue;
            }
            if (i % numDBmerge == 0) {
                pingColor = input2Color[i];
                continue;
            }
            int pongColor = input2Color[i];
            for (auto opIdxMergedDB : colorNode_[pongColor]) {
                opOriList[opIdxMergedDB].UpdateSubgraphID(pingColor);
                colorNode_[pingColor].push_back(opIdxMergedDB);
            }
            colorCycles_[pingColor] += colorCycles_[pongColor];
            hashColor[pingColor] += hashColor[pongColor];
            colorCycles_[pongColor] = 0;
            colorNode_[pongColor].clear();
            hashColor[pongColor] = 0;
            APASS_LOG_DEBUG_F(Elements::Operation, "SubGraph Merge: %lu, %lu.", pingColor, pongColor);
        }
    }
}

Status NBufferMerge::MergeProcessForMulityInOut(const OperationsViewer &opOriList, const std::map<uint64_t, std::vector<int>> &hashMap, 
    const std::map<uint64_t, size_t> &hashMergeNum, std::vector<uint64_t> &hashColor) {
    std::vector<uint64_t> hashMapKeys;
    for (const auto &entry : hashMap) {
        hashMapKeys.push_back(entry.first);
    }
    DFSSortUtils::DFSSortColor(color_, inColor_, outColor_, dfsColorOrder_);
    ParallelTool::Instance().Parallel_for(0, hashMapKeys.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for(int hashMapKeyIdx = st; hashMapKeyIdx < et; hashMapKeyIdx++) {
            uint64_t colorHashValue = hashMapKeys[hashMapKeyIdx];
            if (colorHashValue == 0) continue;
            auto it = hashMap.find(colorHashValue);
            if (it == hashMap.end()) continue;
            std::vector<int> colorValues = it->second;
            if (colorValues.empty()) continue;
            std::vector<std::vector<int>> sortedColors;
            sortedColors.push_back(colorValues);
            size_t numDBMerge =
                (vecNBuffermode == autoMulityInOutMerge) ? hashMergeNum.at(colorHashValue) : hashMergeNum.at(hashOrder[colorHashValue]);
            MergePingPong(sortedColors, opOriList, hashColor, numDBMerge);
        }
    });
    return SUCCESS;
}

Status NBufferMerge::MergeProcess(const OperationsViewer &opOriList, 
                                      std::map<uint64_t, std::vector<int>> &hashMap, 
                                      std::map<uint64_t, size_t> &hashMergeNum, 
                                      std::vector<uint64_t> &hashColor) {
    std::vector<uint64_t> hashMapKeys;
    for (const auto &entry : hashMap) {
        hashMapKeys.push_back(entry.first);
    }
    ParallelTool::Instance().Parallel_for(0, hashMapKeys.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for(int hashMapKeyIdx = st; hashMapKeyIdx < et; hashMapKeyIdx++) {
            uint64_t colorHashValue = hashMapKeys[hashMapKeyIdx];
            if (colorHashValue == 0) continue;
            std::vector<int> &colorValues = hashMap[colorHashValue];
            auto sortedColors = SortColorWithInput(colorValues);
            if (sortedColors.empty()) continue;
            size_t numDBMerge =
                (vecNBuffermode == 1) ? hashMergeNum[colorHashValue] : hashMergeNum[hashOrder[colorHashValue]];
            MergePingPong(sortedColors, opOriList, hashColor, numDBMerge);
        }
    });
    return SUCCESS;
}

std::map<uint64_t, size_t> NBufferMerge::SetNumDB(std::map<uint64_t, std::vector<int>> &hashMap) {
    std::map<uint64_t, size_t> numDBList;
    auto it = vecNBufferSetting.find(VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY);
    if (it != vecNBufferSetting.end()) {
        int defaultVal = it->second;
        for (uint64_t i = 0; i < static_cast<uint64_t>(hashMap.size()); i++) {
            numDBList[i] = defaultVal;
        }
        vecNBufferSetting.erase(it);
    } else {
        for (uint64_t i = 0; i < static_cast<uint64_t>(hashMap.size()); i++) {
            numDBList[i] = 1;
        }
    }
    for (const auto &entry : vecNBufferSetting) {
        if (entry.first >= 0 && entry.first < static_cast<int>(hashMap.size())) {
            numDBList[entry.first] = entry.second;
        }
    }
    return numDBList;
}

Status NBufferMerge::NBufferMergeProcess(Function &func) {
    if (Init(func) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init Failed; Please check the Init method.");
        return FAILED;
    }
    if (color_ == 0) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Operation, "User set nbuffer mode: %d", vecNBuffermode);
    // 获取节点和子图的hash
    auto opOriList = func.Operations();
    std::vector<uint64_t> hashColor(color_, 0);
    std::map<uint64_t, std::vector<int>> hashMap;
    GetColorHash(opOriList, hashColor, hashMap);
    std::map<uint64_t, size_t> hashMergeNum;
    if (vecNBuffermode == autoMerge ||  vecNBuffermode == autoMulityInOutMerge) {
        if (vecNBufferSetting.size() != 0) {
            APASS_LOG_ERROR_F(Elements::Config, "VEC_NBUFFER_MODE is manually set to %d; Please set VEC_NBUFFER_SETTING to empty.", vecNBuffermode);
            return FAILED;
        }
        APASS_LOG_INFO_F(Elements::Config, "Manually set VEC_NBUFFER_MODE to %d, automatically calculate mergeNum.", vecNBuffermode);
        hashMergeNum = GetIsoColorMergeNum(hashMap);
    } else {
        if (CheckVecNBufferSettingForManualMerge() == FAILED) {
            APASS_LOG_ERROR_F(Elements::Config, "Check VEC_NBUFFER_SETTING for manualMerge failed; Please check the VEC_NBUFFER_SETTING config.");
            return FAILED;
        }
        APASS_LOG_INFO_F(Elements::Config, "Manually set VEC_NBUFFER_MODE to %d.", vecNBuffermode);
        hashMergeNum = SetNumDB(hashMap);
    }
    if (vecNBuffermode == autoMulityInOutMerge ||  vecNBuffermode == manualMulityInOutMerge) {
        if (MergeProcessForMulityInOut(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "MergeProcessForMulityInOut failed; Please check the MergeProcessForMulityInOut method.");
            return FAILED;
        }
    } else {
        if (MergeProcess(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "MergeProcess failed; Please check the MergeProcess method.");
            return FAILED;
        }
    }
    
    if (CheckAndFixColorOrder(opOriList, color_, colorCycles_, colorNode_) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndFixColorOrder failed; Please check the CheckAndFixColorOrder method.");
        return FAILED;
    }
    func.SetTotalSubGraphCount(color_);
    APASS_LOG_DEBUG_F(Elements::Operation, "After Nbuffer merge.");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

Status NBufferMerge::CheckVecNBufferSettingForManualMerge() {
    if (vecNBufferSetting.size() == 0) {
        APASS_LOG_ERROR_F(Elements::Config, "VEC_NBUFFER_MODE is manually set to %d; Please set vecNBufferSetting to non-empty.", vecNBuffermode);
        return FAILED;
    }
    for (const auto& pair : vecNBufferSetting) {
        if (pair.first < VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY || pair.first > static_cast<int64_t>(hashOrder.size()) - 1) {
            APASS_LOG_ERROR_F(Elements::Config, "The VEC_NBUFFER_SETTING key %ld is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder %ld.", pair.first, static_cast<int64_t>(hashOrder.size()) - 1);
            return FAILED;
        }
        if (pair.second <= 0 || pair.second > static_cast<int64_t>(INT_MAX)) {
            APASS_LOG_ERROR_F(Elements::Config, "The value %ld of the key %ld in VEC_NBUFFER_SETTING is incorrect; Please set values of VEC_NBUFFER_SETTING more than 0 and not exceeding the INT_MAX %d.", pair.second, pair.first, INT_MAX);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status NBufferMerge::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> Start NBufferMerge.");
    vecNBuffermode = function.paramConfigs_.vecNBuffermode;
    if (vecNBuffermode < noMerge || vecNBuffermode > manualMulityInOutMerge) {
        APASS_LOG_ERROR_F(Elements::Config, "VEC_NBUFFER_MODE is set to %d; Please set VEC_NBUFFER_MODE to be a value in ModeType.", vecNBuffermode);
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Config, "VEC_NBUFFER_MODE is set to %d.", vecNBuffermode);
    if (vecNBuffermode == noMerge) {
        APASS_LOG_INFO_F(Elements::Config, "Manually set VEC_NBUFFER_MODE to 0, skip NBufferMerge.");
        return SUCCESS;
    }
    mgVecParallelLb = function.paramConfigs_.mgVecParallelLb;
    vecNBufferSetting = function.paramConfigs_.vecNBufferSetting;
    if (NBufferMergeProcess(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "NBufferMergeProcess failed; Please check the NBufferMergeProcess method.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> Finish NBufferMerge.");
    return SUCCESS;
}
}  // namespae npu::tile_fwk

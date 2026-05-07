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
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

inline std::vector<uint64_t> GetGMInputFeature(const Operation& op)
{ // 提取GM tensor的特征
    auto ioperand = op.GetIOperands()[0];
    if (ioperand == nullptr) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "op %s %d ioperand is nullptr.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
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
    for (auto& opImm : opImmList) {
        auto offset = opImm.GetSpecifiedValue();
        if (offset.ConcreteValid()) {
            vec.push_back(offset);
            continue;
        }
        std::hash<std::string> hasher;
        auto offsetHash = hasher(opImm.Dump());
        vec.push_back(static_cast<uint64_t>(offsetHash));
    }
    for (auto& dim : op.GetOOperands()[0]->GetShape()) {
        vec.push_back(dim);
    }
    vec.push_back(static_cast<int>(op.GetOpcode()));
    return vec;
}

bool L1CopyInReuseRunner::CanReuse(const Operation& op)
{
    if (op.GetIOperands().size() != 0 && op.GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        op.GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_CONVERT) {
            return true;
        }
    }
    return false;
}

int L1CopyInReuseRunner::GetModeBySetting(
    const std::map<int64_t, int64_t>& setting,
    const std::map<std::string, int64_t>& settingByFunc)
{
    if (settingByFunc.size() == 1) {
        auto defaultIt = settingByFunc.find(FUNC_HASH_ORDER_DEFAULT_KEY);
        if (defaultIt != settingByFunc.end() && defaultIt->second == 1) {
            return 0;
        }
    }
    std::map<int64_t, int64_t> skipSetting = {{-1, 1}};
    if (setting == skipSetting) {
        return 0;
    }
    return 1;
}

// key : 需要被删除的copyin op, value: 保留的copyin op
Status L1CopyInReuseRunner::GetDuplicateOps(std::vector<Operation*>& opOriList, const std::vector<int>& opIdx)
{
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
            APASS_LOG_ERROR_F(
                Elements::Operation, "GetDuplicateOps: op %s %d GetGMInputFeature failed. %s",
                opOriList[i]->GetOpcodeStr().c_str(), opOriList[i]->GetOpMagic(),
                GetFormatBacktrace(*opOriList[i]).c_str());
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

void L1CopyInReuseRunner::TackleOp(
    int i, Operation* op, std::vector<std::vector<int>>& replacedInputs, std::vector<std::vector<int>>& replacedOutputs)
{
    if (CanReuse(*op)) {
        auto allocedL1BufId = op->GetOOperands()[0]->GetRawTensor()->GetRawMagic();
        if (tensormagic2Op_.find(allocedL1BufId) != tensormagic2Op_.end()) {
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Remove useless op [%d, %s].", op->GetOpMagic(), op->GetOpcodeStr().c_str());
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

void L1CopyInReuseRunner::GetOriList(Function& func, std::vector<Operation*>& oriList)
{
    for (auto& op : func.Operations()) {
        oriList.emplace_back(&op);
    }
}

void L1CopyInReuseRunner::MergeProcessIdUpdate(Function& func, std::vector<std::vector<int>>& colorNode, int color)
{
    std::vector<Operation*> oriList;
    GetOriList(func, oriList);
    int colorCount = 0;
    for (int j = 0; j < color; j++) {
        if (colorNode[j].empty()) {
            continue;
        }
        colorCount++;
        for (int i : colorNode[j]) {
            oriList[i]->UpdateSubgraphID(colorCount - 1);
        }
    }
    func.SetTotalSubGraphCount(colorCount);
}

// 合并重复的L1_COPY_IN和L1_ALLOC节点
Status L1CopyInReuseRunner::MergeDupL1CopyIn(Function& func, std::vector<std::vector<int>>& colorNode, int color)
{
    std::vector<Operation*> oriList;
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
        for (auto& replacedInput : replacedInputs) {
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Relink op [%d] input [%d] to op [%d] output [%d].",
                oriList[replacedInput[0]]->GetOpMagic(), replacedInput[1], oriList[replacedInput[2]]->GetOpMagic(),
                replacedInput[3]);
            FunctionUtils::RelinkOperationInput(
                oriList[replacedInput[0]], replacedInput[1], oriList[replacedInput[2]], replacedInput[3]);
        }
        for (auto& replacedOutput : replacedOutputs) {
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

int L1CopyInReuseRunner::GetMaxInColor(const std::vector<int>& nodes, const OperationsViewer& opOriList, int curColor)
{
    int maxInColor = -1;
    for (int j : nodes) {
        for (int k : inGraph_[j]) {
            auto opColor = opOriList[k].GetSubgraphID();
            if (opColor != curColor) {
                maxInColor = std::max(maxInColor, opColor);
            }
        }
    }
    return maxInColor;
}

std::vector<int> L1CopyInReuseRunner::GetCopyIn(
    const OperationsViewer& opOriList, int color, std::vector<std::vector<int>>& colorNode)
{
    // 获取子图L1CopyIn数据量
    std::vector<int> colorCopyIn(color, 0);
    for (int i = 0; i < color; i++) {
        for (int j : colorNode[i]) {
            if (CanReuse(opOriList[j])) {
                int volume = BytesOf(opOriList[j].GetOOperands()[0]->Datatype());
                for (auto& k : opOriList[j].GetOOperands()[0]->GetShape()) {
                    volume *= k;
                }
                colorCopyIn[i] = colorCopyIn[i] + volume;
            }
        }
    }
    return colorCopyIn;
}

void L1CopyInReuseRunner::GetOpHash(std::vector<uint64_t>& hashList, const std::string op, int idx)
{
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

void L1CopyInReuseRunner::GetColorHash(
    const Function& func, const OperationsViewer& opOriList, std::vector<uint64_t>& hashColor,
    const std::vector<std::vector<int>>& colorNode)
{
    std::vector<uint64_t> hashTileOp(opOriList.size(), 0);
    for (size_t i = 0; i < opOriList.size(); i++) {
        GetOpHash(hashTileOp, opOriList[i].GetOpcodeStr(), i);
    }
    uint64_t a = 0x12345678;
    uint64_t p = 23;
    const uint64_t mod = 0xFFFFFFFFFFFFF;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        if (CanReuse(opOriList[i])) {
            mulaccGraph_.insert(opOriList[i].GetSubgraphID());
        }
        hashColor[opOriList[i].GetSubgraphID()] =
            (hashColor[opOriList[i].GetSubgraphID()] * p + (hashTileOp[i] ^ a)) % mod;
    }
    int order = 0;
    for (int i : mulaccGraph_) {
        hashMap_[hashColor[i]].push_back(i);
        if (hashMap_[hashColor[i]].size() == 1) {
            hashOrder_[hashColor[i]] = order;
            order++;
        }
    }
    int funcMagic = func.GetFuncMagic();
    for (auto& entry : hashMap_) {
        int hashOrderVal = hashOrder_[entry.first];
        std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" + std::to_string(hashOrderVal);
        size_t subgraphCount = entry.second.size();
        for (auto subgraphId : entry.second) {
            if (!mulaccGraph_.count(subgraphId))
                continue;
            for (auto opIdx : colorNode[subgraphId]) {
                opOriList[opIdx].SetHashOrderInfo(
                    OpAttributeKey::l1ReuseHashOrder,
                    OpAttributeKey::l1ReuseSubgraphCount,
                    fullHashOrder, subgraphCount);
            }
        }
    }
}

void L1CopyInReuseRunner::HashUpdate(
    Function& func, int color, const std::vector<uint64_t>& hashColor, OperationsViewer& opOriList,
    std::vector<std::vector<int>>& colorNode)
{
    for (auto entry = hashMap_.begin(); entry != hashMap_.end();) {
        if (entry->second.empty()) {
            entry = hashMap_.erase(entry);
            continue;
        }
        entry++;
    }

    hashOrder_.clear();
    int order = 0;
    int funcMagic = func.GetFuncMagic();
    for (int i = 0; i < color; i++) {
        if (hashMap_.find(hashColor[i]) != hashMap_.end() && hashOrder_.find(hashColor[i]) == hashOrder_.end()) {
            hashOrder_[hashColor[i]] = order;
            std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" + std::to_string(order);
            size_t subgraphCount = hashMap_[hashColor[i]].size();
            for (auto subgraphId : hashMap_[hashColor[i]]) {
                for (auto opIdx : colorNode[subgraphId]) {
                    opOriList[opIdx].SetHashOrderInfo(
                        OpAttributeKey::cubeMergeHashOrder,
                        OpAttributeKey::cubeMergeSubgraphCount,
                        fullHashOrder, subgraphCount);
                }
            }
            order++;
        }
    }

    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview.", func.GetMagicName().c_str());
    for (auto& entry : hashOrder_) {
        auto& subgraphIds = hashMap_[entry.first];
        std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" + std::to_string(entry.second);
        APASS_LOG_INFO_F(
            Elements::Function,
            "Cube nbuffer hashOrder: %s, Subgraph count: %zu, Subgraph IDs: %s",
            fullHashOrder.c_str(), subgraphIds.size(), IntVecToStr(subgraphIds).c_str());
    }
    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview end.", func.GetMagicName().c_str());
}

Status L1CopyInReuseRunner::ApplyByFuncConfig(
    int currentFuncMagic, const std::map<std::string, int64_t>& configMapByFunc,
    std::map<int, int>& resultMap, const std::string& configName)
{
    auto defaultIt = configMapByFunc.find(FUNC_HASH_ORDER_DEFAULT_KEY);
    if (defaultIt != configMapByFunc.end()) {
        if (defaultIt->second < 1) {
            APASS_LOG_ERROR_F(Elements::Config,
                "Invalid merge count for DEFAULT: merge count=%ld, please check.",
                static_cast<long>(defaultIt->second));
            return FAILED;
        }
        int defaultVal = static_cast<int>(defaultIt->second);
        for (const auto& [hashVal, order] : hashOrder_) {
            (void) hashVal;
            resultMap[order] = defaultVal;
        }
    }

    for (const auto& entry : configMapByFunc) {
        if (entry.first == FUNC_HASH_ORDER_DEFAULT_KEY) {
            continue;
        }
        int funcMagic, localOrder;
        if (!ParseFuncHashOrder(entry.first, funcMagic, localOrder)) {
            APASS_LOG_WARN_F(Elements::Config,
                "Invalid func hashOrder format: %s in %s, ignored.", entry.first.c_str(), configName.c_str());
            continue;
        }
        if (funcMagic == currentFuncMagic) {
            if (entry.second < 1) {
                APASS_LOG_ERROR_F(Elements::Config,
                    "Invalid merge count for func hashOrder %s: merge count=%ld, please check.",
                    entry.first.c_str(), entry.second);
                return FAILED;
            }
            resultMap[localOrder] = static_cast<int>(entry.second);
        }
    }
    return SUCCESS;
}

Status L1CopyInReuseRunner::ApplyGlobalConfig(
    const std::map<int64_t, int64_t>& configMap, std::map<int, int>& resultMap, const std::string& configName)
{
    auto defaultEntry = configMap.find(-1);
    int defaultValue = -1;
    if (defaultEntry != configMap.end()) {
        if (defaultEntry->second < 1) {
            APASS_LOG_ERROR_F(Elements::Config,
                "Invalid default merge count for %s: Default merge count=%ld, please check.",
                configName.c_str(), static_cast<long>(defaultEntry->second));
            return FAILED;
        }
        defaultValue = defaultEntry->second;
    }
    for (auto& [hashcolor, order] : hashOrder_) {
        (void) hashcolor;
        resultMap[order] = defaultValue;
    }
    for (auto& entry : configMap) {
        int hashOrderKey = static_cast<int>(entry.first);
        if (hashOrderKey < 0) {
            continue;
        }
        bool found = false;
        for (auto& [hashcolor, order] : hashOrder_) {
            (void) hashcolor;
            if (order == hashOrderKey) {
                found = true;
                break;
            }
        }
        if (!found) {
            APASS_LOG_WARN_F(
                Elements::Config, "Invalid hashOrder: %d in %s, ignored.", hashOrderKey, configName.c_str());
            continue;
        }
        if (entry.second < 1) {
            APASS_LOG_ERROR_F(Elements::Config,
                "Invalid merge count for hashOrder %d: merge count=%ld, please check.",
                hashOrderKey, static_cast<long>(entry.second));
            return FAILED;
        }
        resultMap[hashOrderKey] = static_cast<int>(entry.second);
    }
    return SUCCESS;
}

Status L1CopyInReuseRunner::SetNumFromConfig(
    const Function& func, const std::map<int64_t, int64_t>& configMap,
    const std::map<std::string, int64_t>& configMapByFunc, std::map<int, int>& resultMap,
    const std::string& configName)
{
    if (!configMapByFunc.empty()) {
        return ApplyByFuncConfig(func.GetFuncMagic(), configMapByFunc, resultMap, configName);
    }
    return ApplyGlobalConfig(configMap, resultMap, configName);
}

Status L1CopyInReuseRunner::L1MergeProcess(
    OperationsViewer& opOriList, std::vector<std::vector<int>>& colorNode, std::vector<uint64_t>& hashColor,
    std::vector<int>& colorCopyIn, std::map<std::vector<uint64_t>, int>& l1InputList, int& tmpColor,
    std::vector<int>& mergedNum, int& i)
{
    for (auto opIdx : colorNode[i]) {
        if (opOriList[opIdx].HasAttribute(OpAttributeKey::isCube) &&
            !opOriList[opIdx].GetBoolAttribute(OpAttributeKey::isCube)) {
            return SUCCESS;
        }
        if (!CanReuse(opOriList[opIdx])) {
            continue;
        }
        auto vec = GetGMInputFeature(opOriList[opIdx]);
        if (vec.size() == 0) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "L1MergeProcess: op %d %s GetGMInputFeature failed. %s", opOriList[opIdx].GetOpMagic(),
                opOriList[opIdx].GetOpcodeStr().c_str(), GetFormatBacktrace(opOriList[opIdx]).c_str());
            return FAILED;
        }
        l1InputList[vec] = tmpColor;
    } // 记录当前子图所有的L1_COPY_IN搬入的tensor特征
    if (tmpColor != i) {
        for (auto t : colorNode[i]) {
            opOriList[t].UpdateSubgraphID(tmpColor);
            colorNode[tmpColor].push_back(t);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph merge: %d, %d.", i, tmpColor);
        colorNode[i].clear();
        colorCopyIn[tmpColor] = colorCopyIn[tmpColor] + colorCopyIn[i];
        mergedNum[i] = 0;
        mergedNum[tmpColor] += 1;
        hashMap_[hashColor[i]].erase(std::find(hashMap_[hashColor[i]].begin(), hashMap_[hashColor[i]].end(), i));
        hashMap_[hashColor[tmpColor]].erase(
            std::find(hashMap_[hashColor[tmpColor]].begin(), hashMap_[hashColor[tmpColor]].end(), tmpColor));
        hashColor[tmpColor] += hashColor[i];
        hashColor[i] = 0;
        hashMap_[hashColor[tmpColor]].push_back(tmpColor);
    } // 合入子图
    return SUCCESS;
}

void L1CopyInReuseRunner::GetL1ReuseOpOrder(
    std::vector<std::pair<int, int>>& opOrder, std::map<uint64_t, int>& mgRem, std::vector<int>& numLRList,
    std::vector<uint64_t>& hashColor, int color)
{
    std::map<uint64_t, int> mp;
    for (int i = 0; i < color; i++) {
        opOrder[i] = std::make_pair(hashOrder_[hashColor[i]], i);
        mp[hashOrder_[hashColor[i]]]++;
    }
    // 按照子图hashOrder进行排序
    std::sort(opOrder.begin(), opOrder.end());
    int coreNum = Platform::Instance().GetSoc().GetAICCoreNum();
    if (coreNum == 0) {
        APASS_LOG_WARN_F(Elements::Config, "Failed to get number of cores. L1Reuse will not be applied.");
        return;
    }
    // Use hashOrder grouping: all subgraphs of the same hashOrder share the auto-calc result.
    std::map<uint64_t, int> autoCalcCache;
    std::map<uint64_t, int> autoRemCache;
    for (int i = 0; i < color; i++) {
        uint64_t ho = hashOrder_[hashColor[i]];
        if (numLRList[i] == -1) {
            if (autoCalcCache.find(ho) == autoCalcCache.end()) {
                autoCalcCache[ho] = mp[ho] / (coreNum * NUM2);
                autoRemCache[ho] = mp[ho] % (coreNum * NUM2);
            }
            numLRList[i] = autoCalcCache[ho];
            mgRem[ho] = autoRemCache[ho];
        } else {
            mgRem[ho] = 0;
        }
    }
}

bool L1CopyInReuseRunner::GetMergedL1(
    int maxInColor, std::vector<int>& mergedNum, int maxMergeNum, int& tmpColor, int i,
    std::map<std::vector<uint64_t>, int>& l1InputList, std::vector<uint64_t>& vec, std::vector<int>& colorCopyIn,
    std::map<uint64_t, int>& mgRem, uint64_t idx)
{
    auto copyId = l1InputList.find(vec);
    if (copyId != l1InputList.end() && copyId->second >= maxInColor &&
        colorCopyIn[copyId->second] + colorCopyIn[i] <= mgCopyInUpperBound_ && mergedNum[copyId->second] > 0 &&
        (mergedNum[copyId->second] < maxMergeNum || (mergedNum[copyId->second] == maxMergeNum && mgRem[idx] > 0))) {
        tmpColor = copyId->second;
        mgRem[idx] -= (mergedNum[copyId->second] == maxMergeNum ? 1 : 0);
        return true;
    }
    return false;
}

Status L1CopyInReuseRunner::Phase1(
    Function& func, int color, std::vector<std::vector<int>>& colorNode, std::vector<int>& colorCopyIn,
    std::vector<uint64_t>& hashColor)
{
    // 针对matmul的L1 copy reuse进行子图合并
    auto opOriList = func.Operations();
    std::map<std::vector<uint64_t>, int> l1InputList;
    std::map<int, int> numLRMap;
    // CubeL1ReuseMode - apply global and function-granularity settings
    if (SetNumFromConfig(func, numLRMap_, numLRMapByFunc_, numLRMap, "cubeL1ReuseSetting") == FAILED) {
        APASS_LOG_ERROR_F(Elements::Config, "Invalid configuration: %s.", "cubeL1ReuseSetting");
        return FAILED;
    }
    // Expand hashOrder-granularity numLRMap to subgraph-granularity numLRList
    std::vector<int> numLRList(color, -1);
    for (int i = 0; i < color; i++) {
        int order = hashOrder_[hashColor[i]];
        auto it = numLRMap.find(order);
        if (it != numLRMap.end()) {
            numLRList[i] = it->second;
        }
    }
    // Apply semantic label settings for L1 reuse (higher priority than hashorder settings)
    if (ApplySemanticLabelSettingsL1Reuse(opOriList, numLRList, hashColor, color) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Config,
            "ApplySemanticLabelSettingsL1Reuse failed; Please check the semantic labels in cube_l1_reuse_setting.");
        return FAILED;
    }
    std::vector<int> mergedNum(color, 1);
    std::vector<std::pair<int, int>> opOrder(color);
    std::map<uint64_t, int> mgRem;
    GetL1ReuseOpOrder(opOrder, mgRem, numLRList, hashColor, color);
    for (int ii = 0; ii < color; ii++) {
        int i = opOrder[ii].second;
        int tmpColor = -1;
        auto maxInColor = GetMaxInColor(colorNode[i], opOriList, i);
        size_t j = 0;
        while (colorCopyIn[i] <= mgCopyInUpperBound_ && j < colorNode[i].size()) {
            auto opIdx = colorNode[i][j];
            if (!CanReuse(opOriList[opIdx])) {
                j++;
                continue;
            }
            auto vec = GetGMInputFeature(opOriList[opIdx]);
            if (vec.size() == 0) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "Phase1: op %s %d GetGMInputFeature failed. %s",
                    opOriList[opIdx].GetOpcodeStr().c_str(), opOriList[opIdx].GetOpMagic(),
                    GetFormatBacktrace(opOriList[opIdx]).c_str());
                return FAILED;
            }
            if (GetMergedL1(
                    maxInColor, mergedNum, numLRList[i], tmpColor, i, l1InputList, vec, colorCopyIn, mgRem,
                    hashColor[i])) {
                break;
            }
            j++;
        }
        if (tmpColor == -1) {
            tmpColor = i;
        }
        if (L1MergeProcess(opOriList, colorNode, hashColor, colorCopyIn, l1InputList, tmpColor, mergedNum, i) ==
            FAILED) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "L1MergeProcess failed; Please check the L1MergeProcess method.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status L1CopyInReuseRunner::ApplySemanticLabelSettingsL1Reuse(
    const OperationsViewer& opOriList, std::vector<int>& numLRList, const std::vector<uint64_t>& /* hashColor */,
    int color)
{
    if (numLRMapByLabel_.empty()) {
        return SUCCESS;
    }

    // Build a map from semantic label to the subgraph colors that contain ops with that label
    auto labelToColors = BuildLabelToColorsMap(opOriList);

    // For L1Reuse, string keys set only the specific subgraphs containing the labeled ops,
    // NOT the whole isomorphic group.
    // First step: collect override per subgraph color. If multiple labels target the same
    // subgraph, take the max among them.
    std::map<int, int> subgraphOverrides;
    for (const auto& [label, mergeNum] : numLRMapByLabel_) {
        auto it = labelToColors.find(label);
        if (it == labelToColors.end()) {
            APASS_LOG_ERROR_F(
                Elements::Config,
                "Semantic label '%s' specified in cube_l1_reuse_setting not found in any operation. "
                "Please check that the label matches an operation's semantic_label.",
                label.c_str());
            return FAILED;
        }

        for (int colorId : it->second) {
            if (colorId >= color) {
                continue;
            }
            auto overIt = subgraphOverrides.find(colorId);
            if (overIt != subgraphOverrides.end()) {
                overIt->second = std::max(overIt->second, static_cast<int>(mergeNum));
            } else {
                subgraphOverrides[colorId] = static_cast<int>(mergeNum);
            }
        }
    }

    // Second step: replace the numLRList with collected overrides
    for (const auto& [colorId, val] : subgraphOverrides) {
        if (colorId < static_cast<int>(numLRList.size())) {
            numLRList[colorId] = val;
            APASS_LOG_INFO_F(
                Elements::Config, "Applied L1 reuse semantic label override: subgraph_color=%d, merge_num=%d", colorId,
                val);
        }
    }

    return SUCCESS;
}

Status L1CopyInReuseRunner::ApplySemanticLabelSettingsCubeNBuffer(
    const OperationsViewer& opOriList, std::map<int, int>& hashMergeNumMap, const std::vector<uint64_t>& hashColor,
    int color)
{
    if (numDBMapByLabel_.empty()) {
        return SUCCESS;
    }

    // Build a map from semantic label to the subgraph colors that contain ops with that label
    auto labelToColors = BuildLabelToColorsMap(opOriList);

    // First step: collect override value per hashOrder from all labels.
    // If multiple labels target the same isomorphic group, take max among them.
    std::map<int, int> labelOverrides;
    for (const auto& [label, mergeNum] : numDBMapByLabel_) {
        auto it = labelToColors.find(label);
        if (it == labelToColors.end()) {
            APASS_LOG_ERROR_F(
                Elements::Config,
                "Semantic label '%s' specified in cube_nbuffer_setting not found in any operation. "
                "Please check that the label matches an operation's semantic_label.",
                label.c_str());
            return FAILED;
        }

        for (int colorId : it->second) {
            if (colorId >= color) {
                continue;
            }
            uint64_t colorHash = hashColor[colorId];
            auto hashOrderIt = hashOrder_.find(colorHash);
            if (hashOrderIt == hashOrder_.end()) {
                continue;
            }
            int order = hashOrderIt->second;
            auto overIt = labelOverrides.find(order);
            if (overIt != labelOverrides.end()) {
                overIt->second = std::max(overIt->second, static_cast<int>(mergeNum));
            } else {
                labelOverrides[order] = static_cast<int>(mergeNum);
            }
        }
    }

    // Second step: replace the hashMergeNumMap with collected label overrides
    for (const auto& [order, val] : labelOverrides) {
        hashMergeNumMap[order] = val;
        APASS_LOG_INFO_F(
            Elements::Config, "Applied cube nbuffer semantic label override: hash_order=%d, merge_num=%d", order, val);
    }

    return SUCCESS;
}

inline std::vector<int> AdjustNumDBCore(int color, int numDB, int mx)
{
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
            coreNum *= NUM2;
            rm = color;
        } else {
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

void L1CopyInReuseRunner::CubeMergeProcess(
    std::vector<std::vector<int>>& colorNode, OperationsViewer& opOriList, std::map<int, int>& hashMergeNumMap,
    std::vector<int>& colorCopyIn)
{
    for (auto& entry : hashMap_) {
        uint64_t colorHashValue = entry.first;
        std::vector<int>& colorValues = entry.second;
        int sz = colorCopyIn[colorValues[0]];
        if (sz > mgCopyInUpperBound_) {
            continue;
        }
        int pingColor = -1;
        int mxMerge = mgCopyInUpperBound_ / sz;
        std::vector<int> pingColorList =
            AdjustNumDBCore(colorValues.size(), hashMergeNumMap[hashOrder_[colorHashValue]], mxMerge);
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
            APASS_LOG_INFO_F(Elements::Operation, "Subgraph merge: %d, %d.", pingColor, pongColor);
            colorNode[pongColor].clear();
        }
    }
}

Status L1CopyInReuseRunner::Run(Function& func, int color, std::vector<std::vector<int>>& colorNode)
{
    auto opOriList = func.Operations();
    std::vector<uint64_t> hashColor(color, 0);
    hashOrder_.clear();
    GetColorHash(func, opOriList, hashColor, colorNode);
    // print hashorder with function-granularity format
    int funcMagic = func.GetFuncMagic();
    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview.", func.GetMagicName().c_str());
    for (auto& entry : hashMap_) {
        std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" + std::to_string(hashOrder_[entry.first]);
        APASS_LOG_INFO_F(
            Elements::Function, "L1 reuse hashOrder: %s, Subgraph count: %zu, Subgraph IDs: %s",
            fullHashOrder.c_str(), entry.second.size(), IntVecToStr(entry.second).c_str());
    }
    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview end.", func.GetMagicName().c_str());
    auto colorCopyIn = GetCopyIn(opOriList, color, colorNode); // 记录各子图的大小
    mgCopyInUpperBound_ = func.paramConfigs_.sgMgCopyInUpperBound;
    numLRMap_ = func.paramConfigs_.cubeL1ReuseSetting;
    numDBMap_ = func.paramConfigs_.cubeNBufferSetting; // 合并阈值参数设置
    numLRMapByFunc_ = func.paramConfigs_.cubeL1ReuseSettingByFunc;
    numDBMapByFunc_ = func.paramConfigs_.cubeNBufferSettingByFunc;
    numLRMapByLabel_ = func.paramConfigs_.cubeL1ReuseSettingByLabel;
    numDBMapByLabel_ = func.paramConfigs_.cubeNBufferSettingByLabel;
    L1ReuseMode_ = GetModeBySetting(numLRMap_, numLRMapByFunc_);
    cubeNBufferMode_ = GetModeBySetting(numDBMap_, numDBMapByFunc_);
    APASS_LOG_INFO_F(Elements::Operation, "Param Setting mgCopyInUpperBound %d.", mgCopyInUpperBound_);
    if ((L1ReuseMode_ == 1 || !numLRMapByLabel_.empty()) && hashMap_.size() != 0) {
        if (Phase1(func, color, colorNode, colorCopyIn, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Function, "Phase1 failed; Please check the Phase1 method.");
            return FAILED;
        }
        if (MergeDupL1CopyIn(func, colorNode, color) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Function, "Run: MergeDupL1CopyIn failed.");
            return FAILED;
        }
        HashUpdate(func, color, hashColor, opOriList, colorNode);
    }
    std::map<int, int> hashMergeNumMap;
    // NBuffer参数设置 - apply global and function-granularity settings
    if (SetNumFromConfig(func, numDBMap_, numDBMapByFunc_, hashMergeNumMap, "cubeNBufferSetting") == FAILED) {
        APASS_LOG_ERROR_F(Elements::Config, "Invalid configuration: %s.", "cubeNBufferSetting");
        return FAILED;
    }
    // Apply semantic label settings for cube nbuffer (higher priority than hashorder settings)
    if (ApplySemanticLabelSettingsCubeNBuffer(opOriList, hashMergeNumMap, hashColor, color) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Config,
            "ApplySemanticLabelSettingsCubeNBuffer failed; Please check the semantic labels in cube_nbuffer_setting.");
        return FAILED;
    }
    CubeMergeProcess(colorNode, opOriList, hashMergeNumMap, colorCopyIn);
    MergeProcessIdUpdate(func, colorNode, color);
    for (auto& op : func.Operations()) {
        if (static_cast<size_t>(op.GetSubgraphID()) > func.GetTotalSubGraphCount()) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "Run: op SubGraph ID %d out of range. %s", op.GetSubgraphID(),
                GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    RemoveUselessViews(func); // 删除节点
    func.EraseOperations(true);
    APASS_LOG_DEBUG_F(Elements::Operation, "After L1CopyInReuse.");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

void L1CopyInReuseRunner::RemoveUselessViews(Function& func) const
{
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

Status L1CopyInReuseMerge::InitColorNode(Function& func, std::vector<std::vector<int>>& colorNode) const
{
    int colorMax{0};
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (L1CopyInReuseRunner::CanReuse(opOriList[i])) {
            auto feature = GetGMInputFeature(opOriList[i]);
            if (feature.size() == 0) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "Get Feature FAILED. %s", GetFormatBacktrace(opOriList[i]).c_str());
                return FAILED;
            }
            APASS_LOG_INFO_F(Elements::Operation, "Op %zu feature: %s.", i, IntVecToStr(feature).c_str());
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

Status L1CopyInReuseMerge::CheckOpListValid(Function& func) const
{
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetIOperands().size() != 0 &&
            opOriList[i].GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            opOriList[i].GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            if (opOriList[i].GetOpcode() == Opcode::OP_VIEW || opOriList[i].GetOpcode() == Opcode::OP_CONVERT ||
                opOriList[i].GetOpcode() == Opcode::OP_L1_COPY_IN_CONV) {
                // 符合预期且合法
                continue;
            } else if (opOriList[i].GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                // 预期之外，先放行，安排计划评审修复
                continue;
            } else {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Unexpected operation %s. "
                    "Please check if the operation is within the expected range",
                    opOriList[i].Dump().c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status L1CopyInReuseMerge::L1CopyInReuse(Function& func) const
{
    // Check mode: support both legacy (-1: 1) and new format (FUNC_HASH_ORDER_DEFAULT_KEY: 1)
    auto L1ReuseMode = L1CopyInReuseRunner::GetModeBySetting(
        func.paramConfigs_.cubeL1ReuseSetting, func.paramConfigs_.cubeL1ReuseSettingByFunc);
    auto cubeNBufferMode = L1CopyInReuseRunner::GetModeBySetting(
        func.paramConfigs_.cubeNBufferSetting, func.paramConfigs_.cubeNBufferSettingByFunc);
    bool hasLabelSetting = !func.paramConfigs_.cubeL1ReuseSettingByLabel.empty() ||
                           !func.paramConfigs_.cubeNBufferSettingByLabel.empty();
    if (L1ReuseMode == 0 && cubeNBufferMode == 0 && !hasLabelSetting) {
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

    std::vector<Operation*> opList;
    for (auto& op : func.Operations()) {
        opList.emplace_back(&op);
    }
    auto inOutGraph = RescheduleUtils::GetInOutGraphs(opList, func.GetFuncMagic());
    auto& inGraph = inOutGraph[0];
    L1CopyInReuseRunner runner(inGraph);
    if (runner.Run(func, colorNode.size(), colorNode) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "L1CopyInReuse: Run failed.");
        return FAILED;
    }
    return SUCCESS;
}

void L1CopyInReuseMerge::DoHealthCheckAfter(Function& function, const std::string& folderPath)
{
    APASS_LOG_INFO_F(Elements::Function, "After L1CopyInReuseMerge, Health Report: TileGraph START.");
    std::string fileName = GetDumpFilePrefix(function);
    HealthCheckTileGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Function, "After L1CopyInReuseMerge, Health Report: TileGraph END.");
}
} // namespace npu::tile_fwk

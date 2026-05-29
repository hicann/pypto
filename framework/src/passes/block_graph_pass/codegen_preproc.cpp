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
 * \file codegen_preproc.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "passes/pass_interface/pass.h"
#include "codegen_preproc.h"
#include "passes/pass_log/pass_log.h"

#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <unordered_map>

#define MODULE_NAME "CodegenPreproc"

namespace npu {
namespace tile_fwk {
const std::string REDUCE_AXIS = OP_ATTR_PREFIX + "AXIS";

namespace {
const SymbolicScalar& GetParamAddrSymbol()
{
    static const SymbolicScalar kGetParamAddr = []() {
        IRBuilder builder;
        return builder.CreateScalarVar(AddRuntimePrefix("GET_PARAM_ADDR"));
    }();
    return kGetParamAddr;
}

const SymbolicScalar& GetRuntimeParamSymbol()
{
    static const SymbolicScalar kRuntimeParam = []() {
        IRBuilder builder;
        return builder.CreateScalarVar(AddRuntimePrefix("param"));
    }();
    return kRuntimeParam;
}
} // namespace

// only save general gm input/output, not contain spill-out scene
bool CodegenPreproc::IsCopyNeedSave(const Operation& op) const
{
    return OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) && (!op.IsNeedStackGM());
}

void CodegenPreproc::SetTensorParamAddr(
    LogicalTensor& tensor, int64_t tensorParamIdx, const SymbolicScalar& attrOffsetScalar, int opMagic) const
{
    IRBuilder builder;
    SymbolicScalar paramAddr =
        GetParamAddrSymbol()(GetRuntimeParamSymbol(), builder.CreateConstInt(tensorParamIdx), attrOffsetScalar);
    std::map<int, SymbolicScalar> opParamAddrs;
    tensor.GetAttr<std::map<int, SymbolicScalar>>(TensorAttributeKey::tensorAddr, opParamAddrs);
    opParamAddrs[opMagic] = paramAddr;
    tensor.SetAttr<std::map<int, SymbolicScalar>>(TensorAttributeKey::tensorAddr, opParamAddrs);
}

// only used in DYNAMIC_LOOP_PATH scene
Status CodegenPreproc::SaveGmTensorParamIdxToOp(Function& func) const
{
    IRBuilder builder;
    if (!func.IsUnderDynamicFunction()) {
        return SUCCESS;
    }

    // op magic -> <tensor magic, param offset in call op>
    std::unordered_map<int, std::unordered_map<int, int>> gmParamOffsetInOp;
    std::map<int, std::vector<Operation*>> gmParamInCallFunc;
    for (auto& subProgram : func.rootFunc_->programs_) {
        gmParamInCallFunc.clear();
        for (auto& op : subProgram.second->Operations(false)) {
            if (IsCopyNeedSave(op)) {
                int coaIndex =
                    OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
                gmParamInCallFunc[coaIndex].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1 || op.GetOpcode() == Opcode::OP_GATHER_IN_UB) {
                gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(2)].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_GATHER) {
                gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_PERMUTE || op.GetOpcode() == Opcode::OP_PERMUTE_ELEMENT) {
                gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
            }
        }

        APASS_LOG_INFO_F(
            Elements::Operation, "%d:%sgmParamInCallFunc size: %zu", __LINE__, __FUNCTION__, gmParamInCallFunc.size());

        int64_t tensorParamIdx{0};
        for (auto param : gmParamInCallFunc) {
            for (auto op : param.second) {
                op->SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, tensorParamIdx++);
            }
        }

        for (auto& op : subProgram.second->Operations(false)) {
            int64_t gmTensorParamIdx{0};
            if (op.HasAttribute(OpAttributeKey::gmTensorParamIdxInCall)) {
                op.GetAttr(OpAttributeKey::gmTensorParamIdxInCall, gmTensorParamIdx);
            }
            int attrOffset{0};
            for (size_t i = 0; i < op.GetIOperands().size(); ++i) {
                auto& tensor = op.GetIOperands()[i];
                // attroffset of shem follows natural incremental order
                if (OpcodeManager::Inst().IsSharedMemory(op.GetOpcode())) {
                    attrOffset = i;
                }
                if (tensor->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                    SetTensorParamAddr(
                        *tensor, gmTensorParamIdx, builder.CreateConstInt(op.GetIOpAttrOffset(attrOffset++)),
                        op.GetOpMagic());
                }
            }
            attrOffset = 0;
            for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
                auto& tensor = op.GetOOperands()[i];
                // attroffset of shem follows natural incremental order
                if (OpcodeManager::Inst().IsSharedMemory(op.GetOpcode())) {
                    attrOffset = i;
                }
                if (tensor->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                    SetTensorParamAddr(
                        *tensor, gmTensorParamIdx, builder.CreateConstInt(op.GetOOpAttrOffset(attrOffset++)),
                        op.GetOpMagic());
                }
            }
        }
    }

    return SUCCESS;
}

inline bool IsUBCopy(Operation& op)
{
    if (IsCopyIn(op.GetOpcode())) {
        auto outTensor = *(op.GetOOperands().begin());
        if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            return true;
        }
    }
    if (IsCopyOut(op.GetOpcode())) {
        auto inTensor = *(op.GetIOperands().begin());
        if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            return true;
        }
    }
    return false;
}

bool ReduceNeedCombineAxis(const Operation& op)
{
    if (OpcodeManager::Inst().GetOpCalcType(op.GetOpcode()) != OpCalcType::REDUCE) {
        return true;
    }
    if (op.GetOpcode() == Opcode::OP_ROWSUMLINE) {
        auto inputs = op.GetIOperands();
        if (op.GetIOperands().size() != 1 || !op.HasAttr(REDUCE_AXIS)) {
            return false;
        }
        auto axis = op.GetIntAttribute(REDUCE_AXIS);
        int64_t shapeSize = static_cast<int64_t>(inputs.front()->shape.size());
        return shapeSize != 1 && axis != (shapeSize - 2);
    }
    return false;
}

void CodegenPreproc::FixExpandDimForAxisCombine(Operation& op, int dimSize) const
{
    if (op.GetOpcode() == Opcode::OP_EXPAND) {
        auto axes = op.GetVectorIntAttribute(OpAttributeKey::expandDims);
        bool updated = false;
        for (auto& axis : axes) {
            if (axis == dimSize - NUM2) {
                axis = axis + 1;
                updated = true;
            }
        }
        if (updated) {
            op.SetAttribute(OpAttributeKey::expandDims, axes);
        }
    }
    // 隐式expand场景
    if (dimSize >= NUM2 && op.HasAttr(OpAttributeKey::brcOperand)) {
        auto brcOperand = op.GetVectorIntAttribute(OpAttributeKey::brcOperand);
        int operand = brcOperand[dimSize - NUM2];
        if (operand != 0) {
            brcOperand[dimSize - NUM2] = static_cast<int64_t>(0);
            brcOperand[dimSize - 1] = static_cast<int64_t>(operand);
            op.SetAttribute(OpAttributeKey::brcbIdx, static_cast<int64_t>(operand));
            op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
    }
}

inline bool SkipInputCombineOps3510(Operation& op)
{
    if (SUPPORT_BRC_INLINE.count(op.GetOpcode()) == 0) {
        return false;
    }
    auto lhs = op.GetIOperands()[0];
    auto rhs = op.GetIOperands()[1];
    if ((lhs->GetShape() == rhs->GetShape()) ||
        (lhs->tensor->rawshape.back() == 1 && lhs->tensor->rawshape.back() == rhs->tensor->rawshape.back())) {
        return false;
    }
    return true;
}

inline bool SkipInputCombineOps(Operation& op, int dimSize)
{
    if (op.GetOpcode() == Opcode::OP_BRCB) {
        return false;
    }
    if (op.GetOpcode() == Opcode::OP_EXPAND) {
        auto axes = op.GetVectorIntAttribute(OpAttributeKey::expandDims);
        for (auto& axis : axes) {
            if (axis == dimSize - NUM1) { // 尾轴expand不支持换轴
                return false;
            }
        }
    }
    return true;
}

Status CodegenPreproc::ForceCombineAxisForAxisCombine(Function& func) const
{
    for (auto& subProgram : func.rootFunc_->programs_) {
        for (auto& op : subProgram.second->Operations(false)) {
            if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) != OpCoreType::AIV && !IsUBCopy(op)) {
                continue;
            }
            if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && SkipInputCombineOps3510(op)) {
                continue;
            }
            std::vector<bool> inputCombineAxis;
            LogicalTensors inputs = op.GetIOperands();
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i]->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
                    inputs[i]->tensor->rawshape.back() == 1 &&
                    SkipInputCombineOps(op, static_cast<int>(inputs[i]->tensor->rawshape.size()))) {
                    inputCombineAxis.push_back(true);
                } else {
                    inputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::inputCombineAxis, inputCombineAxis);
            std::vector<bool> outputCombineAxis;
            auto outputs = op.GetOOperands();
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (outputs[i]->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
                    outputs[i]->tensor->rawshape.back() == 1 && ReduceNeedCombineAxis(op)) {
                    outputCombineAxis.push_back(true);
                    // OP_EXPAND 只有单输出，此处只会执行一次
                    FixExpandDimForAxisCombine(op, static_cast<int>(outputs[i]->tensor->rawshape.size()));
                } else {
                    outputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::outputCombineAxis, outputCombineAxis);
        }
    }
    return SUCCESS;
}

std::string CodegenPreproc::DumpOpList(Function& function)
{
    std::stringstream ss;
    int idx = 0;
    for (auto& subProgram : function.rootFunc_->programs_) {
        ss << "==================== OP_LIST Codegen_Preproc " << idx << " ====================="
           << "\n";
        for (auto& op : subProgram.second->Operations(false)) {
            if (!op.oOperand.empty()) {
                bool needAlloc = false;
                op.oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "], needAlloc: " << static_cast<int>(needAlloc)
                   << ", memId: " << op.oOperand[0]->memoryrange.memId << "\n";
            } else {
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "]"
                   << "\n";
            }
        }
        idx++;
    }
    return ss.str();
}

void CodegenPreproc::SetNeedAllocAttr(Function& function)
{
    for (auto& subProgram : function.rootFunc_->programs_) {
        std::unordered_set<int> appearedMemId;
        for (auto& op : subProgram.second->Operations(false)) {
            for (auto& outTensor : op.GetOOperands()) {
                if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    continue;
                }
                auto it = appearedMemId.find(outTensor->memoryrange.memId);
                if (it == appearedMemId.end()) {
                    outTensor->SetAttr(OpAttributeKey::needAlloc, true);
                    appearedMemId.insert(outTensor->memoryrange.memId);
                }
            }
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "%s", DumpOpList(function).c_str());
}

static void GetEventsInfo(
    int subgraphNum,
    const std::vector<std::set<int>>& subgraphOutGraph,
    const std::vector<int>& subgraphLatency,
    std::vector<std::tuple<int, bool, int>> &events)
{
    std::vector<int> inDegree(subgraphNum, 0);
    for (int i = 0; i < subgraphNum; ++i) {
        for (int consumer : subgraphOutGraph[i]) {
            if (consumer >= 0 && consumer < subgraphNum) {
                inDegree[consumer]++;
            }
        }
    }

    std::vector<int> earliestStart(subgraphNum, 0);
    std::queue<int> q;

    for (int i = 0; i < subgraphNum; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
            earliestStart[i] = 0;
        }
    }

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        if (node < 0 || node >= subgraphNum) {
            continue;
        }

        int endTime = earliestStart[node] + subgraphLatency[node];

        for (int consumer : subgraphOutGraph[node]) {
            if (consumer >= 0 && consumer < subgraphNum) {
                earliestStart[consumer] = std::max(earliestStart[consumer], endTime);
                inDegree[consumer]--;
                if (inDegree[consumer] == 0) {
                    q.push(consumer);
                }
            }
        }
    }

    for (int i = 0; i < subgraphNum; ++i) {
        int startTime = earliestStart[i];
        int endTime = startTime + subgraphLatency[i];
        events.push_back({startTime, true, i});
        events.push_back({endTime, false, i});
    }
}

inline std::pair<int, int> EstimateRequiredCores(
    int subgraphNum,
    const std::vector<bool>& isCubeGraph,
    const std::vector<std::set<int>>& subgraphOutGraph,
    const std::vector<int>& subgraphLatency)
{
    if (subgraphNum == 0) {
        return {0, 0};
    }

    std::vector<std::tuple<int, bool, int>> events;
    GetEventsInfo(subgraphNum, subgraphOutGraph, subgraphLatency, events);

    auto cmp = [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    };
    std::sort(events.begin(), events.end(), cmp);

    int maxCCores = 0;
    int maxVCores = 0;
    int currentCCores = 0;
    int currentVCores = 0;

    for (const auto& event : events) {
        bool isStart = std::get<1>(event);
        int nodeId = std::get<2>(event);

        if (isStart) {
            if (isCubeGraph[nodeId]) {
                currentCCores++;
                maxCCores = std::max(maxCCores, currentCCores);
            } else {
                currentVCores++;
                maxVCores = std::max(maxVCores, currentVCores);
            }
        } else {
            if (isCubeGraph[nodeId]) {
                currentCCores--;
            } else {
                currentVCores--;
            }
        }
    }

    return {maxCCores, maxVCores};
}

inline void EstimateCVCores(Function &function)
{
    int subgraphNum = function.GetTotalSubGraphCount();
    std::vector<bool> isCubeGraph(subgraphNum, false);
    std::vector<std::set<int>> subgraphOutGraph(subgraphNum);
    std::vector<int> subgraphLatency(subgraphNum, 0);
    for (auto &op : function.Operations()) {
        int subgraphID = op.GetSubgraphID();
        if (op.HasAttribute(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube)) {
            isCubeGraph[subgraphID] = true;
        }
        subgraphLatency[subgraphID] += op.GetLatency();
        for (auto nextOp : op.ConsumerOps()) {
            if (nextOp->GetSubgraphID() != subgraphID) {
                subgraphOutGraph[subgraphID].insert(nextOp->GetSubgraphID());
            }
        }
    }
    auto maxCVCores = EstimateRequiredCores(subgraphNum, isCubeGraph, subgraphOutGraph, subgraphLatency);
    function.SetMaxCVCoreUsage(maxCVCores);

    if (function.GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH) {
        Function *rootFuntion = function.GetRootFunction();
        if (rootFuntion != nullptr) {
            rootFuntion->SetMaxCVCoreUsage(maxCVCores);
        }
    }
}

Status CodegenPreproc::RunOnFunction(Function &function)
{
    EstimateCVCores(function);
    combineAxis = function.paramConfigs_.combineAxis;
    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Start CodegenPreproc.");
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW_TYPE) {
            op.SetOpCode(Opcode::OP_VIEW);
        }
    }
    if (SaveGmTensorParamIdxToOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "CodegenPreproc RunOnFunction failed at function SaveGmTensorParamIdxToOp.");
        return FAILED;
    }

    if (combineAxis) {
        if (ForceCombineAxisForAxisCombine(function) != SUCCESS) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "CodegenPreproc RunOnFunction failed at function ForceCombineAxisForAxisCombine.");
            return FAILED;
        }
    }

    SetNeedAllocAttr(function);
    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Finish CodegenPreproc.");
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu

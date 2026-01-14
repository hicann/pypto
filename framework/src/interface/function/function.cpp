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
 * \file function.cpp
 * \brief
 */

#include "interface/function/function.h"
#include <queue>
#include <algorithm>
#include <unordered_map>
#include "interface/inner/pre_def.h"
#include "interface/cache/hash.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/tensor_offset.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation_impl.h"
#include "interface/utils/serialization.h"
#include "interface/interpreter/flow_verifier.h"

using namespace npu::tile_fwk;

namespace {
const std::string PREFIX = "  ";
const int SPACE_NUM_THREE = 3;
const int LAST_TWO = -2;

const std::set<Opcode> SPECIAL_OPCODE_SET = {
    Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE, Opcode::OP_CALL, Opcode::OP_CONVERT,
    Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT
};
struct ViewKey {
    ViewKey(const int magic, const std::vector<int64_t> &newShape,
            const std::vector<int64_t> &newOffset,
            const std::vector<SymbolicScalar> &tmpDynOffset)
        : rawMagic(magic), shape(newShape), offset(newOffset), dynOffset(tmpDynOffset) {}

    bool operator<(const ViewKey &x) const {
        if (shape != x.shape) {
            return shape < x.shape;
        } else if (offset != x.offset) {
            return offset < x.offset;
        }
        if (dynOffset.size() != x.dynOffset.size()) {
            return dynOffset.size() < x.dynOffset.size();
        }
        for (size_t i = 0; i < dynOffset.size(); i++) {
            if (dynOffset[i].Raw() != x.dynOffset[i].Raw()) {
                return dynOffset[i].Raw() < x.dynOffset[i].Raw();
            }
        }

        return rawMagic < x.rawMagic;
    }

    int rawMagic;
    Shape shape;
    Offset offset;
    std::vector<SymbolicScalar> dynOffset;
};
} // namespace

std::vector<Operation *> OperationsViewer::DuplicatedOpList() const {
    std::vector<Operation *> opList;
    opList.reserve(operations_.size());
    for (auto op : operations_) {
        opList.emplace_back(op.get());
    }
    return opList;
}

struct CompareTensorPtr {
    bool operator()(const std::shared_ptr<LogicalTensor> &a,
                    const std::shared_ptr<LogicalTensor> &b) const {
        return a->offset < b->offset;
    }
};

std::string DynloopFunctionPathNode::Dump() const
{
    int indent = 2;
    std::ostringstream oss;
    std::function<void(const DynloopFunctionPathNode *, int)> dump =
        [&oss, &indent, &dump](const DynloopFunctionPathNode *node, int level) {
        if (!node->cond.IsValid()) {
            oss << std::setw(level * indent) << ' '
                << node->root->GetRawName()
                << "(" << node->root->GetFunctionHash() << ")\n";
        } else {
            oss << std::setw(level * indent) << ' '
                << node->cond.Dump() << "\n";
            if (node->branchNodeList[0] != nullptr) {
                dump(node->branchNodeList[0].get(), level + 1);
            }
            if (node->branchNodeList[1] != nullptr) {
                dump(node->branchNodeList[1].get(), level + 1);
            }
        }
    };
    dump(this, 0);
    return oss.str();
}

std::shared_ptr<DynloopFunctionPathNode> DynloopFunctionAttribute::BuildPathNode() {
    std::shared_ptr<DynloopFunctionPathNode> root = std::make_shared<DynloopFunctionPathNode>();
    if (pathList.size() == 1) {
        root->root = pathList[0].GetRoot();
    } else {
        for (size_t i = 0; i < pathList.size(); i++) {
            auto node = root;
            for (size_t j = 0; j < pathList[i].pathCondList.size(); j++) {
                auto &pathCond = pathList[i].pathCondList[j];
                if (!node->cond.IsValid()) {
                    node->cond = pathCond.GetCond();
                }
                if (node->branchNodeList[pathCond.IsSat()] == nullptr) {
                    node->branchNodeList[pathCond.IsSat()] = std::make_shared<DynloopFunctionPathNode>();
                }
                node = node->branchNodeList[pathCond.IsSat()];
            }
            node->root = pathList[i].GetRoot();
        }
    }
    return root;
}

std::string DynloopFunctionAttribute::DumpBranch() const {
    std::ostringstream oss;
    for (size_t i = 0; i < pathList.size(); i++) {
        auto &path = pathList[i];
        oss << "Branch-" << i << ": " << "\n";
        for (size_t j = 0; j < path.pathCondList.size(); j++) {
            auto &cond = path.pathCondList[j];
            oss << "  " << cond.GetFile()
                << ":" << cond.GetLine()
                << "] " << cond.GetCond().Dump()
                << ":" << cond.IsSat() << "\n";
        }
    }
    oss << "current:" << currIndex << "\n";
    for (size_t i = 0; i < currPathCond.size(); i++) {
        oss << "  " << currPathCond[i].GetFile()
            << ":" << currPathCond[i].GetLine()
            << "] " << currPathCond[i].GetCond().Dump()
            << ":" << currPathCond[i].IsSat() << "\n";
    }
    return oss.str();
}

std::vector<DynloopFunctionPathCondition> DynloopFunctionAttribute::GenCondWithBeginEnd(
    const std::vector<DynloopFunctionPathCondition> &conds) const {
    std::vector<DynloopFunctionPathCondition> resultPathCond = conds;
    for (auto &cond : resultPathCond) {
        if (!cond.cond_.IsExpression()) {
            continue;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(cond.cond_.Raw());
        if (expr->IsLoopEndCall()) {
            std::vector<RawSymbolicScalarPtr> operandList{
                expr->OperandList()[0],
                expr->OperandList()[1],
                RawSymbolicExpression::CreateBopSub(expr->OperandList()[2], loopRange.Step().Raw())
            };
            auto newExpr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL, operandList);
            cond.cond_ = SymbolicScalar(newExpr);
        }
    }
    return resultPathCond;
}

bool DynloopFunctionAttribute::IterationEnd(int unroll, Function *pathFunc, Operation *operation) {
    auto resultPathCond = GenCondWithBeginEnd(currPathCond);
    unrollTimes = unroll;
    pathList.emplace_back(pathFunc, resultPathCond, operation);

    bool finished = true;
    for (size_t idx = 0; idx < currPathCond.size(); idx++) {
        if (!currPathCond[idx].IsSat() && !currPathCond[idx].isConst_) {
            const auto &cond = currPathCond[idx].cond_;
            if (IsLoopBeginOrEndExpr(cond)) {
                if (!cond.IsLoopBegin() && !cond.IsLoopEnd()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(
                    cond.Raw())->IsLoopBeginCall() && !cond.IsLoopBegin()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(
                    cond.Raw())->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            finished = false;
            break;
        }
    }
    return finished;
}

bool DynloopFunctionAttribute::GuessCondResult(const SymbolicScalar &cond, bool &result) {
    if (cond.ConcreteValid()) {
        result = cond.Concrete();
        return true;
    }
    auto condstr = cond.Dump();
    for (auto &pcond : currPathCond) {
        if (condstr == pcond.GetCond().Dump()) {
            result = pcond.IsSat();
            return true;
        }
    }
    return false;
}

bool DynloopFunctionAttribute::AppendCond(const SymbolicScalar &cond, const std::string &file, int line) {
    bool result = false;
    if (currIndex < currPathCond.size()) {
        result = currPathCond[currIndex].IsSat();
    } else {
        bool isConst = GuessCondResult(cond, result);
        currPathCond.emplace_back(result, isConst, cond, file, line);
    }
    currIndex++;
    return result;
}

void DynloopFunctionAttribute::CreateCurrCond() {
    if (pathList.size() == 0) {
        currPathCond.clear();
        currIndex = 0;
        return;
    }
    for (size_t idx = currPathCond.size() - 1; idx != static_cast<size_t>(-1); idx--) {
        if ((!currPathCond[idx].IsSat()) && (!currPathCond[idx].isConst_)) {
            const auto &cond = currPathCond[idx].cond_;
            if (cond.IsExpression()) {
                auto expr = std::static_pointer_cast<RawSymbolicExpression>(cond.Raw());
                if (expr->IsLoopBeginCall() && !cond.IsLoopBegin()) {
                    continue;
                }
                if (expr->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            currPathCond[idx].IsSat() = true;
            currPathCond.erase(currPathCond.begin() + idx + 1, currPathCond.end());
            break;
        }
    }
    currIndex = 0;
}

Function::Function(const Program &belongTo, const std::string &funcMagicName,
                   const std::string &funcRawName, Function *parentFunc)
                 : funcMagicName_(funcMagicName), funcRawName_(funcRawName),
                   tensorMap_(*this), belongTo_(belongTo) {
    parent_ = parentFunc;
    functionMagic_ = IdGen<IdType::FUNCTION>::Inst().NewId();

    opSeed_ = FUNCTION_MAX_INCASTS;
}

OperationsViewer Function::Operations(bool sorted) {
    if (!sorted_ && sorted) {
        sorted_ = true;
        SortOperations();
    }
    return OperationsViewer(operations_, opPosition_);
}

bool Function::IsCube() const {
    auto isL1CopyIn = [](const Operation &op) {
        return (op.GetOpcode() == Opcode::OP_COPY_IN &&
                !(op.oOperand.empty()) &&
                op.oOperand[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) ||
                op.GetOpcode() == Opcode::OP_GATHER_IN_L1;
    };

    for (const auto &oper : OperationsViewer(operations_, opPosition_)) {
        if (isL1CopyIn(oper)) {
            return true;
        }
    }
    return false;
}

OperationsViewer Function::OperationsAfterOOO()
{
    return OperationsViewer(operationsAfterOOO_, opPositionAfterOOO_);
}

void Function::RecordOOOSeq()
{
    operationsAfterOOO_ = operations_;
    opPositionAfterOOO_ = opPosition_;
}

std::vector<OperationPtr> &Function::GetProgramOp() {
    ASSERT(graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::GetProgramOp called. Current graph type: " << static_cast<int>(graphType_);
    return operations_;
}

void Function::SetProgramOp(const std::vector<OperationPtr> &operations) {
    ASSERT(graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::SetProgramOp called. Current graph type: " << static_cast<int>(graphType_);
    operations_ = operations;

    RefreshOpPosition();
    sorted_ = true;
}

void Function::UpdateBelongToThis() {
    ASSERT(graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::UpdateBelongToThis called. Current graph type: " << static_cast<int>(graphType_);
    for (auto &ele : operations_) {
        ele->function_ = this;
    }
}

const SubfuncInvokeInfoTy &Function::GetSubFuncInvokeInfo(const size_t i) const {
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(operations_[i]->GetOpAttribute());
    ASSERT(callAttr != nullptr)
        << "Operation at index " << i << " must have a CallOpAttribute";
    return *(callAttr->invokeInfo_);
}

int Function::GetParamIndex(const std::shared_ptr<RawTensor> &rawTensor) {
    if (slotScope_ == nullptr) {
        return -1;
    }
    auto slots = slotScope_->LoopupArgSlot(rawTensor);
    for (auto slot : slots) {
        for (int idx = 0; idx < (int)explicitArgSlots_.size(); idx++) {
            if (slot == explicitArgSlots_[idx]) {
                return idx;
            }
        }
    }
    return -1;
}

void *Function::GetParamAddress(int index) {
    return explicitArgAddrs_[index];
}

bool Function::HasCallOperation() {
    for (const auto &op : Operations()) {
        if (op.GetOpcode() == Opcode::OP_CALL) {
            return true;
        }
    }
    return false;
}

void Function::CreateLeafInAndOutCast(const LogicalTensorPtr &inOrOut,
                                      LogicalTensors &inOrOutList) const {
    inOrOutList.emplace_back(inOrOut->Clone(*parent_));
}

static int GetTensorDataLookupOutcast(Function *func, Operation *import) {
    auto importTensor = import->GetIOperands()[0];
    auto consumerSet = importTensor->GetConsumers();
    if (consumerSet.size() != 2) {
        return INVALID_IOINDEX;
    }
    for (auto consumer : consumerSet) {
        if (consumer != import) {
            auto outcast = consumer->GetOOperands()[0];
            auto outcastIndex = func->GetOutcastIndex(outcast);
            return outcastIndex;
        }
    }
    return INVALID_IOINDEX;
}

static int GetTensorDataLookupIncast(Function *func, Operation *import) {
    auto importTensor = import->GetIOperands()[0];
    auto producerSet = importTensor->GetProducers();
    if (producerSet.size() != 1) {
        return INVALID_IOINDEX;
    }
    auto producer = *producerSet.begin();
    auto incast = producer->GetIOperands()[0];
    auto incastIndex = func->GetIncastIndex(incast);
    return incastIndex;
}

GetTensorDataIODescDict Function::GetTensorDataForTensorGraph() {
    GetTensorDataIODescDict iodescDict;
    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    if (currDynFunc == nullptr) {
        return iodescDict;
    }
    auto currDynAttr = currDynFunc->GetDyndevAttribute();
    for (auto &op : Operations(false)) {
        if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_IMPORT)) {
            continue;
        }
        int getTensorDataIndex = GetTensorDataGetIndex(&op);
        ASSERT(getTensorDataIndex != -1)
            << "Failed to get tensor data index for operation";
        ASSERT(currDynAttr->getTensorDataUsageDict.count(this))
            << "Current function not found in getTensorDataUsageDict";
        std::unordered_map<int, Operation *> &importDict = currDynAttr->getTensorDataUsageDict[this].importDict;
        ASSERT(importDict.count(getTensorDataIndex))
            << "Import index " << getTensorDataIndex << " not found in importDict";
        auto import = importDict[getTensorDataIndex];
        int outcastIndex = GetTensorDataLookupOutcast(this, import);
        if (outcastIndex != INVALID_IOINDEX) {
            iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, outcastIndex, 0);
        } else {
            int incastIndex = GetTensorDataLookupIncast(this, import);
            if (incastIndex != INVALID_IOINDEX) {
                iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
            } else {
                // Impossible
                ASSERT(false)
                    << "Both outcast and incast indices are invalid";
            }
        }
    }
    return iodescDict;
}

GetTensorDataIODescDict Function::GetTensorDataForLeafGraph() {
    GetTensorDataIODescDict iodescDict;
    for (auto &op : Operations(false)) {
        if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_IMPORT)) {
            continue;
        }
        int getTensorDataIndex = GetTensorDataGetIndex(&op);
        ASSERT(getTensorDataIndex != -1)
            << "Failed to get tensor data index for operation";
        auto tensor = op.GetIOperands()[0];
        auto incastIndex = GetIncastIndex(tensor);
        if (incastIndex != INVALID_IOINDEX) {
            iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
        }
    }
    return iodescDict;
}

void Function::GetTensorDataRefreshIO(const GetTensorDataIODescDict &iodescDict) {
    for (auto &op : Operations(false)) {
        std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList = op.GetDynamicAttributeList();
        for (auto &attr : dynamicAttributeList) {
            attr.get() = GetTensorDataFillIO(iodescDict, attr.get());
        }
    }
}

void Function::BeginFunction(const std::vector<std::reference_wrapper<const Tensor>> &explicitOpArgs) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    for (auto &arg : explicitOpArgs) {
        explicitArgSlots_.push_back(TensorSlot::CreateTensor(arg));
        explicitArgAddrs_.push_back(arg.get().GetData());
    }
}

bool HasCalleeConsumer(Function &func, Function &calleeFunc, size_t outcastIdx) {
    auto outcast = calleeFunc.GetOutcast()[outcastIdx];
    auto outcastSlots = calleeFunc.GetOutCastSlot(outcast);
    for (auto otherCallee : func.GetCalleeFunctionList()) {
        ASSERT(otherCallee != nullptr) << func.GetRawName() << "has nullptr callee";
        for (auto &incast : otherCallee->GetIncast()) {
            auto incastSlots = otherCallee->GetInCastSlot(incast);
            if (TensorSlotManager::HasSameSlot(incastSlots, outcastSlots)) {
                return true;
            }
        }
    }
    return false;
}

void CalleeSlotNoConsumer(Function &calleeFunc, Function &func, const std::map<size_t, size_t> &outcasts,
    std::map<size_t, size_t> &outcastIdx2parent) {
    for (size_t calleeOutcastIdx = 0; calleeOutcastIdx < calleeFunc.GetOutcast().size(); calleeOutcastIdx++) {
        auto caleeOutcast = calleeFunc.GetOutcast()[calleeOutcastIdx];
        auto incastSlots = calleeFunc.GetOutCastSlot(caleeOutcast);
        for (const auto &[outcastIdx, val] : outcasts) {
            (void)val;
            auto outcast = func.GetOutcast()[outcastIdx];
            auto outcastSlots = func.GetOutCastSlot(outcast);
            if (TensorSlotManager::HasSameSlot(incastSlots, outcastSlots) &&
                !HasCalleeConsumer(func, calleeFunc, calleeOutcastIdx)) {
                outcastIdx2parent[calleeOutcastIdx] = outcastIdx;
                break;
            }
        }
    }
}

void Function::EraseCallOpOpnd(const FunctionHash &calleeHash, size_t index) {
    for (auto callop : GetCallopList()) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        ASSERT(callopAttr != nullptr) << "Processing CallOp:" << callop->Dump();
        if (callopAttr->GetCalleeHash() != calleeHash) {
            continue;
        }
        ASSERT(index < callop->oOperand.size())
            << "Index " << index
            << " out of bounds for oOperand size " << callop->oOperand.size();
        ASSERT(callop->oOpAttrOffset.empty())
            << "oOpAttrOffset is not empty for CallOp:"
            << callop->Dump();
        ASSERT(callopAttr->GetArgList().empty())
            << "ArgList is not empty for CallOp:"
            << callop->Dump();
        ASSERT(callopAttr->GetOutCastIndexToExpr().empty())
            << "OutCastIndexToExpr is not empty for CallOp:"
            << callop->Dump();
        for (auto &consumer : callop->oOperand[index]->GetConsumers()) {
            if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                consumer->SetAsDeleted();
            }
        }
        callop->oOperand.erase(callop->oOperand.begin() + index);
    }
    EraseOperations(true, false);
}

void Function::CheckAndUpdateGetTensorData(size_t currOutcastIdx, size_t newOutcastIdx) {
    for (auto &op : Operations(false)) {
        if (!op.IsCall()) {
            for (auto &attr : op.GetDynamicAttributeList()) {
                attr.get() = UpdateGetTensorDataIOIndex(currOutcastIdx, newOutcastIdx, attr.get());
            }
        }
    }
}

void Function::CleanRedundantOutcast(
    std::map<Function *, std::set<size_t>> &removeRecord, std::map<Function*, std::set<size_t>> &getTensorDataRecord) {
    for (auto &[func, removeList] : removeRecord) {
        for (auto it = removeList.rbegin(); it != removeList.rend(); ++it) {
            auto outCastIdx = *it;
            func->Parent().EraseCallOpOpnd(func->GetFunctionHash(), outCastIdx);
            func->RemoveOutcast(outCastIdx);
        }
        if (getTensorDataRecord.count(func) <= 0) {
            continue;
        }
        auto &tensorDataList = getTensorDataRecord[func];
        for (auto currOutcastIdx : tensorDataList) {
            auto it = std::lower_bound(removeList.begin(), removeList.end(), currOutcastIdx);
            size_t newOutcastIdx = currOutcastIdx - std::distance(removeList.begin(), it);
            if (currOutcastIdx != newOutcastIdx) {
                func->CheckAndUpdateGetTensorData(currOutcastIdx, newOutcastIdx);
            }
        }
    }
}

void RedundantOutCastCheck(std::map<Function *, std::set<size_t>> &removeRecord,
    std::map<Function *, std::set<size_t>> &getTensorDataRecord, Function *func, std::map<size_t, size_t> &outcasts) {
    for (auto calleeFunc : func->GetCalleeFunctionList()) {
        ASSERT(calleeFunc != nullptr) << func->GetMagicName() << "has nullptr calleeFunc";
        std::map<size_t, size_t> outcastIdx2parent; // key: callee outcastIdx, value: caller outcastIdx
        CalleeSlotNoConsumer(*calleeFunc, *func, outcasts, outcastIdx2parent);
        if (!outcastIdx2parent.empty()) {
            RedundantOutCastCheck(removeRecord, getTensorDataRecord, calleeFunc, outcastIdx2parent);
        }
        auto &calleeOutCasts = calleeFunc->GetOutcast();
        for (auto &[outCastIdx, val] : outcastIdx2parent) {
            (void)val;
            ASSERT(calleeOutCasts[outCastIdx].get() != nullptr)
                << "Outcast at index " << outCastIdx << " should not be null";
            if (calleeOutCasts[outCastIdx]->IsGetTensorDataOutcast()) {
                getTensorDataRecord[calleeFunc].insert(outCastIdx);
                ASSERT(outcastIdx2parent.count(outCastIdx) > 0)
                    << "Outcast index " << outCastIdx << " should be in outcastIdx2parent";
                getTensorDataRecord[func].insert(outcastIdx2parent[outCastIdx]);
            } else if (getTensorDataRecord[calleeFunc].count(outCastIdx) > 0) {
                ASSERT(outcastIdx2parent.count(outCastIdx) > 0)
                    << "Outcast index " << outCastIdx << " should be in outcastIdx2parent";
                getTensorDataRecord[func].insert(outcastIdx2parent[outCastIdx]);
            } else {
                removeRecord[calleeFunc].insert(outCastIdx);
            }
        }
    }
}

void Function::CleanRedundantOutCast() {
    auto slotMngr = Program::GetInstance().GetTensorSlotManager();
    std::vector<int> outputSlots;
    for (const auto &slot : slotMngr->outputSlotList) {
        outputSlots.push_back(slotMngr->slotIndexDict[slot]);
    }

    // support flatten input output
    if (slotMngr->outputSlotList.size() == 0) {
        for (const auto &slot : slotMngr->inputSlotList) {
            outputSlots.push_back(slotMngr->slotIndexDict[slot]);
        }
    }
    std::map<Function *, std::set<size_t>> removeRecord;
    std::map<Function *, std::set<size_t>> getTensorDataRecord;
    auto calleeLists = GetCalleeFunctionList();
    auto &calleeOutCasts = GetOutcast();
    std::map<size_t, size_t> outputMap;
    for (size_t outCastIdx = 0; outCastIdx < calleeOutCasts.size(); outCastIdx++) {
        auto outcastSlots = GetOutCastSlot(calleeOutCasts[outCastIdx]);
        if ((!TensorSlotManager::HasSameSlot(outputSlots, outcastSlots)) &&
            !HasCalleeConsumer(Parent(), *this, outCastIdx)) {
            outputMap[outCastIdx] = 0;
        }
    }
    if (!outputMap.empty()) {
        RedundantOutCastCheck(removeRecord, getTensorDataRecord, this, outputMap);
    }
    for (auto &[outCastIdx, val] : outputMap) {
        (void)val;
        if (getTensorDataRecord[this].count(outCastIdx) > 0) {
            ASSERT(outputMap.count(outCastIdx) > 0)
                << "outputMap does not contain outCastIdx " << outCastIdx;
            getTensorDataRecord[parent_].insert(outputMap[outCastIdx]);
        } else {
            removeRecord[this].insert(outCastIdx);
        }
    }
    CleanRedundantOutcast(removeRecord, getTensorDataRecord);
}

FunctionCallArgs Function::EndFunction(const std::shared_ptr<TensorSlotScope> &scope) {
    // Deduce Incast and Outcast here, need by TENSOR_GRAPH & STATIC_TILE_GRAPH
    std::vector<Operation *> operationList = Operations(false).DuplicatedOpList();
    if (IsGraphType(GraphType::TENSOR_GRAPH) ||
        IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        OrderedSet<LogicalTensorPtr> incasts;
        OrderedSet<LogicalTensorPtr> outcasts;

        for (auto &op : operationList) {
            for (auto &iOperand : op->iOperand) {
                if (op->IsCall() || (tensorMap_.tensorMap_.count(iOperand->tensor->rawmagic) == 0 &&
                   (&iOperand->BelongFunction() != this))) {
                    incasts.Insert(iOperand);
                }
            }
            for (auto &oOperand : op->oOperand) {
                if (op->IsCall() || oOperand->tensor->GetRefCount() > 0) {
                    outcasts.Insert(oOperand);
                    ASSERT(incasts.count(oOperand) == 0)
                        << "Error: Output operand " << oOperand->tensor->rawmagic
                        << " is found in incasts. Operation: " << op->Dump();
                }
            }
        }
        for (const auto &incast : incasts) {
            AddOriginIncast(incast);
        }
        for (const auto &outcast : outcasts) {
            AddOriginOutcast(outcast);
        }
    }

    LogicalTensors inArgumentList, outArgumentList;
    if (IsGraphType(GraphType::TENSOR_GRAPH) ||
        IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        SetCallOpSlot();
        inArgumentList = MakeIncasts(scope);
        outArgumentList = MakeOutcasts(scope);
        auto iodescDict = GetTensorDataForTensorGraph();
        GetTensorDataRefreshIO(iodescDict);
        /* tensor graph function should keep the order of operations */
        AddOperationGroup(operationList);
        SortOperations();
        ClearOperationGroups();
        if (Program::GetInstance().GetCurrentDynamicFunction()) {
            DyndevFunctionAttribute::ValueDependDesc desc = LookupValueDepend();
            auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
            if (currDynFuncAttr != nullptr) {
                currDynFuncAttr->valueDependDescDict[this] = desc;
            }
        }
    } else if (graphType_ == GraphType::EXECUTE_GRAPH) {
    } else if (graphType_ == GraphType::BLOCK_GRAPH) {
        for (auto &out : outCasts_) {
            CreateLeafInAndOutCast(out, outArgumentList);
        }
        for (auto &in : inCasts_) {
            /* actualRawmagic存在的场景下，前序 LogicalTensor 应该已经创建好，直接获取 */
            CreateLeafInAndOutCast(in, inArgumentList);
        }
        for (const auto &op : operations_) {
            opSeed_ = std::max(opSeed_, op->GetOpMagic() + 1);
        }
    } else {
        ASSERT(false) << "Not support connecting other type of function currently";
    }
    std::vector<int> iOffset;
    std::vector<int> oOffset;
    std::map<int, SymbolicScalar> outIndexToExpr;
    std::vector< std::vector<SymbolicScalar>> argList;
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        argList = NormalizeCoa(iOffset, oOffset);
        GetOutcastSymbolicExpr(outIndexToExpr);
    }
    ComputeHash();
    return {std::move(inArgumentList), std::move(outArgumentList),
            std::move(iOffset), std::move(oOffset),
            std::move(outIndexToExpr), std::move(argList)};
}

void Function::AddWhenNotExistOrAssert(const std::shared_ptr<LogicalTensor> &tensor,
                                             std::map<int, int> &magicToRawMagic,
                                             std::map<int, std::shared_ptr<LogicalTensor>> &magicToLogicalTensor) {
    if (auto it = magicToRawMagic.find(tensor->magic); it != magicToRawMagic.end()) {
        if (it->second != tensor->tensor->GetRawMagic()) {
            ALOG_INFO("Diff Magic Same RawMagic: ", it->second, magicToLogicalTensor[tensor->magic]->Dump(),
                      tensor->tensor->GetRawMagic(), tensor->Dump());
        }
    }
    magicToRawMagic[tensor->magic] = tensor->tensor->GetRawMagic();
    magicToLogicalTensor[tensor->magic] = tensor;
}

// Tensor magic should be only in the function, so same magic have same rawmagic
void Function::TensorMagicCheck() const {
    std::map<int, int> magicToRawMagic;
    std::map<int, std::shared_ptr<LogicalTensor>> magicToLogicalTensor;
    for (const auto &op : operations_) {
        std::map<int, int> subGraphIDCount;
        // Count subGraphID occurrences in iOperand
        for (const auto &tensor : op->iOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }

        // Count subGraphID occurrences in oOperand
        for (const auto &tensor : op->oOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }
    }
}

void Function::OperationLoopCheck(const std::string &errorMsg) {
    std::map<LogicalTensor *, std::vector<Operation *>> producers;
    std::map<LogicalTensor *, std::vector<Operation *>> consumers;

    for (auto &&op : operations_) {
        for (auto &&iop : op->GetIOperands()) {
            consumers[iop.get()].emplace_back(op.get());
        }
        for (auto &&oop : op->GetOOperands()) {
            producers[oop.get()].emplace_back(op.get());
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    for (auto &&op : operations_) {
        int dupOpMagic = -1;
        auto cycleDetection = [&states, &dupOpMagic, &consumers](Operation *curr, auto self) -> bool {
            int magic = curr->GetOpMagic();
            if (states[magic] == DfsState::DONE) {
                return false;
            }

            if (states[magic] == DfsState::IN_STACK) {
                dupOpMagic = magic;
                ALOG_ERROR("[OperationLoopCheck] Cycle detected: ");
                ALOG_ERROR("[OperationLoopCheck]     Operation: ", curr->Dump());
                return true;
            }

            states[magic] = DfsState::IN_STACK;

            for (auto &&oop : curr->GetOOperands()) {
                for (auto *consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (dupOpMagic != -1) {
                            ALOG_ERROR("[OperationLoopCheck]     Tensor:    ", oop->Dump());
                            ALOG_ERROR("[OperationLoopCheck]     Operation: ", curr->Dump());
                            if (magic == dupOpMagic) {
                                dupOpMagic = -1; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[magic] = DfsState::DONE;

            return false;
        };

        ASSERT(!cycleDetection(op.get(), cycleDetection)) << errorMsg;
    }
}

bool Function::OperationLoopCheck()
{
    std::unordered_map<Operation*, int> inLinkNum;
    std::unordered_set<Operation*> visitedOp;
    std::vector<Operation*> visitStack;
    for (std::shared_ptr<Operation> op : operations_) {
        inLinkNum[op.get()] = op->ProducerOps().size();
        if (inLinkNum[op.get()] == 0) {
            visitStack.push_back(op.get());
        }
    }
    while (!visitStack.empty()) {
        Operation* currOp = visitStack.back();
        visitStack.pop_back();
        visitedOp.insert(currOp);
        for (Operation* nextOp : currOp->ConsumerOps()) {
            inLinkNum[nextOp] -= 1;
            if (inLinkNum[nextOp] == 0) {
                visitStack.push_back(nextOp);
            }
            if (inLinkNum[nextOp] < 0) {
                ALOG_ERROR("[OperationLoopCheck]     Operation:", nextOp->Dump());
                return false;
            }
        }
    }
    if (visitedOp.size() != operations_.size()) {
        ALOG_ERROR_F("[OperationLoopCheck]     Loop Detected.");
        return false;
    }
    return true;
}

void Function::GetAnIslandIncastsOutcasts(const std::map<int, int> &opToSubgraph, const int subgraphID,
    const std::vector<Operation *> &operations, std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    std::vector<std::shared_ptr<LogicalTensor>> &oOperands) const {
    std::set<std::shared_ptr<LogicalTensor>> allLogicalTensors;
    std::set<std::shared_ptr<LogicalTensor>> notOutcasts;
    std::set<std::shared_ptr<LogicalTensor>> notIncasts;
    for (const auto &opPtr : operations) {
        const auto &op = *opPtr;
        for (auto &&operand : op.GetIOperands()) {
            allLogicalTensors.insert(operand);
            bool usedbyotherfunction = false;
            for (auto &consumer : operand->GetConsumers()) {
                auto magic = consumer->GetOpMagic();
                if (consumer->GetOpcode() == Opcode::OP_CALL) {
                    continue;
                }
                ASSERT(opToSubgraph.find(magic) != opToSubgraph.end())
                    << "Consumer magic " << magic << " not found in opToSubgraph. " << "\n"
                    << "Operation: " << op.Dump();

                if (opToSubgraph.at(magic) != subgraphID) {
                    usedbyotherfunction = true;
                    break;
                }
            }
            if (!usedbyotherfunction) {
                notOutcasts.insert(operand);
            }
        }

        for (auto &&operand : op.GetOOperands()) {
            allLogicalTensors.insert(operand);
            notIncasts.insert(operand);
        }
    }
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notIncasts.begin(), notIncasts.end(),
        std::inserter(iOperands, iOperands.begin()));
    std::set<std::shared_ptr<LogicalTensor>> tryOOperands;
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notOutcasts.begin(), notOutcasts.end(),
        std::inserter(tryOOperands, tryOOperands.begin()));
    std::set_difference(tryOOperands.begin(), tryOOperands.end(), iOperands.begin(), iOperands.end(),
        std::inserter(oOperands, oOperands.begin()));

    std::sort(iOperands.begin(), iOperands.end(), TensorPtrComparator());
    std::sort(oOperands.begin(), oOperands.end(), TensorPtrComparator());
}

auto Function::AnnotateOperation() {
    std::map<int, std::vector<Operation *>> subgraphs;
    std::map<int, int> opToSubgraph;
    for (auto &&op : Operations()) {
        // same op magic shall only appear once
        ASSERT(opToSubgraph.find(op.GetOpMagic()) == opToSubgraph.end())
                        << "Same op magic shall only appear once." << "\n"
                        << "Duplicate OpMagic found: " << op.GetOpMagic() << "\n" << "Operation: " << op.Dump();
        if (op.GetSubgraphID() < 0) {
            ALOG_DEBUG("Op magic: ", op.GetOpMagic(), "less than 0 graph: ", op.GetSubgraphID());
            continue;
        }
        subgraphs[op.GetSubgraphID()].emplace_back(&op);
        opToSubgraph[op.GetOpMagic()] = op.GetSubgraphID();
        ALOG_DEBUG("Operation: ", op.GetOpMagic(), "Belong To subgraph: ", op.GetSubgraphID());
    }

    for (const auto &pair : subgraphs) {
        ALOG_DEBUG("Subgraph ID: ", pair.first);
        for (const auto &op : pair.second) {
            ALOG_DEBUG("Operation: ", op->Dump());
        }
    }
    return std::make_pair(std::move(subgraphs), std::move(opToSubgraph));
}

std::unordered_set<int> Function::LoopCheck() {
    if (totalSubGraphCount_ == 0) {
        return {};
    }
    ALOG_INFO("LoopCheck begin.");

    auto [subgraphs, opToSubgraph] = AnnotateOperation();
    std::map<LogicalTensor *, std::vector<int>> producers;
    std::map<LogicalTensor *, std::vector<int>> consumers;

    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> iOperands;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> oOperands;

    for (auto &&[subgraphID, operations] : subgraphs) {
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        GetAnIslandIncastsOutcasts(opToSubgraph, subgraphID, operations,
                                   iOperands[subgraphID], oOperands[subgraphID]);

        for (auto &&iop : iOperands[subgraphID]) {
            consumers[iop.get()].push_back(subgraphID);
        }
        for (auto &&oop : oOperands[subgraphID]) {
            producers[oop.get()].push_back(subgraphID);
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    std::unordered_set<int> subGraphInCycle;
    for (auto &&[subgraphID, operations] : subgraphs) {
        (void)operations;
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        int duplicatedSubgraphID = -2;
        auto cycleDetection = [&states, &duplicatedSubgraphID, &oOperands, &consumers,
                               &subGraphInCycle](int currSubgraph, auto self) -> bool {
            if (states[currSubgraph] == DfsState::DONE) {
                return false;
            }

            if (states[currSubgraph] == DfsState::IN_STACK) {
                duplicatedSubgraphID = currSubgraph;
                ALOG_ERROR("[Cycle Detection] Cycle detected: ");
                ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                subGraphInCycle.emplace(currSubgraph);
                return true;
            }

            states[currSubgraph] = DfsState::IN_STACK;

            for (auto &&oop : oOperands[currSubgraph]) {
                for (int consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (duplicatedSubgraphID != -2) {
                            ALOG_ERROR("[Cycle Detection]     tensor:      ", oop->Dump());
                            ALOG_ERROR("[producer]=");
                            for (const auto &producer : oop->GetProducers()) {
                                ALOG_ERROR(producer->GetOpMagic());
                            }
                            ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                            subGraphInCycle.emplace(currSubgraph);
                            if (currSubgraph == duplicatedSubgraphID) {
                                duplicatedSubgraphID = -2; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[currSubgraph] = DfsState::DONE;
            return false;
        };
        if (cycleDetection(subgraphID, cycleDetection)) {
            return subGraphInCycle;
        }
    }
    return std::unordered_set<int>{};
}

std::vector<std::shared_ptr<Operation>> Function::GetSortedOperations() const {
    std::unordered_map<const Operation *, int> opToIndex;
    std::unordered_map<const Operation *, std::set<std::pair<int, int>>> usageDict;

    for (size_t idx = 0; idx < operations_.size(); idx++) {
        auto op = operations_[idx].get();
        ASSERT(opToIndex.count(op) == 0)
            << "Duplicate operation found: " << op->Dump();
        opToIndex.emplace(op, idx);
        if (!op->IsCall()) {
            auto attrList = op->GetDynamicAttributeList();
            usageDict[op] = GetTensorDataUsage(attrList);
        }
    }
    std::vector<int> outDegree(operations_.size(), 0);
    std::vector<int> prevOperation(operations_.size(), -1);

    auto addProd = [&] (auto operation, auto ioperand) {
        for (const auto &prod : ioperand->GetProducers()) {
            if (prod->BelongTo() != this || prod == operation) {
                continue;
            }
            ASSERT(opToIndex.count(prod) != 0)
                << "Producer not found in opToIndex: " << prod->Dump();
            outDegree[opToIndex[prod]]++;
        }
    };

    for (auto &op : operations_) {
        for (auto &iop : op->iOperand) {
           addProd(op.get(), iop);
        }
        for (auto &dop : op->dependOperand) {
            addProd(op.get(), dop);
        }
        for (auto [type, index] : usageDict[op.get()]) {
            if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                addProd(op.get(), inCasts_[index]);
            } else if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                addProd(op.get(), outCasts_[index]);
            }
        }
    }
    for (auto &opGroup : operationGroups_) {
        for (size_t idx = 1; idx < opGroup.size(); idx++) {
            prevOperation[opToIndex[opGroup[idx]]] = opToIndex[opGroup[idx - 1]];
            outDegree[opToIndex[opGroup[idx - 1]]]++;
        }
    }
    std::queue<int> q;
    for (size_t idx = 0; idx < operations_.size(); idx++) {
        if (outDegree[idx] == 0) {
            q.emplace(idx);
        }
    }

    auto visit = [&] (auto operation, auto ioperand) {
        for (const auto &producer : ioperand->GetProducers()) {
            if (producer->BelongTo() != this || producer == operation) {
                continue;
            }
            auto nxtOpIndex = opToIndex[producer];
            if (--outDegree[nxtOpIndex] == 0) {
                q.emplace(nxtOpIndex);
            }
        }
    };

    std::vector<std::shared_ptr<Operation>> sortedOperations;
    while (!q.empty()) {
        const auto &op = operations_[q.front()];
        q.pop();
        sortedOperations.emplace_back(op);
        int prevOpIndex = prevOperation[opToIndex[op.get()]];
        if (prevOpIndex >= 0) {
            if (--outDegree[prevOpIndex] == 0) {
                q.emplace(prevOpIndex);
            }
        }
        for (auto &iop : op->iOperand) {
            visit(op.get(), iop);
        }
        for (auto &dop : op->dependOperand) {
            visit(op.get(), dop);
        }
        for (auto [type, index] : usageDict[op.get()]) {
            if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                visit(op.get(), inCasts_[index]);
            } else if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                visit(op.get(), outCasts_[index]);
            }
        }
    }
    for (auto &op : operations_) {
        ASSERT(outDegree[opToIndex[op.get()]] == 0)
            << "cycle detected: " << op->Dump();
    }
    ASSERT(operations_.size() == sortedOperations.size())
        << "Sorted operations size mismatch: " << sortedOperations.size()
        << " and original size " << operations_.size();
    std::reverse(sortedOperations.begin(), sortedOperations.end());
    return sortedOperations;
}

void Function::SortOperations() {
    std::vector<std::shared_ptr<Operation>> sortedOperations = GetSortedOperations();
    operations_ = sortedOperations;
    RefreshOpPosition();
    sorted_ = true;
}

void Function::ScheduleBy(const std::vector<Operation *> &newList, bool needRefresh) {
    if (needRefresh) {
        RefreshOpPosition();
    }
    ASSERT(newList.size() == operations_.size())
        << "Size mismatch: newList size = " << newList.size()
        << ", operations_ size = " << operations_.size();
    std::vector<std::shared_ptr<Operation>> newOperations;
    for (auto op : newList) {
        ASSERT(opPosition_.count(op) > 0)
            << "Operation not found in opPosition_:" << op->Dump();
        newOperations.emplace_back(operations_[opPosition_.at(op)]);
    }
    operations_ = newOperations;
    RefreshOpPosition();

    sorted_ = true;
}

void Function::AddOperationGroup(std::vector<Operation *> operationGroup) {
    size_t groupID = operationGroups_.size();
    for (const auto &operation : operationGroup) {
        ASSERT(operation->GroupID() == NON_GROUP)
            << "Operation already in a group:" << operation->Dump();
        operation->SetGroupID(groupID);
    }
    operationGroups_.emplace_back(std::move(operationGroup));
    sorted_ = false;
}

void Function::ClearOperationGroups() {
    for (auto &opGroup : operationGroups_) {
        for (auto &op : opGroup) {
            op->SetGroupID(NON_GROUP);
        }
    }
    operationGroups_.clear();
}

void Function::CheckGroupValid() const {
    std::unordered_set<const Operation *> inGroupOp;
    for (size_t i = 0; i < operationGroups_.size(); i++) {
        for (auto &operation : operationGroups_[i]) {
            ASSERT(operation->GroupID() == i)
                << "Operation GroupID mismatch:\n" << "Expected: " << i << ", Actual: " << operation->GroupID() << "\n"
                << "Operation:" << operation->Dump();
            ASSERT(inGroupOp.count(operation) == 0) << "Duplicate operation in group:" << operation->Dump();
            inGroupOp.emplace(operation);
        }
    }
    for (const auto &operation : operations_) {
        ASSERT(inGroupOp.count(operation.get()) == (operation->GroupID() != NON_GROUP))
            << "Operation group membership mismatch:\n" << "Operation: " << operation->Dump() << "\n"
            << "GroupID: " << operation->GroupID();
    }
}

void Function::RefreshOpPosition() {
    opPosition_.clear();
    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        ASSERT(opPosition_.count(operations_[idx].get()) == 0)
            << "Duplicate operation found in opPosition_:\n"
            << operations_[idx]->Dump();
        opPosition_.emplace(operations_[idx].get(), idx);
    }
}

bool Function::enableMagicLookupRecord_{false};
std::map<std::pair<int, int>, std::set<Operation *, LogicalTensor::CompareOp>> Function::tensorAndSubgraphToProducer_;

void Function::ProducerMagicLookup(const Function *function, const LogicalTensorPtr &tensor,
                                   const std::set<Operation *, LogicalTensor::CompareOp> &producers,
                                   const int subGraphId, int &index, std::unordered_map<int, int> &magic2index,
                                   std::stringstream &ss)
{
    for (auto &op : producers) {
        if (subGraphId != INT32_MIN && op->GetSubgraphID() != subGraphId) {
            continue;
        }
        if (op->GetOOperands().size() > 1) {
            for (size_t idx = 0; idx < op->GetOOperands().size(); idx++) {
                if (op->GetOutputOperand(idx) == tensor) {
                    ss << "ooperand " << idx << " ";
                }
            }
        }
        bool isInBoundary = OpcodeManager::Inst().IsBoundaryIn(op->GetOpcode());
        if (isInBoundary) {
            /* 除了最高轴之外的所有内轴都纳入到hash的计算中 */
            for (size_t idx = 1; idx < op->iOperand[0]->tensor->rawshape.size(); idx++) {
                ss << op->iOperand[0]->tensor->rawshape[idx] << " ";
            }
        }
        bool isOutBoundary = OpcodeManager::Inst().IsBoundaryOut(op->GetOpcode());
        if (isOutBoundary) {
            /* 除了最高轴之外的所有内轴都纳入到hash的计算中 */
            for (size_t idx = 1; idx < op->oOperand[0]->tensor->rawshape.size(); idx++) {
                ss << op->oOperand[0]->tensor->rawshape[idx] << " ";
            }
        }
        ss << " " << op->GetOpcodeStr(true);
        for (const auto &attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : "
               << op->DumpAttr(attr) << "]";
        }
        if (function->GetGraphType() != GraphType::BLOCK_GRAPH) {
            ss << op->GetTileShape().ToString();
        }
        if (op->GetOpAttribute() != nullptr) {
            if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (!op->oOperand[0]->isSubGraphBoundary) {
                    ss << " " << op->GetOpAttribute()->Dump();
                }
            } else if ((!IsCopyIn(op->GetOpcode()) && !IsCopyOut(op->GetOpcode())) ||
                function->GetGraphType() != GraphType::BLOCK_GRAPH) {
                ss << " " << op->GetOpAttribute()->Dump();
            }
        }
        MagicLookup(function, op->iOperand, subGraphId, index, magic2index, ss);
    }
}

void Function::MagicLookup(const Function *function, const std::vector<LogicalTensorPtr> &operand,
                           const int subGraphId, int &index, std::unordered_map<int, int> &magic2index,
                           std::stringstream &ss)
{
    for (auto &t : operand) {
        if (magic2index.count(t->GetMagic()) && (function->inCastsSet_.count(t) == 0) &&
            t->GetProducers().size() != 0) {
            continue;
        }
        magic2index[t->GetMagic()] = index++;
        ss << "(" << " " << static_cast<int>(t->tensor->datatype) << " ";
        // Add shape information
        for (const auto &dim : t->shape) {
            ss << dim << " ";
        }
        if (function->IsFunctionType(FunctionType::STATIC)) {
            for (const auto &dim : t->oriShape) {
                ss << dim << " ";
            }
        }
        if (t->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            for (const auto &dim : t->offset) {
                ss << dim << " ";
            }
        }
        if (!enableMagicLookupRecord_) {
            ProducerMagicLookup(function, t, t->GetProducers(), subGraphId, index, magic2index, ss);
        } else if (tensorAndSubgraphToProducer_.count({t->GetMagic(), subGraphId}) > 0) {
            ProducerMagicLookup(function, t, tensorAndSubgraphToProducer_[{t->GetMagic(), subGraphId}],
                                subGraphId, index, magic2index, ss);
        }
        ss << ")";
    }
}

unsigned long Function::ComputeHashOrderless() const {
    std::stringstream ss;
    ss << std::to_string(static_cast<int>(functionType_)) << " ";
    ss << std::to_string(static_cast<int>(graphType_)) << " ";
    if (!IsGraphType({GraphType::BLOCK_GRAPH, GraphType::LEAF_VF_GRAPH}) &&
        !IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TENSOR_GRAPH)) {
        ss << GetMagicName() << " ";
    }

    // Build using Polish Notation
    int index = 0;
    std::unordered_map<int, int> magic2index;
    // 只有leaf graph需要判断边界
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        if (operations_.size()) {
            MagicLookup(
                this, GetOutcast(), operations_[operations_.size() - 1]->GetSubgraphID(), index, magic2index, ss);
        }
    } else {
        MagicLookup(this, GetOutcast(), INT32_MIN, index, magic2index, ss);
    }

    // 补充一些没有输出的Op的hash
    for (size_t i = 0; i < operations_.size(); i++) {
        if (operations_[i]->oOperand.empty()) {
            ss << " " << operations_[i]->GetOpcodeStr(true);
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(operations_[i]->GetOpcode())) {
                ss << " attr: [" << attr << " : "
                   << operations_[i]->DumpAttr(attr) << "]";
            }
            ss << operations_[i]->GetTileShape().ToString();
            MagicLookup(this, operations_[i]->GetIOperands(), operations_[0]->GetSubgraphID(), index, magic2index, ss);
        }
    }
    index = 0;
    for (auto &i : inCasts_) {
        ss << "(i" << index++ << ")";
        bool isGlobal = (globalTensors_.count(i) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }
    for (auto &o : outCasts_) {
        ss << "(o" << index++ << ")";
        bool isGlobal = (globalTensors_.count(o) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }

    // fill symbol and loop range attr of dyndev tensor graph for dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC_LOOP && dynloopAttr_ != nullptr) {
        ss << "symbol name:[" << dynloopAttr_->iterSymbolName << "]";
        ss << "loop range:[" << dynloopAttr_->loopRange.Dump() << "]";
    }
    // temporary avoidance, switch SUPPORT_DYNAMIC_ALIGNED has an unexpected effect on dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC) {
        ss << "dynamic unaligned:" << config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_ALIGNED);
    }
    std::hash<std::string> hasher;
    auto result = hasher(ss.str());
    ALOG_DEBUG_F("Hash for function %d %s is %s hash value is %lu\n",
                 functionMagic_, GetMagicName().c_str(),
                 ss.str().c_str(), result);
    return result;
}

void Function::EraseOperations(bool eraseRelatedTensor, bool sorted) {
    std::unordered_set<LogicalTensorPtr> inOutCastSet(inCasts_.begin(), inCasts_.end());
    inOutCastSet.insert(outCasts_.begin(), outCasts_.end());
    std::vector<std::shared_ptr<Operation>> operations;
    std::unordered_set<std::shared_ptr<LogicalTensor>> removeCandidiateTensor;
    std::unordered_set<std::shared_ptr<LogicalTensor>> removeProducerTensor;
    for (auto &op : operations_) {
        if (!op->IsDeleted()) {
            operations.emplace_back(op);
            continue;
        }
        ASSERT(op->IsDeleted())
            << "Operation not marked as deleted:" << op->Dump();
        for (auto &input : op->GetIOperands()) {
            input->RemoveConsumer(op.get());
            removeCandidiateTensor.insert(input);
        }

        for (auto &output : op->GetOOperands()) {
            output->RemoveProducer(op.get());
            removeCandidiateTensor.insert(output);
            removeProducerTensor.insert(output);
        }

        for (auto &depend : op->GetDependOperands()) {
            depend->RemoveDependOp(op.get());
            removeCandidiateTensor.insert(depend);
        }
    }
    operations_ = operations;

    if (eraseRelatedTensor) {
        for (auto tensorPtr : removeCandidiateTensor) {
            if (inOutCastSet.count(tensorPtr) != 0) {
                continue;
            }
            if (tensorPtr->GetProducers().empty() && tensorPtr->GetConsumers().empty()) {
                GetTensorMap().Erase(tensorPtr);
            }
            else if (removeProducerTensor.count(tensorPtr) > 0 && tensorPtr->GetProducers().empty()) {
                GetTensorMap().Erase(tensorPtr);
                for (auto &consumer : tensorPtr->GetConsumers()) {
                    if (consumer->BelongTo() == this) {
                        consumer->EraseInput(tensorPtr);
                    }
                }
            }
        }
    }
    if (sorted) {
        SortOperations();
    }
}

void Function::EraseOperations(const OperationDeleter &deleter) {
    if (!sorted_) {
        SortOperations();
    }
    for (auto &op : operations_) {
        if (deleter(op, *this)) {
            op->SetAsDeleted();
        }
    }

    EraseOperations();
}

FunctionHash Function::ComputeHash() {
    if (functionHash_.GetHash() != 0 &&
        (functionType_ != FunctionType::DYNAMIC_LOOP && functionType_ != FunctionType::DYNAMIC)) {
        /* 动态类型的graph里面的op和tensor会随着循环的展开而变化，每次都需要刷新 */
        return functionHash_;
    }
    for (auto &ele : inCasts_) {
        inCastsSet_.emplace(ele);
    }
    functionHash_ = ComputeHashOrderless();
    return functionHash_;
}

void Function::AddOriginIncast(const std::shared_ptr<LogicalTensor> tensor) {
    originInCasts_.push_back(tensor);
}

void Function::AddOriginOutcast(const std::shared_ptr<LogicalTensor> tensor) {
    originOutCasts_.push_back(tensor);
}

Operation &Function::AddOperation(const std::string &opName, LogicalTensors iOperands,
    const LogicalTensors &oOperands, const bool updateTensorMap) {
    return AddOperation(FindOpcode(opName), iOperands, oOperands, updateTensorMap);
}

Operation &Function::AddOperation(const Opcode opCode, LogicalTensors iOperands,
                                  const LogicalTensors &oOperands, const bool updateTensorMap) {
    for (auto &iOperand : iOperands) {
        ASSERT(iOperand->shape.size() != 0) << "tensor shape size invalid";
        iOperand = ConnectWithOverlap(iOperand);
    }
    return AddRawOperation(opCode, iOperands, oOperands, updateTensorMap);
}

void Function::UpdateTensorDataUsage(Operation &op) {
    auto dynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    if (dynFunc == nullptr) {
        return;
    }
    auto dynDevAttr = dynFunc->GetDyndevAttribute();
    if (dynDevAttr == nullptr) {
        return;
    }
    auto &descDict = dynDevAttr->getTensorDataDescDict;
    auto &importDict = dynDevAttr->getTensorDataUsageDict[this].importDict;

    auto dynAttrList = op.GetDynamicAttributeList();
    auto dict = GetTensorDataDict(dynAttrList);
    for (auto &[index, callList] : dict) {
        (void)callList;
        ASSERT(descDict.count(index)) << "Invalid index" << op.Dump();
        if (importDict.count(index)) {
            continue;
        }
        auto assemble = descDict[index].assembleTensor;
        std::vector<int64_t> importShape(assemble->GetShape().size(), 1);
        std::vector<int64_t> importOffset(assemble->GetShape().size(), 0);
        auto import = View(*assemble, importShape, importOffset);
        auto importOp = *import.GetStorage()->GetProducers().begin();
        SetEmuOpcode(importOp, EMUOP_TENSOR_GETDATA_IMPORT);
        GetTensorDataSetIndex(importOp, index);
        importDict[index] = importOp;
    }
}

Operation &Function::AddRawOperation(const Opcode opCode, const LogicalTensors &iOperands,
                                     const LogicalTensors &oOperands, bool updateTensorMap) {
    if (IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::EXECUTE_GRAPH, GraphType::BLOCK_GRAPH})) {
        updateTensorMap = false;
        sorted_ = true;
    } else {
        sorted_ = functionType_ == FunctionType::DYNAMIC;
    }
    auto &op =
        operations_.emplace_back(std::make_shared<Operation>(*this, opCode, iOperands, oOperands, updateTensorMap));
    opPosition_.emplace(op.get(), operations_.size() - 1);
    operations_.back()->SetScopeId(config::GetPassOption<int>(SG_SET_SCOPE));
    return *operations_.back();
}

void Function::SetSameMemId(const LogicalTensorPtr &operand, LogicalTensorPtr &dst) {
    ASSERT(operand->Datatype() == dst->Datatype()) << "Check Dtype failed!";

    auto dstRaw = dst->GetRawTensor();
    auto operandRaw = operand->GetRawTensor();
    dstRaw->memoryId = operandRaw->memoryId;
    outIncastLinkMap[dstRaw] = operandRaw;
}

std::vector<Operation *> Function::GetAllInputOperations(const Operation &op) const {
    std::vector<Operation *> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr &inTensor : op.GetIOperands()) {
        if (inTensor == nullptr) {
            continue;
        }
        for (auto &producer : inTensor->GetProducers()) {
            retOps.push_back(producer);
        }
    }
    return retOps;
}

std::vector<Operation *> Function::GetAllOutputOperations(const Operation &op) const {
    std::vector<Operation *> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr &outTensor : op.GetOOperands()) {
        if (outTensor == nullptr) {
            continue;
        }
        for (auto &consumer : outTensor->GetConsumers()) {
            retOps.push_back(consumer);
        }
    }
    return retOps;
}

std::vector<Operation *> Function::GetCallopList() const {
    std::vector<Operation *> callopList;
    for (auto &op : operations_) {
        if (op->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        callopList.push_back(op.get());
    }
    return callopList;
}

std::vector<std::shared_ptr<CallOpAttribute>> Function::GetCallopAttrList() const {
    std::vector<Operation *> callopList = GetCallopList();
    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        callopAttrList.push_back(callopAttr);
    }
    return callopAttrList;
}

std::vector<Function *> Function::GetCalleeFunctionList() const {
    std::vector<Operation *> callopList = GetCallopList();
    std::vector<Function *> calleeFuncList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        auto calleeFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
        ASSERT(calleeFunc) << callopAttr->GetCalleeMagicName() << " is not in functionmap!";
        calleeFuncList.push_back(calleeFunc);
    }
    return calleeFuncList;
}

void Function::SubstituteIn(std::shared_ptr<LogicalTensor> oldTensor,
                            std::shared_ptr<LogicalTensor> newTensor) {
    for (auto &operation : operations_) {
        auto &cur = *operation;
        std::unordered_set<std::shared_ptr<LogicalTensor>> replaced;
        for (size_t i = 0; i < cur.GetInputOperandSize(); i++) {
            LogicalTensorPtr inputTensor = cur.GetInputOperand(i);
            if (inputTensor != oldTensor) {
                continue;
            }
            if (replaced.count(inputTensor) == 0) {
                ASSERT(inputTensor->HasConsumer(cur))
                    << "Tensor is not a consumer of the operation:\n"
                    << "Tensor: " << inputTensor->Dump() << "\n" << "Operation: " << cur.Dump();
                replaced.emplace(inputTensor);
            }
            cur.ReplaceIOperand(i, newTensor);
            if (cur.GetOpAttribute() == nullptr) {
                continue;
            }
            if (auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(cur.GetOpAttribute())) {
                // VIEW操作的offset要相应被修改。
                auto &fromOffset = viewOpAttribute->GetFrom();
                for (size_t j = 0; j < fromOffset.size(); j++) {
                    fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
                }
            } else if (auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(cur.GetOpAttribute())) {
                // CopyIn操作的offset要相应被修改。
                if (!copyOpAttribute->IsCopyOut()) {
                    auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                    (void)memType;
                    for (size_t j = 0; j < fromOffset.size(); j++) {
                        fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
                    }
                    copyOpAttribute->SetFromOffset(fromOffset);
                }
            }
        }
    }
}

void Function::SubstituteOut(std::shared_ptr<LogicalTensor> oldTensor,
                             std::shared_ptr<LogicalTensor> newTensor) {
    for (auto &operation : operations_) {
        auto &cur = *operation;
        for (size_t i = 0; i < cur.GetOOperands().size(); i++) {
            if (cur.GetOOperands()[i] == oldTensor) {
                ASSERT(cur.GetOOperands()[i]->shape == newTensor->shape)
                    << "Shape mismatch:\n" << "Old Tensor Shape: "
                    << StringUtils::ToString(cur.GetOOperands()[i]->shape) << "\n"
                    << "New Tensor Shape: "
                    << StringUtils::ToString(newTensor->shape) << "\n"
                    << "Operation: " << cur.Dump();
                ASSERT(cur.GetOOperands()[i]->HasProducer(cur))
                    << "Tensor is not a producer of the operation:\n"
                    << "Tensor: " << cur.GetOOperands()[i]->Dump() << "\n" << "Operation: " << cur.Dump();
                cur.ReplaceOOperand(i, newTensor);
            }
        }
    }
}

void Function::Substitute(std::shared_ptr<LogicalTensor> oldTensor,
                          std::shared_ptr<LogicalTensor> newTensor) {
    SubstituteIn(oldTensor, newTensor);
    SubstituteOut(oldTensor, newTensor);
}

void Function::RemoveOriginIncastConsumer(const std::shared_ptr<LogicalTensor> &originIncast) const {
    // 因为originIncast不输出本function，因此originIncast的producer的oOperand在本function外，需要移除对function内的消费者
    for (const auto &producer : originIncast->GetProducers()) {
        auto targetFunc = this;
        while (targetFunc != producer->BelongTo()) {
            if (!targetFunc->HasParent()) {
                targetFunc = nullptr;
                break;
            }
            targetFunc = &targetFunc->Parent();
        }
        ASSERT(targetFunc != nullptr)
            << "Failed to find the target function for producer:\n"
            << "Producer: " << producer->Dump();

        for (auto &oOperandForProducerOp : producer->oOperand) {
            auto &consumers = oOperandForProducerOp->GetConsumers();
            for (auto it = consumers.begin(); it != consumers.end();) {
                if ((*it)->BelongTo() == this) {
                    it = consumers.erase(it);
                } else {
                    it++;
                }
            }
        }
    }
}

void Function::UpdateLinkMap(const std::shared_ptr<LogicalTensor> &oriLogicalTensor,
                             const std::shared_ptr<LogicalTensor> &newLogicalTensor, const bool isOutCast) {
    if (isOutCast) {
        //  update outcast
        auto it = outIncastLinkMap.find(oriLogicalTensor->tensor);
        if (it != outIncastLinkMap.end()) {
            outIncastLinkMap[newLogicalTensor->tensor] = it->second;
            newLogicalTensor->tensor->memoryId = it->second->memoryId;
            ALOG_DEBUG_F("UpdateLinkMap memoryId to %d  \n", it->second->memoryId);
            outIncastLinkMap.erase(it);
        }
    } else {
        //  update incast
        for (auto &ele : outIncastLinkMap) {
            if (ele.second == oriLogicalTensor->tensor) {
                ele.second = newLogicalTensor->tensor;
            }
        }
    }
}

std::shared_ptr<LogicalTensor> Function::CreateIncastTensor(const std::shared_ptr<LogicalTensor> &inArgument) {
    auto idx = inCasts_.size();
    auto newSymbol = inArgument->tensor->GetSymbol();
    if (newSymbol == "") {
        newSymbol = "INCAST_SYMBOL" + std::to_string(idx);
    }
    auto incastSymbol = std::make_shared<LogicalTensor>(*this, inArgument->tensor->datatype, inArgument->shape,
        inArgument->tensor->GetDynRawShape(), inArgument->Format(), newSymbol, NodeType::INCAST);
    tensorMap_.Insert(incastSymbol);
    inCasts_.push_back(incastSymbol);
    incastToInArgumentDict[incastSymbol] = inArgument;

    UpdateLinkMap(inArgument, incastSymbol);
    return incastSymbol;
}

void Function::CreateFromIncast(const std::shared_ptr<LogicalTensor> &symbol,
                                      const std::shared_ptr<LogicalTensor> &newIncast,
                                      const std::shared_ptr<LogicalTensor> &originIncast) {
    DEFINE_SOURCE_LOCATION();
    auto &incastOp = AddOperation(Opcode::OP_VIEW, {symbol}, {newIncast});
    incastOp.SetAttr(OpAttributeKey::isGlobalInput, true);

    auto validShape = originIncast->GetDynValidShape();
    if (validShape.empty()) {
        validShape = GetViewValidShape(symbol->GetDynValidShape(), originIncast->GetOffset(),
            originIncast->GetDynOffset().empty() ? SymbolicScalar::FromConcrete(originIncast->GetOffset()) : originIncast->GetDynOffset(),
                newIncast->GetShape());
    }
    incastOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(originIncast->GetOffset(),
        originIncast->GetDynOffset().empty() ? SymbolicScalar::FromConcrete(originIncast->GetOffset()) : originIncast->GetDynOffset(), validShape));
    newIncast->UpdateDynValidShape(validShape);
    newIncast->GetRawTensor()->UpdateDynRawShape(symbol->GetDynValidShape());
}

void Function::ReplaceMaybeParams(const std::shared_ptr<LogicalTensor> &newIncast,
                                        const std::shared_ptr<LogicalTensor> &originIncast) {
    auto it = std::find(originInCasts_.begin(), originInCasts_.end(), originIncast);
    ASSERT(it != originInCasts_.end())
        << "OriginIncast not found in originInCasts_:\n"
        << "OriginIncast: " << originIncast->Dump();
    *it = newIncast;
}

LogicalTensors Function::MakeIncasts(const std::shared_ptr<TensorSlotScope> &scope) {
    ASSERT(IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH))
        << "Invalid function type or graph type";
    ASSERT(HasParent())
        << "Function does not have a parent";
    LogicalTensors inArgumentList;
    std::unordered_set<int> appearedRawIncasts;
    std::vector<std::shared_ptr<RawTensor>> rawIncasts;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> incastWithSameRaw;
    std::map<std::shared_ptr<RawTensor>, std::shared_ptr<LogicalTensor>> rawToIncast;
    size_t iOperandIndex = 0;
    for (auto &originIncast : originInCasts_) {
        incastWithSameRaw[originIncast->tensor->rawmagic].emplace_back(originIncast);
        if (appearedRawIncasts.count(originIncast->tensor->rawmagic) != 0) {
            ++iOperandIndex;
            continue;
        }
        appearedRawIncasts.emplace(originIncast->tensor->rawmagic);
        rawIncasts.emplace_back(originIncast->tensor);
        rawToIncast[originIncast->tensor] = originIncast;
        if (GetSlotScope() != nullptr && GetSlotScope()->oriIncastReadSlotSet.size() > iOperandIndex) {
            GetSlotScope()->incastReadSlotSet.push_back(GetSlotScope()->oriIncastReadSlotSet[iOperandIndex]);
            GetSlotScope()->ioslot.incastSlot.push_back(GetSlotScope()->originalIocastsSlot.incastSlot[iOperandIndex]);
        }
        ++iOperandIndex;
    }

    for (auto &[rawMagic, sameRawIncasts] : incastWithSameRaw) {
        (void)rawMagic;
        sort(sameRawIncasts.begin(), sameRawIncasts.end(),
            [](const auto &a, const auto &b) -> bool { return a->shape < b->shape; });
    }

    int idx = 0;
    for (auto &rawIncast : rawIncasts) {
        const auto &sameRawIncasts = incastWithSameRaw[rawIncast->rawmagic];

        std::vector<int64_t> zeroOffset(rawIncast->rawshape.size(), 0);
        auto inArgument = std::make_shared<LogicalTensor>(Parent(), rawIncast, zeroOffset, rawIncast->rawshape,
            NodeType::LOCAL);
        inArgumentList.push_back(inArgument);

        auto incastSymbol = CreateIncastTensor(inArgument);
        if (scope) {
            scope->incastToInArgumentDict[incastSymbol] = inArgument;
            scope->incastToInOriginalDict[incastSymbol].insert(sameRawIncasts.begin(), sameRawIncasts.end());
        }

        std::map<ViewKey, std::shared_ptr<LogicalTensor>> newincastMap;
        for (auto &originIncast : sameRawIncasts) {
            auto viewKey = ViewKey(originIncast->tensor->rawmagic, originIncast->shape,
                                   originIncast->offset, originIncast->GetDynOffset());
            std::shared_ptr<LogicalTensor> newIncast;
            if (newincastMap.count(viewKey) != 0) {
                newIncast = newincastMap[viewKey];
            } else {
                newIncast = std::make_shared<LogicalTensor>(*this, originIncast->tensor->datatype, originIncast->shape,
                    originIncast->Format(), "INCAST_LOCAL_BUF" + std::to_string(idx++), NodeType::LOCAL);
                ASSERT(originIncast->conflicterTensors.empty())
                    << "OriginIncast has conflicter tensors:" << originIncast->Dump();
                newIncast->CopyMemoryType(originIncast);

                newincastMap.emplace(viewKey, newIncast);
                CreateFromIncast(incastSymbol, newIncast, originIncast);
            }

            Substitute(originIncast, newIncast);

            ReplaceMaybeParams(newIncast, originIncast);

            for (const auto &producer : inArgument->GetProducers()) {
                ASSERT(producer->BelongTo() != this)
                    << inArgument->magic << "-> producer. funcMagic = " << producer->BelongTo()->GetFuncMagic()
                    << " producer = " << producer->GetOpMagic() << "inArgument = " << inArgument->Dump() << std::endl;
            }
            for (const auto &consumer : inArgument->GetConsumers()) {
                ASSERT(consumer->BelongTo() != this)
                    << inArgument->magic << "-> consumer. funcMagic = " << consumer->BelongTo()->GetFuncMagic()
                    << " consumer = " << consumer->GetOpMagic() << "inArgument = " << inArgument->Dump() << std::endl;
            }
        }
    }

    for (auto &rawIncast : rawIncasts) {
        const auto &sameRawIncasts = incastWithSameRaw[rawIncast->rawmagic];
        for (const auto &originIncast : sameRawIncasts) {
            RemoveOriginIncastConsumer(originIncast);
        }
    }

    return inArgumentList;
}

LogicalTensors Function::MakeOutcasts(const std::shared_ptr<TensorSlotScope> &scope) {
    ASSERT(IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH))
        << "Invalid function type or graph type";
    ASSERT(HasParent())
        << "Function does not have a parent";
    LogicalTensors outArgumentList;
    std::unordered_set<int> appearedRawOutcasts;
    std::vector<std::shared_ptr<RawTensor>> rawOutcasts;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> outcastWithSameRaw;
    std::map<std::shared_ptr<RawTensor>, std::shared_ptr<LogicalTensor>> rawToOutcast;
    size_t oOperandIndex = 0;
    for (const auto &originOutcast : originOutCasts_) {
        ASLOGI("originOut cast name %d %d", originOutcast->magic, originOutcast->GetRawMagic());
        outcastWithSameRaw[originOutcast->tensor->rawmagic].emplace_back(originOutcast);
        if (appearedRawOutcasts.count(originOutcast->tensor->rawmagic) != 0) {
            ++oOperandIndex;
            continue;
        }
        appearedRawOutcasts.emplace(originOutcast->tensor->rawmagic);
        rawOutcasts.emplace_back(originOutcast->tensor);
        rawToOutcast[originOutcast->tensor] = originOutcast;
        if (GetSlotScope() != nullptr && GetSlotScope()->oriOutcastWriteSlotSet.size() > oOperandIndex) {
            GetSlotScope()->outcastWriteSlotSet.push_back(GetSlotScope()->oriOutcastWriteSlotSet[oOperandIndex]);
            GetSlotScope()->ioslot.outcastSlot.push_back(GetSlotScope()->originalIocastsSlot.outcastSlot[oOperandIndex]);
        }
        ++oOperandIndex;
    }

    ASLOGI("raw out cast number %zu", rawOutcasts.size());
    for (const auto &rawOutcast : rawOutcasts) {
        auto &sameRawOutcasts = outcastWithSameRaw[rawOutcast->rawmagic];
        std::vector<int64_t> nonOffsets(rawOutcast->rawshape.size(), 0);

        auto idx = outCasts_.size();
        auto newSymbol = rawOutcast->GetSymbol();
        if (newSymbol == "") {
            newSymbol = "OUTCAST_SYMBOL" + std::to_string(idx);
        }
        auto rawSymbol = std::make_shared<LogicalTensor>(*this, rawOutcast->datatype, rawOutcast->rawshape,
            rawOutcast->GetDynRawShape(), rawToOutcast[rawOutcast]->Format(), newSymbol, NodeType::OUTCAST);
        auto rawBuf = std::make_shared<LogicalTensor>(*this, rawOutcast->datatype, rawOutcast->rawshape,
            rawOutcast->GetDynRawShape(), rawToOutcast[rawOutcast]->Format(), "OUTCAST_LOCAL_BUF" + std::to_string(idx),
            NodeType::LOCAL);
        auto outArgument = std::make_shared<LogicalTensor>(Parent(), rawOutcast, nonOffsets,
                                                           rawOutcast->rawshape, NodeType::LOCAL);
        rawSymbol->tensor->UpdateDynRawShape(rawOutcast->GetDynRawShape());
        rawBuf->tensor->UpdateDynRawShape(rawOutcast->GetDynRawShape());
        Parent().tensorMap_.Insert(outArgument);
        outArgumentList.push_back(outArgument);
        UpdateLinkMap(outArgument, rawSymbol, true);
        outCasts_.emplace_back(rawSymbol);
        outcastToOutArgumentDict[rawSymbol] = outArgument;

        if (scope) {
            scope->outcastToOutArgumentDict[rawSymbol] = outArgument;
            scope->outcastToOutOriginalDict[rawSymbol].insert(sameRawOutcasts.begin(), sameRawOutcasts.end());
        }

        std::vector<std::vector<int64_t>> newOutcastOffsets;
        std::vector<std::shared_ptr<LogicalTensor>> iOperand;
        std::vector<std::shared_ptr<LogicalTensor>> oOperand = {rawSymbol};
        ASLOGI("same raw out cast number %zu", sameRawOutcasts.size());

        std::shared_ptr<LogicalTensor> newOutcast = nullptr;
        for (auto &originOutcast : sameRawOutcasts) {
            if (newOutcast == nullptr) {
                newOutcast = rawBuf->View(*this, originOutcast->shape, originOutcast->offset);
                newOutcast->UpdateDynValidShape(originOutcast->GetDynValidShape());
                newOutcastOffsets.emplace_back(originOutcast->offset);
                iOperand.emplace_back(newOutcast);
            }
            auto oldConsumers = originOutcast->GetConsumers(); // only for check
            tensorMap_.Insert(newOutcast);
            Substitute(originOutcast, newOutcast);
            ASSERT(newOutcast->GetConsumers() == oldConsumers)
                << "Consumers mismatch after substitution:\n" << "NewOutcast: " << newOutcast->Dump() << "\n"
                << "OldConsumers: " << oldConsumers.size();
            ASSERT(originOutcast->GetProducers().empty())
                << "OriginOutcast has producers:" << originOutcast->Dump();
            auto it = std::find(originOutCasts_.begin(), originOutCasts_.end(), originOutcast);
            ASSERT(it != originOutCasts_.end())
                << "OriginOutcast not found in originOutCasts_:" << originOutcast->Dump();
            *it = newOutcast;
        }
        ASSERT(rawSymbol->GetProducers().empty())
            << "RawSymbol has producers:" << rawSymbol->Dump();
        ASSERT(iOperand.size() == newOutcastOffsets.size())
            << "iOperand size does not match newOutcastOffsets size:\n" << "iOperand size: " << iOperand.size() << "\n"
            << "newOutcastOffsets size: " << newOutcastOffsets.size();
        for (size_t i = 0; i < iOperand.size(); i++) {
            auto producerSet = iOperand[i]->GetProducers(); // deep copy
            auto partitalAssemble = std::any_of(producerSet.begin(), producerSet.end(), [](Operation *op) {
                return (op->GetOpcode() == Opcode::OP_ASSEMBLE && op->HasAttribute("dassemble")) ||
                        op->GetOpcode() == Opcode::OP_ASSEMBLE_SSA;
            });
            if (partitalAssemble) {
                for (auto producer : producerSet) {
                    auto producerAttr = std::static_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
                    ASSERT(producerAttr) << "mix assemble and common operation for same output \n" << producer->Dump();
                    auto [offset, dynOffset] = TensorOffset::Add(iOperand[i]->GetOffset(), iOperand[i]->GetDynOffset(),
                                                                 producerAttr->GetToOffset(), producerAttr->GetToDynOffset());
                    producer->ReplaceOOperand(0, rawSymbol);
                    producer->SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset, dynOffset));
                }
                auto consumers = iOperand[i]->GetConsumers(); // deep copy
                for (auto consumer : consumers) {
                    DEFINE_SOURCE_LOCATION();
                    for (size_t j = 0; j < consumer->GetIOperands().size(); j++) {
                        if (consumer->GetInputOperand(j) == iOperand[i]) {
                            consumer->ReplaceIOperand(j, rawSymbol);
                        }
                    }
                }
                if (scope) {
                    scope->partialUpdateOutcastDict[rawSymbol] = partitalAssemble;
                }
            } else {
                DEFINE_SOURCE_LOCATION();
                auto &assembleOp = AddOperation(Opcode::OP_ASSEMBLE, {iOperand[i]}, oOperand);
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(newOutcastOffsets[i], SymbolicScalar::FromConcrete(newOutcastOffsets[i])));
            }
        }
        // Substitute alive tensor magics
        Program::GetInstance().UpdateAliveTensorsParent(rawOutcast->rawmagic, Parent());
    }

    for (auto it = tensorMap_.inverseMap_.begin(); it != tensorMap_.inverseMap_.end();) {
        if (appearedRawOutcasts.count(it->second->tensor->rawmagic) > 0) {
            it = tensorMap_.inverseMap_.erase(it);
        } else {
            it++;
        }
    }

    std::vector<int> magicToRemove;
    for (auto it = tensorMap_.tensorMap_.begin(); it != tensorMap_.tensorMap_.end();) {
        if (appearedRawOutcasts.count(it->first) > 0) {
            magicToRemove.push_back(it->first);
            it++;
        } else {
            it++;
        }
    }
    for(auto rmagic: magicToRemove){
        tensorMap_.EraseRawMagic(rmagic);
    }

    for (const auto &tensor : outArgumentList) {
        ASSERT(tensor->GetProducers().empty())
            << "Tensor has producers:" << tensor->Dump();
    }

    return outArgumentList;
}

bool Function::IsFlattening() const {
    return IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH});
}

FunctionType Function::GetFunctionType() const {
    return functionType_;
}

void Function::SetFunctionType(FunctionType type) {
    functionType_ = type;
}

std::string Function::GetFunctionTypeStr() const{
    return GetFunctionTypeNameDict().Find(functionType_);
}

GraphType Function::GetGraphType() const {
    return graphType_;
}

void Function::SetGraphType(GraphType type) {
    graphType_ = type;
}

bool Function::IsFunctionType(FunctionType type) const {
    return functionType_ == type;
}

bool Function::IsFunctionType(std::set<FunctionType> types) const {
    return types.count(functionType_) != 0;
}

bool Function::IsGraphType(GraphType type) const {
    return graphType_ == type;
}

bool Function::IsGraphType(std::set<GraphType> types) const {
    return types.count(graphType_) != 0;
}

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, GraphType graphType) const {
    return functionType_ == funcType && graphType_ == graphType;
}

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, std::set<GraphType> graphTypes) const {
    return IsFunctionType(funcType) && IsGraphType(graphTypes);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, GraphType graphType) const {
    return IsFunctionType(funcTypes) && IsGraphType(graphType);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, std::set<GraphType> graphTypes) const {
    return IsFunctionType(funcTypes) && IsGraphType(graphTypes);
}

void Function::DumpJsonFile(std::string fileName) {
    auto filePath = config::LogTopFolder() + "/" + funcRawName_ + ".json";
    if (!fileName.empty()) {
        filePath = fileName;
    }
    std::ofstream file(filePath);
    Json progDump;
    progDump["version"] = T_VERSION;
    progDump["functions"].push_back(DumpJson());
    progDump["entryhash"] = this->GetFunctionHash().Data();
    file << progDump.dump(1) << std::endl;
    file.close();
}

struct RawTensorCompare {
    bool operator()(const std::shared_ptr<RawTensor>& a,
                    const std::shared_ptr<RawTensor>& b) const {
        return a->rawmagic < b->rawmagic;
    }
};

struct TensorCompare {
    bool operator()(const std::shared_ptr<LogicalTensor>& a,
                    const std::shared_ptr<LogicalTensor>& b) const {
        return a->magic < b->magic;
    }
};

Json Function::DumpJson(bool useTable) {
    Json funcDump;
    funcDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_FUNCTION);
    funcDump["rawname"] = funcRawName_;
    funcDump["funcmagic"] = GetFuncMagic();
    if (parent_ != nullptr) {
        funcDump["parent_funcmagic"] = parent_->GetFuncMagic();
    }
    funcDump["functype"] = functionType_;
    funcDump["graphtype"] = graphType_;
    funcDump["func_magicname"] = funcMagicName_;
    funcDump["_opseed"] = opSeed_;
    funcDump["_rawid"] = IdGen<IdType::RAW_TENSOR>::Inst().CurId();
    funcDump["_funcid"] = IdGen<IdType::FUNCTION>::Inst().CurId();
    funcDump["_l1_reuse_num"] = paramConfigs_.l1ReuseNum;
    funcDump["_cube_nbuffer_mode"] = paramConfigs_.cubeNBufferMode;
    funcDump["_sg_pg_upperbound"] = paramConfigs_.sgPgUpperBound;
    funcDump["_sg_pg_lowerbound"] = paramConfigs_.sgPgLowerBound;
    funcDump["_sg_parallel_num"] = paramConfigs_.sgParallelNum;
    funcDump["_sg_mg_copyin_upper_bound"] = paramConfigs_.sgMgCopyInUpperBound;
    funcDump["_vec_nbuffer_mode"] = paramConfigs_.vecNBuffermode;
    funcDump["_mg_vec_parallel_lb"] = paramConfigs_.mgVecParallelLb;
    funcDump["_sg_cube_parallel_num"] = paramConfigs_.sgCubeParallelNum;
    funcDump["_pg_skip_partition"] = paramConfigs_.pgSkipPartition;
    funcDump["_total_subgraph_count"] = totalSubGraphCount_;
    funcDump["_ooo_preschedule_method"] = paramConfigs_.OoOPreScheduleMethod;
    if (sourceLocation_ != nullptr) {
        funcDump["file"] = sourceLocation_->GetFileName();
        funcDump["line"] = sourceLocation_->GetLineno();
    }

    if (useTable) {
        std::vector<std::pair<int, std::vector<int>>> incasts;
        std::vector<std::pair<int, std::vector<int>>> outcasts;
        size_t inSize = inCasts_.size();
        for (size_t i = 0; i < inSize; i++) {
            std::pair<int, std::vector<int>> incast;
            incast.first = inCasts_[i]->GetMagic();
            if (slotScope_ != nullptr && i < slotScope_->ioslot.incastSlot.size()) {
                incast.second = slotScope_->ioslot.incastSlot[i];
            } else {
                std::vector<int> emptyIncast;
                incast.second = emptyIncast;
            }
            incasts.push_back(incast);
        }
        size_t outSize = outCasts_.size();
        for (size_t i = 0; i < outSize; i++) {
            std::pair<int, std::vector<int>> outcast;
            outcast.first = outCasts_[i]->GetMagic();
            if (slotScope_ != nullptr && i < slotScope_->ioslot.outcastSlot.size()) {
                outcast.second = slotScope_->ioslot.outcastSlot[i];
            } else {
                std::vector<int> emptyOutcast;
                outcast.second = emptyOutcast;
            }
            outcasts.push_back(outcast);
        }

        funcDump["incasts"] = incasts;
        funcDump["outcasts"] = outcasts;
    } else {
        Json incasts = Json::array();
        Json outcasts = Json::array();
        Json globalTensors = Json::array();
        for (auto &i : inCasts_) {
            incasts.push_back(i->DumpJson(true));
        }
        for (auto &o : outCasts_) {
            outcasts.push_back(o->DumpJson(true));
        }
        funcDump["incasts"] = incasts;
        funcDump["outcasts"] = outcasts;
    }

    std::set<int> globalTensorSet;
    for (auto &t : globalTensors_) {
        globalTensorSet.emplace(t->GetMagic());
    }
    std::vector<int> globalTensorVec;
    for (auto &tMagic : globalTensorSet) {
        globalTensorVec.emplace_back(tMagic);
    }
    funcDump["global_tensors"] = globalTensorVec;
    funcDump["static"]["global_tensors"] = funcDump["global_tensors"];

    Json operations = Json::array();
    if (useTable) {
        for (const auto &op : operations_) {
            operations.push_back(op->DumpJson(false));
        }
    } else {
        for (const auto &op : operations_) {
            operations.push_back(op->DumpJson(true));
        }
    }
    funcDump["operations"] = operations;
    funcDump["hash"] = functionHash_.Data();

    if (leafFuncAttr_ != nullptr && leafFuncAttr_->coreType != CoreType::INVALID) {
        funcDump["leaf_func_attr"]["coretype"] = leafFuncAttr_->coreType;
    }

    if (rootFunc_ != nullptr) {
        funcDump["root_func_magic"] = rootFunc_->GetFuncMagic();
    }
    if (!programs_.empty()) {
        Json programsJson;
        for (auto &ele : programs_) {
            programsJson[ele.first] = ele.second->GetFuncMagic();
        }
        funcDump["programs"] = programsJson;
        funcDump["topo"] = topoInfo_.DumpJson();
        funcDump["static"]["topo"] = funcDump["topo"];
    }
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        funcDump["subfunc_param"] = parameter_.ToJson();
        funcDump["static"]["subfunc_param"] = funcDump["subfunc_param"];
    }

    auto aicIt = readySubGraphIds_.find(CoreType::AIC);
    if (aicIt != readySubGraphIds_.end() && !aicIt->second.empty()) {
        funcDump["aic_ready_subgraph_ids"] = aicIt->second;
        funcDump["static"]["aic_ready_subgraph_ids"] = funcDump["aic_ready_subgraph_ids"];
    }

    auto aivIt = readySubGraphIds_.find(CoreType::AIV);
    if (aivIt != readySubGraphIds_.end() && !aivIt->second.empty()) {
        funcDump["aiv_ready_subgraph_ids"] = aivIt->second;
        funcDump["static"]["aiv_ready_subgraph_ids"] = funcDump["aiv_ready_subgraph_ids"];
    }

    auto aicpuIt = readySubGraphIds_.find(CoreType::AICPU);
    if (aicpuIt != readySubGraphIds_.end() && !aicpuIt->second.empty()) {
        funcDump["aicpu_ready_subgraph_ids"] = aicpuIt->second;
        funcDump["static"]["aicpu_ready_subgraph_ids"] = funcDump["aicpu_ready_subgraph_ids"];
    }

    if (useTable) {
        std::set<std::shared_ptr<RawTensor>, RawTensorCompare> rawTensorSet;
        std::set<std::shared_ptr<LogicalTensor>, TensorCompare> tensorSet;
        for (const auto &incast : inCasts_) {
            tensorSet.insert(incast);
            rawTensorSet.insert(incast->tensor);
        }
        for (const auto &outcast : outCasts_) {
            tensorSet.insert(outcast);
            rawTensorSet.insert(outcast->tensor);
        }
        for (const auto &op : operations_) {
            for (auto &i : op->GetIOperands()) {
                tensorSet.insert(i);
                rawTensorSet.insert(i->tensor);
            }
            for (auto &o : op->GetOOperands()) {
                tensorSet.insert(o);
                rawTensorSet.insert(o->tensor);
            }
        }
        std::vector<std::shared_ptr<RawTensor>> rawTensorList(rawTensorSet.begin(), rawTensorSet.end());
        std::vector<std::shared_ptr<LogicalTensor>> tensorList(tensorSet.begin(), tensorSet.end());
        std::sort(rawTensorList.begin(), rawTensorList.end(),
            [](auto l, auto r) { return l->GetRawMagic() < r->GetRawMagic(); });
        std::sort(tensorList.begin(), tensorList.end(), [](auto l, auto r) { return l->GetMagic() < r->GetMagic(); });

        Json rawtensors = Json::array();
        Json tensors = Json::array();
        for (auto &rawTensor : rawTensorList) {
            rawtensors.push_back(rawTensor->DumpJson());
        }
        for (auto &tensor : tensorList) {
            tensors.push_back(tensor->DumpJson(false));
        }
        funcDump["rawtensors"] = rawtensors;
        funcDump["tensors"] = tensors;
    }
    if (functionType_ == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        if (loopAttr != nullptr) {
            //itername
            std::string itername = loopAttr->iterSymbolName;
            funcDump["dynamic"]["itername"] = itername;

            // begin
            SymbolicScalar begin = loopAttr->Begin();
            auto jbegin = ToJson(begin);
            if (jbegin.size() > 0) {
                funcDump["dynamic"]["begin"] = jbegin;
            }

            // end
            SymbolicScalar end = loopAttr->End();
            auto jend = ToJson(end);
            if (jend.size() > 0) {
                funcDump["dynamic"]["end"] = jend;
            }

            // step
            SymbolicScalar step = loopAttr->Step();
            auto jstep = ToJson(step);
            if (jstep.size() > 0) {
                funcDump["dynamic"]["step"] = jstep;
            }

            // originalBegin
            SymbolicScalar originalBegin = loopAttr->originalRange.Begin();
            auto jOriBegin = ToJson(originalBegin);
            if (jOriBegin.size() > 0) {
                funcDump["dynamic"]["originalBegin"] = jOriBegin;
            }

            // originalEnd
            SymbolicScalar originalEnd = loopAttr->originalRange.End();
            auto jOriEnd = ToJson(originalEnd);
            if (jOriEnd.size() > 0) {
                funcDump["dynamic"]["originalEnd"] = jOriEnd;
            }

            // unrollTimes
            int unrollTimes = loopAttr->unrollTimes;
            funcDump["dynamic"]["unrollTimes"] = unrollTimes;

            // pathList
            Json loopFuncPathList = Json::array();
            for (auto &path : loopAttr->pathList) {
                Json pathJson = Json::array();
                auto opmagic = path.callop->GetOpMagic();
                for (auto &pathCond : path.pathCondList) {
                    Json pathCondJson = Json::array();
                    SymbolicScalar cond = pathCond.GetCond();
                    auto jcond = ToJson(cond);
                    pathCondJson.push_back(jcond);
                    pathCondJson.push_back(pathCond.IsSat());
                    pathJson.push_back(pathCondJson);
                }
                if (pathJson.size() > 0) {
                    loopFuncPathList.push_back({opmagic, pathJson});
                }
            }
            if (loopFuncPathList.size() > 0) {
                funcDump["dynamic"]["paths"] = loopFuncPathList;
            }
        }
    }
    return funcDump;
}

void Function::LoadTensorJson(const std::shared_ptr<Function> &func, const Json &funcDump,
                              const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawTensorDict,
                              std::unordered_map<int, std::shared_ptr<LogicalTensor>> &tensorDict) {
    if (funcDump.count("tensors") != 0) {
        for (auto &tensorDump : funcDump["tensors"]) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, tensorDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto &iDump : funcDump["incasts"]) {
        if (!iDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, iDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto &oDump : funcDump["outcasts"]) {
        if (!oDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, oDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }

    for (auto &tDump : funcDump["global_tensors"]) {
        int magic = tDump.get<int>();
        auto &t = tensorDict[magic];
        func->globalTensors_.emplace(t);
    }

    for (auto &iDump : funcDump["incasts"]) {
        int magic = iDump[0].is_number() ? iDump[0].get<int>() : iDump["magic"].get<int>();
        auto &in = tensorDict[magic];
        func->inCasts_.push_back(in);
        func->GetTensorMap().Insert(in);
    }
    for (auto &oDump : funcDump["outcasts"]) {
        int magic = oDump[0].is_number() ? oDump[0].get<int>() : oDump["magic"].get<int>();
        func->outCasts_.push_back(tensorDict[magic]);
    }
    /* 补充对于临时workspace的tensor添加 */
    for (auto &ele : tensorDict) {
        func->GetTensorMap().Insert(ele.second, false);
    }

    for (auto &opDump : funcDump["operations"]) {
        auto op = Operation::LoadJson(*func, tensorDict, opDump);
        func->operations_.push_back(op);
    }
}

std::shared_ptr<Function> Function::LoadJson(Program &belongTo, const Json &funcDump) {
    ASSERT(funcDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_FUNCTION))
        << "Invalid function kind in JSON";
    int funcmagic = funcDump["funcmagic"].get<int>();
    std::string rawname = funcDump["rawname"].get<std::string>();
    std::shared_ptr<Function> func =
        std::make_shared<Function>(belongTo, rawname + "_" + std::to_string(funcmagic), rawname, nullptr);
    func->funcMagicName_ = funcDump["func_magicname"];
    func->functionMagic_ = funcmagic;
    func->functionType_ = static_cast<FunctionType>(funcDump["functype"].get<int>());
    func->graphType_ = static_cast<GraphType>(funcDump["graphtype"].get<int>());
    func->sorted_ = true;
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    if (funcDump.count("rawtensors") != 0) {
        for (auto &rawTensorDump : funcDump["rawtensors"]) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(rawTensorDump);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto &iDump : funcDump["incasts"]) {
        if (!iDump[0].is_number() && !iDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(iDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto &oDump : funcDump["outcasts"]) {
        if (!oDump[0].is_number() && !oDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(oDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }

    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorDict;
    LoadTensorJson(func, funcDump, rawTensorDict, tensorDict);
    func->opSeed_ = funcDump["_opseed"].get<int>();
    int rawid = funcDump["_rawid"].get<int>();
    IdGen<IdType::RAW_TENSOR>::Inst().SetId(rawid);
    int funcid = funcDump["_funcid"].get<int>();
    IdGen<IdType::FUNCTION>::Inst().SetId(funcid);
    func->paramConfigs_.l1ReuseNum = funcDump["_l1_reuse_num"].get<int>();
    func->paramConfigs_.cubeNBufferMode = funcDump["_cube_nbuffer_mode"].get<int>();
    func->paramConfigs_.sgPgUpperBound = funcDump["_sg_pg_upperbound"].get<int>();
    func->paramConfigs_.sgPgLowerBound = funcDump["_sg_pg_lowerbound"].get<int>();
    func->paramConfigs_.sgParallelNum = funcDump["_sg_parallel_num"].get<int>();
    func->paramConfigs_.sgMgCopyInUpperBound = funcDump["_sg_mg_copyin_upper_bound"].get<int>();
    func->paramConfigs_.vecNBuffermode = funcDump["_vec_nbuffer_mode"].get<int>();
    func->paramConfigs_.mgVecParallelLb = funcDump["_mg_vec_parallel_lb"].get<int>();
    func->paramConfigs_.sgCubeParallelNum = funcDump["_sg_cube_parallel_num"].get<int>();
    func->paramConfigs_.pgSkipPartition = funcDump["_pg_skip_partition"].get<bool>();
    auto subGraphCount = funcDump["_total_subgraph_count"].get<size_t>();
    func->SetTotalSubGraphCount(subGraphCount);

    std::vector<std::vector<int>> incastSlot;
    for (auto &iDump : funcDump["incasts"]) {
        std::vector<int> iSlot;
        if (iDump[0].is_number()) {
            for (auto &slot : iDump[1]) {
                iSlot.push_back(slot.get<int>());
            }
            incastSlot.push_back(iSlot);
        }
    }
    std::vector<std::vector<int>> outcastSlot;
    for (auto &oDump : funcDump["outcasts"]) {
        std::vector<int> oSlot;
        if (oDump[0].is_number()) {
            for (auto &slot : oDump[1]) {
                oSlot.push_back(slot.get<int>());
            }
            outcastSlot.push_back(oSlot);
        }
    }
    IncastOutcastSlot ioSlot;
    ioSlot.incastSlot = incastSlot;
    ioSlot.outcastSlot = outcastSlot;
    std::shared_ptr<TensorSlotScope> tensorSlotScope = std::make_shared<TensorSlotScope>(func.get());
    tensorSlotScope->ioslot = ioSlot;
    func->slotScope_ = tensorSlotScope;

    func->ComputeHashOrderless();
    func->functionHash_ = std::stoull(funcDump["hash"].get<std::string>());

    if (func->GetGraphType() == GraphType::BLOCK_GRAPH && func->GetLeafFuncAttribute() == nullptr) {
        std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
        func->SetLeafFuncAttribute(attr);
    }
    if (funcDump.count("leaf_func_attr") != 0 && funcDump["leaf_func_attr"].count("coretype") != 0) {
        std::shared_ptr<LeafFuncAttribute> attr = func->GetLeafFuncAttribute();
        attr->coreType = static_cast<CoreType>(funcDump["leaf_func_attr"]["coretype"].get<int>());
    }

    if (funcDump.count("root_func_magic") != 0) {
        func->rootFunc_ = belongTo.GetFunctionByMagic(funcDump["root_func_magic"].get<int>()).get();
    }
    if (funcDump.count("programs") != 0) {
        uint64_t index = 0;
        for (auto &programMagic : funcDump["programs"]) {
            func->programs_.emplace(std::make_pair(index++, belongTo.GetFunctionByMagic(programMagic.get<int>()).get()));
        }
    }

    if (funcDump.count("topo") != 0) {
        func->topoInfo_.LoadJson(funcDump["topo"]);
    }

    if (funcDump.count("subfunc_param") != 0) {
        func->parameter_.FromJson(funcDump["subfunc_param"]);
    }

    if (funcDump.count("aic_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIC, funcDump["aic_ready_subgraph_ids"].get<std::vector<int>>());
    }
    if (funcDump.count("aiv_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIV, funcDump["aiv_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcDump.count("aicpu_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AICPU, funcDump["aicpu_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcDump.count("dynamic") != 0) {
        auto iterName = funcDump["dynamic"]["itername"];
        auto beginJson = funcDump["dynamic"]["begin"];
        SymbolicScalar begin = LoadSymbolicScalar(beginJson);
        auto endJson = funcDump["dynamic"]["end"];
        SymbolicScalar end = LoadSymbolicScalar(endJson);
        auto stepJson = funcDump["dynamic"]["step"];
        SymbolicScalar step = LoadSymbolicScalar(stepJson);
        LoopRange range(begin, end, step);
        auto originalBeginJson = funcDump["dynamic"]["originalBegin"];
        SymbolicScalar originalBegin = LoadSymbolicScalar(originalBeginJson);
        auto originalEndJson = funcDump["dynamic"]["originalEnd"];
        SymbolicScalar originalEnd = LoadSymbolicScalar(originalEndJson);
        LoopRange originalRange(originalBegin, originalEnd);
        auto attr = std::make_shared<DynloopFunctionAttribute>(iterName, range, originalRange);
        attr->unrollTimes = funcDump["dynamic"]["unrollTimes"];
        auto dynFuncDump = funcDump["dynamic"];
        if (dynFuncDump.count("paths") != 0) {
            std::vector<DynloopFunctionPath> pathList;
            auto pathsJson = funcDump["dynamic"]["paths"];
            for (auto &pathJson : pathsJson) {
                Function *root = func.get();
                std::vector<DynloopFunctionPathCondition> pathCondList;
                auto callOpMagic = pathJson[0];
                Operation *callop = nullptr;
                for (auto &op : func->Operations().DuplicatedOpList()) {
                    if (op->GetOpMagic() == callOpMagic) {
                        callop = op;
                        break;
                    }
                }
                for (auto &pathCondJson : pathJson[1]) {
                    bool isSat = static_cast<bool>(pathCondJson[1]);
                    SymbolicScalar cond = LoadSymbolicScalar(pathCondJson[0]);
                    DynloopFunctionPathCondition pathCond;
                    pathCond.isSat_ = isSat;
                    pathCond.cond_ = cond;
                    pathCondList.push_back(pathCond);
                }
                DynloopFunctionPath path(root, pathCondList, callop);
                pathList.push_back(path);
            }
            attr->pathList = pathList;
        }
        func->SetDynloopAttribute(attr);
    }
    func->RefreshOpPosition();
    return func;
}

static const SymbolicScalar RUNTIME_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET");
static const SymbolicScalar RUNTIME_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE");
static const SymbolicScalar RUNTIME_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM");

static void MaybeNormalizeValue(
        const SymbolicScalar &coaFunc,
        std::vector<SymbolicScalar> &operandCoaList,
        int operandCoaIndex,
        std::vector<OpImmediate> &opImmList,
        int coaIndex,
        bool valueToIndex) {
    for (size_t dimIndex = 0; dimIndex < opImmList.size(); dimIndex++) {
        auto &opImm = opImmList[dimIndex];
        SymbolicScalar scalar = opImm.GetSpecifiedValue();
        auto getTensorDataDict = GetTensorDataDict(scalar);
        if (getTensorDataDict.size() == 0) {
            OpImmediate::NormalizeValue(operandCoaList[operandCoaIndex + dimIndex], opImm,
                coaFunc(opImmList.size(), coaIndex, dimIndex), valueToIndex);
        }
    }
};

static void MaybeNormalizeValue(
        std::vector<SymbolicScalar> &valueCoa,
        SymbolicScalar &value,
        int coaIndex,
        bool valueToIndex) {
    auto getTensorDataDict = GetTensorDataDict(value);
    if (getTensorDataDict.size() == 0) {
        valueCoa.push_back(value);
        if (valueToIndex) {
            value = RUNTIME_COA_GetParam(coaIndex);
        }
    }
}

static std::vector<SymbolicScalar> NormalizeCopyIn(Operation *op, int coaIndexBase, bool valueToIndex) {
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    auto opImmList = copyAttr->GetFromOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex,
                        opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normal
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetToDynValidShape();
    MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex,
                        opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetToDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeCopyOut(Operation *op, int coaIndexBase, bool valueToIndex) {
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    auto opImmList = copyAttr->GetToOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex,
                        opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetToOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normals
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetFromDynValidShape();
    MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex,
                        opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeTensor(
    LogicalTensorPtr operand, int coaIndexBase, bool valueToIndex, bool isNop = false) {
    auto offset = OpImmediate::Specified(operand->GetOffset());
    auto dynOffset = OpImmediate::Specified(operand->GetDynOffset());
    auto shape = OpImmediate::Specified(operand->GetShape());
    auto rawshape = OpImmediate::Specified(operand->GetRawTensor()->GetRawShape());
    auto dynRawshape = OpImmediate::Specified(operand->GetRawTensor()->GetDynRawShape());
    auto dynValidShape = OpImmediate::Specified(operand->GetDynValidShape());
    if (isNop) {
        offset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        dynOffset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        shape = OpImmediate::Specified(Shape(operand->GetShape().size()));
        dynValidShape = OpImmediate::Specified(Shape(operand->GetShape().size()));
    }

    int dim = shape.size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    if (!dynOffset.empty()) {
        MaybeNormalizeValue(
            RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, dynOffset, coaIndexBase, valueToIndex);
        operand->UpdateOffset(TensorOffset{operand->GetOffset(), OpImmediate::ToSpecified(dynOffset)});
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, offset, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynRawshape.empty()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynRawshape, coaIndex, valueToIndex);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, rawshape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynValidShape.empty()) {
        MaybeNormalizeValue(
            RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, dynValidShape, coaIndexBase, valueToIndex);
        operand->UpdateDynValidShape(OpImmediate::ToSpecified(dynValidShape));
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    return operandCoaList;
}

void Function::GetOutcastSymbolicExpr(std::map<int, SymbolicScalar> &tabel) {
    for (size_t idx = 0; idx < outCasts_.size(); idx++) {
        auto op = *outCasts_[idx]->GetProducers().begin();
        if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                tabel[idx] = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
            }
        }
    }
}

std::vector<std::vector<SymbolicScalar>> Function::NormalizeCoa(std::vector<int> &iOffset, std::vector<int> &oOffset) {
    std::unordered_map<int, Operation *> opmagicToOp;
    std::unordered_map<LogicalTensorPtr, int> processedOperands;

    opmagicToOp.reserve(operations_.size());
    for (auto &op : operations_) {
        opmagicToOp[op->GetOpMagic()] = op.get();
    }

    int coaIndex = COA_INDEX_BASE;
    std::vector<std::vector<SymbolicScalar>> coaLists;
    coaLists.reserve(incastPosition.size() + outcastPosition.size());
    NormalizeCoaForInCasts(iOffset, coaLists, coaIndex, processedOperands, opmagicToOp);
    NormalizeCoaForOutCasts(oOffset, coaLists, coaIndex, processedOperands, opmagicToOp);
    NormalizeCoaForNormalOperands(coaLists, coaIndex, processedOperands);
    NormalizeCoaForSpecialInfo(coaLists, coaIndex);

    return coaLists;
}

void Function::NormalizeCoaForInCasts(std::vector<int> &iOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
    int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
    const std::unordered_map<int, Operation *> &opmagicToOp) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    iOffset.clear();
    iOffset.reserve(incastPosition.size());
    for (auto [opmagic, k] : incastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetIOpAttrOffset(k) != -1) {
            continue;
        }
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyIn(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyIn(op, coaIndex, valueToIndex);
            if (CheckEmuOpcode(op, EMUOP_TENSOR_GETDATA_DEPEND)) {
                GetTensorDataSetCoaIndex(op, coaIndex);
            }
        } else {
            auto iOperand = op->GetInputOperand(k);
            auto it = processedOperands.find(iOperand);
            if (it != processedOperands.end()) {
                op->SetIOpAttrOffset(k, it->second);
                iOffset.emplace_back(it->second);
                continue;
            }
            operandCoaList = NormalizeTensor(iOperand, coaIndex, false, op->GetOpcode() == Opcode::OP_NOP);
            processedOperands.emplace(iOperand, coaIndex);
        }
        op->SetIOpAttrOffset(k, coaIndex);
        iOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void Function::NormalizeCoaForOutCasts(std::vector<int> &oOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
    int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
    const std::unordered_map<int, Operation *> &opmagicToOp) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    oOffset.reserve(outcastPosition.size());
    for (auto [opmagic, k] : outcastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetOOpAttrOffset(k) != -1) {
            continue;
        }
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyOut(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyOut(op, coaIndex, valueToIndex);
        } else {
            auto oOperand = op->GetOutputOperand(k);
            auto it = processedOperands.find(oOperand);
            if (it != processedOperands.end()) {
                op->SetOOpAttrOffset(k, it->second);
                oOffset.emplace_back(it->second);
                continue;
            }
            operandCoaList = NormalizeTensor(oOperand, coaIndex, false);
            processedOperands.emplace(oOperand, coaIndex);
        }
        op->SetOOpAttrOffset(k, coaIndex);
        oOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void Function::NormalizeCoaForNormalOperands(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex,
    std::unordered_map<LogicalTensorPtr, int> &processedOperands) {
    std::unordered_set<LogicalTensorPtr> inOutCasts;
    inOutCasts.insert(inCasts_.begin(), inCasts_.end());
    inOutCasts.insert(outCasts_.begin(), outCasts_.end());
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_NOP) {
            continue;
        }
        for (size_t i = 0; i < op->GetInputOperandSize(); i++) {
            auto iOperand = op->GetInputOperand(i);
            if ((op->GetIOpAttrOffset(i) != -1) || (inOutCasts.count(iOperand) > 0)) {
                continue;
            }
            auto it = processedOperands.find(iOperand);
            if (it != processedOperands.end()) {
                op->SetIOpAttrOffset(i, it->second);
                continue;
            }
            if (!iOperand->GetDynOffset().empty() || !iOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(iOperand, coaIndex, valueToIndex);
                processedOperands.emplace(iOperand, coaIndex);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
        for (size_t i = 0; i < op->GetOutputOperandSize(); i++) {
            auto oOperand = op->GetOutputOperand(i);
            if ((op->GetOOpAttrOffset(i) != -1) || (inOutCasts.count(oOperand) > 0)) {
                continue;
            }
            if (oOperand->GetConsumers().empty()) {
                continue;
            }
            auto it = processedOperands.find(oOperand);
            if (it != processedOperands.end()) {
                op->SetOOpAttrOffset(i, it->second);
                continue;
            }
            if (!oOperand->GetDynOffset().empty() || !oOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(oOperand, coaIndex, valueToIndex);
                processedOperands.emplace(oOperand, coaIndex);
                op->SetOOpAttrOffset(i, coaIndex);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
    }
}

void Function::NormalizeCoaForSpecialInfo(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_VEC_DUP || op->GetOpcode() == Opcode::OP_RANGE) {
            if (op->HasAttr(OpAttributeKey::dynScalar)) {
                SymbolicScalar dynScalar = op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, dynScalar, coaIndex, valueToIndex);
                op->SetAttribute(OpAttributeKey::dynScalar, dynScalar);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        } else if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                SymbolicScalar bindTensor = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, bindTensor, coaIndex, valueToIndex);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        }
    }
}

void Function::DumpTopoFile(const std::string &fileName) const
{
    Json totalTopoJson;
    for (const auto &topo : topoInfo_.GetTopology()) {
        Json sJson;
        sJson["taskId"] = topo.esgId;
        sJson["successors"] = Json::array();
        for (const auto &successor : topo.outGraph) {
            sJson["successors"].push_back(successor);
        }
        int id = operations_[topo.esgId]->GetProgramId();
        if (static_cast<size_t>(id) >= calleeMagicNameList_.size()) {
            continue;
        }
        sJson["funcName"] = calleeMagicNameList_[id];
        sJson["semanticLabel"] = operations_[topo.esgId]->GetSemanticLabelStr();
        totalTopoJson.push_back(sJson);
    }
    std::ofstream ofs(fileName);
    ofs << totalTopoJson.dump(1) << std::endl;
    ofs.close();
}


std::string Function::DumpSSATitle() const {
    std::stringstream ss;
    ss << GetMagicName() << "[" << functionMagic_ << "]"
       << " " << GetFunctionHash()
       << " " << GetFunctionTypeNameDict().Find(GetFunctionType())
       << " " << GetGraphTypeNameDict().Find(GetGraphType());
    return ss.str();
}

std::string Function::DumpSSARawTensor(int indent) const {
    std::string prefix(indent, ' ');
    std::unordered_set<int> dumpedRawTensor;
    auto dumped = [&dumpedRawTensor](const std::shared_ptr<LogicalTensor> &tensor) -> bool {
        if (dumpedRawTensor.count(tensor->GetRawTensor()->GetRawMagic()) == 0) {
            dumpedRawTensor.insert(tensor->GetRawTensor()->GetRawMagic());
            return false;
        } else {
            return true;
        }
    };
    std::stringstream ss;
    int rawIndex = 0;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        if (!dumped(inCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE)
               << std::setfill(' ') << rawIndex++ << "] "
               << inCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        if (!dumped(outCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE)
               << std::setfill(' ') << rawIndex++ << "] "
               << outCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < operations_.size(); ++i) {
        for (auto &input : operations_[i]->GetIOperands()) {
            if (!dumped(input)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE)
                   << std::setfill(' ') << rawIndex++ << "] "
                   << input->GetRawTensor()->DumpSSA() << "\n";
            }
        }
        for (auto &output : operations_[i]->GetOOperands()) {
            if (!dumped(output)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE)
                   << std::setfill(' ') << rawIndex++ << "] "
                   << output->GetRawTensor()->DumpSSA() << "\n";
            }
        }
    }
    return ss.str();
}
std::string Function::DumpSSAIncast(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        ss << prefix << "INCAST[" << std::setw(SPACE_NUM_THREE)
           << std::setfill(' ') << i << "]  "
           << inCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.incastSlot.size()) {
            auto &incastSlotList = slotScope_->ioslot.incastSlot[i];
            ss << " fromSlot[";
            for (size_t k = 0; k < incastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << incastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}
std::string Function::DumpSSAOutcast(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        ss << prefix << "OUTCAST[" << std::setw(SPACE_NUM_THREE)
           << std::setfill(' ') << i << "]  "
           << outCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.outcastSlot.size()) {
            auto &outcastSlotList = slotScope_->ioslot.outcastSlot[i];
            ss << " toSlot[";
            for (size_t k = 0; k < outcastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << outcastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}

std::string Function::DumpSSAAttribute(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    if (IsDynloop()) {
        auto attr = GetDynloopAttribute();
        ss << prefix << "LOOP SYMBOL " << attr->iterSymbolName << "\n";
        ss << prefix << "LOOP BEGIN  " << attr->Begin().Dump() << "\n";
        ss << prefix << "LOOP END    " << attr->End().Dump() << "\n";
        ss << prefix << "LOOP STEP   " << attr->Step().Dump() << "\n";
    }
    return ss.str();
}

constexpr int INDENT_TWO = 2;

std::string Function::DumpSSA() const {
    std::stringstream ss;
    ss << "\n-------------\n";
    ss << "Function " << DumpSSATitle() << " {\n";
    ss << DumpSSARawTensor(INDENT_TWO) << "\n";
    ss << DumpSSAIncast(INDENT_TWO) << "\n";
    ss << DumpSSAOutcast(INDENT_TWO) << "\n";
    ss << DumpSSAAttribute(INDENT_TWO) << "\n";
    for (size_t i = 0; i < operations_.size(); ++i) {
        auto op = operations_[i];
        ss << op->DumpSSA(PREFIX); // Operation dump
    }
    ss << "}\n";
    return ss.str();
}

std::string Function::Dump() const {
    return DumpSSA();
}

void Function::DumpFile(const std::string &filePath) const {
    std::ofstream fout(filePath);
    fout << Dump();
    fout.close();
}

void Function::UpdateOperandBeforeRemoveOp(Operation &op, const bool keepOutTensor) {
    // relink, replace input of following op with the input of current op
    if (!op.GetIOperands().empty() && !op.GetOOperands().empty()) {
        LogicalTensorPtr inputTensor = op.GetIOperands().at(0);
        LogicalTensorPtr outputTensor = op.GetOOperands().at(0);
        bool isOutCast = std::find(outCasts_.begin(), outCasts_.end(), outputTensor) != outCasts_.end();
        if (isOutCast || keepOutTensor) {
            outputTensor->RemoveProducer(op);
            for (auto &producer : inputTensor->GetProducers()) {
                outputTensor->AddProducer(*producer);
                producer->ReplaceOutputOperand(inputTensor, outputTensor);
            }
            inputTensor->GetProducers().clear();
        } else {
            inputTensor->RemoveConsumer(op);
            for (const auto &consumer : outputTensor->GetConsumers()) {
                inputTensor->AddConsumer(consumer);
                consumer->ReplaceInputOperand(outputTensor, inputTensor);
            }
            outputTensor->GetConsumers().clear();
        }
    }
}

/**
 * @brief handle input and output control edges
 * all input ctrl edges shall be moved to output ops of current op
 * all output ctrl edges shall be moved to input ops of current op
 * @param op
 */
void Function::HandleControlOps(Operation &op, std::vector<Operation *> &toRemoveOps) const {
    const auto &inputCtrlOpSet = op.GetInCtrlOperations();
    if (!inputCtrlOpSet.empty()) {
        auto outputOps = GetAllOutputOperations(op);
        for (auto peerCtrlOp : inputCtrlOpSet) {
            if (peerCtrlOp == nullptr) {
                continue;
            }
            if (peerCtrlOp->OnlyHasCtrlEdgeToOp(op)) {
                op.RemoveInCtrlOperation(*peerCtrlOp);
                toRemoveOps.push_back(peerCtrlOp);
            } else {
                for (auto &outputOp : outputOps) {
                    outputOp->AddInCtrlOperation(*peerCtrlOp);
                }
            }
        }
        op.ClearInCtrlOperations();
    }
    const auto &outputCtrlOpSet = op.GetOutCtrlOperations();
    if (!outputCtrlOpSet.empty()) {
        auto inputOps = GetAllInputOperations(op);
        for (auto &inputOp : inputOps) {
            for (auto outCtrlOp : outputCtrlOpSet) {
                inputOp->AddOutCtrlOperation(*outCtrlOp);
            }
        }
        op.ClearOutCtrlOperations();
    }
}

Operation *Function::GetOpByOpMagic(const int opMagic) const {
    for (auto op : operations_) {
        if (op->GetOpMagic() == opMagic) {
            return op.get();
        }
    }
    return nullptr;
}

bool Function::TensorReuse(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor) {
    if (dstTensor == nullptr || srcTensor == nullptr) {
        return false;
    }
    if (dstTensor->Datatype() != srcTensor->Datatype() ||
        dstTensor->tensor->GetRawShapeSize() != srcTensor->tensor->GetRawShapeSize()) {
        ASLOGI("Data type or raw shape size of src and dst tensor is not same.");
        return false;
    }

    if (dstTensor->tensor->rawshape == srcTensor->tensor->rawshape) {
        dstTensor->tensor = srcTensor->tensor;
    } else {
        dstTensor->tensor->actualRawmagic = srcTensor->tensor->actualRawmagic == -1 ?
                                                 srcTensor->tensor->rawmagic :
                                                 srcTensor->tensor->actualRawmagic;
    }
    return true;
}

bool Function::IsFromInCast(const std::shared_ptr<LogicalTensor> &tensor) {
    for (auto &t : inCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromOutCast(const std::shared_ptr<LogicalTensor> &tensor) {
    for (auto &t : outCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromDummyOutCast(int rawMagic) {
    for (auto &t : outCasts_) {
        if (t->tensor->rawmagic == rawMagic) {
            return true;
        }
    }
    return false;
}

int Function::GetIncastIndex(std::shared_ptr<LogicalTensor> &tensor) const {
    for (size_t idx = 0; idx < inCasts_.size(); idx++) {
        if (inCasts_[idx] == tensor) {
            return (int)idx;
        }
    }
    return INVALID_IOINDEX;
}

int Function::GetOutcastIndex(std::shared_ptr<LogicalTensor> &tensor) const {
    for (size_t idx = 0; idx < outCasts_.size(); idx++) {
        if (outCasts_[idx] == tensor) {
            return (int)idx;
        }
    }
    return INVALID_IOINDEX;
}

TensorGraphInfo Function::GetGraphInfo() {
    std::vector<LogicalTensors> callopInCasts, callopOutCasts;
    std::set<std::shared_ptr<Operation>> viewOpSet, assembleOpSet;
    std::set<std::shared_ptr<LogicalTensor>> iOperandSet, oOperandSet;
    std::vector<std::shared_ptr<Operation>> operations;
    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOpSet.emplace(op);
            continue;
        }
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleOpSet.emplace(op);
            continue;
        }
        ASSERT(op->GetOpcode() == Opcode::OP_CALL)
            << "Invalid operation code: " << static_cast<int>(op->GetOpcode()) << "\n"
            << "Operation: " << op->Dump();
        operations.emplace_back(op);
        LogicalTensors incasts;
        LogicalTensors outcasts;
        for (auto &iOperand : op->GetIOperands()) {
            auto& viewOp = *iOperand->GetProducers().begin();
            auto& incast = viewOp->GetIOperands()[0];

            iOperandSet.emplace(iOperand);
            incasts.push_back(incast);
        }
        for (auto &oOperand : op->GetOOperands()) {
            auto &assembleOp = *oOperand->GetConsumers().begin();
            auto &outcast = assembleOp->GetOOperands()[0];

            oOperandSet.emplace(oOperand);
            outcasts.push_back(outcast);
        }
        callopInCasts.emplace_back(incasts);
        callopOutCasts.emplace_back(outcasts);
    }
    operations_ = operations;
    return std::make_tuple(std::move(callopInCasts), std::move(callopOutCasts),
        std::move(viewOpSet), std::move(assembleOpSet),
        std::move(iOperandSet), std::move(oOperandSet)
    );
}

void Function::ClearUselessLink(TensorGraphInfo &graphInfo) {
    auto& callopInCasts = std::get<0>(graphInfo);
    auto& callopOutCasts = std::get<1>(graphInfo);
    auto& viewOpSet = std::get<2>(graphInfo);
    auto& assembleOpSet = std::get<3>(graphInfo);
    auto& iOperandSet = std::get<4>(graphInfo);
    auto& oOperandSet = std::get<5>(graphInfo);

    for (auto iOperand : iOperandSet) {
        iOperand->GetProducers().clear();
        iOperand->GetConsumers().clear();
    };
    for (auto oOperand : oOperandSet) {
        oOperand->GetProducers().clear();
        oOperand->GetConsumers().clear();
    };

    for (auto viewOp : viewOpSet) {
        viewOp->GetIOperands().clear();
        viewOp->GetOOperands().clear();
    };

    for (auto assembleOp : assembleOpSet) {
        assembleOp->GetIOperands().clear();
        assembleOp->GetOOperands().clear();
    };

    for (auto incasts : callopInCasts) {
        for (auto incast : incasts) {
            incast->GetConsumers().clear();
        }
    };

    for (auto outcasts : callopOutCasts) {
        for (auto outcast : outcasts) {
            outcast->GetProducers().clear();
        }
    };

    for (auto operation : operations_) {
        operation->GetIOperands().clear();
        operation->GetOOperands().clear();
    };
}

void Function::LinkIoWithCallOp(std::vector<LogicalTensors> &callopInCasts,
                                std::vector<LogicalTensors> &callopOutCasts) {
    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        auto &incasts = callopInCasts[idx];
        for (auto incast : incasts) {
            incast->AddConsumer(*operations_[idx]);
            operations_[idx]->iOperand.emplace_back(incast);
        }
    }

    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        auto &outcasts = callopOutCasts[idx];
        for (auto outcast : outcasts) {
            outcast->AddProducer(*operations_[idx]);
            operations_[idx]->oOperand.emplace_back(outcast);
        }
    }
}

void Function::RemoveCallOpViewAssemble() {
    auto graphInfo = GetGraphInfo();
    ClearUselessLink(graphInfo);
    LinkIoWithCallOp(std::get<0>(graphInfo), std::get<1>(graphInfo));
}

void Function::UpdateOriIocastSlot(const std::shared_ptr<TensorSlotScope> scope) {
    ASSERT(slotScope_ != nullptr) << "slotScope_ is null";
    auto& incastDst = slotScope_->oriIncastReadSlotSet;
    incastDst.insert(incastDst.end(), scope->incastReadSlotSet.begin(), scope->incastReadSlotSet.end());

    auto& outcastDst = slotScope_->oriOutcastWriteSlotSet;
    outcastDst.insert(outcastDst.end(), scope->outcastWriteSlotSet.begin(), scope->outcastWriteSlotSet.end());

    auto& iSlot = slotScope_->originalIocastsSlot.incastSlot;
    auto& oSlot = slotScope_->originalIocastsSlot.outcastSlot;

    iSlot.insert(iSlot.end(), scope->ioslot.incastSlot.begin(), scope->ioslot.incastSlot.end());
    oSlot.insert(oSlot.end(), scope->ioslot.outcastSlot.begin(), scope->ioslot.outcastSlot.end());
}

void Function::SetCallOpSlot() {
    // op all OP_CALL
    bool isAllCallOp = std::all_of(operations_.begin(), operations_.end(),
        [](const auto& op) {
            return op->GetOpcode() == Opcode::OP_CALL;
        });
    if (!isAllCallOp) {
        return;
    }
    std::vector<Function *> calleeList = GetCalleeFunctionList();
    for (auto callee: calleeList) {
        if (callee == nullptr) {
            continue;
        }
        const std::shared_ptr<TensorSlotScope> calleeScope = callee->GetSlotScope();
        // callee incast -> call op iOperand, callee outcast -> call op oOperand
        UpdateOriIocastSlot(calleeScope);
    }
    return;
}

std::vector<int> Function::GetInCastSlot(const std::shared_ptr<LogicalTensor> &incast) {
    std::vector<int> ret;
    for (size_t idx = 0; idx < inCasts_.size(); ++idx) {
        if (inCasts_[idx] == incast) {
            auto &scope = GetSlotScope();
            ASSERT(scope != nullptr) << "SlotScope is null";
            ret = scope->ioslot.incastSlot[idx];
        }
    }
    return ret;
}

std::vector<int> Function::GetOutCastSlot(const std::shared_ptr<LogicalTensor> &outcast) {
    std::vector<int> ret;
    for (size_t idx = 0; idx < outCasts_.size(); ++idx) {
        if (outCasts_[idx] == outcast) {
            auto &scope = GetSlotScope();
            ASSERT(scope != nullptr) << "SlotScope is null";
            ret = scope->ioslot.outcastSlot[idx];
        }
    }
    return ret;
}

void Function::ResetOperations() {
    operations_.clear();
    opPosition_.clear();
    sorted_ = false;
    tensorMap_.Reset();
    for (auto &i : inCasts_) {
        tensorMap_.Insert(i);
    }
    for (auto &t : inCasts_) {
        t->GetConsumers().clear();
    }
    for (auto &t : outCasts_) {
        t->GetProducers().clear();
    }
}

std::set<Operation *, LogicalTensor::CompareOp> Function::FindConsumers(const Operation &op) const {
    std::set<Operation *, LogicalTensor::CompareOp> consumers;
    for (const auto &output : op.oOperand) {
        for (auto &consumer : output->GetConsumers()) {
            if (consumer->BelongTo() == this) {
                consumers.emplace(consumer);
            }
        }
    }
    return consumers;
}

std::set<Operation *, LogicalTensor::CompareOp> Function::FindProducers(const Operation &op) const {
    std::set<Operation *, LogicalTensor::CompareOp> producers;
    for (const auto &input : op.iOperand) {
        for (auto &producer : input->GetProducers()) {
            if (producer->BelongTo() == this) {
                producers.emplace(producer);
            }
        }
    }
    return producers;
}

std::vector<OriArgInfo> Function::GetOpOriginArgsInfo() {
    std::map<int, OriArgInfo> args;
    int maxSubscript = 0;
    for (const auto &incast : inCasts_) {
        auto subscript = GetParamIndex(incast->GetRawTensor());
        if (subscript == -1) {
            continue;
        }
        maxSubscript = std::max(maxSubscript, subscript);
        OriArgInfo info{reinterpret_cast<uint64_t>(GetParamAddress(subscript)), incast->MemorySize(),
            incast->GetCachePolicy(CachePolicy::PREFETCH)};
        if (args.count(subscript) > 0) {
            ASSERT(args.at(subscript) == info);
        } else {
            args.emplace(subscript, info);
        }
    }
    for (const auto &outcast : outCasts_) {
        auto subscript = GetParamIndex(outcast->GetRawTensor());
        if (subscript == -1) {
            continue;
        }
        maxSubscript = std::max(maxSubscript, subscript);
        OriArgInfo info{reinterpret_cast<uint64_t>(GetParamAddress(subscript)), outcast->MemorySize(),
            outcast->GetCachePolicy(CachePolicy::PREFETCH)};
        if (args.count(subscript) > 0) {
            ASSERT(args.at(subscript) == info);
        } else {
            args.emplace(subscript, info);
        }
    }

    std::vector<OriArgInfo> argsInfo(maxSubscript + 1);
    for (int idx = 0; idx <= maxSubscript; idx++) {
        if (args.count(idx) > 0) {
            argsInfo[idx] = args.at(idx);
        } else {
            argsInfo[idx] = OriArgInfo{0, 0, false};
        }
    }
    return argsInfo;
}

void Function::OpValidCheck(Operation &op) const {
    std::unordered_set<const Operation *> opMap;
    std::unordered_set<std::shared_ptr<LogicalTensor>> incasts(GetIncast().begin(), GetIncast().end());
    if (SPECIAL_OPCODE_SET.count(op.GetOpcode()) != 0) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ASSERT(op.GetIOperands().size() == 1) << "OP_VIEW expects 1 input operand, but got " << op.GetIOperands().size();
            ASSERT(op.GetOOperands().size() <= 1) << "OP_VIEW expects at most 1 output operand, but got " << op.GetOOperands().size();
            auto opAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            ASSERT(opAttr != nullptr)
                << "OP_VIEW should have a ViewOpAttribute, but it is null";
            ASSERT(op.GetIOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size())
                << "OP_VIEW input operand offset size does not match attribute from offset size";
            if (!op.GetOOperands().empty()) {
                ASSERT(op.GetOOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size())
                    << "OP_VIEW output operand offset size does not match attribute from offset size";
            }
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ASSERT(op.GetIOperands().size() == 1)
                << "OP_ASSEMBLE should have exactly 1 input operand, but has " << op.GetIOperands().size();
            ASSERT(op.GetOOperands().size() <= 1)
                << "OP_ASSEMBLE should have at most 1 output operand, but has " << op.GetOOperands().size();
            auto opAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            ASSERT(opAttr != nullptr)
                << "OP_ASSEMBLE should have an AssembleOpAttribute, but it is null";
            if (!op.GetIOperands().empty()) {
                ASSERT(op.GetIOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size())
                    << "OP_ASSEMBLE input operand offset size does not match attribute to offset size";
            }
            ASSERT(op.GetOOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size())
                << "OP_ASSEMBLE output operand offset size does not match attribute to offset size";
        }
    } else {
        ASSERT(op.GetOpAttribute() == nullptr)
            << "Non-special operation should not have an operation attribute";
    }

    ASSERT(op.GetOpMagic() >= 0 && op.GetOpMagic() < opSeed_)
        << "Operation magic number is out of bounds: " << op.GetOpMagic()
        << ", function opSeed_ is: " << opSeed_;
    if (!op.IsCall()) { // call 允许多输出，其余操作目前不允许
        ASSERT(op.GetOOperands().size() <= 1)
            << "Non-call operation should have at most 1 output operand, but has " << op.GetOOperands().size();
    }
    for (auto &oOperand : op.GetOOperands()) {
        ASSERT(oOperand->GetShape().size() == oOperand->GetOffset().size())
            << "Output operand shape size does not match offset size";
        ASSERT(&oOperand->BelongFunction() == this)
            << "Output operand does not belong to the current function";
        auto tmp = GetTensorMap().GetTensorByMagic(oOperand->magic);
        ASSERT(tmp == oOperand)
            << "Tensor map does not match output operand";
        ASSERT(oOperand->HasProducer(op))
            << "Output operand does not have the current operation as a producer, opmagic: " << op.GetOpMagic()
            << ", operand magic: " << oOperand->magic;
    }
    for (auto &iOperand : op.GetIOperands()) {
        ASSERT(iOperand->GetShape().size() == iOperand->GetOffset().size())
            << "Input operand shape size does not match offset size";
        ASSERT(&iOperand->BelongFunction() == this)
            << "Input operand does not belong to the current function";
        if (!iOperand->GetProducers().empty() || incasts.count(iOperand) != 0) {
            auto tmp = GetTensorMap().GetTensorByMagic(iOperand->magic);
            ASSERT(tmp == iOperand)
                << "Tensor map does not match input operand";
        }

        ASSERT(iOperand->HasConsumer(op))
            << "Input operand does not have the current operation as a consumer";
        for (const auto &producer : iOperand->GetProducers()) {
            ASSERT(producer->BelongTo() == this)
                << "Producer does not belong to the current function";
            ASSERT(producer->GetOpMagic() >= 0 && producer->GetOpMagic() < opSeed_)
                << "Producer magic number is out of bounds: " << producer->GetOpMagic() << ", function opSeed_ is: " << opSeed_
                << ", producer in tensor(" << iOperand->magic << "," << iOperand->tensor->rawmagic << ")";
            if (producer->IsDeleted()) {
                continue;
            }
            ASSERT(opMap.find(producer) != opMap.end())
                << "Producer not found in operation map";
        }
    }

    ASSERT(opMap.count(&op) == 0)
        << "Operation is already in the operation map";
    opMap.emplace(&op);
}

DyndevFunctionAttribute::ValueDependDesc Function::LookupValueDepend() {
    struct ValueDependSearcher {
        static void Search(DyndevFunctionAttribute::ValueDependDesc &desc, const SymbolicScalar &attr) {
            std::vector<RawSymbolicScalarPtr> callList = LookupExpressionByOpcode(attr.Raw(), SymbolicOpcode::T_MOP_CALL);
            for (auto &call : callList) {
                auto caller = call->GetExpressionOperandList()[0];
                if (!caller->IsSymbol()) {
                    continue;
                }
                std::string name = caller->GetSymbolName();
                if (CallIsGetInputData(name)) {
                    desc.getInputDataCount++;
                } else if (CallIsGetTensorData(name)) {
                    desc.getTensorDataCount++;
                }
            }
        }
    };

    DyndevFunctionAttribute::ValueDependDesc desc;
    if (GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        ValueDependSearcher::Search(desc, loopAttr->Begin());
        ValueDependSearcher::Search(desc, loopAttr->End());
        ValueDependSearcher::Search(desc, loopAttr->Step());
        for (auto &path : loopAttr->GetPathList()) {
            for (auto &cond : path.GetPathCondList()) {
                ValueDependSearcher::Search(desc, cond.GetCond());
            }
        }

    } else {
        for (auto &op : Operations(false)) {
            std::vector<std::reference_wrapper<SymbolicScalar>> attrList = op.GetDynamicAttributeList();
            for (auto &attr : attrList) {
                ValueDependSearcher::Search(desc, attr.get());
            }
        }
    }
    return desc;
}

void Function::ValidCheck() const {
    int opMagic = -1000000;
    for (auto &op : const_cast<Function &>(*this).Operations()) {
        opMagic = std::max(opMagic, op.GetOpMagic());
    }
    ASSERT(opMagic + 1 <= opSeed_)
        << "Invalid opMagic range: max opMagic is " << opMagic << ", function opSeed_ is: " << opSeed_;

    TensorMagicCheck();

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::vector<Operation *>> used;
    for (auto &op : const_cast<Function &>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        for (const auto &operand : op.GetOOperands()) {
            if (used.count(operand) > 0) {
                for (auto innerOp : used.at(operand)) {
                    ASSERT(innerOp->ComputeHash() != op.ComputeHash()) << "Duplicate operation detected with the same hash: " << op.ComputeHash();
                }
            }
            used[operand].emplace_back(&op);
        }
    }

    for (auto &op : const_cast<Function &>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        OpValidCheck(op);
    }
}

std::shared_ptr<OpAttribute> Function::CreateCallOpAttribute(const std::vector<std::vector<SymbolicScalar>> &argList,
                                                             const std::map<int, SymbolicScalar> &outIndexToExpr) {
    FunctionHash hash;
    if (rootFunc_ != nullptr) {
        /* has rootFunc, then current function is cutted */
        hash = rootFunc_->ComputeHash();
    } else {
        hash = ComputeHash();
    }
    auto opAttribute = std::make_shared<CallOpAttribute>(hash, argList, GetMagicName(), outIndexToExpr);
    return opAttribute;
}

std::shared_ptr<LogicalTensor> Function::ConnectWithOverlap(std::shared_ptr<LogicalTensor> iOperand) {
    auto matches = GetTensorMap().Find(iOperand);
    if (matches.empty()) {
        return iOperand;
    }
    auto overlapStatus = CalcOverlap(iOperand, matches);
    ASSERT(!matches.empty())
        << "Matches should not be empty";

    std::vector<std::vector<int64_t>> offsetOfOverlaps;
    std::vector<std::shared_ptr<LogicalTensor>> needAddConsumer;
    sort(matches.begin(), matches.end(), [](const auto &a, const auto &b) -> bool { return a->offset < b->offset; });

    std::vector<int64_t> minimumOffsets = matches.front()->offset;

    for (auto &m : matches) {
        offsetOfOverlaps.emplace_back(m->offset);
        for (size_t idx = 0; idx < minimumOffsets.size(); idx++) {
            minimumOffsets[idx] = std::min(minimumOffsets[idx], m->offset[idx]);
        }
    }
    for (auto &offsetOfOverlap : offsetOfOverlaps) {
        for (size_t idx = 0; idx < offsetOfOverlap.size(); idx++) {
            offsetOfOverlap[idx] -= minimumOffsets[idx];
        }
    }

    switch (overlapStatus) {
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL: {
            auto assembleResult = std::make_shared<LogicalTensor>(*this, iOperand->Datatype(), iOperand->shape,
                iOperand->GetDynValidShape(), iOperand->Format(), "Assemble_" + matches[0]->Symbol(),
                iOperand->nodetype);
            ASSERT(assembleResult->GetProducers().empty()) << "Assemble result should have no producers";
            for (size_t idx = 0; idx < matches.size(); idx++) {
                auto &assembleOp = AddRawOperation(Opcode::OP_ASSEMBLE, {matches[idx]}, {assembleResult});
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offsetOfOverlaps[idx], SymbolicScalar::FromConcrete(offsetOfOverlaps[idx])));
            }
            return assembleResult;
        }
        case OverlapStatus::PERFECTLY_MATCH: {
            // IF there is an existing tensor that fully matches GetIOperands(), change GetIOperands() to
            // the existing tensor
            return matches.front();
        }
        case OverlapStatus::BE_COVERED: {
            auto viewResult = std::make_shared<LogicalTensor>(*this, matches.front()->tensor->datatype, iOperand->shape,
                iOperand->Format(), "View_" + matches.front()->tensor->symbol, matches.front()->nodetype);
            auto &viewOp = AddRawOperation(Opcode::OP_VIEW, {matches.front()}, {viewResult});
            viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(
                iOperand->GetOffset(), iOperand->GetDynOffset(), iOperand->GetDynValidShape()));
            if (!iOperand->GetDynValidShape().empty()) {
                viewResult->UpdateDynValidShape(iOperand->GetDynValidShape());
            }
            return viewResult;
        }
        case OverlapStatus::BE_COVERED_BY_ALL: {
            std::vector<int64_t> minimumOffset;
            std::vector<int64_t> maximumShape;
            CalcShapeAndOffsetOfGroup(matches, minimumOffset, maximumShape);

            auto assembleResult = std::make_shared<LogicalTensor>(*this, matches[0]->Datatype(), maximumShape,
                iOperand->Format(), "Assemble_" + matches[0]->Symbol(), iOperand->nodetype);
            ASSERT(assembleResult->GetProducers().empty()) "Assemble result should have no producers";
            for (size_t idx = 0; idx < matches.size(); idx++) {
                auto &assembleOp = AddRawOperation(Opcode::OP_ASSEMBLE, {matches[idx]}, {assembleResult});
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offsetOfOverlaps[idx], SymbolicScalar::FromConcrete(offsetOfOverlaps[idx])));
            }

            auto viewResult = std::make_shared<LogicalTensor>(*this, assembleResult->Datatype(), iOperand->shape,
                iOperand->Format(), "View_" + assembleResult->Symbol(), assembleResult->nodetype);
            auto &viewOp = AddRawOperation(Opcode::OP_VIEW, {assembleResult}, {viewResult});
            std::vector<int64_t> newOffset = TensorOffset::Sub(iOperand->GetOffset(), minimumOffset);
            std::vector<SymbolicScalar> newDynOffset = TensorOffset::Sub(iOperand->GetDynOffset(), minimumOffset);
            // fill valid shape
            viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffset, newDynOffset, iOperand->GetDynValidShape()));
            return viewResult;
        }
        default: ASSERT(false) << "unexpected behavior";
    }

    ASSERT(false);
    return nullptr;
}

using SameSlotSetIndex = std::map<std::vector<int>, std::vector<int>>;

template <typename T>
void RemoveDupIndices(std::vector<T>& data, std::vector<int> indexList) {
    std::sort(indexList.begin(), indexList.end(), std::greater<int>());
    for (int index : indexList) {
        if (index >= 0 && index < static_cast<int>(data.size())) {
            data.erase(data.begin() + index);
        }
    }
}

SameSlotSetIndex ClassifyIocasts(const std::vector<std::vector<int>>& vec) {
    SameSlotSetIndex classification;
    for (size_t idx = 0; idx < vec.size(); ++idx) {
        classification[vec[idx]].push_back(idx);
    }

    // if value vector size < 2，delete this pair
    for (auto it = classification.begin(); it != classification.end(); ) {
        if (it->second.size() == 1) {
            it = classification.erase(it);
        } else {
            ++it;
        }
    }
    return classification;
}

void Function::DoMergeFunctionDupIncast() {
    auto sameSlotSetIndex = ClassifyIocasts(GetSlotScope()->ioslot.incastSlot);
    std::vector<int> removeIdx;
    for (auto& pair : sameSlotSetIndex) {
        auto& slotSetIndex = pair.second;
        ASSERT(slotSetIndex.size() > 1)
            << "Slot set index should have more than one element";
        removeIdx.insert(removeIdx.end(), slotSetIndex.begin() + 1, slotSetIndex.end());
        auto oriIncast = inCasts_[slotSetIndex[0]];
        auto newIncast = std::make_shared<LogicalTensor>(*this, oriIncast->tensor->datatype, oriIncast->shape,
            oriIncast->tensor->GetDynRawShape(), oriIncast->Format(), oriIncast->tensor->GetSymbol(), NodeType::INCAST);

        for (auto incastIdx : slotSetIndex) {
            ASSERT(inCasts_[incastIdx]->GetConsumers().size() > 0)
                << "Incast at index " << incastIdx << " should have at least one consumer";
            auto op = *inCasts_[incastIdx]->GetConsumers().begin();
            op->ReplaceIOperand(0, newIncast);
            tensorMap_.Insert(newIncast);
        }
        inCasts_[slotSetIndex[0]] = newIncast;
    }
    RemoveDupIndices(inCasts_, removeIdx);
    RemoveDupIndices(GetSlotScope()->incastReadSlotSet, removeIdx);
    RemoveDupIndices(GetSlotScope()->ioslot.incastSlot, removeIdx);
}

void Function::DoMergeFunctionDupOutcast() {
    auto sameSlotSetIndex = ClassifyIocasts(GetSlotScope()->ioslot.outcastSlot);
    std::vector<int> removeIdx;
    for (auto& pair : sameSlotSetIndex) {
        auto& slotSetIndex = pair.second;
        ASSERT(slotSetIndex.size() > 1)
            << "Slot set index should have more than one element";
        removeIdx.insert(removeIdx.end(), slotSetIndex.begin() + 1, slotSetIndex.end());
        auto oriOutcast = outCasts_[slotSetIndex[0]];
        auto newOutcast = std::make_shared<LogicalTensor>(*this, oriOutcast->tensor->datatype, oriOutcast->shape,
            oriOutcast->tensor->GetDynRawShape(), oriOutcast->Format(), oriOutcast->tensor->GetSymbol(), NodeType::OUTCAST);

        for (auto incastIdx : slotSetIndex) {
            ASSERT(outCasts_[incastIdx]->GetProducers().size() > 0)
                << "Outcast at index " << incastIdx << " should have at least one producer";
            auto& op = *outCasts_[incastIdx]->GetProducers().begin();
            op->ReplaceOOperand(0, newOutcast);
            tensorMap_.Insert(newOutcast);
        }
        outCasts_[slotSetIndex[0]] = newOutcast;
    }
    RemoveDupIndices(outCasts_, removeIdx);
    RemoveDupIndices(GetSlotScope()->outcastWriteSlotSet, removeIdx);
    RemoveDupIndices(GetSlotScope()->ioslot.outcastSlot, removeIdx);
}

void Function::MergeFunctionDupIocast() {
    DoMergeFunctionDupIncast();
    DoMergeFunctionDupOutcast();
}

bool Function::InsertLoopIdxNameList(const std::string &idxName) {
    if (parent_ == nullptr) {
        loopIdxNameList_.insert(idxName);
        return true;
    }

    auto realParent = parent_->parent_;
    if (realParent->GetFunctionType() == FunctionType::DYNAMIC_LOOP &&
        realParent->LoopIdxNameList().find(idxName) != realParent->LoopIdxNameList().end()) {
        return false;
    }

    loopIdxNameList_.insert(idxName);
    for (const auto &it : realParent->LoopIdxNameList()) {
        loopIdxNameList_.insert(it);
    }
    return true;
}

DefineProg::DefineProg(const std::string &name) : isRecording_(true) {
    Program::GetInstance().SetName(name);
}
DefineProg::~DefineProg() {
    if (isRecording_) {
        ALOG_INFO("prog.end: name=", Program::GetInstance().Name());
    }
}

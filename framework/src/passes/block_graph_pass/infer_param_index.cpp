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
 * \file infer_param_index.cpp
* \brief
 */

#include <queue>
#include <vector>
#include "infer_param_index.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/opcode.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InferParamIndex"

namespace npu {
namespace tile_fwk {
std::string InferParamIndex::DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable)
{
    std::ostringstream ss;
    for (auto paramInfo : dynParamTable) {
        ss << "param: " << paramInfo.first << " ( ";
        ss << "tensorIdx: " << paramInfo.second.tensorIndex << ", ";
        ss << "dimsize: " << paramInfo.second.dimSize << ", ";
        ss << "type: " << static_cast<int>(paramInfo.second.type) << ", ";
        ss << "addrCoaIdx: " << paramInfo.second.tensorBaseAddrCoaIndex << ", ";
        ss << "dimIdx: " << paramInfo.second.dimIndex << " )" << std::endl;
    }
    return ss.str();
}

Status InferParamIndex::ResetOutputDynValidShape(const Operation &op) {
    std::vector<SymbolicScalar> validShape;
    const std::set<Opcode> specifiedOps = {Opcode::OP_VEC_DUP, Opcode::OP_EXPAND, Opcode::OP_RESHAPE,Opcode::OP_GATHER,
        Opcode::OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_L1, Opcode::OP_LOAD};
    for (auto outOperand : op.GetOOperands()) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) || specifiedOps.count(op.GetOpcode())) {
            for (size_t dimIdx = 0U; dimIdx < outOperand->GetShape().size(); ++dimIdx) {
                validShape.push_back(SymbolicScalar("sym_" +  std::to_string(outOperand->GetMagic()) +
                                                    "_dim_" + std::to_string(dimIdx)));
            }
        }
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) { // Assemble的oOperand保持validShape不变
            outOperand->UpdateDynValidShape(validShape);
        }
    }
    return SUCCESS;
}

Status InferParamIndex::ResetViewDynValidShape(const Operation &op) {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return SUCCESS;
    }
    auto newDynValidShape = viewOpAttribute->GetToDynValidShape();
    std::vector<int> newValidShape;
    for (auto validSym : newDynValidShape) {
        if (validSym.ConcreteValid()) {
            newValidShape.push_back(validSym.Concrete());
        }
    }
    if (newValidShape.size() == newDynValidShape.size()) {
        op.GetOOperands()[0]->UpdateDynValidShape(newDynValidShape);
        return SUCCESS;
    }
    viewOpAttribute->SetToDynValidShape(std::vector<SymbolicScalar>());
    return SUCCESS;
}

Status InferParamIndex::ResetAssembleDynValidShape(const Operation &op) {
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
    if (assembleOpAttribute != nullptr) {
        auto emptyValidShape = std::vector<SymbolicScalar>();
        assembleOpAttribute->SetFromDynValidShape(emptyValidShape);
    }
    return SUCCESS;
}

Status InferParamIndex::ResetDynValidShape(Function& function) {
    for (auto &op : function.Operations(false)) {
        if (ResetOutputDynValidShape(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Fail to reset the output operand shape of operation %d in function %s. Please check whether the shape is valid in your input graph.%s", op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        // 清空view和assemble的属性中的dynvalidshape，以便后续重新推导符号化的dynvalidshape
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (ResetViewDynValidShape(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Fail to reset the output operand shape of VIEW operation %d in function %s. %s", op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (ResetAssembleDynValidShape(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Fail to reset the output operand shape of ASSEMBLE operation %d in function %s. %s", op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::InferShape(Function &function)
{
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    std::vector<Operation*> opList = function.Operations(false).DuplicatedOpList();
    if (opList.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "There is no operation in function %s. Please check the operation list of the input graph", function.GetRawName().c_str());
        return FAILED;
    }
    for (auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    for (auto op : opList) {
        for (auto producer : op->ProducerOps()) {
            opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
            opOutGraph[opMagic2Idx[producer->GetOpMagic()]].push_back(opMagic2Idx[op->GetOpMagic()]);
        }
    }
    bool isParamIndex = true;
    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph, isParamIndex);
    return SUCCESS;
}

Status InferParamIndex::UpdateValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified) {
    for (auto &op : subFunc.Operations(false)) {
        int tensorBaseAddrCoaIndex = IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
        if (tensorBaseAddrCoaIndex == -1) {
            continue;
        }
        if (addr2ValidShape.find(tensorBaseAddrCoaIndex) == addr2ValidShape.end()) {
            addr2ValidShape[tensorBaseAddrCoaIndex] = op.GetOOperands()[0]->GetDynValidShape();
            if (IsCopyIn(op.GetOpcode())) {
                auto attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                if (attr->GetToDynValidShape().size() != 0 && attr->GetToDynValidShape()[0].IsSpecified()) {
                    addr2ValidShapeSpecified[tensorBaseAddrCoaIndex] = OpImmediate::ToSpecified(attr->GetToDynValidShape());
                }
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::SetSubValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified) {
    std::set<std::string> visitedSymbol;
    int tensorIndex{0};
    for (auto validShape : addr2ValidShape) {
        int dimIdx{0};
        for (auto dim : validShape.second) {
            if (!dim.IsSymbol()) {
                continue;
            }
            if (visitedSymbol.count(dim.Dump()) > 0) {
                continue;
            }
            auto tensorBaseAddrCoaIndex = validShape.first;
            SymbolicScalar dynDim;
            if (addr2ValidShapeSpecified.count(tensorBaseAddrCoaIndex)) {
                dynDim = addr2ValidShapeSpecified[tensorBaseAddrCoaIndex][dimIdx];
            }
            auto paramInfo = DynParamInfo{
                static_cast<int>(validShape.second.size()), tensorIndex, tensorBaseAddrCoaIndex, DynParamInfoType::VALID_SHAPE,
                dimIdx, dynDim, false, ""};
            subFunc.InsertDynParam(dim.Dump(), paramInfo);
            dimIdx++;
        }
        tensorIndex++;
    }
    return SUCCESS;
}

Status InferParamIndex::UpdateParamIndex(Function &function) {
    for (auto &subProgram : function.rootFunc_->programs_) {
        auto &subFunc = *subProgram.second;
        if (ResetDynValidShape(subFunc) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ResetDynValidShape failed; Please check the ResetDynValidShape method.");
            return FAILED;
        }
        if (InferShape(subFunc) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Function, "Print function before update: %s\n", subFunc.Dump().c_str());
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShape;
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShapeSpecified;
        if (UpdateValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Update valid shape for the function %s failed. Please check above for more information.", function.GetRawName().c_str());
            return FAILED;
        }
        if (SetSubValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Update valid shape for the function %s failed. Please check above for more information.", function.GetRawName().c_str());
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Function, "Print function after update: %s\n", DumpParamIndex(subFunc.GetDynParamTable()).c_str());
    }
    return SUCCESS;
}

Status InferParamIndex::RunOnFunction(Function &function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferParamIndex.");
    if (UpdateParamIndex(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "UpdateParamIndex failed; Please check the UpdateParamIndex method.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InferParamIndex By Sequential Execution.");
    return SUCCESS;
}
}  // namespace tile_fwk
}  // namespace npu
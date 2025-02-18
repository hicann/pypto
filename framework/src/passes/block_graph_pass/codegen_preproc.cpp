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
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "passes/pass_interface/pass.h"
#include "codegen_preproc.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "CodegenPreproc"

namespace npu {
namespace tile_fwk {
// only save general gm input/output, not contain spill-out scene
bool CodegenPreproc::IsNeedSave(const Operation &op) const {
    return OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) && (!op.IsNeedStackGM());
}

// only used in DYNAMIC_LOOP_PATH scene
Status CodegenPreproc::SaveGmTensorParamIdxToOp(Function &func) const {
    if (!func.IsUnderDynamicFunction()) {
        return SUCCESS;
    }

    std::map<int, std::vector<Operation *>> gmParamInCallFunc;
    for (auto &subProgram : func.rootFunc_->programs_) {
        gmParamInCallFunc.clear();
        for (auto &op : subProgram.second->Operations(false)) {
            if (IsNeedSave(op)) {
                int coaIndex = IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
                gmParamInCallFunc[coaIndex].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(2)].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_GATHER_IN_UB) {
                gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
                gmParamInCallFunc[op.GetIOpAttrOffset(2)].emplace_back(&op);
            }
            if (op.GetOpcode() == Opcode::OP_LOAD) {
                int addrPos = op.GetIOpAttrOffset(0);
                gmParamInCallFunc[addrPos].emplace_back(&op);
            }
        }
        APASS_LOG_INFO_F(Elements::Operation, "%d:%sgmParamInCallFunc size: %zu", __LINE__, __FUNCTION__, gmParamInCallFunc.size());
        int tensorParamIdx{0};
        for (auto param : gmParamInCallFunc) {
            for (auto op : param.second) {
                op->SetAttribute("GmTensorParamIdxInCallFunc", tensorParamIdx);
                ++tensorParamIdx;
            }
        }
    }
    return SUCCESS;
}

void CodegenPreproc::CombineTailAxis(std::vector<int64_t> &shape, size_t shapeSize) const {
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = 1;
}

void CodegenPreproc::CombineLastAxis(std::vector<SymbolicScalar> &shape, size_t shapeSize) const {
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = SymbolicScalar(1);
}

Status CodegenPreproc::ProcessAxis(Operation &op, std::vector<bool> attr, bool isInput) const {
    LogicalTensors operands = isInput ? op.GetIOperands() : op.GetOOperands();
    if (attr.size() < operands.size()) {
        for (size_t i = 0; i < operands.size() - attr.size(); ++i) {
            attr.emplace_back(false);
        }
    }
    if (attr.size() != operands.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d %s attr size(%zu) is not equal to operands size(%zu), ProcessAxis failed.", op.GetOpMagic(), op.GetOpcodeStr().c_str(), attr.size(), operands.size());
        return FAILED;
    }
    for (size_t i = 0; i < operands.size(); ++i) {
        if (attr[i]) {
            size_t shapeSize = operands[i]->shape.size();
            CombineTailAxis(operands[i]->shape, shapeSize);
            CombineTailAxis(operands[i]->oriShape, shapeSize);
            CombineTailAxis(operands[i]->tensor->rawshape, shapeSize);
            if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false)) {
                CombineLastAxis(operands[i]->dynValidShape_, shapeSize);
            } 
        }
    }
    return SUCCESS;
}

Status CodegenPreproc::ForceCombineAxis(Function &func) const {
    for (auto &subProgram : func.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations(false)) {
            if (op.HasAttr(OP_ATTR_PREFIX + "input_combine_axis")) {
                std::vector<bool> attrIn;
                op.GetAttr(OP_ATTR_PREFIX + "input_combine_axis", attrIn);
                op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
                if (ProcessAxis(op, attrIn, true) != SUCCESS) { 
                    APASS_LOG_ERROR_F(Elements::Operation, "ForceCombineAxis failed at function ProcessAxis(input) for subProgram(%lu).", subProgram.first);
                    return FAILED; 
                }
                if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
                    op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
                    auto output = op.GetOOperands()[0];
                    CombineTailAxis(output->tensor->rawshape, output->tensor->rawshape.size());
                }
            }
            if (op.HasAttr(OP_ATTR_PREFIX + "output_combine_axis")) {
                std::vector<bool> attrOut;
                op.GetAttr(OP_ATTR_PREFIX + "output_combine_axis", attrOut);
                op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
                if (ProcessAxis(op, attrOut, false) !=SUCCESS) { 
                    APASS_LOG_ERROR_F(Elements::Operation, "ForceCombineAxis failed at function ProcessAxis(out) for subProgram(%lu).", subProgram.first);
                    return FAILED; 
                }
                if (op.GetOpcode() == Opcode::OP_COPY_IN) {
                    op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
                    auto input = op.GetIOperands()[0];
                    CombineTailAxis(input->tensor->rawshape, input->tensor->rawshape.size());
                }
            }
        }
    }
    return SUCCESS;
}

inline bool IsUBCopy(Operation& op) {
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

Status CodegenPreproc::ForceCombineAxisForAxisCombine(Function &func) const {
    const std::set<Opcode> skipInputCombineOps = {Opcode::OP_BRCB, Opcode::OP_EXPAND};
    for (auto &subProgram : func.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations(false)) {
            if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) != OpCoreType::AIV && !IsUBCopy(op)) {
                continue;
            }
            std::vector<bool> inputCombineAxis;
            for (size_t i = 0; i < op.GetIOperands().size(); ++i) {
                LogicalTensors operands = op.GetIOperands();
                if (operands[i]->tensor->rawshape.back() == 1 && skipInputCombineOps.count(op.GetOpcode()) == 0) {
                    inputCombineAxis.push_back(true);
                } else {
                    inputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::inputCombineAxis, inputCombineAxis);
            std::vector<bool> outputCombineAxis;
            for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
                LogicalTensors operands = op.GetOOperands();
                if (operands[i]->tensor->rawshape.back() == 1 && OpcodeManager::Inst().GetOpCalcType(op.GetOpcode()) != OpCalcType::REDUCE) {
                    outputCombineAxis.push_back(true);
                } else {
                    outputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::outputCombineAxis, outputCombineAxis);
        }
    }
    return SUCCESS;
}

std::string CodegenPreproc::DumpOpList(Function &function) {
    std::stringstream ss;
    int idx = 0;
    for (auto &subProgram : function.rootFunc_->programs_) {
        ss << "==================== OP_LIST Codegen_Preproc " << idx << " =====================" << "\n";
        for (auto &op : subProgram.second->Operations(false)) {
            if (!op.oOperand.empty()) {
                bool needAlloc = false;
                op.oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "], needAlloc: " << static_cast<int>(needAlloc)
                    << ", memId: " << op.oOperand[0]->memoryrange.memId << "\n";
            } else {
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "]" << "\n";
            }
        }
        idx++;
    }
    return ss.str();
}

void CodegenPreproc::SetNeedAllocAttr(Function &function) {
    for (auto &subProgram : function.rootFunc_->programs_) {
        std::unordered_set<int> appearedMemId;
        for (auto &op : subProgram.second->Operations(false)) {
            for (auto &outTensor : op.GetOOperands()) {
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

Status CodegenPreproc::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Start CodegenPreproc.");
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW_TYPE) {
            op.SetOpCode(Opcode::OP_VIEW);
        }
    }
    if (SaveGmTensorParamIdxToOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreproc RunOnFunction failed at function SaveGmTensorParamIdxToOp.");
        return FAILED;
    }
    
    if (ConfigManager::Instance().GetOperationConfig("COMBINE_AXIS", false)) {
        if (ForceCombineAxisForAxisCombine(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreproc RunOnFunction failed at function ForceCombineAxisForAxisCombine.");
            return FAILED;
        }
    } else {
        if (ForceCombineAxis(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreproc RunOnFunction failed at function ForceCombineAxis.");
            return FAILED;
        }
    }

    SetNeedAllocAttr(function);
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Finish CodegenPreproc.");
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu

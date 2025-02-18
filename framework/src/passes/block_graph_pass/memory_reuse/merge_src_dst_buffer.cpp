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
 * \file merge_src_dst_buffer.cpp
 * \brief
 */

#include "merge_src_dst_buffer.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SrcDstBufferMerge"

namespace npu::tile_fwk {

void SrcDstBufferMergeImpl::InitializeTensorMemorymap(Operation &op) const {
    for (auto &input : op.GetIOperands()) {
        TileRange range;
        range.memId = input->tensor->GetRawMagic();
        input->memoryrange = range;
    }
    for (auto &output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memoryrange = range;
    }
}

void SrcDstBufferMergeImpl::InitTensorMaxSize(const LogicalTensorPtr &output) {
    for (auto &consumer : output->GetConsumers()) {
        tensorConsumers_[output->memoryrange.memId].insert(consumer->GetOpMagic());
        if (tensorMaxSize_.find(output->memoryrange.memId) == tensorMaxSize_.end()) {
            tensorMaxSize_[output->memoryrange.memId] = output->GetDataSize();
            continue;
        }
        tensorMaxSize_[output->memoryrange.memId] =
            std::max(tensorMaxSize_[output->memoryrange.memId], output->GetDataSize());
    }
}

Status SrcDstBufferMergeImpl::CheckOpValid(const Operation *op, int opId) {
    if (op == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op:%d is null.%s", opId, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

void SrcDstBufferMergeImpl::InitOpOutput(const Operation &op) {
    int outId = 0;
    for (auto &output : op.GetOOperands()) {
        if (output == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Op:%s, magic:%d, output:%d is null.",
                op.GetOpcodeStr().c_str(), op.GetOpMagic(), outId);
            ++outId;
            continue;
        }
        if (output->memoryrange.memId == -1) {
            output->memoryrange.memId = output->GetMagic();
        }
        InitTensorMaxSize(output);
        ++outId;
    }
}

Status SrcDstBufferMergeImpl::Init(const std::vector<Operation *> &opList) {
    if (opList.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "OpList empty.");
        return FAILED;
    }
    if (opList.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "First op is null.");
        return FAILED;
    }
    
    int opId = 0;
    for (auto &op : opList) {
        if (CheckOpValid(op, opId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CheckOpValid failed.");
            return FAILED;
        }
        InitializeTensorMemorymap(*op);
        InitOpOutput(*op);
        ++opId;
    }

    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckIgnoreScene(const Operation &oriOps) {
    /* use opcode is unfavorable for reading and modification, maybe use opcalctype */
    const std::set<Opcode> ignoreOps = {Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT};
    if (ignoreOps.count(oriOps.GetOpcode()) != 0) {
        return true;
    }
    
    if (OpcodeManager::Inst().HasStaticAttribute(oriOps.GetOpcode(), OpAttributeKey::excludeBufferReuse)) {
        return true;
    }

    if (OpcodeManager::Inst().GetCoreType(oriOps.GetOpcode()) == OpCoreType::AIC) {
        return true;
    }
    for (auto &output : oriOps.GetOOperands()) {
        if (output == nullptr) {
            return true;
        }
    }
    return false;
}

std::pair<bool, Status> SrcDstBufferMergeImpl::CheckHasInplaced(const Operation &oriOps, const Operation &ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors) {
    if (oriOps.HasAttr(OpAttributeKey::inplaceInfo)) {
        std::map<int, int> inplaceInfo;
        if (!oriOps.GetAttr(OpAttributeKey::inplaceInfo, inplaceInfo)) {
            APASS_LOG_ERROR_F(Elements::Tensor, "OriOps:%s[%d] get inplaceInfo error.%s", oriOps.GetOpcodeStr().c_str(), oriOps.GetOpMagic(), GetFormatBacktrace(oriOps).c_str());
            return std::make_pair(false, FAILED);
        }
        for (auto &[iIdx, oIdx] : inplaceInfo) {
            auto in = ops.GetIOperands()[iIdx];
            auto out = ops.GetOOperands()[oIdx];
            out->memoryrange.memId = in->memoryrange.memId;
            tensorConsumers_[in->memoryrange.memId].insert(
                tensorConsumers_[out->memoryrange.memId].begin(),
                tensorConsumers_[out->memoryrange.memId].end());
            replacedTensors[out->memoryrange.memId] = in;
        }
        return std::make_pair(true, SUCCESS);
    }
    return std::make_pair(false, SUCCESS);
}

bool SrcDstBufferMergeImpl::FindReplaced(const Operation &oriOps, const Operation &ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors) {
    if (ops.GetOOperands().size() == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Operation %s[%d] has no outOperans", ops.GetOpcodeStr().c_str(), ops.GetOpMagic());
        return false;
    }
    auto out = ops.GetOOperands()[0];
    auto outTensorMagic = out->memoryrange.memId;
    for (auto in : oriOps.GetIOperands()) {
        if (in != nullptr && CanSrcDstReuse(oriOps, in, out)) {
            // 当前输出复用输入
            auto inTensorMagic = in->memoryrange.memId;
            if (inTensorMagic == outTensorMagic) {
                continue;
            }
            APASS_LOG_DEBUG_F(Elements::Tensor, "Set out tensor %d reuse src tensor %d",
                out->GetMagic(), in->GetMagic());
            out->memoryrange.memId = in->memoryrange.memId;
            if (tensorConsumers_[outTensorMagic].size() > tensorConsumers_[inTensorMagic].size()) {
                tensorConsumers_[inTensorMagic] = tensorConsumers_[outTensorMagic];
            }
            replacedTensors[outTensorMagic] = in;
            return true;
        }
    }

    return false;
}

void SrcDstBufferMergeImpl::NotFindReplacedProcess(const Operation &ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors) {
    for (auto &out : ops.GetOOperands()) {
        auto outTensorMagic = out->memoryrange.memId;
        APASS_LOG_DEBUG_F(Elements::Tensor, "Op %d out tensor magic: %d",
            ops.GetOpMagic(), outTensorMagic);
        if (replacedTensors.find(outTensorMagic) != replacedTensors.end()) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Find tensor: %d replaced by tensor: %d",
                outTensorMagic, replacedTensors[outTensorMagic]->memoryrange.memId);
            out->memoryrange.memId =
                replacedTensors[outTensorMagic]->memoryrange.memId;
        }
    }
}

Status SrcDstBufferMergeImpl::Run(Function &func) {
    if (func.rootFunc_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "RootFunc is null.");
        return FAILED;
    }
    for (auto &subProgram : func.rootFunc_->programs_) {
        APASS_LOG_INFO_F(Elements::Operation, "Merge src dst for program id : [%lu]",
            subProgram.first);
        auto opList = subProgram.second->Operations(false).DuplicatedOpList();
        if (Init(opList) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Init failed; Please check the Init method.");
            return FAILED;
        }
        auto oriOps(opList);
        std::unordered_map<int, std::shared_ptr<LogicalTensor>> replacedTensors;
        for (size_t i = 0; i < oriOps.size(); i++) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Try reuse op [%d] input by out tensor.",
                oriOps[i]->GetOpMagic());
            if (CheckIgnoreScene(*oriOps[i])) {
                continue;
            }
            auto hasInplaced = CheckHasInplaced(*oriOps[i], *opList[i], replacedTensors);
            if (hasInplaced.second == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "CheckHasInplaced failed; Please check the CheckHasInplaced method.");
                return FAILED;
            }
            if (hasInplaced.first) {
                continue;
            }
            bool findReplaced = FindReplaced(*oriOps[i], *opList[i], replacedTensors);
            if (!findReplaced) {
                NotFindReplacedProcess(*opList[i], replacedTensors);
            }
        }
    }
    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckAssembleReuse(const LogicalTensorPtr &outOperand) {
    for (auto consumer : outOperand->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (auto assembleOutTensor : consumer->GetOOperands()) {
            if (assembleOutTensor->memoryrange.memId == outOperand->memoryrange.memId) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Assemble cannot be reused.");
                return false;
            }
        }
    }
    return true;
}

bool SrcDstBufferMergeImpl::CanSrcDstReuse(const Operation &ops, std::shared_ptr<LogicalTensor> iOperand, std::shared_ptr<LogicalTensor> oOperand) {
    if (std::find(SCATTER_ELEMENT_OPS.begin(), SCATTER_ELEMENT_OPS.end(), ops.GetOpcode()) != SCATTER_ELEMENT_OPS.end()) {
        if (iOperand == ops.GetIOperands()[0]) {
            return true;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Try reuse src %d dst %d",
        iOperand->GetMagic(), oOperand->GetMagic());
    if (oOperand->GetMemoryTypeOriginal() != iOperand->GetMemoryTypeOriginal()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Memtype is not same.");
        return false;
    }
    if (tensorMaxSize_[oOperand->memoryrange.memId] != tensorMaxSize_[iOperand->memoryrange.memId]) {
        return false;
    }
    if (oOperand->Datatype() != iOperand->Datatype()) {
        return false;
    }
    if (!CheckAssembleReuse(oOperand)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Check Assemble op which cannot be reused.");
        return false;
    }
    // 确保复用UB buffer后不会被覆写
    auto iter = tensorConsumers_.find(iOperand->memoryrange.memId);
    if (iter != tensorConsumers_.end() && iter->second.size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has more than 1 output.", ops.GetOpcodeStr().c_str(), ops.GetOpMagic());
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Reusable, iOperand magic: %d, memId: %d, oOperand magic: %d, memId: %d, op:%s[%d]", iOperand->GetMagic(), iOperand->memoryrange.memId, 
        oOperand->GetMagic(), oOperand->memoryrange.memId, ops.GetOpcodeStr().c_str(), ops.GetOpMagic());
    return true;
}
} // namespace npu::tile_fwk
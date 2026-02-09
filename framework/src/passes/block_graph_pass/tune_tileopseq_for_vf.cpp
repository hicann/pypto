/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tune_tileopseq_for_vf.cpp
 * \brief
 */

#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "TuneTileOpSeqForVF"

namespace npu {
namespace tile_fwk {
bool TuneTileOpSeqForVF::IsGroupMergeable(PipeSync &ps, size_t left, size_t k, int groupNum) {
    size_t tempIdx = left;
    for (auto &groupOp : mergedOps[groupNum]) {
        if (ps.HasDataDependency(*groupOp, *opList_[k], --tempIdx, k)) {
            return false;
        }
    }
    return true;
}

bool TuneTileOpSeqForVF::IsMergeable(std::unordered_set<Operation *> &moveFrontOp, size_t left, size_t right, PipeSync &ps, int groupNum) {
    for (size_t k = left + 1; k < right; k++) {
        if (opList_[k]->GetOpcode() == Opcode::OP_VIEW || opList_[k]->GetOpcode() == Opcode::OP_ASSEMBLE ||
            opList_[k]->GetOpcode() == Opcode::OP_NOP || opList_[k]->GetOpcode() == Opcode::OP_HUB) {
            continue;
        }
        // 如果该op和vecTileop0和vecTileop1都存在依赖关系，则不能融合
        if (ps.HasDataDependency(*opList_[left], *opList_[k], left, k) && ps.HasDataDependency(*opList_[k], *opList_[right], k, right)) {
            return false;
        }
        // vecTileop0 op(set) vecTileop1(wait) 这种情况下两个vecTileop中间的op需要前移
        if (ps.HasDataDependency(*opList_[k], *opList_[right], k, right)) {
            // left后，k前的op需要判断与k是否有依赖关系
            for (size_t i = left + 1; i < k; i++) {
                if (ps.HasDataDependency(*opList_[i], *opList_[k], i, k) &&
                    std::find(moveFrontOp.begin(), moveFrontOp.end(), opList_[i]) == moveFrontOp.end()) {
                    return false;
                }
            }
            // 需要进一步判断和vecTileop0 group中的op是否有依赖关系
            if (groupNum == -1) {
                moveFrontOp.insert(opList_[k]);
            } else {
                if (!IsGroupMergeable(ps, left, k, groupNum)) {
                    return false;
                }
                moveFrontOp.insert(opList_[k]);
            }
        }
    }
    return true;
}

void TuneTileOpSeqForVF::MoveOpsForMerge(const std::unordered_set<Operation *> &moveFrontOp, size_t left, size_t right, int groupNum) {
    std::vector<Operation *> moveLeft;
    std::vector<Operation *> moveRight;
    for (size_t k = left + 1; k < right; k++) {
        if (moveFrontOp.count(opList_[k])) {
            moveLeft.emplace_back(opList_[k]);
        } else {
            moveRight.emplace_back(opList_[k]);
        }
    }
    // 删除left和right中间的op
    std::vector<size_t> toMoveIdx;
    for (size_t k = left + 1; k < right; k++) {
        toMoveIdx.emplace_back(k);
    }
    for (auto it = toMoveIdx.rbegin(); it != toMoveIdx.rend(); it++) {
        opList_.erase(opList_.begin() + *it);
    }
    // 在vecTileop1的右侧将moveRight的op插入
    auto insertPosR = opList_.begin() + left + 2;
    opList_.insert(insertPosR, moveRight.begin(), moveRight.end());
    // 在vecTileop0 group的左侧将moveLeft的op插入
    auto insertPosL = opList_.begin() + left - mergedOps[groupNum].size() + 2;
    opList_.insert(insertPosL, moveLeft.begin(), moveLeft.end());
}

void TuneTileOpSeqForVF::FindPipeVIdx(std::vector<size_t> &pipeVIdx, AIVCore coreType) {
    PipeSync ps;
    for (size_t i = 0; i < opList_.size(); i++) {
        if (opList_[i]->GetOpcode() == Opcode::OP_VEC_DUP) {
            continue;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList_[i]->GetOpcode());
        ps.AdjustOpCfg(opcfg, *opList_[i]);
        if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList_[i]->GetAIVCore() == coreType) {
            pipeVIdx.emplace_back(i);
        }
    }
}

void TuneTileOpSeqForVF::ChangeOpSeq(PipeSync &ps, bool isAIV1) {
    AIVCore coreType;
    if (!isAIV1) {
        coreType = AIVCore::AIV0;
    } else {
        coreType = AIVCore::AIV1;
    }
    mergedOps.clear();

    std::vector<size_t> pipeVIdx;
    FindPipeVIdx(pipeVIdx, coreType);
    if (pipeVIdx.size() <= 1) {
        return;
    }

    for (size_t idx = 0; idx + 1 < pipeVIdx.size(); idx++) {
        size_t left = pipeVIdx[idx];
        size_t right = pipeVIdx[idx + 1];
        APASS_LOG_DEBUG_F(Elements::Operation, "Try to merge %d %s and %d %s", opList_[left]->GetOpMagic(), opList_[left]->GetOpcodeStr().c_str(),
            opList_[right]->GetOpMagic(), opList_[right]->GetOpcodeStr().c_str());
        
        // 先看vecTileop0是否已经在mergedOps中
        int groupNum = -1;
        for (size_t i = 0; i < mergedOps.size(); i++) {
            for (size_t j = 0; j < mergedOps[i].size(); j++) {
                if (mergedOps[i][j] == opList_[left]) {
                    groupNum = i;
                    break;
                }
            }
        }

        std::unordered_set<Operation *> moveFrontOp;
        if (!IsMergeable(moveFrontOp, left, right, ps, groupNum)) {
            continue;
        }
        // 可以融合
        // 将vecTileop0和vecTileop1添加到mergedOps中
        APASS_LOG_DEBUG_F(Elements::Operation, "Need merge.");
        if (groupNum == -1) {
            // 将vecTileop0和vecTileop1添加到mergedOps中
            std::vector<Operation *> newOp = {opList_[left], opList_[right]};
            mergedOps.emplace_back(newOp);
            groupNum = mergedOps.size() - 1;
        } else {
            mergedOps[groupNum].emplace_back(opList_[right]);
        }

        MoveOpsForMerge(moveFrontOp, left, right, groupNum);

        // 由于移动，pipeVop的idx会发生变化，需要重新更新pipeVIdx
        pipeVIdx.clear();
        FindPipeVIdx(pipeVIdx, coreType);
    }
}

Status TuneTileOpSeqForVF::RunOnFunction(Function &function) {
    if (!config::GetPassGlobalConfig(KEY_ENABLE_VF, false)) {
        APASS_LOG_DEBUG_F(Elements::Function, "TuneTileOpSeqForVF is skipped for ENABLE_VF is false.");
        return SUCCESS;
    }
    size_t funcId = 0;
    for (auto &program : function.rootFunc_->programs_) {
        std::vector<Operation *> opList(program.second->Operations(false).DuplicatedOpList());
        opList_ = opList;
        PipeSync ps;
        APASS_LOG_DEBUG_F(Elements::Function, "=======================function %d ======================", funcId);
        for (const auto &op : opList_) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Input Operation %d %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
            ps.BuildTensorRangeMap(op);
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
            if (opcfg.pipeIdStart_ != PipeType::PIPE_V) {
                continue;
            }
            // 假定：pipe_V的op的AIV类型只能是AIV0或AIV1
            if (op->GetAIVCore() != AIVCore::AIV0 && op->GetAIVCore() != AIVCore::AIV1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Pipe_V op %d %s AIV type is neither AIV0 nor AIV1, RunOnFunction failed.", op->GetOpMagic(), op->GetOpcodeStr().c_str());
                return FAILED;
            }
        }
        // AIV0和AIV1各调整一次
        ChangeOpSeq(ps, false);
        ChangeOpSeq(ps, true);
        // 将调整后的oplist刷新到function中去
        program.second->ScheduleBy(opList_, true);
        APASS_LOG_DEBUG_F(Elements::Function, "---------------------------------------------------");
        for (const auto &op : opList_) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Output Operation %d %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
        }
        funcId++;

        // TODO 增加拓扑逻辑校验
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
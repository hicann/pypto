/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main_block.cpp
 * \brief
 */

#include "main_block.h"
#include "codegen/codegen.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
MainBlockCondBulider::MainBlockCondBulider() = default;

void MainBlockCondBulider::AddUniqueCondition(const SymbolicScalar &newCond)
{
    SymbolicScalar cond = SymbolicScalar(false);
    std::string condStr = newCond.Dump();
    if ((mainBlockStrSet_.find(condStr) != mainBlockStrSet_.end()) ||
        (mainBlockStrSet_.find(cond.Dump()) != mainBlockStrSet_.end())) {
        return;
    }

    mainBlockStrSet_.insert(condStr);
    mainBlockCondGroup_.push_back(newCond);
}

bool MainBlockCondBulider::CheckShapeEquality(const Shape &shape, const std::vector<SymbolicScalar> &dynShape)
{
    SymbolicScalar cond = SymbolicScalar(false);
    if (shape.size() != dynShape.size()) {
        AddUniqueCondition(cond);
        return false;
    }

    for (uint32_t i = 0; i < shape.size(); i++) {
        if (shape[i] == -1) {  // -1: copy_in, copy_out and callop dynamic axis shape
            continue;
        }
        cond = (shape[i] == dynShape[i]);
        AddUniqueCondition(cond);
        if (cond.IsImmediate() && (cond == 0)) {
            return false;
        }
    }
    return true;
}

bool MainBlockCondBulider::GetValidShapeFromCoa(const std::vector<SymbolicScalar> &argList,
        Shape &shape, std::vector<SymbolicScalar> &dynValidShape)
{
    if (argList.empty()) {
        MACHINE_LOGW("argList is empty!");
        return false;
    }

    int dim = (argList.size() -1 + COA_INDEX_TYPE_COUNT - 1) / COA_INDEX_TYPE_COUNT;
    int validShapeDim = argList.size() -1 - dim * (COA_INDEX_TYPE_COUNT - 1);
    int coaIndex = COA_INDEX_DIM_BASE;

    shape.reserve(dim);
    dynValidShape.reserve(validShapeDim);

    // coa: [offset, shape, rawshape, validshape]
    for (int i = 0; i < dim; i++) {
        shape.push_back(argList[coaIndex + dim + i]);
    }

    coaIndex += dim * (COA_INDEX_TYPE_COUNT - 1);
    for (int i = 0; i < validShapeDim; i++) {
        dynValidShape.push_back(argList[coaIndex + i]);
    }

    return true;
}

void MainBlockCondBulider::CollectCallopMainBlockConds(Function *func)
{
    if (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) != 1) {
        AddUniqueCondition(SymbolicScalar(false));
        return;
    }

    auto checkOperand = [&](auto &op, auto &shape, auto &validshape, const char* tag) -> bool {
        auto cond = CheckShapeEquality(shape, validshape);
        if (!cond) {
            MACHINE_LOGW("get mainBlock flag false, op code %s, %s shape is %s, validShape is %s",
               op.GetOpcodeStr().c_str(),
               tag,
               IntVecToStr(shape).c_str(),
               IntVecToStr(validshape).c_str());
        }
        return cond;
    };

    for (auto &op : func->Operations()) {
        for (auto &iop : op.GetIOperands()) {
            if (!checkOperand(op, iop->shape, iop->GetDynValidShape(), "iop")) {
                return;
            }
        }
        for (auto &oop : op.GetOOperands()) {
            if (!checkOperand(op, oop->shape, oop->GetDynValidShape(), "oop")) {
                return;
            }
        }
    }
}

void MainBlockCondBulider::CollectCoaMainBlockConds(const std::vector<std::vector<SymbolicScalar>> &argList)
{
    if (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) != 1) {
        AddUniqueCondition(SymbolicScalar(false));
        return;
    }

    for (const auto& iter : argList) {
        Shape shape;
        std::vector<SymbolicScalar> dynValidShape;
        if (!GetValidShapeFromCoa(iter, shape, dynValidShape)) {
            AddUniqueCondition(SymbolicScalar(false));
            return;
        }
        auto cond = CheckShapeEquality(shape, dynValidShape);
        if (!cond) {
            MACHINE_LOGW("get mainBlock flag false, coa shape is %s, validShape is %s",
               IntVecToStr(shape).c_str(),
               IntVecToStr(dynValidShape).c_str());
            return;
        }
    }
}

SymbolicScalar MainBlockCondBulider::BuildMainBlockExpression()
{
    SymbolicScalar runtimeSelect("RUNTIME_Select");
    SymbolicScalar runtimeAnd("RUNTIME_And");
    SymbolicScalar cond = false;
    if (mainBlockCondGroup_.empty()) {
        return runtimeSelect(cond, 1, 0);
    }

    cond = true;
    for (const auto &iter : mainBlockCondGroup_) {
        cond = runtimeAnd(cond, iter);
    }
    
    cond = runtimeSelect(cond, 1, 0);
    return cond;
}

void MainBlockCondBulider::Gencode(Function *function, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset)
{
    if (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1) {
        npu::tile_fwk::CodeGenCtx codeGenCtxMainBlock("", GetEmitPath("kernel_aicore"), true);
        npu::tile_fwk::CodeGen codeGenMainBlock(codeGenCtxMainBlock);
        codeGenMainBlock.GenCode(*function, invokeParaOffset);
    }
}

const std::vector<SymbolicScalar>& MainBlockCondBulider::GetCondGroup() const {
    return mainBlockCondGroup_;
}

const std::unordered_set<std::string>& MainBlockCondBulider::GetCondStrSet() const {
    return mainBlockStrSet_;
}
}

    
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
 * \file calc_cube.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"
#include "calc.h"
#include "interface/operation/operation_impl.h"

namespace npu::tile_fwk {

void ExecuteOpAMulB(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO || ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);

    auto &cubeTile = ctx->op->GetTileShape().GetCubeTile();
    int k1 = cubeTile.k[1];
    int k2 = cubeTile.k[2];
    int kStep = std::gcd(k1, k2);
    bool transA = (ctx->op->HasAttr(Matrix::A_MUL_B_TRANS_A)) ? ctx->op->GetBoolAttribute(Matrix::A_MUL_B_TRANS_A) : false;
    bool transB = (ctx->op->HasAttr(Matrix::A_MUL_B_TRANS_B)) ? ctx->op->GetBoolAttribute(Matrix::A_MUL_B_TRANS_B) : false;
    MatMulParam param = {transA, transB, kStep};
    switch (ctx->op->GetOpcode()) {
        case Opcode::OP_A_MUL_B: calc::MatMul(ret, lhs, rhs, param); break;
        case Opcode::OP_A_MULACC_B: {
            ASSERT(ctx->ioperandDataViewList->size() == SIZE_THREE);
            auto acc = ctx->ioperandDataViewList->at(2);
            calc::AccMatMul(ret, lhs, rhs, acc, param);
        } break;
        default: ASSERT(false); break;
    }
}
REGISTER_CALC_OP(OP_A_MUL_B, Opcode::OP_A_MUL_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_AT_MUL_B, Opcode::OP_AT_MUL_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_AT_MUL_BT, Opcode::OP_AT_MUL_BT, ExecuteOpAMulB);

void ExecuteOpAlloc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= 1);
    ASSERT(ctx->ioperandDataViewList->size() == 0);
}
REGISTER_CALC_OP(OP_UB_ALLOC, Opcode::OP_UB_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L1_ALLOC, Opcode::OP_L1_ALLOC, ExecuteOpAlloc);

void ExecuteDuplicate(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &oper = ctx->ioperandDataViewList->at(0);
    Opcode opCode = ctx->op->GetOpcode();
    bool trans = opCode == Opcode::OP_L1_TO_L0_BT || opCode == Opcode::OP_L1_TO_L0_AT;
    calc::Copy(ret, oper, trans);
}
REGISTER_CALC_OP(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_CONVERT, Opcode::OP_CONVERT, ExecuteDuplicate);
}
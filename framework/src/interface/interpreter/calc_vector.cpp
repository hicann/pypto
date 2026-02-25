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
 * \file calc_vector.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

template <Opcode opcode>
void ExecuteOpBinary(ExecuteOperationContext *ctx) {
    if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
        opcode == Opcode::OP_DIV_BRC) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
    } else if (opcode == Opcode::OP_BITWISEXOR || opcode == Opcode::OP_COPYSIGN || opcode == Opcode::OP_POW) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);

    if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
        opcode == Opcode::OP_DIV_BRC) {
        bool axisCombine = ctx->op->GetBoolAttribute("input_combine_axis_done");
        lhs->SetAxisCombine(axisCombine);
        rhs->SetAxisCombine(axisCombine);
    }

    switch (opcode) {
        case Opcode::OP_ADD: calc::Add(ret, lhs, rhs); break;
        case Opcode::OP_ADD_BRC: calc::Add(ret, lhs, rhs); break;
        case Opcode::OP_PAIRSUM: calc::PairSum(ret, lhs, rhs); break;
        case Opcode::OP_SUB: calc::Sub(ret, lhs, rhs); break;
        case Opcode::OP_SUB_BRC: calc::Sub(ret, lhs, rhs); break;
        case Opcode::OP_MUL: calc::Mul(ret, lhs, rhs); break;
        case Opcode::OP_MUL_BRC: calc::Mul(ret, lhs, rhs); break;
        case Opcode::OP_DIV: calc::Div(ret, lhs, rhs); break;
        case Opcode::OP_DIV_BRC: calc::Div(ret, lhs, rhs); break;
        case Opcode::OP_POW: calc::Pow(ret, lhs, rhs); break;
        case Opcode::OP_S_MAX: calc::Max(ret, lhs, rhs); break;
        case Opcode::OP_PAIRMAX: calc::PairMax(ret, lhs, rhs); break;
        case Opcode::OP_PAIRMIN: calc::PairMin(ret, lhs, rhs); break;
        case Opcode::OP_S_MIN: calc::Min(ret, lhs, rhs); break;
        case Opcode::OP_BITWISEAND: calc::BitwiseAnd(ret, lhs, rhs); break;
        case Opcode::OP_BITWISEOR: calc::BitwiseOr(ret, lhs, rhs); break;
        case Opcode::OP_BITWISEXOR: calc::BitwiseXor(ret, lhs, rhs); break;
        case Opcode::OP_COPYSIGN: calc::CopySign(ret, lhs, rhs); break;
        default: ASSERT(false);
    }
}
REGISTER_CALC_OP(OP_ADD, Opcode::OP_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CALC_OP(OP_ADD_BRC, Opcode::OP_ADD_BRC, ExecuteOpBinary<Opcode::OP_ADD_BRC>);
REGISTER_CALC_OP(OP_SUB, Opcode::OP_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CALC_OP(OP_SUB_BRC, Opcode::OP_SUB_BRC, ExecuteOpBinary<Opcode::OP_SUB_BRC>);
REGISTER_CALC_OP(OP_MUL, Opcode::OP_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CALC_OP(OP_MUL_BRC, Opcode::OP_MUL_BRC, ExecuteOpBinary<Opcode::OP_MUL_BRC>);
REGISTER_CALC_OP(OP_DIV, Opcode::OP_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CALC_OP(OP_DIV_BRC, Opcode::OP_DIV_BRC, ExecuteOpBinary<Opcode::OP_DIV_BRC>);
REGISTER_CALC_OP(OP_POW, Opcode::OP_POW, ExecuteOpBinary<Opcode::OP_POW>);
REGISTER_CALC_OP(OP_S_ADD, Opcode::OP_S_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CALC_OP(OP_S_SUB, Opcode::OP_S_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CALC_OP(OP_S_MUL, Opcode::OP_S_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CALC_OP(OP_S_DIV, Opcode::OP_S_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CALC_OP(OP_PAIRMAX, Opcode::OP_PAIRMAX, ExecuteOpBinary<Opcode::OP_PAIRMAX>);
REGISTER_CALC_OP(OP_PAIRMIN, Opcode::OP_PAIRMIN, ExecuteOpBinary<Opcode::OP_PAIRMIN>);
REGISTER_CALC_OP(OP_PAIRSUM, Opcode::OP_PAIRSUM, ExecuteOpBinary<Opcode::OP_PAIRSUM>);
REGISTER_CALC_OP(OP_S_MAX, Opcode::OP_S_MAX, ExecuteOpBinary<Opcode::OP_S_MAX>);
REGISTER_CALC_OP(OP_S_MIN, Opcode::OP_S_MIN, ExecuteOpBinary<Opcode::OP_S_MIN>);
REGISTER_CALC_OP(OP_MAXIMUM, Opcode::OP_MAXIMUM, ExecuteOpBinary<Opcode::OP_S_MAX>);
REGISTER_CALC_OP(OP_MINIMUM, Opcode::OP_MINIMUM, ExecuteOpBinary<Opcode::OP_S_MIN>);
REGISTER_CALC_OP(OP_BITWISEAND, Opcode::OP_BITWISEAND, ExecuteOpBinary<Opcode::OP_BITWISEAND>);
REGISTER_CALC_OP(OP_BITWISEOR, Opcode::OP_BITWISEOR, ExecuteOpBinary<Opcode::OP_BITWISEOR>);
REGISTER_CALC_OP(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, ExecuteOpBinary<Opcode::OP_BITWISEXOR>);
REGISTER_CALC_OP(OP_COPYSIGN, Opcode::OP_COPYSIGN, ExecuteOpBinary<Opcode::OP_COPYSIGN>);

void ExecuteOpFmod(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    calc::Fmod(ret, lhs, rhs);
}
REGISTER_CALC_OP(OP_MOD, Opcode::OP_MOD, ExecuteOpFmod);

void ExecuteOpFmods(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::FmodS(ret, lhs, element);
}
REGISTER_CALC_OP(OP_MODS, Opcode::OP_MODS, ExecuteOpFmods);

void ExecuteOpVecDup(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 0);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::ExpandS(ret, element);
}
REGISTER_CALC_OP(OP_VEC_DUP, Opcode::OP_VEC_DUP, ExecuteOpVecDup);

void ExecuteOpWhereTT(ExecuteOperationContext *ctx) {
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto input = ctx->ioperandDataViewList->at(1);
    auto other = ctx->ioperandDataViewList->at(2);
    calc::WhereTT(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_TT, Opcode::OP_WHERE_TT, ExecuteOpWhereTT);

void ExecuteOpWhereTS(ExecuteOperationContext *ctx) {
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto input = ctx->ioperandDataViewList->at(1);
    auto other = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    // OpAttributeKey::dynScalar
    calc::WhereTS(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_TS, Opcode::OP_WHERE_TS, ExecuteOpWhereTS);

void ExecuteOpWhereST(ExecuteOperationContext *ctx) {
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto other = ctx->ioperandDataViewList->at(1);
    auto input = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    // OpAttributeKey::dynScalar
    calc::WhereST(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_ST, Opcode::OP_WHERE_ST, ExecuteOpWhereST);

void ExecuteOpWhereSS(ExecuteOperationContext *ctx) {
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto input = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    auto other = ctx->op->GetElementAttribute(OpAttributeKey::dynScalar);
    calc::WhereSS(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_SS, Opcode::OP_WHERE_SS, ExecuteOpWhereSS);

template <Opcode opcode>
void ExecuteOpReduce(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    if (oop->GetShape()[axis] != 1) {
        std::vector<int64_t> oopShape = oop->GetShape();
        oopShape[axis] = 1;
        oop = oop->View(oopShape, std::vector<int64_t>(oopShape.size(), 0));
    }

    switch (opcode) {
        case Opcode::OP_ROWSUM_SINGLE: calc::RowSumSingle(oop, iop, axis); break;
        case Opcode::OP_ROWMAX_SINGLE: calc::RowMaxSingle(oop, iop, axis); break;
        case Opcode::OP_ROWMIN_SINGLE: calc::RowMinSingle(oop, iop, axis); break;
        case Opcode::OP_ROWSUMLINE: calc::RowSumExpand(oop, iop, axis); break;
        case Opcode::OP_ROWMAXLINE: calc::RowMaxLine(oop, iop, axis); break;
        case Opcode::OP_ROWMINLINE: calc::RowMinLine(oop, iop, axis); break;
        default: ASSERT(false) << "opcode not support" << ctx->op->GetOpcodeStr();
    }
}
REGISTER_CALC_OP(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ExecuteOpReduce<Opcode::OP_ROWSUM_SINGLE>);
REGISTER_CALC_OP(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ExecuteOpReduce<Opcode::OP_ROWSUMLINE>);
REGISTER_CALC_OP(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMAX_SINGLE>);
REGISTER_CALC_OP(OP_ROWMAXLINE, Opcode::OP_ROWMAXLINE, ExecuteOpReduce<Opcode::OP_ROWMAXLINE>);
REGISTER_CALC_OP(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMIN_SINGLE>);
REGISTER_CALC_OP(OP_ROWMINLINE, Opcode::OP_ROWMINLINE, ExecuteOpReduce<Opcode::OP_ROWMINLINE>);

void ExecuteOpCast(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    CastMode mode = static_cast<CastMode>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "mode"));
    calc::Cast(ret, iop, mode);
}
REGISTER_CALC_OP(OP_CAST, Opcode::OP_CAST, ExecuteOpCast);

template <Opcode opcode>
void ExecuteOpUnary(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    switch (opcode) {
        case Opcode::OP_EXP: calc::Exp(ret, iop); break;
        case Opcode::OP_NEG: calc::Neg(ret, iop); break;
        case Opcode::OP_SIGN: calc::Sign(ret, iop); break;
        case Opcode::OP_RSQRT: calc::Rsqrt(ret, iop); break;
        case Opcode::OP_SQRT: calc::Sqrt(ret, iop); break;
        case Opcode::OP_RECIPROCAL: calc::Reciprocal(ret, iop); break;
        case Opcode::OP_RELU: calc::Relu(ret, iop); break;
        case Opcode::OP_BITWISENOT: calc::BitwiseNot(ret, iop); break;
        case Opcode::OP_ABS: calc::Abs(ret, iop); break;
        case Opcode::OP_BRCB: calc::Brcb(ret, iop); break;
        case Opcode::OP_LN: calc::Ln(ret, iop); break;
        case Opcode::OP_ISFINITE: calc::IsFinite(ret, iop); break;
        default: ASSERT(false);
    }
}
REGISTER_CALC_OP(OP_EXP, Opcode::OP_EXP, ExecuteOpUnary<Opcode::OP_EXP>);
REGISTER_CALC_OP(OP_NEG, Opcode::OP_NEG, ExecuteOpUnary<Opcode::OP_NEG>);
REGISTER_CALC_OP(OP_SIGN, Opcode::OP_SIGN, ExecuteOpUnary<Opcode::OP_SIGN>);
REGISTER_CALC_OP(OP_RSQRT, Opcode::OP_RSQRT, ExecuteOpUnary<Opcode::OP_RSQRT>);
REGISTER_CALC_OP(OP_SQRT, Opcode::OP_SQRT, ExecuteOpUnary<Opcode::OP_SQRT>);
REGISTER_CALC_OP(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ExecuteOpUnary<Opcode::OP_RECIPROCAL>);
REGISTER_CALC_OP(OP_RELU, Opcode::OP_RELU, ExecuteOpUnary<Opcode::OP_RELU>);
REGISTER_CALC_OP(OP_BITWISENOT, Opcode::OP_BITWISENOT, ExecuteOpUnary<Opcode::OP_BITWISENOT>);
REGISTER_CALC_OP(OP_ABS, Opcode::OP_ABS, ExecuteOpUnary<Opcode::OP_ABS>);
REGISTER_CALC_OP(OP_BRCB, Opcode::OP_BRCB, ExecuteOpUnary<Opcode::OP_BRCB>);
REGISTER_CALC_OP(OP_LN, Opcode::OP_LN, ExecuteOpUnary<Opcode::OP_LN>);
REGISTER_CALC_OP(OP_ISFINITE, Opcode::OP_ISFINITE, ExecuteOpUnary<Opcode::OP_ISFINITE>);

void ExecuteOpCeil(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    calc::Ceil(ret, iop);
}
REGISTER_CALC_OP(OP_CEIL, Opcode::OP_CEIL, ExecuteOpCeil);

void ExecuteOpFloor(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    calc::Floor(ret, iop);
}
REGISTER_CALC_OP(OP_FLOOR, Opcode::OP_FLOOR, ExecuteOpFloor);

void ExecuteOpTrunc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    calc::Trunc(ret, iop);
}
REGISTER_CALC_OP(OP_TRUNC, Opcode::OP_TRUNC, ExecuteOpTrunc);

void ExecuteOpRound(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    auto &input = ctx->ioperandDataViewList->at(0);

    int decimals = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "decimals");
    calc::Round(output, input, decimals);
}
REGISTER_CALC_OP(OP_ROUND, Opcode::OP_ROUND, ExecuteOpRound);

void ExecuteOpOneHot(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    int numClasses = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    calc::OneHot(ret, iop, numClasses);
}
REGISTER_CALC_OP(OP_ONEHOT, Opcode::OP_ONEHOT, ExecuteOpOneHot);

void ExecuteOpExpand(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    calc::Expand(oop, iop);
}
REGISTER_CALC_OP(OP_EXPAND, Opcode::OP_EXPAND, ExecuteOpExpand);

void ExecuteOpTransposeMoveOut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);

    std::vector<int64_t> axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    if (std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute())) {
        auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
        std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetShape());
        if (ctx->op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) {
            std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
            std::vector<int64_t> iopShape = iop->GetShape();
            std::swap(iopShape[axises[0]], iopShape[axises[1]]);
            auto oopCopy = std::make_shared<LogicalTensorData>(oop->GetData(), iopShape, toOffset);
            return calc::Transpose(oopCopy, iop, axises[0], axises[1]);
        } else {
            std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetFromOffset());
            std::vector<int64_t> oopShape = oop->GetShape();
            std::swap(oopShape[axises[0]], oopShape[axises[1]]);
            auto iopCopy = std::make_shared<LogicalTensorData>(iop->GetData(), oopShape, fromOffset);
            return calc::Transpose(oop, iopCopy, axises[0], axises[1]);
        }
    }
    calc::Transpose(oop, iop, axises[0], axises[1]);
}
REGISTER_CALC_OP(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, ExecuteOpTransposeMoveOut);
REGISTER_CALC_OP(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, ExecuteOpTransposeMoveOut);

void ExecuteOpTranspose(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    auto axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    calc::Transpose(oop, iop, axises[0], axises[1]);
}
REGISTER_CALC_OP(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, ExecuteOpTranspose);

void ExecuteOpLogicalNot(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    calc::LogicalNot(oop, iop);
}
REGISTER_CALC_OP(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, ExecuteOpLogicalNot);

void ExecuteOpLogicalAnd(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    calc::LogicalAnd(ret, lhs, rhs);
}
REGISTER_CALC_OP(OP_LOGICALAND, Opcode::OP_LOGICALAND, ExecuteOpLogicalAnd);

void ExecuteOpIndexOutcast(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto index = ctx->ioperandDataViewList->at(1);
    auto dst = ctx->ioperandDataViewList->at(2);
    int axis = ctx->op->GetIntAttribute("axis");
    int blockSize = ctx->op->GetIntAttribute(OpAttributeKey::panzBlockSize);
    std::string cacheMode = ctx->op->GetStringAttribute(OpAttributeKey::cacheMode);

    calc::ScatterUpdate(oop, src, index, dst, axis, cacheMode, blockSize);
}
REGISTER_CALC_OP(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, ExecuteOpIndexOutcast);

void ExecuteOpScatterElement(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    auto src = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, src);
    int reduce = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");

    calc::ScatterElement(oop, self, indices, src, axis, reduce);
}
REGISTER_CALC_OP(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ExecuteOpScatterElement);

void ExecuteOpScatter(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    auto src = ctx->ioperandDataViewList->at(2);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int reduce = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");

    calc::Scatter(oop, self, indices, src, axis, reduce);
}
REGISTER_CALC_OP(OP_SCATTER, Opcode::OP_SCATTER, ExecuteOpScatter);

template <typename T, DataType dataType>
Element GetEndBySize(Element start, Element size, Element step) {
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T endValue = startValue + size.GetSignedData() * stepValue - stepValue / 2;
    Element end(dataType, endValue);
    return end;
}

void ExecuteOpRange(ExecuteOperationContext *ctx) {
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto start = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "START");
    auto size = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "SIZE");
    auto step = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "STEP");
    Element curStart = start;
    if (ctx->op->HasAttr(OpAttributeKey::dynScalar)) {
        SymbolicScalar tileIdx = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
        if (tileIdx.ConcreteValid()) {
            int64_t tileIdxVal = tileIdx.Concrete();
            Element tileIdxElem = Element(start.GetDataType(), tileIdxVal);
            curStart = start + step * tileIdxElem;
        }
    }
    Element end;
    if (start.GetDataType() == DT_INT32) {
        end = GetEndBySize<int32_t, DT_INT32>(curStart, size, step);
    } else if (start.GetDataType() == DT_INT64) {
        end = GetEndBySize<int64_t, DT_INT64>(curStart, size, step);
    } else if (start.GetDataType() == DT_FP32) {
        end = GetEndBySize<float, DT_FP32>(curStart, size, step);
    } else {
        std::string errorMessage = "Unsupported DataType " + DataType2String(start.GetDataType());
        throw std::invalid_argument(errorMessage.c_str());
    }
    calc::Range(oop, curStart, end, step);
}
REGISTER_CALC_OP(OP_RANGE, Opcode::OP_RANGE, ExecuteOpRange);

void ExecuteOpCompare(ExecuteOperationContext *ctx) {
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto iop_other = ctx->ioperandDataViewList->at(1);
    auto operation = static_cast<CmpOperationType>(
        ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation")
    );
    auto mode = static_cast<CmpModeType>(
        ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode")
    );
    calc::Compare(oop, iop_self, iop_other, operation, mode);
}
REGISTER_CALC_OP(OP_CMP, Opcode::OP_CMP, ExecuteOpCompare);

void ExecuteOpCmps(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    
    auto operation = static_cast<CmpOperationType>(
        ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation")
    );
    auto mode = static_cast<CmpModeType>(
        ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode")
    );
    calc::Cmps(oop, iop_self, element, operation, mode);
}
REGISTER_CALC_OP(OP_CMPS, Opcode::OP_CMPS, ExecuteOpCmps);

void ExecuteOpHypot(ExecuteOperationContext *ctx) {
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto iop_other = ctx->ioperandDataViewList->at(1);
    calc::Hypot(oop, iop_self, iop_other);
}
REGISTER_CALC_OP(OP_HYPOT, Opcode::OP_HYPOT, ExecuteOpHypot);

void ExecuteOpExtract(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto maskMode = ctx->op->GetIntAttribute("op_attr_makeMode");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::Extract(oop, src, maskMode, descending);
}
REGISTER_CALC_OP(OP_EXTRACT, Opcode::OP_EXTRACT, ExecuteOpExtract);
REGISTER_CALC_OP(OP_EXTRACT_SINGLE, Opcode::OP_EXTRACT_SINGLE, ExecuteOpExtract);

void ExecuteOpGather(ExecuteOperationContext *ctx) {
    auto output = ctx->ooperandInplaceDataViewList->at(0);
    auto parmas = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute("op_attr_axis");
    calc::Gather(output, parmas, indices, axis);
}
REGISTER_CALC_OP(OP_GATHER, Opcode::OP_GATHER, ExecuteOpGather);
void ExecuteOpGatherINUB(ExecuteOperationContext *ctx) {
    auto output = ctx->ooperandInplaceDataViewList->at(0);
    auto parmas = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    auto pageTable = ctx->ioperandDataViewList->at(2);
    int blocksize = ctx->op->GetIntAttribute(OpAttributeKey::blockSize);
    calc::GatherINUB(output, parmas, indices, pageTable, blocksize, -2);
}
REGISTER_CALC_OP(OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_UB, ExecuteOpGatherINUB);

void ExecuteOpIndexAdd(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &self = ctx->ioperandDataViewList->at(0);
    auto &src = ctx->ioperandDataViewList->at(1);
    auto &indices = ctx->ioperandDataViewList->at(2);
    auto alpha = Element(DT_FP32, 1.0);
    if (ctx->op->HasAttribute(OpAttributeKey::scalar)) {
        alpha = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    }
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    calc::IndexAdd(ret, self, src, indices, axis, alpha);
}
REGISTER_CALC_OP(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, ExecuteOpIndexAdd);

void ExecuteOpTri(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    auto &input = ctx->ioperandDataViewList->at(0);

    auto dia = ctx->op->GetElementAttribute(OpAttributeKey::dynScalar);
    int diagonal = static_cast<int32_t>(dia.GetSignedData());
    bool isUpper = ctx->op->GetBoolAttribute(OpAttributeKey::isUpper);
    isUpper ? calc::TriU(output, input, diagonal) : calc::TriL(output, input, diagonal);
}
REGISTER_CALC_OP(OP_TRIUL, Opcode::OP_TRIUL, ExecuteOpTri);

void ExecuteOpCumSum(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    auto &input = ctx->ioperandDataViewList->at(0);

    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    calc::CumSum(output, input, axis);
}
REGISTER_CALC_OP(OP_CUM_SUM, Opcode::OP_CUM_SUM, ExecuteOpCumSum);

void ExecuteOpIndexPut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() <= SIZE_SIX);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto values = ctx->ioperandDataViewList->at(1);
    std::vector<LogicalTensorDataPtr> indices;
    for (int i = SIZE_TWO; i < static_cast<int>(ctx->ioperandDataViewList->size()); i++) {
        auto indicesTemp = ctx->ioperandDataViewList->at(i);
        indices.push_back(indicesTemp);
    }
    bool accumulate = ctx->op->GetBoolAttribute(OpAttributeKey::accumulate);
    calc::IndexPut(out, self, indices, values, accumulate);
}
REGISTER_CALC_OP(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, ExecuteOpIndexPut);

void ExecuteOpMrgSort(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    auto kValue = ctx->op->GetIntAttribute("op_attr_kvalue");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::Topk(oop, src, topk_axis, kValue, descending);
}
REGISTER_CALC_OP(OP_MRGSORT, Opcode::OP_MRGSORT, ExecuteOpMrgSort);

void ExecuteOpTwoTileMrgSort(ExecuteOperationContext *ctx) {
    auto src = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    calc::TwoTileMrgSort(oop, src);
}
REGISTER_CALC_OP(OP_TWOTILEMRGSORT, Opcode::OP_TWOTILEMRGSORT, ExecuteOpTwoTileMrgSort);

void ExecuteOpSort(ExecuteOperationContext *ctx) {
    auto src = ctx->ioperandDataViewList->at(0);
    auto value = ctx->ooperandInplaceDataViewList->at(0);
    auto index = ctx->ooperandInplaceDataViewList->at(1);
    auto axis = ctx->op->GetIntAttribute("op_attr_axis");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::Sort(value, index, src, axis, descending);
}
REGISTER_CALC_OP(OP_SORT_UB, Opcode::OP_SORT_UB, ExecuteOpSort);

void ExecuteOpTopK(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto outValue = ctx->ooperandInplaceDataViewList->at(0);
    auto outIndex = ctx->ooperandInplaceDataViewList->at(1);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    auto kValue = ctx->op->GetIntAttribute("op_attr_kvalue");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::TopK(outValue, outIndex, src, kValue, topk_axis, descending);
}
REGISTER_CALC_OP(OP_TOPK, Opcode::OP_TOPK, ExecuteOpTopK);

void ExecuteOpBitSort(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    int offset = ctx->op->GetIntAttribute("op_attr_offset");
    calc::BitSort(oop, src, topk_axis, descending, offset);
}
REGISTER_CALC_OP(OP_BITSORT, Opcode::OP_BITSORT, ExecuteOpBitSort);

void ExecuteOpTiledMrgSort(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_FOUR);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src1 = ctx->ioperandDataViewList->at(0);
    auto src2 = ctx->ioperandDataViewList->at(1);
    auto src3 = ctx->ioperandDataViewList->at(2);
    auto src4 = ctx->ioperandDataViewList->at(3);
    auto validBit = ctx->op->GetIntAttribute("op_attr_validBit");
    auto kvalue = ctx->op->GetIntAttribute("op_attr_kvalue");
    calc::TiledMrgSort(oop, src1, src2, src3, src4, validBit, kvalue);
}
REGISTER_CALC_OP(OP_TILEDMRGSORT, Opcode::OP_TILEDMRGSORT, ExecuteOpTiledMrgSort);

void ExecuteOpTopkSort(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 2);  // value + temp

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop_value = ctx->ooperandInplaceDataViewList->at(0);
    auto oop_temp = ctx->ooperandInplaceDataViewList->at(1);

    int startIndex = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "start_index");

    calc::TopkSort(oop_value, oop_temp, iop, startIndex);
}
REGISTER_CALC_OP(OP_TOPK_SORT, Opcode::OP_TOPK_SORT, ExecuteOpTopkSort);

void ExecuteOpTopkMerge(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);

    int mergeSize = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "merge_size");

    calc::TopkMerge(oop, iop, mergeSize);
}
REGISTER_CALC_OP(OP_TOPK_MERGE, Opcode::OP_TOPK_MERGE, ExecuteOpTopkMerge);

void ExecuteOpTopkExtract(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);

    int k = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "k");
    bool isIndex = static_cast<bool>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "is_index"));

    calc::TopkExtract(oop, iop, k, isIndex);
}
REGISTER_CALC_OP(OP_TOPK_EXTRACT, Opcode::OP_TOPK_EXTRACT, ExecuteOpTopkExtract);

void ExecuteOpReduceAcc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    calc::ReduceAcc(ret, *ctx->ioperandDataViewList);
}
REGISTER_CALC_OP(OP_REDUCE_ACC, Opcode::OP_REDUCE_ACC, ExecuteOpReduceAcc);

template <Opcode opcode>
void ExecuteOpBinaryScalar(ExecuteOperationContext *ctx) {
    if (opcode == Opcode::OP_BITWISEXOR) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    bool reverse = ctx->op->GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand");

    switch (opcode) {
        case Opcode::OP_ADDS: calc::AddS(ret, lhs, element); break;
        case Opcode::OP_SUBS: calc::SubS(ret, lhs, element, reverse); break;
        case Opcode::OP_MULS: calc::MulS(ret, lhs, element); break;
        case Opcode::OP_MAXS: calc::MaxS(ret, lhs, element); break;
        case Opcode::OP_MINS: calc::MinS(ret, lhs, element); break;
        case Opcode::OP_DIVS: calc::DivS(ret, lhs, element, reverse); break;
        case Opcode::OP_S_MAXS: calc::MaxS(ret, lhs, element); break;
        case Opcode::OP_S_MINS: calc::MinS(ret, lhs, element);  break;
        case Opcode::OP_BITWISEANDS: calc::BitwiseAndS(ret, lhs, element); break;
        case Opcode::OP_BITWISEORS: calc::BitwiseOrS(ret, lhs, element); break;
        case Opcode::OP_BITWISEXORS: calc::BitwiseXorS(ret, lhs, element); break;
        default: ASSERT(false);
    }
}
REGISTER_CALC_OP(OP_ADDS, Opcode::OP_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CALC_OP(OP_SUBS, Opcode::OP_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CALC_OP(OP_MULS, Opcode::OP_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CALC_OP(OP_DIVS, Opcode::OP_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CALC_OP(OP_MAXS, Opcode::OP_MAXS, ExecuteOpBinaryScalar<Opcode::OP_MAXS>);
REGISTER_CALC_OP(OP_MINS, Opcode::OP_MINS, ExecuteOpBinaryScalar<Opcode::OP_MINS>);
REGISTER_CALC_OP(OP_BITWISEANDS, Opcode::OP_BITWISEANDS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEANDS>);
REGISTER_CALC_OP(OP_BITWISEORS, Opcode::OP_BITWISEORS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEORS>);
REGISTER_CALC_OP(OP_BITWISEXORS, Opcode::OP_BITWISEXORS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEXORS>);
REGISTER_CALC_OP(OP_S_ADDS, Opcode::OP_S_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CALC_OP(OP_S_SUBS, Opcode::OP_S_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CALC_OP(OP_S_MULS, Opcode::OP_S_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CALC_OP(OP_S_DIVS, Opcode::OP_S_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CALC_OP(OP_S_MAXS, Opcode::OP_S_MAXS, ExecuteOpBinaryScalar<Opcode::OP_S_MAXS>);
REGISTER_CALC_OP(OP_S_MINS, Opcode::OP_S_MINS, ExecuteOpBinaryScalar<Opcode::OP_S_MINS>);

void ExecuteOpGatherElement(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &params = ctx->ioperandDataViewList->at(0);
    auto &indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute("op_attr_axis");
    calc::GatherElements(ret, params, indices, axis);
}
REGISTER_CALC_OP(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, ExecuteOpGatherElement);

template <Opcode opcode>
void ExecuteOpBitwiseShift(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);

    switch (opcode) {
        case Opcode::OP_BITWISERIGHTSHIFT: calc::BitwiseRightShift(ret, lhs, rhs); break;
        case Opcode::OP_BITWISELEFTSHIFT: calc::BitwiseLeftShift(ret, lhs, rhs); break;
        default: ASSERT(false);
    }
}

template <Opcode opcode>
void ExecuteOpBitwiseShiftScalar(ExecuteOperationContext *ctx) {
    if (opcode == Opcode::OP_SBITWISERIGHTSHIFT || opcode == Opcode::OP_SBITWISELEFTSHIFT) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_INT32, 0);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);

    switch (opcode) {
        case Opcode::OP_BITWISERIGHTSHIFTS: calc::BitwiseRightShiftS(ret, lhs, element); break;
        case Opcode::OP_BITWISELEFTSHIFTS: calc::BitwiseLeftShiftS(ret, lhs, element); break;
        case Opcode::OP_SBITWISERIGHTSHIFT: calc::SBitwiseRightShift(ret, element, lhs); break;
        case Opcode::OP_SBITWISELEFTSHIFT: calc::SBitwiseLeftShift(ret, element, lhs); break;
        default: ASSERT(false);
    }
}
REGISTER_CALC_OP(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT, ExecuteOpBitwiseShift<Opcode::OP_BITWISERIGHTSHIFT>);
REGISTER_CALC_OP(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, ExecuteOpBitwiseShift<Opcode::OP_BITWISELEFTSHIFT>);
REGISTER_CALC_OP(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS, ExecuteOpBitwiseShiftScalar<Opcode::OP_BITWISERIGHTSHIFTS>);
REGISTER_CALC_OP(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS, ExecuteOpBitwiseShiftScalar<Opcode::OP_BITWISELEFTSHIFTS>);
REGISTER_CALC_OP(OP_SBITWISERIGHTSHIFT, Opcode::OP_SBITWISERIGHTSHIFT, ExecuteOpBitwiseShiftScalar<Opcode::OP_SBITWISERIGHTSHIFT>);
REGISTER_CALC_OP(OP_SBITWISELEFTSHIFT, Opcode::OP_SBITWISELEFTSHIFT, ExecuteOpBitwiseShiftScalar<Opcode::OP_SBITWISELEFTSHIFT>);
}
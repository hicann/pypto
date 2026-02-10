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
 * \file calc_common.cpp
 * \brief
 */

#include "utils/string_utils.h"
#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {
void ExecuteOpAssemble(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() <= NUM_VALUE_2);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);

    auto assemble = std::static_pointer_cast<AssembleOpAttribute>(ctx->op->GetOpAttribute());
    std::vector<int64_t> offset = ctx->opInter->EvaluateOffset(assemble->GetToOffset(), assemble->GetToDynOffset());
    auto ret = oop->View(iop->GetShape(), offset);
    calc::Copy(ret, iop);
}
REGISTER_CALC_OP(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, ExecuteOpAssemble);
REGISTER_CALC_OP(OP_ASSEMBLE_SSA, Opcode::OP_ASSEMBLE_SSA, ExecuteOpAssemble);

void ExecuteOpNone(ExecuteOperationContext *ctx) {
    (void)ctx;
}
REGISTER_CALC_OP(OP_PHASE1, Opcode::OP_PHASE1, ExecuteOpNone);
REGISTER_CALC_OP(OP_PHASE2, Opcode::OP_PHASE2, ExecuteOpNone);
REGISTER_CALC_OP(OP_SYNC_SRC, Opcode::OP_SYNC_SRC, ExecuteOpNone);
REGISTER_CALC_OP(OP_SYNC_DST, Opcode::OP_SYNC_DST, ExecuteOpNone);
REGISTER_CALC_OP(OP_BAR_V, Opcode::OP_BAR_V, ExecuteOpNone);
REGISTER_CALC_OP(OP_BAR_M, Opcode::OP_BAR_M, ExecuteOpNone);
REGISTER_CALC_OP(OP_NOP, Opcode::OP_NOP, ExecuteOpNone);

void ExecuteOpView(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    auto opAttr = std::static_pointer_cast<ViewOpAttribute>(ctx->op->GetOpAttribute());
    auto offset = ctx->opInter->EvaluateOffset(opAttr->GetFromOffset(), opAttr->GetFromDynOffset());
    if (oop->GetData() == iop->GetData()) {
        return;
    }
    auto ret = iop->View(oop->GetShape(), offset);
    calc::Copy(oop, ret);
}
REGISTER_CALC_OP(OP_VIEW, Opcode::OP_VIEW, ExecuteOpView);

void ExecuteOpCopyOut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() <= NUM_VALUE_2);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);

    auto copyout = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
    auto [from, toOffsetAttr] = copyout->GetCopyOutAttr();
    std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetShape());
    std::vector<int64_t> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetRawShape());
    std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, toOffsetAttr);

    std::vector<int64_t> iopShape = iop->GetShape();
    if (oop->GetIsSpilled()) {
        std::fill(toOffset.begin(), toOffset.end(), 0);
    }

    bool axisCombine = ctx->op->GetBoolAttribute("input_combine_axis");
    auto oopValid = std::make_shared<LogicalTensorData>(oop->GetData(), iopShape, toOffset);

    if (from == MemoryType::MEM_L0C) {
        if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
            calc::Add(oopValid, iop, oopValid);
        } else {
            calc::Cast(oopValid, iop);
        }
    } else {
        calc::Copy(oopValid, iop, axisCombine);
    }
}
REGISTER_CALC_OP(OP_COPY_OUT, Opcode::OP_COPY_OUT, ExecuteOpCopyOut);

void ExecuteOpCopyIn(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);

    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());

    bool outputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
    std::vector<int64_t> oopShape = oop->GetShape();
    std::vector<int> axises = {0, 1};
    LogicalTensorDataPtr oopTrans;
    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        std::vector<int64_t> transShape = {oopShape[1], oopShape[0]};
        std::vector<int64_t> transRawShape = {oop->GetData()->GetShape()[1], oop->GetData()->GetShape()[0]};
        oopTrans = LogicalTensorData::CreateEmpty(oop->GetDataType(), transShape, std::vector<int64_t>(0), transRawShape);
    }

    // HACK: copyin's default attribute should be full tensor
    auto iopValid = iop;
    auto oopValid = oop;
    if (copyin != nullptr) {
        std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetShape());
        std::vector<int64_t> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetRawShape());
        std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
        std::vector<int64_t> dynvalidshape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetToDynValidShape());
        if (dynvalidshape.empty()) {
            dynvalidshape = shape;
        }

        iopValid = std::make_shared<LogicalTensorData>(iopValid->GetData(), dynvalidshape, fromOffset);
        if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
            oopTrans = oopTrans->View(dynvalidshape, std::vector<int64_t>(fromOffset.size(), 0));
        } else {
            oopValid = oop->View(dynvalidshape, std::vector<int64_t>(fromOffset.size(), 0));
        }
    }

    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        calc::Copy(oopValid, iopValid, true);
    } else {
        calc::Copy(oopValid, iopValid);
    }
}
REGISTER_CALC_OP(OP_COPY_IN, Opcode::OP_COPY_IN, ExecuteOpCopyIn);

void ExecuteOpCopy(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    calc::Copy(oop, iop);
}
REGISTER_CALC_OP(OP_REGISTER_COPY, Opcode::OP_REGISTER_COPY, ExecuteOpCopy);

std::string FormatString(const std::string &s, OperationInterpreter *opInter,
    const std::vector<LogicalTensorDataPtr> *iopDataView, const std::vector<SymbolicScalar> *scalars) {
    std::stringstream ss;
    size_t pos = 0;
    size_t tpos = 0, spos = 0;
    while (pos < s.size()) {
        if (s[pos] == '{') {
            size_t end = s.find('}', pos + 1);
            if (end == std::string::npos) {
                ss << s.substr(pos);
                break;
            }
            auto symbol = s.substr(pos + 1, end - pos - 1);
            if (symbol == "T") {
                if (iopDataView && tpos < iopDataView->size())
                    ss << (*iopDataView)[tpos++]->ToString();
                else
                    ss << "???";
            } else if ( symbol == "S") {
                if (scalars && spos < scalars->size())
                    ss << opInter->EvaluateSymbolicScalar((*scalars)[spos++]);
                else
                    ss << "???";
            } else {
                ss << '{' << symbol << '}';
            }
            pos = end + 1;
        } else {
            ss << s[pos];
            pos++;
        }
    }
    return ss.str();
}

void ExecutePrint(ExecuteOperationContext *ctx) {
    auto cond = ctx->op->GetSymbolicScalarAttribute(OP_ATTR_PREFIX + "cond");
    if (!ctx->opInter->EvaluateSymbolicScalar(cond)) {
        return;
    }

    std::vector<SymbolicScalar> *scalars = nullptr;
    scalars = ctx->op->GetAttr<std::vector<SymbolicScalar> >(OP_ATTR_PREFIX + "scalars");
    if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "fname")) {
        auto fname = ctx->op->GetStringAttribute(OP_ATTR_PREFIX + "fname");
        uint64_t ts = 0;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        ts = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
        auto baseName = FormatString(fname, ctx->opInter, nullptr, scalars);
        auto basePath = config::LogTopFolder() + "/tensor/" + baseName + "_" + std::to_string(ts);
        auto binPath = basePath + ".data";
        auto csvPath = basePath + ".csv";
        auto &iop = ctx->ioperandDataViewList->at(0);
        auto shape = iop->GetValidShape();
        if (shape.empty()) {
            shape = iop->GetShape();
        }
        auto oop = LogicalTensorData::CreateEmpty(iop->GetDataType(), shape, shape, iop->GetData()->GetShape());
        calc::Copy(oop, iop);
        oop->GetData()->ToFile(binPath);
        std::ofstream csv(csvPath, std::ios::out);
        if (csv) {
            csv << "key,value\n";
            csv << "dtype," << static_cast<int>(oop->GetDataType()) << "\n";
            csv << "shape,";
            if (!shape.empty()) {
                csv << shape[0];
                for (size_t i = 1; i < shape.size(); i++) {
                    csv << "x" << shape[i];
                }
            }
            csv << "\n";
            csv << "element_count," << oop->GetData()->GetDataSize() / oop->GetData()->GetElementSize() << "\n";
            csv.close();
        } else {
            std::cerr << "open csv file " << csvPath << " failed!!!!\n";
        }
    }

    std::string format;
    if (ctx->op->GetAttr(OP_ATTR_PREFIX + "format", format)) {
        std::cout << FormatString(format, ctx->opInter, ctx->ioperandDataViewList, scalars) << std::endl;
    }
}
REGISTER_CALC_OP(OP_PRINT, Opcode::OP_PRINT, ExecutePrint);

void ExecuteOpReshape(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    auto actualIop = std::make_shared<LogicalTensorData>(iop->GetData());
    if (oop->GetSize() > iop->GetSize()) {
        ALOG_EVENT(ctx->op->Dump());
        ALOG_EVENT("iop validShape: ", iop->GetShape(), " ---> oop validShape: ", oop->GetShape());
        ALOG_EVENT("Reshape: input tensor is not enough to reshape to output tensor");
        calc::Reshape(oop, actualIop);
    } else {
        calc::Reshape(oop, iop);
    }
}
REGISTER_CALC_OP(OP_RESHAPE, Opcode::OP_RESHAPE, ExecuteOpReshape);
}
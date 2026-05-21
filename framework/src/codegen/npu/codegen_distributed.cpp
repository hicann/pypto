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
 * \file codegen_distributed.cpp
 * \brief
 */

#include <sstream>
#include <string>
#include "codegen/codegen_common.h"
#include "codegen_op_npu.h"
#include "securec.h"
#include "interface/operation/distributed/distributed_common.h"

namespace npu::tile_fwk {

using AtomicType = Distributed::AtomicType;

static bool HasGetTensorDataInOffsetAttr(
    const std::vector<int64_t>& shape, const std::vector<SymbolicScalar>& offsetAttr)
{
    for (size_t index = 0; index < shape.size(); ++index) {
        if (!offsetAttr[index].IsValid()) {
            continue;
        }
        auto callList = LookupExpressionByOpcode(offsetAttr[index].Raw(), SymbolicOpcode::T_MOP_CALL);
        for (auto& call : callList) {
            auto caller = call->GetExpressionOperandList()[0];
            if (caller->IsSymbol() && CallIsGetTensorData(caller->GetSymbolName())) {
                return true;
            }
        }
    }
    return false;
}

std::string CodeGenOpNPU::GetTemplateDType() const
{
    static const std::unordered_map<Opcode, int32_t> dTypeOperandIndexMap = {
        {Opcode::OP_FFN_BATCHING, 0},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, 0},
        {Opcode::OP_COPY_TO_LOCAL_EXPERT, 0},
        {Opcode::OP_SEND_TO_ROUTING_EXPERT, 1},
        {Opcode::OP_SEND_TO_SHARED_EXPERT, 1},
        {Opcode::OP_FFN_SCHED, 1},
        {Opcode::OP_FFN_BATCHING, 1},
        {Opcode::OP_FFN_VALIDCNT, 1},
        {Opcode::OP_SHMEM_PUT, 1},
        {Opcode::OP_SHMEM_PUT_UB2GM, 1},
        {Opcode::OP_SHMEM_SIGNAL, 1},
        {Opcode::OP_SHMEM_WAIT_UNTIL, 1},
        {Opcode::OP_SHMEM_GET, 1},
        {Opcode::OP_SHMEM_GET_GM2UB, 1},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, 1},
        {Opcode::OP_FFN_COMBINEINFO, 2},
        {Opcode::OP_SHMEM_SET, 3},
        {Opcode::OP_DISPATCH_SET_FLAG, 4},
    };
    auto it = dTypeOperandIndexMap.find(opCode);
    ASSERT(GenCodeErr::OP_CODE_UNSUPPORTED, it != dTypeOperandIndexMap.end())
        << "opcode \"" << opCodeStr << "\" is not distributed opcode";
    int32_t operandIndex = it->second;
    return DataType2CCEStr(operandDtype[operandIndex]);
}

std::string CodeGenOpNPU::GenExtraTemplateParamsForMoeDistributedCombine(int32_t operandIndex) const
{
    Distributed::MoeCombineAttr distOpAttr =
        AnyCast<Distributed::MoeCombineAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t secondToLastIndex = 2;
    int64_t rowShape = shape[operandIndex][shape[operandIndex].size() - secondToLastIndex];
    if (distOpAttr.rowShape != -1) {
        rowShape = distOpAttr.rowShape;
    }
    int64_t colShape = shape[operandIndex][shape[operandIndex].size() - 1];
    std::ostringstream oss;
    oss << "<" << GetTemplateDType() << ", " << distOpAttr.topK << ", " << rowShape << ", " << colShape << ", "
        << distOpAttr.paddedColShape << ">";
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForPutAndGet() const
{
    std::ostringstream oss;
    static const std::unordered_map<Opcode, std::array<int32_t, 2>> opcodeIndexMap = {
        {Opcode::OP_SHMEM_PUT, {3, 4}}, {Opcode::OP_SHMEM_GET, {0, 3}}, {Opcode::OP_SHMEM_GET_GM2UB, {0, 3}}};
    auto [nonShmemDataIndex, shmemDataIndex] = opcodeIndexMap.at(opCode);
    const std::vector<int64_t>& tileShape = shape[shmemDataIndex];
    int64_t tileRowShape = tileShape[tileShape.size() - 2];
    int64_t tileColShape = tileShape[tileShape.size() - 1];

    int64_t bufferRowShape = 0;
    int64_t bufferColShape = 0;
    Distributed::AtomicType atomicType = Distributed::AtomicType::SET;
    if (opCode == Opcode::OP_SHMEM_PUT) {
        Distributed::ShmemPutAttr distOpAttr =
            AnyCast<Distributed::ShmemPutAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
        bufferRowShape = distOpAttr.copyBufferShape[0];
        bufferColShape = distOpAttr.copyBufferShape[1];
        atomicType = distOpAttr.atomicType;
    } else {
        Distributed::ShmemGetAttr distOpAttr =
            AnyCast<Distributed::ShmemGetAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
        bufferRowShape = distOpAttr.copyBufferShape[0];
        bufferColShape = distOpAttr.copyBufferShape[1];
        atomicType = distOpAttr.atomicType;
    }

    const std::vector<int64_t>& shmemTensorRawShape = rawShape[shmemDataIndex];
    const std::vector<int64_t>& nonShmemTensorRawShape = rawShape[nonShmemDataIndex];
    int64_t srcStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
    int64_t dstStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
    if ((opCode == Opcode::OP_SHMEM_GET) || (opCode == Opcode::OP_SHMEM_GET_GM2UB)) {
        srcStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
        dstStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
    }

    oss << "<" << DataType2CCEStr(operandDtype[nonShmemDataIndex]) << ", "
        << DataType2CCEStr(operandDtype[shmemDataIndex]) << ", " << tileRowShape << ", " << tileColShape << ", "
        << bufferRowShape << ", " << bufferColShape << ", " << srcStride << ", " << dstStride << ", "
        << Distributed::ToString(atomicType) << ">";
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForPutUb2Gm() const
{
    std::ostringstream oss;
    int32_t shmemDataIndex = 2;

    int64_t tileRowShape = *(shape[shmemDataIndex].rbegin() + 1);
    int64_t tileColShape = shape[shmemDataIndex].back();

    Distributed::ShmemPutAttr distOpAttr = AnyCast<Distributed::ShmemPutAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

    const std::vector<int64_t>& shmemTensorRawShape = rawShape[shmemDataIndex];
    int64_t dstStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];

    oss << "<" << GetTemplateDType() << ", " << tileRowShape << ", " << tileColShape << ", " << dstStride << ", "
        << Distributed::ToString(distOpAttr.atomicType) << ">";
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForSignal() const
{
    std::ostringstream oss;
    Distributed::ShmemSignalAttr distOpAttr =
        AnyCast<Distributed::ShmemSignalAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

    oss << "<" << distOpAttr.signalValue << ", " << distOpAttr.signalStride << ", "
        << Distributed::ToString(distOpAttr.atomicType) << ", " << (distOpAttr.notifyAll ? "true" : "false") << ", "
        << distOpAttr.worldSize;

    auto paddedShape = distOpAttr.tileShape;
    paddedShape.resize(SHAPE_DIM4, 0);
    for (const auto& val : paddedShape) {
        oss << ", " << val;
    }
    oss << ", " << distOpAttr.tileShape.size() << ">";

    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForMoeDistributedCombineSend() const
{
    std::ostringstream oss;
    int32_t expandXIndex = 4;
    oss << GenExtraTemplateParamsForMoeDistributedCombine(expandXIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForMoeDistributedCombineReceive() const
{
    std::ostringstream oss;
    int32_t outIndex = 0;
    oss << GenExtraTemplateParamsForMoeDistributedCombine(outIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsForSet() const
{
    std::ostringstream oss;
    int32_t shmemTensorIndex = 3;
    Distributed::ShmemSetAttr distOpAttr = AnyCast<Distributed::ShmemSetAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t bufferEleNum = distOpAttr.setBufferShape[0];
    size_t shmemTensorDim = rawShape[shmemTensorIndex].size();
    ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, shmemTensorDim >= SHAPE_DIM2)
        << "shmem tensor dim = " << shmemTensorDim << ", should >= 2.";
    if (distOpAttr.isSetData) {
        oss << "<" << GetTemplateDType() << ", " << bufferEleNum << ">";
    } else {
        oss << "<" << GetTemplateDType() << ", " << rawShape[shmemTensorIndex][0] << ", "
            << Distributed::SHMEM_SIGNAL_STRIDE << ", " << bufferEleNum << ">";
    }
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParamsDefault() const
{
    std::ostringstream oss;
    Distributed::MoeDispatchAttr distOpAttr;
    if (opAttrs.count(OpAttributeKey::distOpAttr) != 0) {
        distOpAttr = AnyCast<Distributed::MoeDispatchAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    }
    if (distOpAttr.extraTemplateParam.empty()) {
        oss << "<" << GetTemplateDType() << ">";
    } else {
        oss << "<" << GetTemplateDType() << ", " << distOpAttr.extraTemplateParam << ">";
    }
    return oss.str();
}

std::string CodeGenOpNPU::GenTemplateParams() const
{
    static const std::unordered_map<Opcode, std::function<std::string(CodeGenOpNPU const*)>> templateParamHandlers = {
        {Opcode::OP_SHMEM_PUT, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_GET, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_PUT_UB2GM, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForPutUb2Gm(); }},
        {Opcode::OP_SHMEM_GET_GM2UB, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_SIGNAL, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForSignal(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND,
         [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForMoeDistributedCombineSend(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE,
         [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForMoeDistributedCombineReceive(); }},
        {Opcode::OP_SHMEM_SET, [](const CodeGenOpNPU* self) { return self->GenTemplateParamsForSet(); }}};

    auto handler = templateParamHandlers.find(opCode);
    if (handler != templateParamHandlers.end()) {
        return handler->second(this);
    } else {
        return GenTemplateParamsDefault();
    }
}

std::string CodeGenOpNPU::GenOffsets(int32_t operandIndex) const
{
    int32_t dim = shape[operandIndex].size();
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_OFFSET)[0];
}

std::string CodeGenOpNPU::GenShapes(int32_t operandIndex) const
{
    int32_t dim = shape[operandIndex].size();
    return GenGetParamMacroPacked(operandIndex, dim, "SHAPE")[0];
}

std::string CodeGenOpNPU::GenRawShapes(int32_t operandIndex) const
{
    int32_t dim = shape[operandIndex].size();
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_RAW_SHAPE)[0];
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapes(int32_t operandIndex) const
{
    return GenOffsets(operandIndex) + ", " + GenRawShapes(operandIndex);
}

std::string CodeGenOpNPU::GenDynOffCoord(int32_t operandIndex) const
{
    size_t dim = shape[operandIndex].size();

    // 如果 offset 没有任何 GetTensorData 类型，则从Coa中获取
    if (!HasGetTensorDataInOffsetAttr(shape[operandIndex], offsetFromAttr[operandIndex])) {
        return GenOffCoord(operandIndex);
    }

    // 如果 offset 有 GetTensorData 类型，则从 copyOpAttribute 获取
    std::ostringstream oss;
    for (size_t index = 0; index < dim; ++index) {
        if (offsetFromAttr[operandIndex][index].IsValid()) {
            oss << SymbolicExpressionTable::BuildExpression(offsetFromAttr[operandIndex][index]);
        } else {
            oss << GenParamIdxExprByIndex(operandIndex, dim, PREFIX_STR_OFFSET)[index];
        }
        if (index != dim - 1) {
            oss << ", ";
        }
    }
    std::string wrappedCoord = WrapParamByParentheses({oss.str()});
    return PrintCoord(dim, wrappedCoord);
}

std::string CodeGenOpNPU::GenOffCoord(int32_t operandIndex) const
{
    size_t dim = shape[operandIndex].size();
    auto OffsetSymbol = GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_OFFSET);
    std::string wrappedCoord = WrapParamByParentheses(OffsetSymbol);
    return PrintCoord(dim, wrappedCoord);
}

std::string CodeGenOpNPU::GenDynValidShape(int32_t operandIndex) const
{
    auto srcDynValidShape = dynamicValidShape[operandIndex];
    FillVecWithDummyInHead<SymbolicScalar>(srcDynValidShape, SHAPE_DIM5 - dynamicValidShape[operandIndex].size(), 1);
    std::vector<std::string> paramList;
    for (auto dynShape : srcDynValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    return tileOpCallParam;
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemPut() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 3;
    int32_t shmemDataIndex = 4;
    int32_t outPutDataIndex = 0;

    oss << ", " << QueryTileTensorNameByIdx(nonShmemDataIndex) << ", " << QueryTileTensorNameByIdx(shmemDataIndex)
        << ", " << GenOffCoord(nonShmemDataIndex) << ", " << GenDynOffCoord(shmemDataIndex) << ", "
        << GenDynValidShape(outPutDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemGet() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 0;
    int32_t shmemDataIndex = 3;

    oss << ", " << QueryTileTensorNameByIdx(nonShmemDataIndex) << ", " << QueryTileTensorNameByIdx(shmemDataIndex)
        << ", " << GenDynOffCoord(nonShmemDataIndex) << ", " << GenOffCoord(shmemDataIndex) << ", "
        << GenDynValidShape(nonShmemDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemPutUB() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 1;
    int32_t shmemDataIndex = 2;
    int32_t outPutDataIndex = 0;

    oss << QueryTileTensorNameByIdx(nonShmemDataIndex) << ", " << QueryTileTensorNameByIdx(shmemDataIndex) << ", "
        << GenOffCoord(nonShmemDataIndex) << ", " << GenDynOffCoord(shmemDataIndex) << ", "
        << GenDynValidShape(outPutDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemGetUB() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 0;
    int32_t shmemDataIndex = 3;
    int32_t outPutDataIndex = 0;

    oss << ", " << QueryTileTensorNameByIdx(nonShmemDataIndex) << ", " << QueryTileTensorNameByIdx(shmemDataIndex)
        << ", " << GenOffCoord(nonShmemDataIndex) << ", " << GenDynOffCoord(shmemDataIndex) << ", "
        << GenDynValidShape(outPutDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemSignal() const
{
    std::ostringstream oss;
    int32_t shmemSignalIndex = 3;

    oss << ", " << QueryTileTensorNameByIdx(shmemSignalIndex) << ", " << GenOffCoord(shmemSignalIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForMoeDistributedCombineSend() const
{
    std::ostringstream oss;
    int32_t expandXIndex = 4;
    oss << ", " << GenOffsets(expandXIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForMoeDistributedCombineReceive() const
{
    std::ostringstream oss;
    int32_t shmemDataIndex = 6;
    Distributed::MoeCombineAttr distOpAttr =
        AnyCast<Distributed::MoeCombineAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    oss << ", " << GenOffsets(shmemDataIndex) << ", " << distOpAttr.rowOffset;
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForSendToRoutingExpert() const
{
    std::ostringstream oss;
    int32_t expertTableIndex = 6;
    int32_t shmemDataIndex = 5;
    oss << ", " << GenOffsetsAndRawShapes(expertTableIndex) << ", " << GenOffsetsAndRawShapes(shmemDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForSendToSharedExpert() const
{
    std::ostringstream oss;
    int32_t tokenIndex = 2;
    int32_t shmemDataIndex = 3;
    oss << ", " << GenOffsetsAndRawShapes(tokenIndex) << ", " << GenOffsetsAndRawShapes(shmemDataIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForCopyToLocalExpert() const
{
    std::ostringstream oss;
    int32_t tokenIndex = 3;
    oss << ", " << GenOffsetsAndRawShapes(tokenIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForDispatchSetFlag() const
{
    std::ostringstream oss;
    int32_t shmemFlagIndex = 5;
    oss << ", " << GenOffsetsAndRawShapes(shmemFlagIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForFfnOperations() const
{
    std::ostringstream oss;
    int32_t shmemIndex = 3;
    oss << ", " << GenOffsetsAndRawShapes(shmemIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForFfnCombineInfo() const
{
    std::ostringstream oss;
    int32_t shmemIndex = 2;
    oss << ", " << GenOffsetsAndRawShapes(shmemIndex);
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesForShmemSet() const
{
    std::ostringstream oss;
    Distributed::ShmemSetAttr distOpAttr = AnyCast<Distributed::ShmemSetAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int32_t shmemTensorIndex = 3;
    std::string shmemTensor = QueryTileTensorNameByIdx(shmemTensorIndex);
    if (distOpAttr.isSetData) {
        oss << ", " << shmemTensor << ", " << GenOffCoord(shmemTensorIndex);
    } else {
        oss << ", " << shmemTensor;
    }
    return oss.str();
}

std::string CodeGenOpNPU::GenOffsetsAndRawShapesDefault() const { return ""; }

std::string CodeGenOpNPU::GenExtraParamsStr() const
{
    static const std::unordered_map<Opcode, std::function<std::string(CodeGenOpNPU const*)>>
        offsetsAndRawShapesHandlers = {
            {Opcode::OP_SHMEM_PUT, [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemPut(); }},
            {Opcode::OP_SHMEM_GET, [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemGet(); }},
            {Opcode::OP_SHMEM_PUT_UB2GM,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutUB(); }},
            {Opcode::OP_SHMEM_GET_GM2UB,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemGetUB(); }},
            {Opcode::OP_SHMEM_SIGNAL,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemSignal(); }},
            {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForMoeDistributedCombineSend(); }},
            {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForMoeDistributedCombineReceive(); }},
            {Opcode::OP_SEND_TO_ROUTING_EXPERT,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForSendToRoutingExpert(); }},
            {Opcode::OP_SEND_TO_SHARED_EXPERT,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForSendToSharedExpert(); }},
            {Opcode::OP_COPY_TO_LOCAL_EXPERT,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForCopyToLocalExpert(); }},
            {Opcode::OP_DISPATCH_SET_FLAG,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForDispatchSetFlag(); }},
            {Opcode::OP_FFN_SCHED,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
            {Opcode::OP_FFN_BATCHING,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
            {Opcode::OP_FFN_VALIDCNT,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
            {Opcode::OP_FFN_COMBINEINFO,
             [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForFfnCombineInfo(); }},
            {Opcode::OP_SHMEM_SET, [](const CodeGenOpNPU* self) { return self->GenOffsetsAndRawShapesForShmemSet(); }}};

    auto handler = offsetsAndRawShapesHandlers.find(opCode);
    if (handler != offsetsAndRawShapesHandlers.end()) {
        return handler->second(this);
    } else {
        return GenOffsetsAndRawShapesDefault();
    }
} // namespace npu::tile_fwk

std::string CodeGenOpNPU::GenTargetRankStr() const
{
    if (opAttrs.count(OpAttributeKey::ownerRank) == 0) {
        return "";
    }
    std::ostringstream oss;
    auto ownerRank = AnyCast<SymbolicScalar>(opAttrs.at(OpAttributeKey::ownerRank));
    if (ownerRank.IsValid()) {
        oss << ", " << SymbolicExpressionTable::BuildExpression(ownerRank);
    }
    return oss.str();
}

std::string CodeGenOpNPU::GenDistOp() const
{
    std::ostringstream oss;
    std::unordered_set<int32_t> skipOperands = {};
    static const std::unordered_map<Opcode, std::unordered_set<int32_t>> skipIndexMap = {
        {Opcode::OP_SHMEM_PUT, {0, 2, 3, 4}},           {Opcode::OP_SHMEM_GET, {0, 2, 3}},
        {Opcode::OP_SHMEM_PUT_UB2GM, {0, 1, 2, 3}},     {Opcode::OP_SHMEM_GET_GM2UB, {0, 2, 3}},
        {Opcode::OP_SHMEM_SIGNAL, {0, 2, 3}},           {Opcode::OP_SHMEM_SET, {0, 2, 3}},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, {0}}, {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, {4}},
    }; // 跳过部分操作数索引，对于 shmem api 只需要获得 ub 地址信息
    auto it = skipIndexMap.find(opCode);
    if (it != skipIndexMap.end()) {
        skipOperands = it->second;
    }
    oss << tileOpName << GenTemplateParams() << "(param, " << GenParamsStr(skipOperands) << GenExtraParamsStr()
        << GenTargetRankStr() << ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk

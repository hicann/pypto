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
#include "codegen_op_cloudnpu.h"
#include "interface/utils/log.h"
#include "securec.h"
#include "interface/operation/distributed/distributed_common.h"

namespace npu::tile_fwk {

using AtomicType = Distributed::AtomicType;
using DistOpAttr = Distributed::DistOpAttr;
constexpr int32_t GM2UB_SHMEMDATA_INDEX = 2;

void CheckInRange(int64_t value)
{
    if (value < std::numeric_limits<uint32_t>::min() || value > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Invalid value: " + std::to_string(value));
    }
}

std::string CodeGenOpCloudNPU::GetTemplateDType() const
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
    ASSERT(it != dTypeOperandIndexMap.end());
    int32_t operandIndex = it->second;
    return DataType2CCEStr(operandDtype[operandIndex]);
}

std::string CodeGenOpCloudNPU::GenExtraTemplateParamsForMoeDistributedCombine(int32_t operandIndex) const {
    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t secondToLastIndex = 2;
    int64_t rowShape = originShape[operandIndex][originShape[operandIndex].size() - secondToLastIndex];
    if (distOpAttr.rowShape != -1) {
        rowShape = distOpAttr.rowShape;
    }
    int64_t colShape = originShape[operandIndex][originShape[operandIndex].size() - 1];
    std::ostringstream oss;
    oss << "<" << GetTemplateDType() << ", " << distOpAttr.topK << ", " << rowShape << ", " << colShape << ", "
        << distOpAttr.paddedColShape << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForPutAndGet() const
{
    std::ostringstream oss;
    static const std::unordered_map<Opcode, std::array<int32_t, 2>> opcodeIndexMap = {
        {Opcode::OP_SHMEM_PUT, {3, 4}},
        {Opcode::OP_SHMEM_GET, {0, 3}},
        {Opcode::OP_SHMEM_PUT_UB2GM, {1, GM2UB_SHMEMDATA_INDEX}},
        {Opcode::OP_SHMEM_GET_GM2UB, {0, 3}}
    };
    auto [nonShmemDataIndex, shmemDataIndex] = opcodeIndexMap.at(opCode);
    const std::vector<int64_t>& tileShape = originShape[shmemDataIndex];
    int64_t tileRowShape = tileShape[tileShape.size() - 2];
    int64_t tileColShape = tileShape[tileShape.size() - 1];

    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

    int64_t bufferRowShape = distOpAttr.copyBufferShape[0];
    int64_t bufferColShape = distOpAttr.copyBufferShape[1];

    const std::vector<int64_t>& shmemTensorRawShape = rawShape[shmemDataIndex];
    const std::vector<int64_t>& nonShmemTensorRawShape = rawShape[nonShmemDataIndex];
    int64_t srcStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
    int64_t dstStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
    if ((opCode == Opcode::OP_SHMEM_GET) || (opCode == Opcode::OP_SHMEM_GET_GM2UB)) {
        srcStride = shmemTensorRawShape[shmemTensorRawShape.size() - 1];
        dstStride = nonShmemTensorRawShape[nonShmemTensorRawShape.size() - 1];
    }

    CheckInRange(tileRowShape);
    CheckInRange(tileColShape);
    CheckInRange(bufferRowShape);
    CheckInRange(bufferColShape);
    CheckInRange(srcStride);
    CheckInRange(dstStride);

    oss << "<" << DataType2CCEStr(operandDtype[nonShmemDataIndex]) << ", " << DataType2CCEStr(operandDtype[shmemDataIndex])
        << ", " << tileRowShape << ", " << tileColShape << ", " << bufferRowShape
        << ", " << bufferColShape << ", " << srcStride << ", " << dstStride << ", "
        << Distributed::AtomicTypeToString(distOpAttr.atomicType) << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForSignal() const
{
    std::ostringstream oss;
    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    oss << "<" << std::to_string(distOpAttr.signalValue) << ", "
        << std::to_string(distOpAttr.signalStride) << ", "
        << std::to_string(distOpAttr.tileRowShape) << ", "
        << std::to_string(distOpAttr.tileColShape) << ", "
        << Distributed::AtomicTypeToString(distOpAttr.atomicType) << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForMoeDistributedCombineSend() const
{
    std::ostringstream oss;
    int32_t expandXIndex = 4;
    oss << GenExtraTemplateParamsForMoeDistributedCombine(expandXIndex);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForMoeDistributedCombineReceive() const
{
    std::ostringstream oss;
    int32_t outIndex = 0;
    oss << GenExtraTemplateParamsForMoeDistributedCombine(outIndex);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForSet() const
{
    std::ostringstream oss;
    int32_t shmemTensorIndex = 3;
    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t bufferEleNum = distOpAttr.setBufferShape[0];
    int32_t rowDimIndex = 2;
    int32_t colDimIndex = 3;
    int64_t rawShapeRow;
    int64_t rawShapeCol;
    if (distOpAttr.setType == 0) {
        rawShapeRow = originShape[shmemTensorIndex][rowDimIndex];
        rawShapeCol = originShape[shmemTensorIndex][colDimIndex];
    } else {
        rawShapeRow = Distributed::MAX_TILE_NUM;
        rawShapeCol = Distributed::SHMEM_SIGNAL_STRIDE;
    }
    oss << "<" << GetTemplateDType() << ", " << originShape[shmemTensorIndex][1] << ", "
            << rawShapeRow << ", " << rawShapeCol << ", " << bufferEleNum << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsDefault() const
{
    std::ostringstream oss;
    DistOpAttr distOpAttr;
    if (opAttrs.count(OpAttributeKey::distOpAttr) != 0) {
        distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    }
    if (distOpAttr.extraTemplateParam.empty()) {
        oss << "<" << GetTemplateDType() << ">";
    } else {
        oss << "<" << GetTemplateDType() << ", " << distOpAttr.extraTemplateParam << ">";
    }
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParams() const
{
    static const std::unordered_map<Opcode,
        std::function<std::string(CodeGenOpCloudNPU const*)>> templateParamHandlers = {
        {Opcode::OP_SHMEM_PUT, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_GET, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_PUT_UB2GM, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_GET_GM2UB, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForPutAndGet(); }},
        {Opcode::OP_SHMEM_SIGNAL, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForSignal(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForMoeDistributedCombineSend(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForMoeDistributedCombineReceive(); }},
        {Opcode::OP_SHMEM_SET, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForSet(); }}
    };

    auto handler = templateParamHandlers.find(opCode);
    if (handler != templateParamHandlers.end()) {
        return handler->second(this);
    } else {
        return GenTemplateParamsDefault();
    }
}

std::string CodeGenOpCloudNPU::GenOffsets(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_OFFSET)[0];
}

std::string CodeGenOpCloudNPU::GenShapes(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, "SHAPE")[0];
}

std::string CodeGenOpCloudNPU::GenRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_RAW_SHAPE)[0];
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenOffsets(operandIndex, dim) + ", " + GenRawShapes(operandIndex, dim);
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemPut() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 3;
    int32_t shmemDataIndex = 4;
    int32_t nonShmemDataDim = originShape[nonShmemDataIndex].size();
    int32_t shmemDataDim = 4;
    std::string viewOffsetStr = dynamicValidShape[shmemDataIndex][2].Dump();
    size_t firstComma = viewOffsetStr.find(",");
    size_t lastComma = viewOffsetStr.rfind(",");
    std::string viewOffset = viewOffsetStr.substr(firstComma + 1, lastComma - firstComma - 1);
    if (viewOffset.find("RUNTIME_GetTensorDataInt32Dim2") != std::string::npos) {
        oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim) << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim) << ", " << viewOffset;
    } else {
        oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim) << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim) << ", " << -1;
    }
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemGet() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = 0;
    int32_t shmemDataIndex = 3;
    int32_t nonShmemDataDim = originShape[nonShmemDataIndex].size();
    int32_t shmemDataDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim) << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemPutAndGetUB() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT_UB2GM) ? 1 : 0;
    int32_t shmemDataIndex = 2;
    int32_t nonShmemDataDim = 2;
    int32_t shmemDataDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim)
        << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemSignal() const
{
    std::ostringstream oss;
    int32_t shmemSignalIndex = 3;
    int32_t shmemSignalDim = 5;
    oss << ", " << GenOffsetsAndRawShapes(shmemSignalIndex, shmemSignalDim)
        << ", " << GenShapes(shmemSignalIndex, shmemSignalDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForMoeDistributedCombineSend() const
{
    std::ostringstream oss;
    int32_t expandXIndex = 4;
    int32_t expandXDim = 2;
    oss << ", " << GenOffsets(expandXIndex, expandXDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForMoeDistributedCombineReceive() const
{
    std::ostringstream oss;
    int32_t shmemDataIndex = 6;
    int32_t shmemDataDim = 4;
    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    oss << ", " << GenOffsets(shmemDataIndex, shmemDataDim) << ", " << distOpAttr.rowOffset;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForSendToRoutingExpert() const
{
    std::ostringstream oss;
    int32_t expertTableIndex = 6;
    int32_t expertTableDim = 2;
    int32_t shmemDataIndex = 5;
    int32_t shmemDataDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(expertTableIndex, expertTableDim) << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForSendToSharedExpert() const
{
    std::ostringstream oss;
    int32_t tokenIndex = 2;
    int32_t tokenDim = 2;
    int32_t shmemDataIndex= 3;
    int32_t shmemDataDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(tokenIndex, tokenDim) << ", "
        << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForCopyToLocalExpert() const
{
    std::ostringstream oss;
    int32_t tokenIndex = 3;
    int32_t tokenDim = 2;
    oss << ", " << GenOffsetsAndRawShapes(tokenIndex, tokenDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForDispatchSetFlag() const
{
    std::ostringstream oss;
    int32_t shmemFlagIndex = 5;
    int32_t shmemFlagDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(shmemFlagIndex, shmemFlagDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForFfnOperations() const
{
    std::ostringstream oss;
    int32_t shmemIndex = 3;
    int32_t shmemDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(shmemIndex, shmemDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForFfnCombineInfo() const
{
    std::ostringstream oss;
    int32_t shmemIndex = 2;
    int32_t shmemDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(shmemIndex, shmemDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemSet() const
{
    std::ostringstream oss;
    DistOpAttr distOpAttr = AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int32_t shmemTensorIndex = 3;
    int32_t shmemTensorDim;
    if (distOpAttr.setType == 0) {
        shmemTensorDim = 4;
        oss << ", " << GenOffsets(shmemTensorIndex, shmemTensorDim);
    } else {
        shmemTensorDim = 5;
        oss << ", " << GenOffsetsAndRawShapes(shmemTensorIndex, shmemTensorDim) << ", " << GenShapes(shmemTensorIndex, shmemTensorDim);
    }
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesDefault() const
{
    return "";
}

std::string CodeGenOpCloudNPU::GenExtraParamsStr() const
{
    static const std::unordered_map<Opcode,
        std::function<std::string(CodeGenOpCloudNPU const*)>> offsetsAndRawShapesHandlers = {
        {Opcode::OP_SHMEM_PUT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPut(); }},
        {Opcode::OP_SHMEM_GET, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemGet(); }},
        {Opcode::OP_SHMEM_PUT_UB2GM, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutAndGetUB(); }},
        {Opcode::OP_SHMEM_GET_GM2UB, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemGet(); }},
        {Opcode::OP_SHMEM_SIGNAL, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemSignal(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForMoeDistributedCombineSend(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForMoeDistributedCombineReceive(); }},
        {Opcode::OP_SEND_TO_ROUTING_EXPERT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForSendToRoutingExpert(); }},
        {Opcode::OP_SEND_TO_SHARED_EXPERT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForSendToSharedExpert(); }},
        {Opcode::OP_COPY_TO_LOCAL_EXPERT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForCopyToLocalExpert(); }},
        {Opcode::OP_DISPATCH_SET_FLAG, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForDispatchSetFlag(); }},
        {Opcode::OP_FFN_SCHED, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
        {Opcode::OP_FFN_BATCHING, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
        {Opcode::OP_FFN_VALIDCNT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForFfnOperations(); }},
        {Opcode::OP_FFN_COMBINEINFO, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForFfnCombineInfo(); }},
        {Opcode::OP_SHMEM_SET, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemSet(); }}
    };

    auto handler = offsetsAndRawShapesHandlers.find(opCode);
    if (handler != offsetsAndRawShapesHandlers.end()) {
        return handler->second(this);
    } else {
        return GenOffsetsAndRawShapesDefault();
    }
}

std::string CodeGenOpCloudNPU::GenDistOp() const
{
    std::ostringstream oss;
    std::unordered_set<int32_t> skipOperands = {};
    static const std::unordered_map<Opcode, std::unordered_set<int32_t>> skipIndexMap = {
        {Opcode::OP_SHMEM_PUT, {0, 2}},
        {Opcode::OP_SHMEM_GET, {2}},
        {Opcode::OP_SHMEM_PUT_UB2GM, {0, 3}},
        {Opcode::OP_SHMEM_GET_GM2UB, {2}},
        {Opcode::OP_SHMEM_SIGNAL, {0, 2}},
        {Opcode::OP_SHMEM_SET, {0, 2}},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, {0}},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, {4}},
    };
    auto it = skipIndexMap.find(opCode);
    if (it != skipIndexMap.end()) {
        skipOperands = it->second;
    }
    oss << tileOpName << GenTemplateParams() << "(" << GenParamsStr(skipOperands) << GenExtraParamsStr()
        << ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk

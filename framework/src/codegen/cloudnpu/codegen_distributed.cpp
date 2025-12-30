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

using AtomicType = npu::tile_fwk::Distributed::AtomicType;
using DistOpAttr = npu::tile_fwk::Distributed::DistOpAttr;
constexpr int32_t GM2UB_SHMEMDATA_INDEX = 2;

static const std::unordered_map<Opcode, std::unordered_set<int32_t>> skipIndexMap = {
    {Opcode::OP_SHMEM_PUT, {0, 4}},
    {Opcode::OP_SHMEM_GET, {2}},
    {Opcode::OP_SHMEM_PUT_UB2GM, {0, 3}},
    {Opcode::OP_SHMEM_GET_GM2UB, {1}},
    {Opcode::OP_SHMEM_SIGNAL, {0, 2}},
    {Opcode::OP_SHMEM_SET, {0, 2}},
};

void CheckInRange(int64_t value)
{
    if (value < std::numeric_limits<uint32_t>::min() || value > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Invalid value: " + std::to_string(value));
    }
}

std::string CodeGenOpCloudNPU::GetTemplateDType() const
{
    int32_t operandIndex{1};
    switch (opCode) {
        case Opcode::OP_FFN_BATCHING:
        case Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE:
        case Opcode::OP_COPY_TO_LOCAL_EXPERT: {
            operandIndex = 0;
            break;
        }
        case Opcode::OP_FFN_COMBINEINFO: {
            operandIndex = 2; // 从 operand 2 获取 T
            break;
        }
        case Opcode::OP_SHMEM_SET: {
            operandIndex = 3; // 从 operand 3 获取 T
            break;
        }
        case Opcode::OP_DISPATCH_SET_FLAG: {
            operandIndex = 4; // 从 operand 4 获取 T
            break;
        }
        default: {
            break;
        }
    }
    return DataType2CCEStr(operandDtype[operandIndex]);
}

void CodeGenOpCloudNPU::GenExtraTemplateParamsForMoeCombine(std::ostringstream& oss, int32_t operandIndex) const {
    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t colShape = originShape[operandIndex][originShape[operandIndex].size() - 1];
    int64_t dataIndex = (opCode == Opcode::OP_SHMEM_MOE_COMBINE_SEND) ? 4 : 6;
    int64_t rowShape = originShape[dataIndex][originShape[dataIndex].size() - 2];
    oss << "<" << GetTemplateDType() << ", " << distOpAttr.topK << ", " << rowShape << ", " << colShape << ", "
        << distOpAttr.paddedColShape << ">";
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForPutAndGet() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT) ? 2 : 0;
    int32_t shmemDataIndex = 3;
    int32_t shapeIndex = 3;
    if (opCode == Opcode::OP_SHMEM_PUT_UB2GM) {
        nonShmemDataIndex = 1;
        shmemDataIndex = GM2UB_SHMEMDATA_INDEX;
        shapeIndex = 1;
    }
    if (opCode == Opcode::OP_SHMEM_GET_GM2UB) {
        nonShmemDataIndex = 0;
        shmemDataIndex = GM2UB_SHMEMDATA_INDEX;
        shapeIndex = GM2UB_SHMEMDATA_INDEX;
    }

    const std::vector<int64_t>& tileShape = originShape[shapeIndex];
    int64_t tileRowShape = tileShape[tileShape.size() - 2];
    int64_t tileColShape = tileShape[tileShape.size() - 1];

    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

    int64_t bufferRowShape = distOpAttr.copyBufferShape[0];
    int64_t bufferColShape = distOpAttr.copyBufferShape[1];

    const std::vector<int64_t>& originTensorShape = rawShape[shapeIndex];
    int64_t stride = originTensorShape[originTensorShape.size() - 1];

    CheckInRange(tileRowShape);
    CheckInRange(tileColShape);
    CheckInRange(bufferRowShape);
    CheckInRange(bufferColShape);
    CheckInRange(stride);

    oss << "<" << DataType2CCEStr(operandDtype[nonShmemDataIndex]) << ", " << DataType2CCEStr(operandDtype[shmemDataIndex])
        << ", " << tileRowShape << ", " << tileColShape << ", " << bufferRowShape
        << ", " << bufferColShape << ", " << stride << ", " << stride << ", "
        << npu::tile_fwk::Distributed::AtomicTypeToString(distOpAttr.atomicType) << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForSignal() const
{
    std::ostringstream oss;
    int32_t shmemSignalIndex = 3;
    int64_t rankShape = originShape[shmemSignalIndex][0];
    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    oss << "<" << std::to_string(distOpAttr.signalValue) << ", " << npu::tile_fwk::Distributed::AtomicTypeToString(distOpAttr.atomicType) << ", " << rankShape << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForMoeCombineSend() const
{
    std::ostringstream oss;
    int32_t dataBufferIndex = 1;
    GenExtraTemplateParamsForMoeCombine(oss, dataBufferIndex);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForMoeCombineReceive() const
{
    std::ostringstream oss;
    int32_t outBufferIndex = 3;
    GenExtraTemplateParamsForMoeCombine(oss, outBufferIndex);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsForSet() const
{
    std::ostringstream oss;
    int32_t shmemTensorIndex = 3;
    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t bufferEleNum = distOpAttr.setBufferShape[0];
    int32_t rowDimIndex = 2;
    int32_t colDimIndex = 3;
    oss << "<" << GetTemplateDType() << ", " << originShape[shmemTensorIndex][1] << ", "
        << originShape[shmemTensorIndex][rowDimIndex] << ", " << originShape[shmemTensorIndex][colDimIndex]
        << ", " << bufferEleNum << ">";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTemplateParamsDefault() const
{
    std::ostringstream oss;
    DistOpAttr distOpAttr;
    if (opAttrs.count(OpAttributeKey::distOpAttr) != 0) {
        distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
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
        {Opcode::OP_SHMEM_MOE_COMBINE_SEND, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForMoeCombineSend(); }},
        {Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE, [](const CodeGenOpCloudNPU* self) { return self->GenTemplateParamsForMoeCombineReceive(); }},
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

std::string CodeGenOpCloudNPU::GenRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_RAW_SHAPE)[0];
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes(int32_t operandIndex, int32_t dim) const
{
    return GenOffsets(operandIndex, dim) + ", " + GenRawShapes(operandIndex, dim);
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemPutAndGet() const
{
    std::ostringstream oss;
    int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT) ? 2 : 0;
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
    int32_t shmemSignalDim = 4;
    oss << ", " << GenOffsetsAndRawShapes(shmemSignalIndex, shmemSignalDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemMoeCombineSend() const
{
    std::ostringstream oss;
    int32_t inIndex = 4;
    int32_t inDim = 2;
    oss << ", " << GenOffsets(inIndex, inDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesForShmemMoeCombineReceive() const
{
    std::ostringstream oss;
    int32_t shmemDataIndex = 6;
    int32_t shmemDataDim = 4;
    oss << ", " << GenOffsets(shmemDataIndex, shmemDataDim);
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
    int32_t shmemTensorIndex = 3;
    int32_t shmemTensorDim = 4;
    oss << ", " << GenOffsets(shmemTensorIndex, shmemTensorDim);
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapesDefault() const
{
    return "";
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes() const
{
    static const std::unordered_map<Opcode,
        std::function<std::string(CodeGenOpCloudNPU const*)>> offsetsAndRawShapesHandlers = {
        {Opcode::OP_SHMEM_PUT, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutAndGet(); }},
        {Opcode::OP_SHMEM_GET, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutAndGet(); }},
        {Opcode::OP_SHMEM_PUT_UB2GM, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutAndGetUB(); }},
        {Opcode::OP_SHMEM_GET_GM2UB, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemPutAndGetUB(); }},
        {Opcode::OP_SHMEM_SIGNAL, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemSignal(); }},
        {Opcode::OP_SHMEM_MOE_COMBINE_SEND, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemMoeCombineSend(); }},
        {Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE, [](const CodeGenOpCloudNPU* self) { return self->GenOffsetsAndRawShapesForShmemMoeCombineReceive(); }},
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
    auto it = skipIndexMap.find(opCode);
    if (it != skipIndexMap.end()) {
        skipOperands = it->second;
    }
    oss << tileOpName << GenTemplateParams() << "(" << GenParamsStr(skipOperands) << GenOffsetsAndRawShapes()
        << ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk

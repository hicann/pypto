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
    {Opcode::OP_SHMEM_REDUCE, {4}},
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
        case Opcode::OP_SHMEM_CLEAR_SIGNAL: {
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

void CodeGenOpCloudNPU::GenExtraTemplateParamsForPutAndGet(std::ostringstream& oss) const {
    int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT) ? 2 : 0;
    // 必须从 shmemData 取 shape，不能从 nonShmemData 取
    // 如果从 nonShmemData 取，ShmemGet 会取到 assemble 后的 shape，不符合预期
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
    
    const std::vector<int64_t>& tileShape = originShape[shapeIndex]; // originShape 是切块后的 shape
    int64_t tileRowShape = tileShape[tileShape.size() - 2];
    int64_t tileColShape = tileShape[tileShape.size() - 1];

    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

    int64_t bufferRowShape = distOpAttr.copyBufferShape[0];
    int64_t bufferColShape = distOpAttr.copyBufferShape[1];

    const std::vector<int64_t>& originTensorShape = rawShape[shapeIndex]; // rawShape 是切块前的 shape
    int64_t stride = originTensorShape[originTensorShape.size() - 1];

    CheckInRange(tileRowShape);
    CheckInRange(tileColShape);
    CheckInRange(bufferRowShape);
    CheckInRange(bufferColShape);
    CheckInRange(stride);

    oss << "<" << DataType2CCEStr(operandDtype[nonShmemDataIndex]) 
        << ", " << DataType2CCEStr(operandDtype[shmemDataIndex]) 
        << ", " << tileRowShape << ", " << tileColShape << ", " << bufferRowShape
        << ", " << bufferColShape << ", " << stride << ", " << stride << ", "
        << npu::tile_fwk::Distributed::AtomicTypeToString(distOpAttr.atomicType) << ">";
}

void CodeGenOpCloudNPU::GenExtraTemplateParamsForMoeCombine(std::ostringstream& oss, int32_t operandIndex) const {
    DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
    int64_t colShape = originShape[operandIndex][originShape[operandIndex].size() - 1];
    int64_t dataIndex = (opCode == Opcode::OP_SHMEM_MOE_COMBINE_SEND) ? 4 : 6;
    int64_t rowShape = originShape[dataIndex][originShape[dataIndex].size() - 2];
    oss << "<" << GetTemplateDType() << ", " << distOpAttr.topK << ", " << rowShape << ", " << colShape << ", "
        << distOpAttr.paddedColShape << ">";
}

std::string CodeGenOpCloudNPU::GenTemplateParams() const
{
    std::ostringstream oss;
    switch (opCode) {
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_GET: 
        case Opcode::OP_SHMEM_PUT_UB2GM:
        case Opcode::OP_SHMEM_GET_GM2UB:{
            GenExtraTemplateParamsForPutAndGet(oss);
            break;
        }
        case Opcode::OP_SHMEM_SIGNAL: {
            DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
            oss << "<" << std::to_string(distOpAttr.signalValue) << ", "
                << npu::tile_fwk::Distributed::AtomicTypeToString(distOpAttr.atomicType) << ">";
            break;
        }
        case Opcode::OP_SHMEM_REDUCE: {
            DistOpAttr distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));

            int32_t outIndex = 0;
            const std::vector<int64_t> outShape = rawShape[outIndex];
            int64_t row = outShape[0];
            int64_t col = outShape[1];

            oss << "<" << GetTemplateDType() << ", " << distOpAttr.extraTemplateParam << ", " << row << ", " << col
                << ">";
            break;
        }
        case Opcode::OP_SHMEM_MOE_COMBINE_SEND: {
            int32_t dataBufferIndex = 1;
            GenExtraTemplateParamsForMoeCombine(oss, dataBufferIndex);
            break;
        }
        case Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE: {
            int32_t outBufferIndex = 3;
            GenExtraTemplateParamsForMoeCombine(oss, outBufferIndex);
            break;
        }
        default: {
            DistOpAttr distOpAttr;
            if (opAttrs.count(OpAttributeKey::distOpAttr) != 0) {
                distOpAttr = npu::tile_fwk::AnyCast<DistOpAttr>(opAttrs.at(OpAttributeKey::distOpAttr));
            }
            if (distOpAttr.extraTemplateParam.empty()) {
                oss << "<" << GetTemplateDType() << ">";
            } else {
                oss << "<" << GetTemplateDType() << ", " << distOpAttr.extraTemplateParam << ">";
            }
            break;
        }
    }
    return oss.str();
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

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes() const
{
    std::ostringstream oss;
    switch (opCode) {
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_GET: {
            int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT) ? 2 : 0;
            int32_t shmemDataIndex = 3;
            int32_t nonShmemDataDim = originShape[nonShmemDataIndex].size();
            int32_t shmemDataDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim)
                << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
            break;
        }
        case Opcode::OP_SHMEM_PUT_UB2GM:
        case Opcode::OP_SHMEM_GET_GM2UB: {
            int32_t nonShmemDataIndex = (opCode == Opcode::OP_SHMEM_PUT_UB2GM) ? 1 : 0;
            int32_t shmemDataIndex = 2;
            int32_t nonShmemDataDim = 2;
            int32_t shmemDataDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim)
                << ", " << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
            break;
        }
        case Opcode::OP_SHMEM_CLEAR_SIGNAL:
        case Opcode::OP_SHMEM_SIGNAL: {
            int32_t shmemSignalIndex = (opCode == Opcode::OP_SHMEM_SIGNAL) ? 3 : 2;
            int32_t shmemSignalDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(shmemSignalIndex, shmemSignalDim);
            break;
        }
        case Opcode::OP_SHMEM_REDUCE: {
            int32_t outIndex = 0;
            int32_t outDim = 2;
            oss << ", " << GenOffsetsAndRawShapes(outIndex, outDim);
            break;
        }
        case Opcode::OP_SHMEM_MOE_COMBINE_SEND: {
            int32_t inIndex = 4;
            int32_t inDim = 2;
            oss << ", " << GenOffsets(inIndex, inDim);
            break;
        }
        case Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE: {
            int32_t shmemDataIndex = 6;
            int32_t shmemDataDim = 4;
            oss << ", " << GenOffsets(shmemDataIndex, shmemDataDim);
            break;
        }
        case Opcode::OP_SEND_TO_ROUTING_EXPERT: {
            int32_t expertTableIndex = 6;
            int32_t expertTableDim = 2;
            int32_t shmemDataIndex = 5;
            int32_t shmemDataDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(expertTableIndex, expertTableDim) << ", "
                << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
            break;
        }
        case Opcode::OP_SEND_TO_SHARED_EXPERT: {
            int32_t tokenIndex = 2;
            int32_t tokenDim = 2;
            int32_t shmemDataIndex= 3;
            int32_t shmemDataDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(tokenIndex, tokenDim) << ", "
                << GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
            break;
        }
        case Opcode::OP_COPY_TO_LOCAL_EXPERT: {
            int32_t tokenIndex = 3;
            int32_t tokenDim = 2;
            oss << ", " << GenOffsetsAndRawShapes(tokenIndex, tokenDim);
            break;
        }
        case Opcode::OP_DISPATCH_SET_FLAG: {
            int32_t shmemFlagIndex = 5;
            int32_t shmemFlagDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(shmemFlagIndex, shmemFlagDim);
            break;
        }
        case Opcode::OP_FFN_SCHED:
        case Opcode::OP_FFN_BATCHING:
        case Opcode::OP_FFN_VALIDCNT: {
            int32_t shmemIndex = 3;
            int32_t shmemDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(shmemIndex, shmemDim);
            break;
        }
        case Opcode::OP_FFN_COMBINEINFO: {
            int32_t shmemIndex = 2;
            int32_t shmemDim = 4;
            oss << ", " << GenOffsetsAndRawShapes(shmemIndex, shmemDim);
            break;
        }
        default: {
            break;
        }
    }
    return oss.str();
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

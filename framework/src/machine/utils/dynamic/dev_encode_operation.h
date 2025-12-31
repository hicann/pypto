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
 * \file dev_encode_operation.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "interface/tensor/runtime_slot.h"

namespace npu::tile_fwk::dynamic {
struct DevAscendOperationOperandInfo {
    int tensorIndex{0};
    int staticOffsetAttrBeginIndex{0};
    int staticShapeAttrBeginIndex{0};
    int staticRawShapeAttrBeginIndex{0};

    DevAscendOperationOperandInfo() {}
    DevAscendOperationOperandInfo(int tTensorIndex, int tStaticAttrBeginIndex, int tStaticDim)
        : tensorIndex(tTensorIndex),
          staticOffsetAttrBeginIndex(tStaticAttrBeginIndex),
          staticShapeAttrBeginIndex(tStaticAttrBeginIndex + tStaticDim),
          staticRawShapeAttrBeginIndex(staticShapeAttrBeginIndex + tStaticDim) {}
    int GetDim() const { return staticShapeAttrBeginIndex - staticOffsetAttrBeginIndex; }
};

struct DevAscendOperation {
    DevLocalVector<DevAscendOperationOperandInfo> ioperandList;
    DevLocalVector<DevAscendOperationOperandInfo> ooperandList;
    DevLocalVector<SymInt> attrList; // opattr[0] -> hash
    int32_t outcastStitchIndex;
    uint32_t depGraphPredCount;
    DevLocalVector<int> depGraphSuccList;
    DevLocalVector<int> depGraphCopyOutResolveSuccIndexList;
    uint64_t debugOpmagic; // DEBUG_ONLY
};

struct DevAscendFunctionCallOperandUse {
    int operationIdx{-1};
    int operandIdx{-1};
    int offsetAttrIdx{-1};
    int shapeAttrIdx{-1};

    DevAscendFunctionCallOperandUse() = default;
    DevAscendFunctionCallOperandUse(int operationIdx_, int operandIdx_, int offsetAttrIdx_, int shapeAttrIdx_)
        : operationIdx(operationIdx_), operandIdx(operandIdx_), offsetAttrIdx(offsetAttrIdx_), shapeAttrIdx(shapeAttrIdx_) {}
};

struct DevAscendFunctionIncast {
    int tensorIndex;
    DevLocalVector<int> fromSlotList;

    int dim;
    int stitchByAllFullMatch;
    DevLocalVector<DevAscendFunctionCallOperandUse> consumerList;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevLocalVector<uint32_t> cellMatchStaticIncastTable;

    DevLocalVector<uint32_t> stitchPolicyFullCoverConsumerAllOpIdxList;
};

struct DevAscendFunctionOutcast {
    int tensorIndex;
    DevLocalVector<int> toSlotList;

    int dim;
    int stitchByAllFullMatch;
    RuntimeSlotDesc desc;

    DevLocalVector<DevAscendFunctionCallOperandUse> producerList;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevLocalVector<uint32_t> cellMatchStaticOutcastTable;
    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTable;

    int stitchPolicyFullCoverProducerHubOpIdx;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    DevLocalVector<uint32_t> stitchPolicyFullCoverProducerAllOpIdxList;
    int exprListIndex;
};

struct InoutOperationAttr {
    int dim;
    std::vector<DevAscendFunctionCallOperandUse> useList;
    int cellMatchSize;

    std::vector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    int stitchPolicyFullCoverProducerHubOpIdx;

    DevCellMatchTableDesc cellMatchTableDesc;

    std::vector<uint32_t> useOpList;
    int bindTensorExprIndex{-1};
};

struct DevAscendFunctionDuppedOperation {
    uint32_t size;
    uint32_t predCountBase;
    uint32_t stitchBase;
    uint32_t stitchCount;
};

struct DevAscendFunctionDuppedVector {
    uint32_t size;
    uint32_t base;
};
}
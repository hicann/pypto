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
 * \file gather_after_prolog.cpp
 * \brief
 */

#include "gather_after_prolog.h"
 
using namespace npu::tile_fwk;
 
namespace npu::tile_fwk {
Tensor CalcOffsetsForGather(const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &actSeqs, const DSIASimpleParams &params, SymbolicScalar b, SymbolicScalar s1) {
    auto n2 = 1; // topKIndcies->shape[2]; // n2
    int blockSize = params.blockSize;
    int topk = params.topk;
    auto maxBlockNumPerBatch = GetInputShape(blockTable, 1);
    constexpr int64_t maxBatch = 128;
    constexpr int64_t maxS1 = 4;

    Tensor offsets(DT_INT32, {maxBatch * maxS1, n2 * params.topk}, "offsets");
    LOOP("gather_and_matmul", FunctionType::DYNAMIC_LOOP, idx, LoopRange(b * s1)) {
        // ASSERT(s1 != 0) << "s1 can`t be zero!";
        auto bIdx = idx / s1;
        auto s1Idx = idx % s1;
        config::SetSemanticLabel("calc_offset");
        SymbolicScalar curKvSeq = std::max(GetTensorData(actSeqs, {bIdx}) - s1 + 1 + s1Idx, 0);
        SymbolicScalar topkLoop = std::min(curKvSeq, topk); // for MTP s1!= 1 casual计算, 并且与topk取min
        constexpr int32_t TILE_0 = 1;
        constexpr int32_t TILE_1 = 256;
        TileShape::Current().SetVecTile(TILE_0, TILE_1); // vector运算存在bug，{1, 8192*2}会导致精度错误
        auto topKIndciesReshape = View(topKIndcies, {1, n2 * topk}, {1, topkLoop}, {idx, 0});
        auto topKIndciesReshapeFP32 = Cast(topKIndciesReshape, DT_FP32);
        topKIndciesReshapeFP32 = Add(topKIndciesReshapeFP32, Element(DT_FP32, 0.5f));
        auto blockIdxInBatchsFP32 = Div(topKIndciesReshapeFP32, Element(DT_FP32, static_cast<float>(blockSize)));
        auto blockIdxInBatchs = Cast(blockIdxInBatchsFP32, DT_INT32, CAST_FLOOR);
        auto tails = Sub(topKIndciesReshape, Mul(blockIdxInBatchs, Element(DT_INT32, blockSize)));
        auto blockTableRawOffsets = Full(bIdx * maxBlockNumPerBatch, DT_INT32, {1, n2 * topk}, {1, topkLoop});
        auto slcBlockIdxs = Load(blockTable, Add(blockTableRawOffsets, blockIdxInBatchs));
        auto blockOffsets = Mul(slcBlockIdxs, Element(DT_INT32, blockSize));
        auto offset = Add(blockOffsets, tails);
        Assemble(offset, {idx, 0}, offsets);
    }
    return offsets;
}

void GatherAfterPrologCompute(Tensor &topKIndcies, Tensor &kNopeCache, Tensor &kRopeCache, Tensor &blockTable, Tensor &actSeqs,
    Tensor &gatherRes, const DSIASimpleParams &params, SymbolicScalar b, SymbolicScalar s1) {
    int dN = kNopeCache.GetShape()[kNopeCache.GetShape().size() - 1];
    int dR = kRopeCache.GetShape()[kRopeCache.GetShape().size() - 1];
    int n2 = topKIndcies.GetShape()[2]; // n2
    int blockSize = params.blockSize;
    int topk = params.topk;
 
    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};
 
    LOOP("loop_b_gather", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
        LOOP("loop_s1_gather", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1, 1)) {
            LOOP("loop_n2_gather", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2, 1)) {
                config::SetSemanticLabel("gather0");
                SymbolicScalar curKvSeq = GetTensorData(actSeqs, {bIdx});
                SymbolicScalar topkLoop = std::min(std::max(curKvSeq - s1 + 1 + s1Idx, 0), topk); // for MTP s1!= 1 casual计算, 并且与topk取min
                LOOP("loop_k_gather", FunctionType::DYNAMIC_LOOP, topKIdx, LoopRange(0, topkLoop, 1), unrollList) {
                    SymbolicScalar topkIndex;
                    TileShape::Current().SetVecTile(1, 1, 1, NUM16);
#if DSIA_DEBUG == 1
                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, n2Idx, topKIdx});
#else
                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, n2Idx, topKIdx});
#endif
                    SymbolicScalar blockIdxInBatch = topkIndex / blockSize;
                    SymbolicScalar tail = topkIndex % blockSize;
                    SymbolicScalar slcBlockIdx = GetTensorData(blockTable, {bIdx, blockIdxInBatch});
                    TileShape::Current().SetVecTile(1, dN);
                    auto kvSlcBlock = View(kNopeCache, {1, dN}, {slcBlockIdx * blockSize + tail, 0});
                    auto krSlcBlock = View(kRopeCache, {1, dR}, {slcBlockIdx * blockSize + tail, 0});
 
                    config::SetSemanticLabel("gather1");
                    auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                    auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                    config::SetSemanticLabel("gather2");
                    auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, gatherRes.GetDataType());
                    auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, gatherRes.GetDataType());
                    SymbolicScalar ofs =
                        bIdx * s1 * n2 * topk + s1Idx * n2 * topk + n2Idx * topk + topKIdx;
 
                    Assemble(kvSlcBlock_fp16, {ofs, 0}, gatherRes);
                    Assemble(krSlcBlock_fp16, {ofs, dN}, gatherRes);
                }
            }
        }
    }
}
 
} // namespace npu::tile_fwk
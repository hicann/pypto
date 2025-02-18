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
 * \file kv_compress.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "tilefwk/tilefwk.h"
#include "interface/utils/common.h"
#include "kv_compress.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void compressKv(const Tensor &kvCache, const Tensor &krCache, const Tensor &cmpKvCache, const Tensor &cmpKrCache,
    const Tensor &blockTable, Tensor &cmpCacheIndex, const Tensor &actSeqLen, const Tensor &mlpWk1,
    const Tensor &mlpWk2, const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpKvCacheOut, Tensor &cmpKrCacheOut,
    Tensor &auxTensor, const int cmpBlockSize, const int cmpStride, const int rs, CmpAttnTile &tileConfig) {
    /* bellows are function params support
    kvCache: [blockNum * blockSize, n2 * dN], fp16/bf16
    krCache: [blockNum * blockSize, n2 * dR], fp16/bf16
    cmpKvCache: [cmpBlockNum, blockSize, n2, dN], fp16/bf16
    cmpKrCache: [cmpBlockNum, blockSize, n2, dR], fp16/bf16
    blockTable: [b, maxBlockNum], int32
    cmpCacheIndex: [b, s1], int32
    actSeqLen: [b], int32
    mlpWk1: [cmpBlockSize*dK, 2*cmpBlockSize*dK], fp16/bf16
    mlpWk2: [2*cmpBlockSize*dK, dK], fp16/bf16
    mlpCos: [b, cmpBlockSize, dr], fp16/bf16
    mlpSin: [b, cmpBlockSize, dR], fp16/bf16
    kvCacheCmpOut: [cmpBlockNum*blockSize, n2*dN], fp16/bf16
    krCacheCmpOut: [cmpBlockNum*blockSize, n2*dR], fp16/bf16
    */
    FUNCTION("CompressKv",
        {kvCache, krCache, cmpKvCache, cmpKrCache, blockTable, cmpCacheIndex, actSeqLen, mlpWk1, mlpWk2, mlpCos,
            mlpSin},
        {cmpKvCacheOut, cmpKrCacheOut, auxTensor}) {
        auto kDtype = kvCache.GetStorage()->Datatype();

        const int b = cmpCacheIndex.GetShape()[NUM_VALUE_0];
        const int s1 = cmpCacheIndex.GetShape()[SHAPE_DIM1];
        const int n2 = cmpKvCache.GetShape()[SHAPE_DIM2];
        const int dN = cmpKvCache.GetShape()[SHAPE_DIM3];
        const int dR = cmpKrCache.GetShape()[SHAPE_DIM3];
        const int blockSize = cmpKvCache.GetShape()[SHAPE_DIM1];
        const int cmpBlockNum = cmpKvCache.GetShape()[NUM_VALUE_0];
        const int rc = auxTensor.GetShape()[NUM_VALUE_0] - rs;
        const int auxVecLen = auxTensor.GetShape()[SHAPE_DIM1];

        ASSERT(n2 == 1);

        ASSERT(cmpBlockSize == NUM_32);
        ASSERT(cmpStride == NUM_16);

        Tensor batchConcatNR(kDtype, {b, cmpBlockSize, dN + dR}, "batchConcatNR");

        LOOP("BEFORE_KV_COMPRESS", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b), {}, true) {
            // Construct align-shape tensors for store dynamic seqs
            auto curKvLen = GetTensorData(actSeqLen, {bIdx});
            auto t1 = curKvLen % cmpStride == 0;
            auto t2 = curKvLen >= cmpBlockSize;
            auto blockStartIdx = (curKvLen - cmpBlockSize) / blockSize;
            auto blockEndIdx = (curKvLen - 1) / blockSize;
            auto blockStartOffset = (curKvLen - cmpBlockSize) % blockSize;
            auto tableLoop = blockEndIdx - blockStartIdx + 1;
            IF(t1 * t2) {}
            ELSE {
                blockStartIdx = 0;
                blockEndIdx = 0;
                blockStartOffset = 0;
                tableLoop = 1;
            }

            Tensor kNopeBlock(kDtype, {cmpBlockSize, dN}, "kNopeBlock");
            Tensor kRopeBlock(kDtype, {cmpBlockSize, dR}, "kRopeBlock");
            config::SetSemanticLabel("BlockConcat");
            IF(tableLoop == 1) {
                auto blockIdx = GetTensorData(blockTable, {bIdx, blockStartIdx});
                kNopeBlock = View(kvCache, {cmpBlockSize, dN}, {blockIdx * blockSize + blockStartOffset, 0});
                kRopeBlock = View(krCache, {cmpBlockSize, dR}, {blockIdx * blockSize + blockStartOffset, 0});
            }
            ELSE { // tableLoop == 2
                auto blockIdx0 = GetTensorData(blockTable, {bIdx, blockStartIdx});
                auto kNopeBlock0 =
                    View(kvCache, {cmpBlockSize / 2, dN}, {(blockIdx0 + 1) * blockSize - cmpBlockSize / 2, 0});
                auto kRopeBlock0 =
                    View(krCache, {cmpBlockSize / 2, dR}, {(blockIdx0 + 1) * blockSize - cmpBlockSize / 2, 0});
                auto blockIdx1 = GetTensorData(blockTable, {bIdx, blockStartIdx + 1});
                auto kNopeBlock1 = View(kvCache, {cmpBlockSize / 2, dN}, {blockIdx1 * blockSize, 0});
                auto kRopeBlock1 = View(krCache, {cmpBlockSize / 2, dR}, {blockIdx1 * blockSize, 0});
                TileShape::Current().SetVecTile(cmpBlockSize, dN);
                kNopeBlock = Cat({kNopeBlock0, kNopeBlock1}, 0);
                TileShape::Current().SetVecTile(cmpBlockSize, dR);
                kRopeBlock = Cat({kRopeBlock0, kRopeBlock1}, 0);
            }

            TileShape::Current().SetVecTile(cmpBlockSize, dN);
            kNopeBlock = Reshape(kNopeBlock, {1, cmpBlockSize, dN});

            // LocalRope
            TileShape::Current().SetVecTile(1, NUM_32, NUM_64);
            config::SetSemanticLabel("MlpLocalRope");
            auto cosTmp = View(mlpCos, {1, cmpBlockSize, dR}, {bIdx, 0, 0});
            auto sinTmp = View(mlpSin, {1, cmpBlockSize, dR}, {bIdx, 0, 0});
            auto kRopeEmbed = BatchMlpSingleRope(kRopeBlock, cosTmp, sinTmp, tileConfig.mlpRopeTile);

            config::SetSemanticLabel("MlpCompress");
            Assemble(kNopeBlock, {bIdx, 0, 0}, batchConcatNR);
            Assemble(kRopeEmbed, {bIdx, 0, dN}, batchConcatNR);
        }

        Tensor batchMlpCompressResult(kDtype, {b, dN + dR}, "batchMlpCompressResult");
        LOOP("KV_COMPRESS", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1), {}, true) {
            (void)bIdx;
            auto reshapeNR = Reshape(batchConcatNR, {b, cmpBlockSize * (dN + dR)});
            batchMlpCompressResult =
                BatchMlpCompress(reshapeNR, mlpWk1, mlpWk2, tileConfig.mlpCmpTile); // (b, n2, dN + dR)

            TileShape::Current().SetVecTile(1, NUM_32, NUM_64);
        }

        Tensor batchNopeResult(kDtype, {b, dN}, "batchNopeResult");
        Tensor batchRopeResult(kDtype, {b, dR}, "batchRopeResult");
        LOOP("AFTER_KV_COMPRESS", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b), {}, true) {
            // Construct align-shape tensors for store dynamic seqs
            auto curKvLen = GetTensorData(actSeqLen, {bIdx});
            auto t1 = curKvLen % cmpStride == 0;
            auto t2 = curKvLen >= cmpBlockSize;
            TileShape::Current().SetVecTile(NUM_32, NUM_64);
            Tensor kNopeCmp(kDtype, {1, dN});
            Tensor kRopeCmp(kDtype, {1, dR});
            IF(t1 * t2) {
                kNopeCmp = View(batchMlpCompressResult, {1, dN}, {bIdx, 0});
                kRopeCmp = View(batchMlpCompressResult, {1, dR}, {bIdx, dN});
            }
            ELSE {
                auto cmpKvCacheDim2 = Reshape(cmpKvCache, {cmpBlockNum * blockSize * n2, dN});
                auto cmpKrCacheDim2 = Reshape(cmpKrCache, {cmpBlockNum * blockSize * n2, dR});
                auto index = GetTensorData(cmpCacheIndex, {bIdx, s1 - 1});
                kNopeCmp = View(cmpKvCacheDim2, {1, dN}, {index, 0});
                kRopeCmp = View(cmpKrCacheDim2, {1, dR}, {index, 0});
            }

            if (kDtype == DT_BF16) {
                kNopeCmp = Cast(kNopeCmp, DT_FP32);
                kRopeCmp = Cast(kRopeCmp, DT_FP32);
                kNopeCmp = Cast(kNopeCmp, DT_BF16);
                kRopeCmp = Cast(kRopeCmp, DT_BF16);
            } else {
                kNopeCmp = Add(kNopeCmp, Element(DT_FP32, 0.0f));
                kRopeCmp = Add(kRopeCmp, Element(DT_FP32, 0.0f));
            }
            Assemble(kNopeCmp, {bIdx, 0}, batchNopeResult);
            Assemble(kRopeCmp, {bIdx, 0}, batchRopeResult);
        }

        LOOP("UPDATE_CMP_KV_CACHE", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1), {}, true) {
            (void)bIdx;
            auto cmpKvCacheDim2 = Reshape(cmpKvCache, {cmpBlockNum * blockSize * n2, dN});
            auto cmpKrCacheDim2 = Reshape(cmpKrCache, {cmpBlockNum * blockSize * n2, dR});
            TileShape::Current().SetVecTile(NUM_32, NUM_64);
            auto index = Reshape(View(cmpCacheIndex, {b, 1}, {0, s1 - 1}), {1, b});
            cmpKvCacheOut =
                Reshape(ScatterUpdate(cmpKvCacheDim2, index, batchNopeResult, 0), {cmpBlockNum, blockSize, 1, dN});
            cmpKrCacheOut =
                Reshape(ScatterUpdate(cmpKrCacheDim2, index, batchRopeResult, 0), {cmpBlockNum, blockSize, 1, dR});
            for (int i = 0; i < rs + rc - 1; i++) {
                auto auxVector = npu::tile_fwk::Full(
                    Element(DT_FP32, float(std::min(i + 1, rc) - std::max(i - rs, 0))), DT_FP32, {1, auxVecLen});
                Assemble(auxVector, {i, 0}, auxTensor);
            }
        }
    }
}

} // namespace npu::tile_fwk

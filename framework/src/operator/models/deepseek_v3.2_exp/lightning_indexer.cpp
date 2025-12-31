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
 * \file lightning_indxer.cpp
 * \brief
 */

#include "lightning_indexer.h"
#include <cfloat>

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void LightningIndexerTopkQuant(const Tensor &query, const Tensor &key, const Tensor &qScale, const Tensor &kScale,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes, const int selectedCount,
    IndexerTile tileConfig, const std::set<int> &unrollList) {
    LightningIndexerTopkImpl(query, key, true, &qScale, &kScale, weights, actSeqKey, blockTable, topkRes, selectedCount,
        tileConfig, unrollList);
}

void LightningIndexerTopk(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    const std::set<int> &unrollList) {
    LightningIndexerTopkImpl(query, key, false, nullptr, nullptr, weights, actSeqKey, blockTable, topkRes,
        selectedCount, tileConfig, unrollList);
}

void LightningIndexerTopkImpl(const Tensor &query, const Tensor &key, bool isQuant, const Tensor *qScale,
    const Tensor *kScale, const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes,
    const int selectedCount, IndexerTile tileConfig, const std::set<int> &unrollList, Tensor *tmpOut,
    Tensor *topkValue) {
    /*
    <no quant>
    query: [B, S1, indexN1, indexD], bf16
    key: [blockNum, blockSize, n2, indexD] bf16

    <quant>
    query: [B, S1, indexN1, indexD], int8
    key: [blockNum, blockSize, n2, indexD] int8
    qScale: [B, S1, indexN1, 1], fp16
    kScale: [blockNum, blockSize, n2, 1] fp16

    <common>
    weights: [B, S1, indexN1], bf16
    actSeqKey: [B], int32
    blockTable: [B, maxBlockNum]
    topkRes: [B, s1, N2, selectedCount], int32
    selectedCount: selectedCount num
    */

    // Symbolization
    SymbolicScalar b = GetInputShape(query, 0);
    SymbolicScalar s1 = GetInputShape(query, 1);
    SymbolicScalar blockNum = GetInputShape(key, 0);

    auto indexN1 = query.GetStorage()->shape[SHAPE_DIM2];
    auto indexD = query.GetStorage()->shape[SHAPE_DIM3];
    auto blockSize = key.GetStorage()->shape[1];
    auto n2 = key.GetStorage()->shape[SHAPE_DIM2];
    auto qkDType = query.GetStorage()->Datatype();
    auto scaleDType = isQuant ? (*qScale).GetStorage()->Datatype() : DT_FP16;
    auto wDType = weights.GetStorage()->Datatype();
    auto group = indexN1 / n2; // 暂时考虑整除
    auto c1Tile = tileConfig.c1Tile;
    constexpr int64_t maxS2 = 128 * 1024;

    Tensor query2D(qkDType, {b * s1 * indexN1, indexD}, "query2D");
    Tensor key2D(qkDType, {blockNum * blockSize, n2 * indexD}, "key2D");
    Tensor qScale2D(scaleDType, {b * s1 * indexN1, 1}, "qScale2D");
    Tensor kScale2D(scaleDType, {blockNum * blockSize, n2}, "kScale2D");
    Tensor weight2D(wDType, {b * s1 * indexN1, 1}, "weight2D");

    constexpr int32_t NUMS_2048 = 2048;
    ASSERT(selectedCount == NUMS_2048);

    LOOP("INPUT_4D_2_2D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        query2D = Reshape(query, {b * s1 * indexN1, indexD}, true);
        key2D = Reshape(key, {blockNum * blockSize, n2 * indexD}, true);
        weight2D = Reshape(weights, {b * s1 * indexN1, 1}, true);
        if (isQuant) {
            qScale2D = Reshape(*qScale, {b * s1 * indexN1, 1}, true);
            kScale2D = Reshape(*kScale, {blockNum * blockSize, n2}, true);
        }
    }

    LOOP("INDEX_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b)) {
        auto curSeq = GetTensorData(actSeqKey, {bIdx});

        LOOP("INDEX_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1)) {
            // 因果推理
            auto casualOffset = s1 - s1Idx - 1;
            auto effSeq = curSeq - casualOffset;
            auto actBlock = (effSeq + blockSize - 1) / blockSize;
            LOOP("INDEX_LOOP_N2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(n2)) {
                auto bs1n2Offset = bIdx * s1 * n2 + s1Idx * n2 + n2Idx;
                auto qOffset = bIdx * s1 * indexN1 + s1Idx * indexN1 + n2Idx * group;

                Tensor localSum(DT_FP32, {1, maxS2}, "localSum");

                // unrolling process template
                auto unrollingProcess = [&](int unrollLength, auto &&firstBlockIdx) {
                    // TileShape::Current().SetVecTile(128, 128);
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0}); // (group, d)
                    std::vector<Tensor> concatSrcs;
                    // static unrolling
                    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});

                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                        auto mmRes = Matrix::Matmul<false, true>(DataType::DT_FP32, curQ, curK); // (group, blockSize)
                        concatSrcs.emplace_back(mmRes); // Matches tile size, no extra Assign needed (?)
                    }

                    TileShape::Current().SetVecTile(tileConfig.weightTile);

                    auto curW = View(weight2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    auto wB32 = Cast(curW, DT_FP32);                      // (group, 1)

                    auto mmRes = Cat(concatSrcs, -1); // (group, superBlockSize)

                    TileShape::Current().SetVecTile(tileConfig.v1Tile);
                    auto reluRes = Maximum(mmRes, Element(DT_FP32, 0.0f)); // (group, superBlockSize)
                    auto mulRes = Mul(reluRes, wB32); // (group, superBlockSize) * (group, 1) -> (group, superBlockSize)
                    auto sumRes = Sum(mulRes, 0, true); // (1, superBlockSize)
                    Assemble(sumRes, {0, firstBlockIdx * blockSize}, localSum);
                    if (tmpOut != nullptr) {
                        // tmpOut: [B*S1*N2, S2]
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, *tmpOut);
                    }
                };

                auto unrollingProcessQuant = [&](int unrollLength, auto &&firstBlockIdx) {
                    // TileShape::Current().SetVecTile(128, 128);
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0});    // (group, d)
                    Tensor curQScale = View(qScale2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    std::vector<Tensor> mmResQuantConcatSrcs;
                    std::vector<Tensor> kScaleConcatSrcs;

                    // static unrolling
                    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});

                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);

                        auto mmRes = Matrix::Matmul<false, true>(DataType::DT_INT32, curQ, curK); // (group, blockSize)
                        mmResQuantConcatSrcs.emplace_back(mmRes);

                        auto curKScale =
                            View(kScale2D, {blockSize, 1}, {std::min(blockSize, effSeq - (blockIdx * blockSize)), 1},
                                {curBlockIdx * blockSize, n2Idx}); // (blockSize, 1)
                        kScaleConcatSrcs.emplace_back(curKScale);
                    }

                    TileShape::Current().SetVecTile(tileConfig.weightTile);

                    auto curW = View(weight2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    auto wF16 = Cast(curW, DT_FP16);                      // (group, 1)

                    TileShape::Current().SetVecTile(tileConfig.v1Tile);
                    auto curKScale = Cat(kScaleConcatSrcs, 0);
                    auto mmResI32 = Cat(mmResQuantConcatSrcs, -1); // (group, superBlockSize)
                    auto mmResFP32 = Mul(Cast(mmResI32, DT_FP32), Element(DT_FP32, AVOID_FP32_TO_FP16_OVERFLOW_SCALE));
                    auto mmResFP16 = Cast(mmResFP32, DT_FP16);
                    auto mmResDequant = Mul(Mul(mmResFP16, curQScale), Transpose(curKScale, {0, 1}));
                    auto reluRes = Maximum(mmResDequant, Element(DT_FP16, 0.0f)); // (group, superBlockSize)
                    auto mulRes = Mul(reluRes, wF16); // (group, superBlockSize) * (group, 1) -> (group, superBlockSize)

                    // RowSumSingle doesn't support non-4-byte types currently
                    auto sumRes = Sum(Cast(mulRes, DT_FP32), 0, true); // (1, superBlockSize)
                    Assemble(sumRes, {0, firstBlockIdx * blockSize}, localSum);
                    if (tmpOut != nullptr) {
                        // tmpOut: [B*S1*N2, S2]
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, *tmpOut);
                    }
                };

                LOOP("INDEX_LOOP_MATMUL", FunctionType::DYNAMIC_LOOP, loopBlockIdx, LoopRange(actBlock), unrollList) {
                    for (int loopUnrollLength : unrollList) {
                        UNROLL(loopUnrollLength) {
                            if (isQuant) {
                                unrollingProcessQuant(loopUnrollLength, loopBlockIdx);
                            } else {
                                unrollingProcess(loopUnrollLength, loopBlockIdx);
                            }
                        }
                    }
                }

                LOOP("LOOP_TEST", FunctionType::DYNAMIC_LOOP, loopBlockIdx, LoopRange(1), {}, true) {
                    (void)loopBlockIdx;
                    DataType xdtype = localSum.GetDataType();
                    DataType idxdtype = topkRes.GetDataType();
                    constexpr int padIdxValue = -1;
                    constexpr int tileSize = 8192;
                    constexpr bool descending = true;
                    constexpr float padValue = descending ? -FLT_MAX : FLT_MAX;
                    constexpr int length2K = NUMS_2048;
                    constexpr int length8K = 1024 * 8;
                    constexpr int length64K = 1024 * 64;
                    constexpr int length128K = maxS2;

                    auto lengthIsLE2K = effSeq <= length2K;
                    auto lengthIsGT2K = effSeq > length2K;
                    TileShape::Current().SetVecTile({1, tileSize});
                    LOOP("2K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsLE2K)) {
                        (void)unused;
                        Tensor padX2K(xdtype, {1, length2K}, "padX2K");
                        config::SetPassOption(PG_SKIP_PARTITION, true);
                        LOOP("2K_PAD", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                            (void)unused1;
                            TileShape::Current().SetVecTile({1, length2K});
                            auto ax = View(localSum, {1, length2K}, {1, effSeq}, {0, 0});
                            auto bx = Full(Element(xdtype, padValue), xdtype, {1, length2K}, {1, length2K - effSeq});
                            Assemble(Assign(ax), {0, 0}, padX2K);
                            Assemble(bx, {0, effSeq}, padX2K);
                        }
                        config::SetPassOption(PG_SKIP_PARTITION, false);
                        LOOP("2K_TOPK", FunctionType::DYNAMIC_LOOP, unused2, LoopRange(1)) {
                            (void)unused2;
                            auto [resValue, resIdx] = TopK(View(padX2K, {1, length2K}, {0, 0}), selectedCount, 1);
                            TileShape::Current().SetVecTile(tileConfig.addsTile);
                            auto topk4D = Reshape(
                                View(resIdx, {1, selectedCount}, {1, effSeq}, {0, 0}), {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                            Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                            auto topkIndicesPad = Full(Element(idxdtype, padIdxValue), idxdtype, {1, 1, 1, selectedCount},
                                {1, 1, 1, selectedCount - effSeq});
                            Assemble(topkIndicesPad, {bIdx, s1Idx, n2Idx, effSeq}, topkRes);

                            if (topkValue != nullptr) {
                                auto topk4DValue = Reshape(View(resValue, {1, selectedCount}, {1, effSeq}, {0, 0}),
                                    {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                                Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                                auto topkValuePad = Full(Element(DT_FP32, padValue), DT_FP32, {1, 1, 1, selectedCount},
                                    {1, 1, 1, selectedCount - effSeq});
                                Assemble(topkValuePad, {bIdx, s1Idx, n2Idx, effSeq}, *topkValue);
                            }
                            TileShape::Current().SetVecTile({1, tileSize});
                        }
                    }

                    auto lengthIsLE8K = effSeq <= length8K;
                    auto lengthIsGT8K = effSeq > length8K;
                    LOOP("8K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT2K * lengthIsLE8K)) {
                        UNUSED(unused);
                        Tensor padX8K(xdtype, {1, length8K}, "padX8K");
                        config::SetPassOption(PG_SKIP_PARTITION, true);
                        LOOP("8K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                            UNUSED(unused0);
                            TileShape::Current().SetVecTile({1, tileSize});
                            auto ax = View(localSum, {1, length8K}, {1, effSeq}, {0, 0});
                            auto bx = Full(Element(xdtype, padValue), xdtype, {1, length8K}, {1, length8K - effSeq});
                            Assemble(Assign(ax), {0, 0}, padX8K);
                            Assemble(bx, {0, effSeq}, padX8K);
                        }
                        config::SetPassOption(PG_SKIP_PARTITION, false);

                        TileShape::Current().SetVecTile({1, tileSize});
                        LOOP("8K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                            UNUSED(unused1);
                            auto [resValue, resIdx] = TopK(View(padX8K, {1, length8K}, {0, 0}), selectedCount, 1);
                            TileShape::Current().SetVecTile(tileConfig.addsTile);
                            auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                            Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                            if (topkValue != nullptr) {
                                TileShape::Current().SetVecTile(tileConfig.addsTile);
                                auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                                Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                            }
                            TileShape::Current().SetVecTile({1, tileSize});
                        }
                    }

                    auto lengthIsLE64K = effSeq <= length64K;
                    auto lengthIsGT64K = effSeq > length64K;
                    LOOP("64K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT8K * lengthIsLE64K)) {
                        UNUSED(unused);
                        Tensor padX64K(xdtype, {1, length64K}, "padX64K");
                        config::SetPassOption(PG_SKIP_PARTITION, true);
                        LOOP("64K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                            UNUSED(unused0);
                            TileShape::Current().SetVecTile({1, tileSize});
                            auto ax = View(localSum, {1, length64K}, {1, effSeq}, {0, 0});
                            auto bx = Full(Element(xdtype, padValue), xdtype, {1, length64K}, {1, length64K - effSeq});
                            Assemble(Assign(ax), {0, 0}, padX64K);
                            Assemble(bx, {0, effSeq}, padX64K);
                        }
                        config::SetPassOption(PG_SKIP_PARTITION, false);

                        TileShape::Current().SetVecTile({1, tileSize});
                        LOOP("64K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                            UNUSED(unused1);
                            auto [resValue, resIdx] = TopK(View(padX64K, {1, length64K}, {0, 0}), selectedCount, 1);
                            TileShape::Current().SetVecTile(tileConfig.addsTile);
                            auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                            Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                            if (topkValue != nullptr) {
                                TileShape::Current().SetVecTile(tileConfig.addsTile);
                                auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                                Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                            }
                            TileShape::Current().SetVecTile({1, tileSize});
                        }
                    }

                    LOOP("128K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT64K)) {
                        UNUSED(unused);
                        Tensor padX128K(xdtype, {1, length128K}, "padX128K");
                        config::SetPassOption(PG_SKIP_PARTITION, true);
                        LOOP("128K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                            UNUSED(unused0);
                            TileShape::Current().SetVecTile({1, tileSize});
                            auto ax = View(localSum, {1, length128K}, {1, effSeq}, {0, 0});
                            auto bx = Full(Element(xdtype, padValue), xdtype, {1, length128K}, {1, length128K - effSeq});
                            Assemble(Assign(ax), {0, 0}, padX128K);
                            Assemble(bx, {0, effSeq}, padX128K);
                        }
                        config::SetPassOption(PG_SKIP_PARTITION, false);

                        TileShape::Current().SetVecTile({1, tileSize});
                        LOOP("128K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                            UNUSED(unused1);
                            auto [resValue, resIdx] = TopK(View(padX128K, {1, length128K}, {0, 0}), selectedCount, 1);
                            TileShape::Current().SetVecTile(tileConfig.addsTile);
                            auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                            Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                            if (topkValue != nullptr) {
                                TileShape::Current().SetVecTile(tileConfig.addsTile);
                                auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                                Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                            }
                            TileShape::Current().SetVecTile({1, tileSize});
                        }
                    }
                }
            }
        }
    }
}

void LightningIndexerImpl(const Tensor &idxQuery, const Tensor &idxQueryScale, const Tensor &idxKeyCache,
    const Tensor &idxKeyScale, const Tensor &idxWeight, const Tensor &actSeqKey, const Tensor &blockTable,
    const int selectedCount, Tensor &topkRes, LightningIndexerConfigs configs, const std::set<int> &unrollList,
    Tensor *firstMm, Tensor *mmOut, Tensor *topkValue) {
    /*
    idxQuery: [t, idxNHeads, indexD], int8
    idxQueryScale: [t, idxNHeads], fp16
    idxKeyCache: [blockNum, blockSize, 1, indexD], int8
    idxKeyScale: [blockNum, blockSize, 1], fp16

    idxWeight: [t, idxNHeads], fp16
    actSeqKey: [b], int32
    blockTable: [b, maxBlockNum], int32
    selectedCount: topk select num

    mmOut: [t, maxBlockNum * blockSize], fp32
    topkValue: [t, 1, selectedCount], fp32
    topkRes: [t, 1, selectedCount], int32
    */

    // graph fuse thresold
    config::SetPassOption("mg_copyin_upper_bound", configs.mgCopyInUpperBound);
    config::SetPassOption("pg_upper_bound", configs.pgUpperBound);
    // vector graph fuse optimization
    config::SetPassOption("vec_nbuffer_mode", configs.vecMergeMode);
    config::SetPassOption("vec_nbuffer_setting", configs.vecNBufferSetting);
    // cube graph fuse optimization
    config::SetPassOption("cube_l1_reuse_setting", configs.cubeL1ReuseSetting);
    // stitch optimization
    config::SetRuntimeOption("stitch_function_inner_memory", configs.maxRecyclePeriod);
    config::SetRuntimeOption("stitch_function_outcast_memory", configs.maxLoopNum);
    // schedule policy selection
    config::SetRuntimeOption("device_sched_mode", static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    // get tile params from configs
    auto s1Tile = configs.s1Tile; // s1 need to be divided by s1Tile
    auto topkTile = configs.topkTile;
    auto c1Tile = configs.c1Tile;
    auto c2Tile = configs.c2Tile;

    // symbolization of params
    SymbolicScalar t = GetInputShape(idxQuery, 0);
    SymbolicScalar b = GetInputShape(actSeqKey, 0);
    SymbolicScalar blockNum = GetInputShape(idxKeyCache, 0);

    auto idxNHeads = idxQuery.GetShape()[SHAPE_DIM1];
    auto indexD = idxQuery.GetShape()[SHAPE_DIM2];
    auto blockSize = idxKeyCache.GetShape()[SHAPE_DIM1];
    auto qkDType = idxQuery.GetDataType();
    auto scaleDType = idxQueryScale.GetDataType();
    auto wDType = idxWeight.GetDataType();

    auto s1 = t / b; // s1 of each batch is equal in decode process
    auto s1Loop = (s1 + s1Tile - 1) / s1Tile;

    Tensor query2D(qkDType, {t * idxNHeads, indexD}, "query2D");
    Tensor qScale3D(scaleDType, {t, 1, idxNHeads}, "qScale3D");
    Tensor key2D(qkDType, {blockNum * blockSize, indexD}, "key2D");
    Tensor kScale2D(scaleDType, {blockNum, blockSize}, "kScale2D");
    Tensor weight3D(wDType, {t, 1, idxNHeads}, "weight3D");
    // static tensor for rawShape assemble
    Tensor maxTensor(DT_FP32, {MAX_LI_BATCH * MAX_LI_S1, MAX_LI_S2}, "maxTensor");

    // reshape will not generate real datamove
    LOOP("LI_RESHAPE_INPLACE", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        Reshape(idxQuery, query2D);
        Reshape(idxQueryScale, qScale3D);
        Reshape(idxKeyCache, key2D);
        Reshape(idxWeight, weight3D);
        Reshape(idxKeyScale, kScale2D);
    }

    LOOP("LI_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b)) {
        auto curSeq = GetTensorData(actSeqKey, {bIdx});
        auto curBlock = (curSeq + blockSize - 1) / blockSize;
        auto lastSeq = curSeq - (curBlock - 1) * blockSize;
        LOOP("LI_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1Loop)) {
            auto qOffset = bIdx * s1 * idxNHeads + s1Idx * s1Tile * idxNHeads;
            config::SetSemanticLabel("LI-W-MUL");
            Tensor wScale(DT_FP16, {s1Tile, 1, idxNHeads}, "wScale");
            LOOP("LI_W_SCALE", FunctionType::DYNAMIC_LOOP, unUsed, LoopRange(1)) {
                (void)unUsed;
                TileShape::Current().SetVecTile(s1Tile, 1, idxNHeads);
                auto curQs = View(
                    qScale3D, {s1Tile, 1, idxNHeads}, {bIdx * s1 + s1Tile * s1Idx, 0, 0}); // (s1Tile, 1, idxNHeads)
                auto curW = View(
                    weight3D, {s1Tile, 1, idxNHeads}, {bIdx * s1 + s1Tile * s1Idx, 0, 0}); // (s1Tile, 1, idxNHeads)
                wScale = Mul(curQs, curW); // (s1Tile, 1, idxNHeads), fp16 * fp16
            }

            LOOP("LOOP_BLOCK_NUM", FunctionType::DYNAMIC_LOOP, bnIdx, LoopRange(curBlock), unrollList) {
                for (auto unrollLoop : unrollList) {
                    UNROLL(unrollLoop) {
                        auto curQ =
                            View(query2D, {s1Tile * idxNHeads, indexD}, {qOffset, 0}); // (s1Tile * idxNHeads, indexD)
                        std::vector<Tensor> firstMmCollect;
                        std::vector<Tensor> kScaleCollect;
                        // static unroll into bigger block to reduce tasks
                        for (int64_t subBnIdx = 0; subBnIdx < unrollLoop; subBnIdx++) {
                            auto idxInBlock = bnIdx + subBnIdx;
                            SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, idxInBlock});
                            auto tailSeq = std::min(blockSize, curSeq - (idxInBlock * blockSize));
                            config::SetSemanticLabel("LI-QK-DOT");
                            auto kBlock = View(key2D, {blockSize, indexD}, {tailSeq, indexD},
                                {curBlockIdx * blockSize, 0}); // (blockSize, indexD)
                            TileShape::Current().SetCubeTile(
                                {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                            if (false) {
                                // use fixpipe
                                auto qkDot = Matrix::Matmul<false, true>(DataType::DT_FP16, curQ, kBlock,
                                    configs.extendParam); // (s1Tile * idxNHeads, blockSize)
                                firstMmCollect.emplace_back(qkDot);
                                if (firstMm != nullptr) {
                                    Assemble(qkDot, {qOffset, idxInBlock * blockSize},
                                        *firstMm); // (s1Tile * idxNHeads, blockSize)
                                }
                            } else {
                                auto qkDot = Matrix::Matmul<false, true>(
                                    DataType::DT_INT32, curQ, kBlock); // (s1Tile * idxNHeads, blockSize)
                                TileShape::Current().SetVecTile(s1Tile * idxNHeads, blockSize);
                                qkDot = Cast(qkDot, DT_FP32);
                                qkDot = Maximum(qkDot, Element(DT_FP32, 0.0f));
                                qkDot = Mul(qkDot, Element(DT_FP32, AVOID_FP32_TO_FP16_OVERFLOW_SCALE));
                                qkDot = Cast(qkDot, DT_FP16); // (s1Tile * idxNHeads, blockSize)
                                firstMmCollect.emplace_back(qkDot);
                                if (firstMm != nullptr) {
                                    Assemble(qkDot, {qOffset, idxInBlock * blockSize},
                                        *firstMm); // (s1Tile * idxNHeads, blockSize)
                                }
                            }

                            auto kSBlock =
                                View(kScale2D, {1, blockSize}, {1, tailSeq}, {curBlockIdx, 0}); // (1, blockSize)
                            kScaleCollect.emplace_back(kSBlock);
                        }

                        config::SetSemanticLabel("LI-KS-Cat");
                        TileShape::Current().SetVecTile(unrollLoop, blockSize);
                        auto kScaleCat = Cat(kScaleCollect, -1); // (1, unrollLoop * blockSize), fp16
                        TileShape::Current().SetVecTile(1, blockSize * unrollLoop);
                        kScaleCat = Cast(kScaleCat, DT_FP32); // (1, unrollLoop * blockSize), fp32

                        config::SetSemanticLabel("LI-W-DOT");
                        TileShape::Current().SetVecTile(std::min(s1Tile * idxNHeads, blockSize), blockSize);
                        auto firstMmCat = Cat(firstMmCollect, -1); // (s1Tile * idxNHeads, unrollLoop * blockSize)
                        auto validCatShape = (unrollLoop - 1) * blockSize + lastSeq;
                        auto qk3D = Reshape(firstMmCat, {s1Tile, idxNHeads, unrollLoop * blockSize},
                            {s1Tile, idxNHeads, std::min(unrollLoop * blockSize, validCatShape)});
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, false);
                        auto wQk = Matrix::BatchMatmul<false, false>(
                            DT_FP32, wScale, qk3D); // (s1Tile, 1, unrollLoop * blockSize);
                        auto secondMm = Reshape(wQk, {s1Tile, unrollLoop * blockSize},
                            {s1Tile, std::min(unrollLoop * blockSize, validCatShape)});

                        config::SetSemanticLabel("LI-K-SCALE");
                        TileShape::Current().SetVecTile(s1Tile, unrollLoop * blockSize);
                        auto kRes = Mul(secondMm, kScaleCat); // (s1Tile, unrollLoop * blockSize)
                        Assemble(kRes, {bIdx * MAX_LI_S1 + s1Idx * s1Tile, bnIdx * blockSize}, maxTensor);
                        if (mmOut != nullptr) {
                            Assemble(kRes, {bIdx * s1 + s1Idx * s1Tile, bnIdx * blockSize}, *mmOut); // pointer of mmOut
                        }
                    }
                }
            }
        }
    }

    Tensor pad2K(DT_FP32, {MAX_LI_BATCH * MAX_LI_S1, selectedCount}, "pad2K");
    Tensor pad128K(DT_FP32, {MAX_LI_BATCH * MAX_LI_S1, MAX_LI_S2}, "pad128K");
    LOOP("LOOP_TOPK_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b), {}, true) {
        auto curSeq = GetTensorData(actSeqKey, {bIdx});
        LOOP("LOOP_TOPK_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1)) {
            auto effSeq = curSeq - (s1 - s1Idx - 1);
            auto srcIdx = bIdx * MAX_LI_S1 + s1Idx;
            auto dstIdx = bIdx * s1 + s1Idx;
            TileShape::Current().SetVecTile(1, selectedCount);
            config::SetPassOption("pg_skip_partition", true); // no tile unroll
            LOOP("TOPK_PAD_2K", FunctionType::DYNAMIC_LOOP, pad2KIdx, LoopRange(effSeq < selectedCount)) {
                (void)pad2KIdx;
                config::SetSemanticLabel("LI-2K-PAD");
                auto topkIn = View(maxTensor, {1, selectedCount}, {1, effSeq}, {srcIdx, 0});
                auto padTensor =
                    Full(Element(DT_FP32, -FLT_MAX), DT_FP32, {1, selectedCount}, {1, selectedCount - effSeq});
                Assemble(Assign(topkIn), {srcIdx, 0}, pad2K);
                Assemble(padTensor, {srcIdx, effSeq}, pad2K);
            }
            config::SetPassOption("pg_skip_partition", false); // tile unroll
            LOOP("TOPK_2K_CALC", FunctionType::DYNAMIC_LOOP, calc2KIdx, LoopRange(effSeq < selectedCount)) {
                (void)calc2KIdx;
                config::SetSemanticLabel("LI-2K-TOPK");
                TileShape::Current().SetVecTile(1, selectedCount);
                auto cur2KIn = View(pad2K, {1, selectedCount}, {srcIdx, 0});
                auto [curRes, curIdx] = TopK(cur2KIn, selectedCount, -1); // (1, selectedCount)
                config::SetSemanticLabel("LI-2K-INDEX");
                auto effIdx = View(curIdx, {1, selectedCount}, {1, effSeq}, {0, 0}); // (1, effSeq)
                effIdx = Reshape(effIdx, {1, 1, selectedCount}, {1, 1, effSeq});
                auto padIdx = Full(Element(DT_INT32, -1), DT_INT32, {1, selectedCount},
                    {1, selectedCount - effSeq}); // (1, selectedCount - effSeq)
                padIdx = Reshape(effIdx, {1, 1, selectedCount}, {1, 1, selectedCount - effSeq});
                TileShape::Current().SetVecTile(1, 1, selectedCount);
                Assemble(Assign(effIdx), {dstIdx, 0, 0}, topkRes);
                Assemble(padIdx, {dstIdx, 0, effSeq}, topkRes);
                if (topkValue != nullptr) {
                    config::SetSemanticLabel("LI-2K-RES");
                    curRes = Reshape(curRes, {1, 1, selectedCount});
                    TileShape::Current().SetVecTile(1, 1, selectedCount);
                    Assemble(Assign(curRes), {dstIdx, 0, 0}, *topkValue); // pointer of topkValue
                }
            }

            if (false) {
                LOOP("TOPK_OVER_2K", FunctionType::DYNAMIC_LOOP, over2kIdx, LoopRange(effSeq >= selectedCount)) {
                    (void)over2kIdx;
                    config::SetSemanticLabel("LI-OVER2K-TOPK");
                    auto topkIn = View(maxTensor, {1, MAX_LI_S2}, {1, effSeq}, {srcIdx, 0});
                    TileShape::Current().SetVecTile(1, topkTile);
                    auto [curRes, curIdx] = TopK(topkIn, selectedCount, -1); // (1, selectedCount)
                    config::SetSemanticLabel("LI-OVER2K-INDEX");
                    curIdx = Reshape(curIdx, {1, 1, selectedCount});
                    TileShape::Current().SetVecTile(1, 1, selectedCount);
                    Assemble(Assign(curIdx), {dstIdx, 0, 0}, topkRes);
                    if (topkValue != nullptr) {
                        config::SetSemanticLabel("LI-OVER2K-RES");
                        curRes = Reshape(curRes, {1, 1, selectedCount});
                        TileShape::Current().SetVecTile(1, 1, selectedCount);
                        Assemble(Assign(curRes), {dstIdx, 0, 0}, *topkValue); // pointer of topkValue
                    }
                }
            } else {
                // Pad to 128k for avoid topk bug
                TileShape::Current().SetVecTile(1, topkTile);
                config::SetPassOption("pg_skip_partition", true); // no tile unroll
                LOOP("TOPK_PAD_128K", FunctionType::DYNAMIC_LOOP, pad128KIdx, LoopRange(effSeq > selectedCount)) {
                    (void)pad128KIdx;
                    config::SetSemanticLabel("LI-128K-PAD");
                    auto topkIn = View(maxTensor, {1, MAX_LI_S2}, {1, effSeq}, {srcIdx, 0});
                    auto padTensor = Full(Element(DT_FP32, -FLT_MAX), DT_FP32, {1, MAX_LI_S2}, {1, MAX_LI_S2 - effSeq});
                    Assemble(Assign(topkIn), {srcIdx, 0}, pad128K);
                    Assemble(padTensor, {srcIdx, effSeq}, pad128K);
                }
                config::SetPassOption("pg_skip_partition", false); // tile unroll
                LOOP("TOPK_128K_CALC", FunctionType::DYNAMIC_LOOP, calc128KIdx, LoopRange(effSeq >= selectedCount)) {
                    (void)calc128KIdx;
                    config::SetSemanticLabel("LI-128K-TOPK");
                    auto cur128KIn = View(pad128K, {1, MAX_LI_S2}, {srcIdx, 0});
                    auto [curRes, curIdx] = TopK(cur128KIn, selectedCount, -1); // (1, selectedCount)
                    config::SetSemanticLabel("LI-128K-INDEX");
                    curIdx = Reshape(curIdx, {1, 1, selectedCount}, {1, 1, selectedCount});
                    TileShape::Current().SetVecTile(1, 1, selectedCount);
                    Assemble(Assign(curIdx), {dstIdx, 0, 0}, topkRes);
                    if (topkValue != nullptr) {
                        config::SetSemanticLabel("LI-128K-RES");
                        curRes = Reshape(curRes, {1, 1, selectedCount});
                        TileShape::Current().SetVecTile(1, 1, selectedCount);
                        Assemble(Assign(curRes), {dstIdx, 0, 0}, *topkValue); // pointer of topkValue
                    }
                }
            }
        }
    }
}

void LightningIndexer(const Tensor &idxQuery, const Tensor &idxQueryScale, const Tensor &idxKeyCache,
    const Tensor &idxKeyScale, const Tensor &idxWeight, const Tensor &actSeqKey, const Tensor &blockTable,
    const int selectedCount, Tensor &topkRes, LightningIndexerConfigs configs, const std::set<int> &unrollList) {
    LightningIndexerImpl(idxQuery, idxQueryScale, idxKeyCache, idxKeyScale, idxWeight, actSeqKey, blockTable,
        selectedCount, topkRes, configs, unrollList);
}

} // namespace npu::tile_fwk
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
 * \file test_dynamic_pa_post.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"

#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicPAPOSTTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

TEST_F(DynamicPAPOSTTest, dynamic_prolog_post_low_lantency) {
    int b = 2;
    int sq = 1;
    int nq = 32;
    int dn = 128;
    int dr = 64;
    int skv = 128;
    int blockSize = 128;
    int nTile = nq;
    int dTile = 64;
    float softmaxScale = 0.8f;
    int vHeadDim = 32;
    int h = 64;

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, dTile};
    tileConfig.c1TileShape = {nTile, nTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, dTile};
    tileConfig.c2TileShape = {nTile, nTile, dTile, dTile, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, dTile};

    const std::vector<int> seq(b, skv);
    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor weightUV(DT_BF16, {nq, dn, vHeadDim}, "weightUV");
    Tensor weightO(DT_BF16, {nq * vHeadDim, h}, "weightO");
    Tensor postOut(DT_FP32, {b, sq, h}, "postOut");

    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    std::vector<uint8_t> devProgBinary;

    PrologPost(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, weightUV, weightO, blockSize,
        softmaxScale, postOut, tileConfig);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * sq * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * sq * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> kRopeCacheData(blockNum * blockSize * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> vNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(nq * dn * vHeadDim, 0);
    std::vector<npu::tile_fwk::bfloat16> weightOData(nq * vHeadDim * h, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", vNopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/weight_uv.bin", weightUVData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/weight_o.bin", weightOData);

    std::vector<float> golden(b * sq * h, 0);
    readInput(GetGoldenDir() + "/post_out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.004f));
}

void testPaAdds(PaTileShapeConfig& tileConfig, int maxUnrollTimes = 1, bool manualUnroll = false, bool outputPaOut = true) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    std::vector<uint8_t> devProgBinary;
    int paramsSize = 8;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    std::vector<int> seq(b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    Tensor postOut(DT_FP32, {b * nq * sq, dn}, "postOut");
    if (!manualUnroll) {
        if (outputPaOut) {
            PageAttentionAddS(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut, postOut,
                tileConfig, maxUnrollTimes);
        } else {
            PageAttentionAddSSingleOutput(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut, postOut,
                tileConfig, maxUnrollTimes);
        }
    }

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * sq * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * sq * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> kRopeCacheData(blockNum * blockSize * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> vNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", vNopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);

    std::vector<float> golden(b * sq * nq * dn, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });
    if (outputPaOut) {
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<float>(paOut, 0),
            RawTensorData::CreateConstantTensor<float>(postOut, 0),
        });

        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
        EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f));
        EXPECT_TRUE(resultCmp(golden, (float *)outs1->data(), 0.0005f));
    } else {
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<float>(postOut, 0),
        });

        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f));
    }
}

TEST_F(DynamicPAPOSTTest, dynamic_page_attention_adds_high_throughput_dview_large) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    testPaAdds(tileConfig);
}

TEST_F(DynamicPAPOSTTest, dynamic_page_attention_adds_high_throughput_dview_large_single_out) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    bool outputPaOut = false;
    testPaAdds(tileConfig, 1, false, outputPaOut);
}

void PageAttentionPost(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &weightUV, Tensor &weightO,
    Tensor &weightOScaleW, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    int batchSize = blockTable.GetShape()[0];
    int nQ = qNope.GetShape()[0] / batchSize; // B*1*N

    auto N = weightUV.GetShape()[0];;
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, weightUV, weightO, weightOScaleW},
        {postOut}) {
        SymbolicScalar nLoop = nQ / nTile;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1),
                     PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false, true); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (IsLoopBegin(bn, 0)) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        auto miNew = Maximum(mi, tildaMij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t1 = Sub(mi, miNew);           // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t6 = Mul(t2, li);       // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto liNew = Add(t6, t5);    // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)

                        auto q3 = Mul(oi, t2); // (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }

        SymbolicScalar B = attentionOut.GetShape()[0] / N; // S=1
        const int64_t bTile = 32;
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, papostiter, LoopRange(B / bTile), {}, true) {
            auto postInUnit = View(attentionOut, {bTile * S * N, kvLoraRank}, {papostiter * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(32L, bTile * S * N), kvLoraRank}); // raw (bTile*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank}); // 128个
            TileShape::Current().SetVecTile({std::min(32L, bTile * S), 1, kvLoraRank}); // raw (bTile*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)    // 128个

            // config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
            TileShape::Current().SetCubeTile({std::min(32L, bTile * S), std::min(32L, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)},
                {vHeadDim, vHeadDim});                                 // raw bTile*1  512   128   // 128/4个
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(1, std::min(bTile, bTile * S), vHeadDim); // raw (128, bTile*1, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim) // 128个
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (bTile*1, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32L, bTile * S), std::min(32L, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)}, {std::min(512L, H), std::min(512L, H)},
                false, true); // raw  bTile*1  16k  7168
            Tensor res =
                npu::tile_fwk::Matrix::Matmul(DT_INT32, quantizedA, weightO); // (bTile*S, H) // 14*8=112个

            TileShape::Current().SetVecTile(std::min(bTile, bTile * S), std::min(bTile, H)); // raw (bTile*1, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);   // (B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1, H)  // 224个
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            std::vector<SymbolicScalar> dynOffset = {papostiter * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}

void testPaPost(PaTileShapeConfig& tileConfig, int maxUnrollTimes = 1, bool manualUnroll = false) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    std::vector<uint8_t> devProgBinary;

    int paramsSize = 8;
    int postParamsSize = 7;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    std::vector<int64_t> params(postParamsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);

    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<int> seq(b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {1, H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    if (!manualUnroll) {
        PageAttentionPost(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize,
                          softmaxScale, weightUV, weightO, weightOScaleW, paOut, postOut, tileConfig, maxUnrollTimes);
    }

    // 读数据
    // PA数据
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * sq * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * sq * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> kRopeCacheData(blockNum * blockSize * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> vNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", vNopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);
    std::vector<float> paGolden(b * sq * nq * dn, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", paGolden);

    // POST数据
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(1 * H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);
    std::vector<npu::tile_fwk::bfloat16> paPostgolden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", paPostgolden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(paPostgolden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.04f));
}

TEST_F(DynamicPAPOSTTest, dynamic_prolog_post_high_throughput_dview_large) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    testPaPost(tileConfig);
}

}

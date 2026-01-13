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
 * \file test_onboard_query_states.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(OnBoardTest, test_query_states_fp16_b32_n2) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int b = 32;        //--> 32
    int num_heads = 2; //--> 32
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::float16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(DataType::DT_FP16, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(DataType::DT_FP16, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            DataType::DT_FP16, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            DataType::DT_FP16, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 128, 1, 64); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 1, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(DataType::DT_FP16, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 1, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(1, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> q(cap);
    std::vector<npu::tile_fwk::float16> golden(outputCap);
    std::vector<npu::tile_fwk::float16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::float16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_fp16_b32_n16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int b = 32;
    int num_heads = 16;
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::float16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(DataType::DT_FP16, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(DataType::DT_FP16, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            DataType::DT_FP16, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            DataType::DT_FP16, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 1, 32, 128); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 32, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(DataType::DT_FP16, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 8, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(2, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> q(cap);
    std::vector<npu::tile_fwk::float16> golden(outputCap);
    std::vector<npu::tile_fwk::float16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::float16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_fp16_b32_n32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int b = 32;
    int num_heads = 32;
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::float16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::float16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(DataType::DT_FP16, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(DataType::DT_FP16, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            DataType::DT_FP16, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            DataType::DT_FP16, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 1, 32, 128); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 32, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(DataType::DT_FP16, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 8, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(2, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> q(cap);
    std::vector<npu::tile_fwk::float16> golden(outputCap);
    std::vector<npu::tile_fwk::float16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::float16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::float16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_bf16_b32_n2) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    DataType dType = DataType::DT_BF16;
    int b = 32;        //--> 32
    int num_heads = 2; //--> 32
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::bfloat16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(dType, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(dType, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            dType, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            dType, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 128, 1, 64); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 1, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(dType, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 1, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(1, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::bfloat16> q(cap);
    std::vector<npu::tile_fwk::bfloat16> golden(outputCap);
    std::vector<npu::tile_fwk::bfloat16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_bf16_b32_n2_concat) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    DataType dType = DataType::DT_BF16;
    int b = 32;        //--> 32
    int num_heads = 2; //--> 32
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::bfloat16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/r2.bin", b * s * num_heads * kvLoraRank);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(dType, {b, s, num_heads, kvLoraRank}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(dType, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            dType, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            dType, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            TileShape::Current().SetVecTile(1, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::bfloat16> q(cap);
    std::vector<npu::tile_fwk::bfloat16> golden(outputCap);
    std::vector<npu::tile_fwk::bfloat16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_bf16_b32_n2_nocat) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    DataType dType = DataType::DT_BF16;
    int b = 32;        //--> 32
    int num_heads = 2; //--> 32
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::bfloat16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(dType, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(dType, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            dType, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            dType, {b, num_heads, s, kvLoraRank}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 128, 1, 64); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 1, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(dType, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 1, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(1, 1, 1, 512);
            query_states = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::bfloat16> q(cap);
    std::vector<npu::tile_fwk::bfloat16> golden(outputCap);
    std::vector<npu::tile_fwk::bfloat16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/t3.bin", golden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_bf16_b32_n16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    DataType dType = DataType::DT_BF16;
    int b = 32;
    int num_heads = 16;
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::bfloat16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(dType, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(dType, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            dType, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            dType, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 1, 32, 128); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 32, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(dType, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 8, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(2, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::bfloat16> q(cap);
    std::vector<npu::tile_fwk::bfloat16> golden(outputCap);
    std::vector<npu::tile_fwk::bfloat16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_query_states_bf16_b32_n32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    DataType dType = DataType::DT_BF16;
    int b = 32;
    int num_heads = 32;
    int s = 1;
    int qkNopeHeadDim = 128;                           // 128
    int qkRopeHeadDim = 64;                            // 64
    int kvLoraRank = 512;                               // 512
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim; // 192

    int cap = b * s * num_heads * q_head_dim;
    int bs = b * s;

    int outputCap = b * num_heads * s * (kvLoraRank + qkRopeHeadDim);
    uint64_t outputSize = outputCap * sizeof(npu::tile_fwk::bfloat16);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("QueryStates") {
        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_q.bin", b * s * num_heads * q_head_dim);
        void *q_pe_rope_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_q_pe_rope.bin", b * s * num_heads * qkRopeHeadDim);
        void *kv_b_proj_w_k_ptr = readToDev<npu::tile_fwk::bfloat16>(
            GetGoldenDir() + "/query_states_kv_b_proj_wk.bin", num_heads * qkNopeHeadDim * kvLoraRank);


        Tensor q(dType, {b, s, num_heads, q_head_dim}, (uint8_t *)q_ptr, "q");
        Tensor q_pe_rope(dType, {b, num_heads, s, qkRopeHeadDim}, (uint8_t *)q_pe_rope_ptr, "q_pe_rope");
        Tensor kvBProjWK(
            dType, {num_heads, qkNopeHeadDim, kvLoraRank}, (uint8_t *)kv_b_proj_w_k_ptr, "kvBProjWK");
        Tensor query_states(
            dType, {b, num_heads, s, kvLoraRank + qkRopeHeadDim}, (uint8_t *)out_ptr, "query_states");

        config::SetBuildStatic(true);
        FUNCTION("QUERY_STATES_T", {q, q_pe_rope, kvBProjWK, query_states}) {
            Tensor q_nope = View(q, {b, s, num_heads, qkNopeHeadDim}, {0, 0, 0, 0});
            TileShape::Current().SetVecTile(1, 1, 32, 128); // --> SetVecTileShapes(1, 1, 128, 64)
            Tensor q_nope1 = Reshape(q_nope, {b * s, num_heads, qkNopeHeadDim});
            TileShape::Current().SetVecTile(1, 32, 128);
            Tensor q_nope2 = Transpose(q_nope1, {0, 1}); // (num_heads, bs, qkNopeHeadDim)

            // bmm: (num_heads, bs, qkNopeHeadDim) * (num_heads, qkNopeHeadDim, kvLoraRank)
            // = (num_heads, bs, kvLoraRank)
            TileShape::Current().SetCubeTile({std::min(128, bs), std::min(128, bs)}, {128, 128}, {128, 128});
            Tensor q_nope_new = Matrix::BatchMatmul(dType, q_nope2, kvBProjWK, false, false);

            TileShape::Current().SetVecTile(1, 8, 512);
            Tensor q_nope_new2 = Transpose(q_nope_new, {0, 1}); //(bs, num_heads, kvLoraRank)
            TileShape::Current().SetVecTile(1, 128, 64);
            Tensor q_nope_new3 = Reshape(q_nope_new2, {b, s, num_heads, kvLoraRank});
            TileShape::Current().SetVecTile(2, 1, 1, 512);
            Tensor q_nope_new4 = Transpose(q_nope_new3, {1, 2}); //(b, num_heads, s, kvLoraRank)
            TileShape::Current().SetVecTile(2, 2, 1, 512);
            query_states = Cat({q_nope_new4, q_pe_rope}, -1); // (b, num_heads, s, kvLoraRank + qkRopeHeadDim)
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::bfloat16> q(cap);
    std::vector<npu::tile_fwk::bfloat16> golden(outputCap);
    std::vector<npu::tile_fwk::bfloat16> res(outputCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/query_states_res.bin", golden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

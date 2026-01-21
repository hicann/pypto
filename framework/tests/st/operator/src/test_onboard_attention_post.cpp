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
 * \file test_onboard_attention_post.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(OnBoardTest, test_attention_post_bf16_real_batch4) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtype_num = params[6];

    DataType dType = DataType::DT_FP32;
    if (dtype_num == 0) {
        dType = DataType::DT_FP32;
    } else if (dtype_num == 1) {
        dType = DataType::DT_FP16;
    } else if (dtype_num == 2) {
        dType = DataType::DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;

    int inputSize = B*N*S*kvLoraRank;
    int wUvSize = N*kvLoraRank*vHeadDim;
    int wOSize = N*vHeadDim*H;
    int outputSize = B*S*H;
    int t1Size = B*S*N*kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* out_ptr = allocDevAddr(outputByteSize);
    PROGRAM("ATTENTION_POST") {
        std::vector<int64_t> inputShape = {B,N,S,kvLoraRank};
        std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
        std::vector<int64_t> wOShape = {N*vHeadDim,H};
        std::vector<int64_t> outputShapeT = {B, S, H};
        std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
        void *input_ptr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
        void *w_uv_ptr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
        void *w_o_ptr = readToDev<T>(GetGoldenDir() + "/w_o.bin", wOSize);

        void *t1_ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

        Tensor input_i(dType, inputShape, (uint8_t *)input_ptr, "A");
        Tensor w_uv_i(dType, wUvShape, (uint8_t *)w_uv_ptr, "B");
        Tensor w_o_i(dType, wOShape, (uint8_t *)w_o_ptr, "C");
        Tensor outputT(dType, outputShapeT, out_ptr, "D1");
        Tensor t1_i(dType, t1Shape, (uint8_t *)t1_ptr, "E");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ATTENTION_POST_T", {input_i, t1_i, w_uv_i, w_o_i, outputT}) {
            // T+R+T fail
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor atten_res0 = Transpose(input_i, {1, 2});
            TileShape::Current().SetVecTile({4, 1, 32, std::min(512, kvLoraRank)});
            Tensor atten_res1 = Reshape(atten_res0, {B * S, N, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, kvLoraRank});
            Tensor t2_res = Transpose(atten_res1, {0, 1});

            TileShape::Current().SetCubeTile({16, 16}, {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
                {std::min(128, vHeadDim), std::min(128, vHeadDim)}); // M 16对齐
            // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
            Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, w_uv_i);

            TileShape::Current().SetVecTile(32, 4, vHeadDim); // 必须切，但是尾轴不能切
            Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,vHeadDim]

            TileShape::Current().SetVecTile({4, 32, vHeadDim});
            Tensor r2_res = Reshape(t3_res, {B * S, N*vHeadDim});

            // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
            TileShape::Current().SetCubeTile({16, 16}, {std::min(256, N * vHeadDim), std::min(256, N * vHeadDim)},
                {std::min(128, H), std::min(128, H)});
            Tensor bmm5_res = Matrix::Matmul(dType, r2_res, w_o_i, false, false);

            TileShape::Current().SetVecTile({4, std::min(2048, H)});
            outputT = Reshape(bmm5_res, {B, S, H});
        }
    }
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) == HOST_COMPILE_END) {
        std::cout << Program::GetInstance().Dump() << std::endl;
    } else {
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        std::vector<T> golden(outputSize);
        std::vector<T> res(outputSize);
        machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputByteSize);
        readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
        int ret = resultCmp<T>(golden, res, 0.005f);
        EXPECT_EQ(ret, true);
    }
}

TEST_F(OnBoardTest, test_attention_post_bf16_real_n128) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtype_num = params[6];

    DataType dType = DataType::DT_FP32;
    if (dtype_num == 0) {
        dType = DataType::DT_FP32;
    } else if (dtype_num == 1) {
        dType = DataType::DT_FP16;
    } else if (dtype_num == 2) {
        dType = DataType::DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;

    int inputSize = B*N*S*kvLoraRank;
    int wUvSize = N*kvLoraRank*vHeadDim;
    int wOSize = N*vHeadDim*H;
    int outputSize = B*S*H;
    int t1Size = B*S*N*kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* out_ptr = allocDevAddr(outputByteSize);
    PROGRAM("ATTENTION_POST") {
        std::vector<int64_t> inputShape = {B,N,S,kvLoraRank};
        std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
        std::vector<int64_t> wOShape = {N*vHeadDim,H};
        std::vector<int64_t> outputShapeT = {B, S, H};
        std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
        void *input_ptr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
        void *w_uv_ptr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
        void *w_o_ptr = readToDev<T>(GetGoldenDir() + "/w_o.bin", wOSize);

        void *t1_ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

        Tensor input_i(dType, inputShape, (uint8_t *)input_ptr, "A");
        Tensor w_uv_i(dType, wUvShape, (uint8_t *)w_uv_ptr, "B");
        Tensor w_o_i(dType, wOShape, (uint8_t *)w_o_ptr, "C");
        Tensor outputT(dType, outputShapeT, out_ptr, "D1");
        Tensor t1_i(dType, t1Shape, (uint8_t *)t1_ptr, "E");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ATTENTION_POST_T", {input_i, t1_i, w_uv_i, w_o_i, outputT}) {
            // T+R+T fail
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor atten_res0 = Transpose(input_i, {1, 2});
            TileShape::Current().SetVecTile({4, 1, 32, kvLoraRank});
            Tensor atten_res1 = Reshape(atten_res0, {B * S, N, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, kvLoraRank});
            Tensor t2_res = Transpose(atten_res1, {0, 1});

            TileShape::Current().SetCubeTile({32, 32}, {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
                {std::min(128, vHeadDim), std::min(128, vHeadDim)}); // M 16对齐
            // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
            Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, w_uv_i);

            TileShape::Current().SetVecTile(32, 4, vHeadDim); // 必须切，但是尾轴不能切
            Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,vHeadDim]

            TileShape::Current().SetVecTile({4, 32, vHeadDim});
            Tensor r2_res = Reshape(t3_res, {B * S, N*vHeadDim});

            // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
            TileShape::Current().SetCubeTile({32, 32}, {std::min(256, N * vHeadDim), std::min(256, N * vHeadDim)},
                {std::min(128, H), std::min(128, H)});
            Tensor bmm5_res = Matrix::Matmul(dType, r2_res, w_o_i, false, false);

            TileShape::Current().SetVecTile({32, std::min(2048, H)});
            outputT = Reshape(bmm5_res, {B, S, H});
        }
    }
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) == HOST_COMPILE_END) {
        std::cout << Program::GetInstance().Dump() << std::endl;
    } else {
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        std::vector<T> golden(outputSize);
        std::vector<T> res(outputSize);
        machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputByteSize);
        readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
        int ret = resultCmp<T>(golden, res, 0.006f);
        EXPECT_EQ(ret, true);
    }
}

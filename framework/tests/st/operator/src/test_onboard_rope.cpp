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
 * \file test_onboard_rope.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class RoPEOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(RoPEOnBoardTest, test_operation_rope_reshape_transpose_reshape_muls) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_128_1_64/";

    std::vector<int64_t> qPeShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, 1, S, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {64, 64}, // for cos/sin->cast
            {1, 64, 64}, // for gather,unsqueeze
            {1, 64, 1, 64},
            {1, 64, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, qEmbed}) {
            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            auto qView = Reshape(q, {B, N, S, qkRopeHeadDim / 2, 2}); // [b,n,s,qk_d//2,2]
            TileShape::Current().SetVecTile(ropeTileConfig.fiveDimsTileShape);
            auto qTrans = Transpose(qView, {3, 4});
            auto qReshape = Reshape(qTrans, {B, N, S, qkRopeHeadDim});    // [b,n,s,qk_d]
            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            qEmbed = Mul(qReshape, Element(DataType::DT_FP32, -1.0));
            // qEmbed = RotateHalf(qReshape); // 待reshape+view+muls精度解决后再验证
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "q_R_T_R_Muls.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_tensorIndex_unsqueeze_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 32;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/32_128_1_64/";

    std::vector<int64_t> qPeShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, 1, S, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, 1, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {64, 64}, // for cos/sin->cast
            {1, 64, 64}, // for gather,unsqueeze
            {1, 64, 1, 64},
            {1, 64, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {cos, sin, positionIds, qEmbed}) {
            // TensorIndex+unsqueeze+mul  ok
            TileShape::Current().SetVecTile(ropeTileConfig.threeDimsTileShape); // TensorIndex, 设置三维Tile
            auto cosTensorIndexes = TensorIndex(cos, positionIds);                             // [s,qk_d],[b,s]->[b,s,qk_d]
            auto sinTensorIndexes = TensorIndex(sin, positionIds);

            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape); // TensorIndex, 设置三维Tile
            auto cosUnsqueeze = Unsqueeze(cosTensorIndexes, 1); // [b,1,s,qk_d]
            auto sinUnsqueeze = Unsqueeze(sinTensorIndexes, 1);

            qEmbed = Mul(sinUnsqueeze, cosUnsqueeze);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "cos_sin_T_U_mul.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_reshape_view_muls) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_128_1_64/";

    std::vector<int64_t> qPeShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, 1, S, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim / 2};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, qEmbed}) {
            TileShape::Current().SetVecTile({1, 64, 1, 64});
            auto qView = Reshape(q, {B, N, S, qkRopeHeadDim / 2, 2}); // [b,n,s,qk_d//2,2]
            TileShape::Current().SetVecTile({1, 64, 1, 64, 64});
            auto qTrans = Transpose(qView, {3, 4});
            auto qReshape = Reshape(qTrans, {B, N, S, qkRopeHeadDim});    // [b,n,s,qk_d]

            Tensor x1 = View(qReshape, {B, N, S, qkRopeHeadDim / 2}, {0, 0, 0, 0});
            qEmbed = Mul(x1, Element(DataType::DT_FP32, -1.0));
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "q_R_T_R_View_Muls.bin", qEmbedGolden);
    // readInput(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_reshape_view_muls_concat) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_128_1_64/";

    std::vector<int64_t> qPeShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, 1, S, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {64, 64}, // for cos/sin->cast
            {1, 64, 64}, // for gather,unsqueeze
            {1, 64, 1, 64},
            {1, 64, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, k, qEmbed, kEmbed}) {
            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            auto qView = Reshape(q, {B, N, S, qkRopeHeadDim / 2, 2}); // [b,n,s,qk_d//2,2]
            TileShape::Current().SetVecTile(ropeTileConfig.fiveDimsTileShape);
            auto qTrans = Transpose(qView, {3, 4});
            auto qReshape = Reshape(qTrans, {B, N, S, qkRopeHeadDim});    // [b,n,s,qk_d]

            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            qEmbed = RotateHalf(qReshape); // view+muls+concat

            // k
            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            auto kView = Reshape(k, {B, 1, S, qkRopeHeadDim / 2, 2}); // [b,n,s,qk_d//2,2]
            TileShape::Current().SetVecTile(ropeTileConfig.fiveDimsTileShape);
            auto kTrans = Transpose(kView, {3, 4});
            auto kReshape = Reshape(kTrans, {B, 1, S, qkRopeHeadDim});    // [b,n,s,qk_d]

            TileShape::Current().SetVecTile(ropeTileConfig.fourDimsTileShape);
            kEmbed = RotateHalf(kReshape); // view+muls+concat
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "q_RTR_rotate_half.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 256) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    // kEmbed
    std::vector<float> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(float));
    std::vector<float> kEmbedGolden(kEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "k_RTR_rotate_half.bin", kEmbedGolden);
    // readInput(GetGoldenDir() + shape_dir_path + "k_R_T_R.bin", kEmbedGolden);
    // readInput(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp(kEmbedGolden, kEmbedRes, 0.001f);

    // for (int i = 0; i < kEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << kEmbedGolden[i] << " AST res:" << kEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======kEmbedSize is " << kEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_deepseekv3) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_128_1_64/";

    std::vector<int64_t> qPeShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, 1, S, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {64, 64}, // for cos/sin->cast
            {1, 64, 64}, // for gather,unsqueeze
            {1, 64, 1, 64},
            {1, 64, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, k, cos, sin, positionIds, qEmbed, kEmbed}) {
            ApplyRotaryPosEmb(q, k, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    // kEmbed
    std::vector<float> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(float));
    std::vector<float> kEmbedGolden(kEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp(kEmbedGolden, kEmbedRes, 0.001f);

    // for (int i = 0; i < kEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << kEmbedGolden[i] << " AST res:" << kEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======kEmbedSize is " << kEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_v2_deepseekv3) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_1_128_64/";

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, 1, qkRopeHeadDim}; // k的N=1
    std::vector<int64_t> cosSinShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, S, 1, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfigNew ropeTileConfig{
            {1, 1, 32}, // (b,s,d)
            {1, 1, 64, 32}, // Q (b,s,n,d)
            {1, 1, 1, 32}, // K (b,s,1,d)
            {1, 1, 64, 32, 2} // (b,s,n,d//2,2)
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, k, cos, sin, qEmbed, kEmbed}) {
            ApplyRotaryPosEmbV2(q, k, cos, sin, qEmbed, kEmbed, 2, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // kEmbed
    std::vector<float> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(float));
    std::vector<float> kEmbedGolden(kEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp(kEmbedGolden, kEmbedRes, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RoPEOnBoardTest, test_operation_rope_v2_deepseekv3_b32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 32;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/32_1_128_64/";

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, 1, qkRopeHeadDim};
    std::vector<int64_t> cosSinShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> qEmbedShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, S, 1, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPE") {

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "q.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "k.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        Tensor q(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "q");
        Tensor k(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "k");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfigNew ropeTileConfig{
            {32, 1, 64}, // (b,s,d)
            {1, 1, 32, 64}, // Q (b,s,n,d)
            {32, 1, 1, 64}, // K (b,s,1,d)
            {32, 1, 1, 32, 2} // (b,s,n,d//2,2)
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {q, k, cos, sin, qEmbed, kEmbed}) {
            ApplyRotaryPosEmbV2(q, k, cos, sin, qEmbed, kEmbed, 2, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // kEmbed
    std::vector<float> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(float));
    std::vector<float> kEmbedGolden(kEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp(kEmbedGolden, kEmbedRes, 0.001f);
    EXPECT_EQ(ret, true);
}

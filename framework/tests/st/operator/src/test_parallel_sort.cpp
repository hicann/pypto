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
 * \file test_parallel_sort.cpp
 * \brief
 */
#include "tilefwk/tilefwk_op.h"
#include "interface/configs/config_manager.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class ParallelSortSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

int64_t Capacity(std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

template <typename T = float, typename idxT = int>
void SortStaticTest(int tileSize){
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    std::vector<int> params(2);
    readInput<int>(GetGoldenDir() + "/params.bin", params);
    int32_t length = params[0];
    bool descending = (bool)params[1];

    std::vector<int64_t> shape = {1, length};

    void *xPtr = readToDev<uint32_t>(GetGoldenDir() + "/x.bin", Capacity(shape));
    uint8_t* yPtr = allocDevAddr(Capacity(shape) * sizeof(float));
    uint8_t* yIdxPtr = allocDevAddr(Capacity(shape) * sizeof(float));

    Tensor x(DataType::DT_FP32, shape, (uint8_t *)xPtr, "x");
    Tensor y(DataType::DT_FP32, shape, (uint8_t *)yPtr, "y");
    Tensor yIdx(DataType::DT_FP32, shape, (uint8_t *)yIdxPtr, "yIdx");

    std::vector<float> yGolden(Capacity(shape));
    std::vector<float> yIdxGolden(Capacity(shape));
    readInput(GetGoldenDir() + "/y.bin", yGolden);
    readInput(GetGoldenDir() + "/yidx.bin", yIdxGolden);

    ConfigManager::Instance();
    config::SetBuildStatic(true);
    FUNCTION("Sort", {x, y, yIdx}) {
        TileShape::Current().SetVecTile({1, tileSize});
        std::tie(y, yIdx) = Sort(x, descending);
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    uint64_t taskTime = DeviceRunner::Get().GetTasksTime();
    std::cout<<"Sort Cost Time is: "<< taskTime << std::endl;

    std::vector<float> yResult(Capacity(shape));
    std::vector<float> yIdxResult(Capacity(shape));
    machine::GetRA()->CopyFromTensor((uint8_t *)yResult.data(), (uint8_t *)yPtr, Capacity(shape) * sizeof(float));
    machine::GetRA()->CopyFromTensor((uint8_t *)yIdxResult.data(), (uint8_t *)yIdxPtr, Capacity(shape) * sizeof(float));
    std::cout << "y" << std::endl;
    bool cmp = resultCmp(yGolden, yResult, 0);
    std::cout << "yIdx" << std::endl;
    bool cmpIdx = resultCmp(yIdxGolden, yIdxResult, 0);
    EXPECT_EQ(cmp && cmpIdx, true);
}

template <typename T = float, typename idxT = int>
void SortTest(int tileSize){
    config::SetHostOption(ONLY_CODEGEN, true);
    
    std::vector<int> params(2);
    readInput<int>(GetGoldenDir() + "/params.bin", params);
    int32_t length = params[0];
    bool descending = (bool)params[1];

    DataType dType = DT_FP32;
    DataType idxDType = DT_INT32;
    std::vector<int64_t> shape = {1, length};

    // input & output
    Tensor x(dType, shape, "x");
    Tensor y(dType, shape, "y");
    Tensor yIdx(idxDType, shape, "yIdx");

    // output golden
    std::vector<T> yGolden = getGoldenVec<T>(shape, "/y.bin");
    std::vector<idxT> yIdxGolden = getGoldenVec<idxT>(shape, "/yidx.bin");

    // input & output data
    auto xData = CreateTensorData<T>(x, shape, "/x.bin");
    auto yData = RawTensorData::CreateConstantTensor<T>(y, 0.0);
    auto yIdxData = RawTensorData::CreateConstantTensor<idxT>(yIdx, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {yData, yIdxData};
    std::vector<RawTensorDataPtr> inputDataList = {xData};

    FUNCTION("Sort", {x}, {y, yIdx}) {
        LOOP("LOOP_1", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1)) {
            UNUSED(bIdx);
            TileShape::Current().SetVecTile({1, tileSize});
            std::tie(y, yIdx)  = Sort(x, descending);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "y ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(yGolden, (T *)yData->data(), 0));
    std::cout << "yIdx ======" << std::endl;
    EXPECT_TRUE(resultCmp<idxT>(yIdxGolden, (idxT *)yIdxData->data(), 0));
}

template <typename T = float, typename idxT = int>
void SortWithIndexTest(int tileSize){
    config::SetHostOption(ONLY_CODEGEN, true);
    
    std::vector<int> params(2);
    readInput<int>(GetGoldenDir() + "/params.bin", params);
    int32_t length = params[0];
    bool descending = (bool)params[1];

    DataType dType = DT_FP32;
    DataType idxDType = DT_INT32;
    std::vector<int64_t> shape = {1, length};

    // input & output
    Tensor x(dType, shape, "x");
    Tensor idx(idxDType, shape, "idx");
    Tensor y(dType, shape, "y");
    Tensor yIdx(idxDType, shape, "yIdx");

    // output golden
    std::vector<T> yGolden = getGoldenVec<T>(shape, "/y.bin");
    std::vector<idxT> yIdxGolden = getGoldenVec<idxT>(shape, "/yidx.bin");

    // input & output data
    auto xData = CreateTensorData<T>(x, shape, "/x.bin");
    auto idxData = CreateTensorData<T>(idx, shape, "/idx.bin");
    auto yData = RawTensorData::CreateConstantTensor<T>(y, 0.0);
    auto yIdxData = RawTensorData::CreateConstantTensor<idxT>(yIdx, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {yData, yIdxData};
    std::vector<RawTensorDataPtr> inputDataList = {xData, idxData};

    FUNCTION("Sort", {x, idx}, {y, yIdx}) {
        LOOP("LOOP_1", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1)) {
            UNUSED(bIdx);
            TileShape::Current().SetVecTile({1, tileSize});
            std::tie(y, yIdx) = SortWithIndex(x, idx, descending);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "y ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(yGolden, (T *)yData->data(), 0));
    std::cout << "yIdx ======" << std::endl;
    EXPECT_TRUE(resultCmp<idxT>(yIdxGolden, (idxT *)yIdxData->data(), 0));
}

template <typename T = float, typename idxT = int>
void TopKTest(int tileSize){
    config::SetHostOption(ONLY_CODEGEN, true);
    
    std::vector<int> params(3);
    readInput<int>(GetGoldenDir() + "/params.bin", params);
    int32_t length = params[0];
    bool descending = (bool)params[1];
    int32_t k = params[2];

    DataType dType = DT_FP32;
    DataType idxDType = DT_INT32;
    std::vector<int64_t> shape = {1, length};
    std::vector<int64_t> kShape = {1, k};

    // input & output
    Tensor x(dType, shape, "x");
    Tensor yIdx(idxDType, kShape, "yIdx");

    // output golden
    std::vector<idxT> yIdxGolden = getGoldenVec<idxT>(kShape, "/yidx.bin");

    // input & output data
    auto xData = CreateTensorData<T>(x, shape, "/x.bin");
    auto yIdxData = RawTensorData::CreateConstantTensor<idxT>(yIdx, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {yIdxData};
    std::vector<RawTensorDataPtr> inputDataList = {xData};

    FUNCTION("TopK", {x}, {yIdx}) {
        LOOP("LOOP_1", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1)) {
            UNUSED(bIdx);
            TileShape::Current().SetVecTile({1, tileSize});
            auto res = Sort(x, descending);
            auto resIdx = std::get<1>(res);
            yIdx = View(resIdx, {1, k}, {0, 0});
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "yIdx ======" << std::endl;
    EXPECT_TRUE(resultCmp<idxT>(yIdxGolden, (idxT *)yIdxData->data(), 0));
}

TEST_F(ParallelSortSTest, sort_static) {
    SortStaticTest(256);
}

TEST_F(ParallelSortSTest, sort) {
    SortTest(256);
}

TEST_F(ParallelSortSTest, sort_index) {
    SortWithIndexTest(256);
}

TEST_F(ParallelSortSTest, topk) {
    TopKTest(2048);
}

TEST_F(ParallelSortSTest, fp32_64k) {
    SortTest(8192);
}

TEST_F(ParallelSortSTest, fp32_128k) {
    SortTest(8192);
}

TEST_F(ParallelSortSTest, topk_128k_2k) {
    TopKTest(8192);
}
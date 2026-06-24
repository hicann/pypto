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
 * \file test_dynamic_cast.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "interface/utils/file_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicCastTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

// 设置测试张量
void SetupTestTensors(
    int b, int sq, int d, DataType iType, DataType oType, Tensor& q, Tensor& actSeqs, Tensor& out, bool ready_on_host)
{
    std::vector<int64_t> qShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    q = Tensor(iType, qShape, "q");
    actSeqs = Tensor(DT_INT32, {b, 1}, "actual_seq");
    out = Tensor(oType, outShape, "out");

    if (ready_on_host) {
        config::SetRuntimeOption<std::vector<std::string>>(READY_ON_HOST_TENSORS, {"actual_seq"});
    }
}

// 准备测试数据
void PrepareTestData(int b, int sq, int d, std::vector<int>& actSeqsData, std::vector<int32_t>& golden)
{
    actSeqsData.resize(b, 20);
    golden.resize(b * sq * d, 0);

    for (int bidx = 0; bidx < b; ++bidx) {
        for (int seq = 0; seq < actSeqsData[bidx]; ++seq) {
            for (int dim = 0; dim < d; ++dim) {
                int idx = bidx * sq * d + seq * d + dim;
                golden[idx] = 1;
            }
        }
    }
}

// 构建计算图
void BuildComputeGraph(const Tensor& q, const Tensor& actSeqs, Tensor& out, int sq, int d, DataType oType)
{
    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0});
            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp = Cast(q0, oType, CAST_ROUND);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }
}

// 验证测试结果
void VerifyResults(int b, int sq, int d, const std::vector<int32_t>& golden)
{
    std::vector<float> x(b * sq * d);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    std::vector<int32_t> outVec(
        reinterpret_cast<int32_t*>(outs->data()),
        reinterpret_cast<int32_t*>(outs->data()) + outs->size() / sizeof(int32_t));
    int ret = resultCmpCast<float, int32_t>(x, golden, outVec, 0.001f);
    EXPECT_EQ(ret, true);
}

// 检查生成的代码
void CheckGeneratedCode(bool ready_on_host)
{
    std::string aicpuDirPath = config::LogTopFolder() + "/kernel_aicpu";
    bool foundWaitAicoreStart = false;

    std::vector<std::string> cppFiles = GetFiles(aicpuDirPath, "cpp");
    for (const auto& fileName : cppFiles) {
        if (fileName.substr(0, 15) == "controlFlow_dev") {
            std::string controlFlowFile = aicpuDirPath + "/" + fileName;
            std::ifstream file(controlFlowFile);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (content.find("WaitAicoreStart") != std::string::npos) {
                foundWaitAicoreStart = true;
                break;
            }
        }
    }

    if (ready_on_host) {
        EXPECT_FALSE(foundWaitAicoreStart);
    } else {
        EXPECT_TRUE(foundWaitAicoreStart);
    }
}

void TestDynCastUnalign(bool ready_on_host)
{
    TileShape::Current().SetVecTile(1, 16);
    int b = 1;
    int sq = 32;
    int d = 64;
    DataType iType = DataType::DT_FP32;
    DataType oType = DataType::DT_INT32;

    // 设置测试张量
    Tensor q, actSeqs, out;
    SetupTestTensors(b, sq, d, iType, oType, q, actSeqs, out, ready_on_host);

    // 准备测试数据
    std::vector<int> actSeqsData;
    std::vector<int32_t> golden;
    PrepareTestData(b, sq, d, actSeqsData, golden);

    // 添加输入、输出和金标准数据
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 0),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(out, golden),
    });

    // 构建计算图
    BuildComputeGraph(q, actSeqs, out, sq, d, oType);

    // 运行测试
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), DeviceLauncherConfig(0, 3)); // 看护可重入，连续执行3次

    // 验证结果
    VerifyResults(b, sq, d, golden);

    // 检查生成的代码
    CheckGeneratedCode(ready_on_host);

    unsetenv("ENABLE_CTRLFLOW_COMPILE");
}

TEST_F(DynamicCastTest, testDynCastUnalign) { TestDynCastUnalign(true); }

TEST_F(DynamicCastTest, testDynCastUnalign1) { TestDynCastUnalign(false); }

TEST_F(DynamicCastTest, testDynCastDevSeparate)
{
#ifdef BUILD_WITH_NEW_CANN
    TileShape::Current().SetVecTile(1, 16);

    int b = 1;
    int sq = 32;
    int d = 64;
    std::vector<int64_t> qShape = {b * sq, d};
    std::vector<int64_t> indicesShape = {b, sq};
    std::vector<int64_t> outShape = {b * sq, d};
    DataType iType = DataType::DT_FP32;
    DataType oType = DataType::DT_INT32;

    Tensor q(iType, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(oType, outShape, "out");

    std::vector<int> actSeqsData(b, 20);
    std::vector<int32_t> golden(b * sq * d, 0);
    for (int bidx = 0; bidx < b; ++bidx) {
        for (int seq = 0; seq < actSeqsData[bidx]; ++seq) {
            for (int dim = 0; dim < d; ++dim) {
                int idx = bidx * sq * d + seq * d + dim;
                golden[idx] = 1;
            }
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 0),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0});
            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp = Cast(q0, oType, CAST_ROUND);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    DeviceLauncherConfig config;
    config.repeatNum = 3;
    config.cpuSeparate = true;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config); // 看护可重入，连续执行3次

    std::vector<float> x(b * sq * d);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    std::vector<int32_t> outVec(
        reinterpret_cast<int32_t*>(outs->data()),
        reinterpret_cast<int32_t*>(outs->data()) + outs->size() / sizeof(int32_t));
    int ret = resultCmpCast<float, int32_t>(x, golden, outVec, 0.001f);
    EXPECT_EQ(ret, true);
#endif
}

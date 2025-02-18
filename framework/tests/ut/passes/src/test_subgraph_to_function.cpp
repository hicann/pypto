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
 * \file test_subgraph_to_function.cpp
 * \brief Unit test for SubgraphToFunction pass.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "tilefwk/data_type.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/tile_graph_pass/static_subgraph_processor.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/statistics/execute_graph_statistic.h"
#include "ut_json/ut_json_tool.h"

namespace npu {
namespace tile_fwk {

class SubgraphToFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }

    void TearDown() override {}
};

bool ArePsgHashesUnique(const Function &function) {
    std::unordered_set<size_t> hashSet;
    for (const auto &[psgId, program] : function.programs_) {
        (void)psgId;
        size_t hashValue = program->ComputeHash().GetHash();
        if (hashSet.find(hashValue) != hashSet.end()) {
            return false;
        }
        hashSet.insert(hashValue);
    }
    return true;
}

bool IsPSgToESgMapOneToOne(const std::multimap<int, int> &PSgToESgMap) {
    std::unordered_map<int, int> esgToPsgMap;
    for (const auto &[psgId, esgId] : PSgToESgMap) {
        if (esgToPsgMap.find(esgId) != esgToPsgMap.end()) {
            return false;
        }
        esgToPsgMap[esgId] = psgId;
    }
    return true;
}

std::multimap<int, int> GetPSgToESgMap(Function *rootFunc) {
    std::multimap<int, int> PSgToESgMap;

    for (size_t i = 0; i < rootFunc->Operations().size(); i++) {
        auto iter = rootFunc->Operations()[i].GetSubFuncInvokeInfo();
        int PSgId = iter.GetProgramId();
        int ESgId = rootFunc->Operations()[i].GetSubgraphID();
        PSgToESgMap.insert({PSgId, ESgId});
        ALOG_INFO_F("PSgId: %d, ESgId: %d", PSgId, ESgId);
    }
    return PSgToESgMap;
}

TEST_F(SubgraphToFunctionTest, DifferentOffset) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_DifferentOffset", "TILE_DifferentOffset", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "USE_MAX_FREQ_LABEL", true);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_DifferentOffset", currFunctionPtr);

    constexpr int totalSubGraphCount = 3;
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    constexpr int subGraphID2 = 2;

    constexpr int opMagic0 = 10032;
    constexpr int opMagic1 = 10039;
    constexpr int opMagic2 = 10043;
    constexpr int opMagic3 = 10021;
    constexpr int opMagic4 = 10024;
    constexpr int opMagic5 = 10023;
    constexpr int opMagic6 = 10026;
    constexpr int opMagic7 = 10029;
    constexpr int opMagic8 = 10030;

    constexpr int tensorMagic0 = 3;
    constexpr int tensorMagic1 = 15;
    constexpr int tensorMagic2 = 66;
    constexpr int tensorMagic3 = 79;
    constexpr int tensorMagic4 = 30;
    constexpr int tensorMagic5 = 35;
    constexpr int tensorMagic6 = 29;
    constexpr int tensorMagic7 = 34;
    constexpr int tensorMagic8 = 7;
    // prepare the graph
    std::vector<int64_t> shape0 = {32, 8, 8};
    std::vector<int64_t> shape1 = {16, 64};
    std::vector<int64_t> shape2 = {16, 32};
    std::vector<int64_t> shape3 = {16, 8, 8};
    auto shape3Imme = OpImmediate::Specified(shape3);
    auto shape2Imme = OpImmediate::Specified(shape2);
    auto shape1Imme = OpImmediate::Specified(shape1);
    std::shared_ptr<LogicalTensor> incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape0);
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    incast->SetMagic(tensorMagic0);
    incast->isSubGraphBoundary = true;

    std::shared_ptr<LogicalTensor> tensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    tensor0->SetMemoryTypeBoth(MEM_UB);
    tensor0->SetMagic(tensorMagic1);
    tensor0->subGraphID = subGraphID0;

    auto &copyopin0 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {tensor0});
    copyopin0.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({16, 0, 0}), MEM_UB, shape3Imme, shape3Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin0.UpdateSubgraphID(subGraphID0);
    copyopin0.opmagic = opMagic0;

    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    tensor1->SetMemoryTypeBoth(MEM_UB);
    tensor1->SetMagic(tensorMagic2);
    tensor1->subGraphID = subGraphID0;

    auto &reshapeop = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor0}, {tensor1});
    reshapeop.UpdateSubgraphID(subGraphID0);
    reshapeop.opmagic = opMagic1;

    std::shared_ptr<LogicalTensor> input_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(tensorMagic3);
    input_tensor->isSubGraphBoundary = true;
    input_tensor->subGraphID = subGraphID0;

    auto &copyoutop0 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {input_tensor});
    copyoutop0.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape1Imme, shape1Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop0.UpdateSubgraphID(subGraphID0);
    copyoutop0.opmagic = opMagic2;

    std::shared_ptr<LogicalTensor> inner_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->subGraphID = subGraphID1;
    inner_tensor1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> inner_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 32});
    inner_tensor2->subGraphID = subGraphID2;
    inner_tensor2->SetMagic(tensorMagic5);

    auto &copyopin1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin1.UpdateSubgraphID(subGraphID1);
    copyopin1.opmagic = opMagic3;

    auto &copyopin2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 32}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(subGraphID2);
    copyopin2.opmagic = opMagic4;

    std::shared_ptr<LogicalTensor> result_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->subGraphID = subGraphID1;
    result_tensor1->SetMagic(tensorMagic6);
    std::shared_ptr<LogicalTensor> result_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->subGraphID = subGraphID2;
    result_tensor2->SetMagic(tensorMagic7);
    auto &expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID1);
    expopin1.opmagic = opMagic5;

    auto &expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID2);
    expopin2.opmagic = opMagic6;

    std::shared_ptr<LogicalTensor> output_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(tensorMagic8);
    output_tensor->isSubGraphBoundary = true;

    auto &copyoutop1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(subGraphID1);
    copyoutop1.opmagic = opMagic7;

    auto &copyoutop2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 32}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(subGraphID2);
    copyoutop2.opmagic = opMagic8;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);
    ALOG_INFO("draw program graph before pass.");

    std::stringstream ssBefore;
    ssBefore << "Before_subgraphToFunction";

    // call the pass
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.PreCheck(*currFunctionPtr);
    subgraphToFunction.RunOnFunction(*currFunctionPtr);
    subgraphToFunction.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_subgraphToFunction";

    // do the expect
    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto &PSgToESgMap = GetPSgToESgMap(rootFunc);

    size_t originalSubgraphCount = currFunctionPtr->GetTotalSubGraphCount();
    std::unordered_set<int> uniquePSgIds;
    for (const auto &pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }
    size_t mergedSubgraphCount = uniquePSgIds.size();
    EXPECT_EQ(mergedSubgraphCount, originalSubgraphCount);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, SameOffset) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_SameOffset", "TILE_SameOffset", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_SameOffset", currFunctionPtr);

    constexpr int totalSubGraphCount = 2;
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;

    constexpr int opMagic3 = 10021;
    constexpr int opMagic4 = 10024;
    constexpr int opMagic5 = 10023;
    constexpr int opMagic6 = 10026;
    constexpr int opMagic7 = 10029;
    constexpr int opMagic8 = 10030;

    constexpr int tensorMagic3 = 79;
    constexpr int tensorMagic4 = 30;
    constexpr int tensorMagic5 = 35;
    constexpr int tensorMagic6 = 29;
    constexpr int tensorMagic7 = 34;
    constexpr int tensorMagic8 = 7;
    // prepare the graph
    std::vector<int64_t> shape1 = {16, 64};
    std::vector<int64_t> shape2 = {16, 32};
    std::vector<int64_t> shape3 = {32, 32};
    auto shape2Imme = OpImmediate::Specified(shape2);
    std::shared_ptr<LogicalTensor> input_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(tensorMagic3);
    input_tensor->isSubGraphBoundary = true;

    std::shared_ptr<LogicalTensor> inner_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->subGraphID = subGraphID0;
    inner_tensor1->SetMagic(tensorMagic4);
    std::shared_ptr<LogicalTensor> inner_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 0});
    inner_tensor2->subGraphID = subGraphID1;
    inner_tensor2->SetMagic(tensorMagic5);
    auto &copyopin1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin1.UpdateSubgraphID(subGraphID0);
    copyopin1.opmagic = opMagic3;
    auto &copyopin2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(subGraphID1);
    copyopin2.opmagic = opMagic4;

    std::shared_ptr<LogicalTensor> result_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->subGraphID = subGraphID0;
    result_tensor1->SetMagic(tensorMagic6);
    std::shared_ptr<LogicalTensor> result_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->subGraphID = subGraphID1;
    result_tensor2->SetMagic(tensorMagic7);
    auto &expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID0);
    expopin1.opmagic = opMagic5;
    auto &expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID1);
    expopin2.opmagic = opMagic6;

    std::shared_ptr<LogicalTensor> output_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(tensorMagic8);
    output_tensor->isSubGraphBoundary = true;

    auto &copyoutop1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(subGraphID0);
    copyoutop1.opmagic = opMagic7;
    auto &copyoutop2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({16, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(subGraphID1);
    copyoutop2.opmagic = opMagic8;

    currFunctionPtr->inCasts_.push_back(input_tensor);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);
    ALOG_INFO("draw program graph before pass.");

    Json progDump;
    progDump["version"] = "2.0";
    progDump["functions"].push_back(currFunctionPtr->DumpJson());
    auto filePath = "Before_subgraphIsomorphismPass.json";
    std::ofstream file(filePath);
    file << progDump.dump() << std::endl;
    file.close();

    std::stringstream ssBefore;
    ssBefore << "Before_subgraphToFunction";

    // call the pass
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.PreCheck(*currFunctionPtr);
    subgraphToFunction.RunOnFunction(*currFunctionPtr);
    subgraphToFunction.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_subgraphIsomorphismPass";

    // do the expect
    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto &PSgToESgMap = GetPSgToESgMap(rootFunc);

    std::unordered_set<int> uniquePSgIds;
    for (const auto &pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }
    size_t mergedSubgraphCount = uniquePSgIds.size();
    EXPECT_EQ(mergedSubgraphCount, 1);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load)
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int64_t> shapeA = {bs, m, k};
    std::vector<int64_t> shapeB = {bs, k, n};
    std::vector<int64_t> shapeC = {bs, m, n};

    config::Reset();
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    config::SetBuildStatic(true);
    FUNCTION("BATCHMATMUL", {matA, matB, matC})
    {
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", "DISABLE_PASS", true);
        matC = npu::tile_fwk::Matrix::BatchMatmul<false, false>(DT_FP32, matA, matB);
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedule", "DISABLE_PASS", false);
    auto programJson = Program::GetInstance().DumpJson();
    auto currentFunctionPtr = Program::GetInstance().GetCurrentFunction();
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 6);
    ASSERT_NE(currentFunctionPtr, nullptr);
    EXPECT_EQ(currentFunctionPtr->Operations().size(), 1);
    EXPECT_EQ(currentFunctionPtr->GetRawName(), "PROGRAM_ENTRY");

    auto batchMatmulFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_BATCHMATMUL");
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(batchMatmulFunc->Operations().size(), 9);
    #endif

    ASSERT_NE(batchMatmulFunc->rootFunc_, nullptr);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 4);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 3);
    auto &oriPrograms = batchMatmulFunc->rootFunc_->programs_;
    EXPECT_EQ(oriPrograms[0]->Operations().size(), 1);
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(oriPrograms[1]->Operations().size(), 8);
    #endif
    auto topoBefore = batchMatmulFunc->rootFunc_->topoInfo_;
    auto &entrysBefore = topoBefore.GetTopology();
    EXPECT_EQ(programJson["functions"].size(), 6);
    SubfuncInvokeInfoTy invokeInfo10000;
    SubfuncInvokeInfoTy invokeInfo10001;
    SubfuncInvokeInfoTy invokeInfo10002;
    SubfuncInvokeInfoTy invokeInfo10003;
    for (auto &op : batchMatmulFunc->rootFunc_->Operations()) {
        EXPECT_EQ(op.GetOpcode(), Opcode::OP_CALL);
        if (op.GetOpMagic() == 10000) {
            invokeInfo10000 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10001) {
            invokeInfo10001 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10002) {
            invokeInfo10002 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10003) {
            invokeInfo10003 = op.GetSubFuncInvokeInfo();
        }
    }

    Program::GetInstance().LoadJson(programJson);
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 6);
    auto newCurrFuncPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(newCurrFuncPtr, nullptr);
    // 校验CallOpAttribute
    ASSERT_NE(newCurrFuncPtr->rootFunc_, nullptr);
    EXPECT_EQ(newCurrFuncPtr->rootFunc_->Operations().size(), 4);

    for (auto &op : newCurrFuncPtr->rootFunc_->Operations()) {
        EXPECT_EQ(op.GetOpcode(), Opcode::OP_CALL);
        if (op.GetOpMagic() == 10000) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10000);
        }

        if (op.GetOpMagic() == 10001) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10001);
        }

        if (op.GetOpMagic() == 10002) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10002);
        }

        if (op.GetOpMagic() == 10003) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10003);
        }
    }
    batchMatmulFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_BATCHMATMUL");
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(batchMatmulFunc->Operations().size(), 9);
    #endif
    ASSERT_NE(batchMatmulFunc->rootFunc_, nullptr);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 4);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 3);

    // 校验Topo
    auto &topo = newCurrFuncPtr->rootFunc_->topoInfo_;
    auto &entrys = topo.GetTopology();
    EXPECT_EQ(entrysBefore.size(), entrys.size());

    for (size_t i = 0; i < entrysBefore.size(); i++) {
        EXPECT_EQ(entrys[i].esgId, entrysBefore[i].esgId);
        EXPECT_EQ(entrys[i].readyState, entrysBefore[i].readyState);
        EXPECT_EQ(entrys[i].outGraph, entrysBefore[i].outGraph);
    }

    // 校验rootFunc_->programs_
    auto &programs = newCurrFuncPtr->rootFunc_->programs_;
    ASSERT_EQ(programs.size(), 3);
    EXPECT_EQ(programs[0]->Operations().size(), 1);
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programs[1]->Operations().size(), 8);
    #endif
    // 校验CopyInCopyoutAttribute
    for (auto &op : programs[1]->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            auto copyInOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyInOpAttr, nullptr);
            auto outputDynShape = op.GetOOperands().front()->GetDynValidShape();
            EXPECT_NE(outputDynShape.size(), 0);
        }
        if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            auto copyOutOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyOutOpAttr, nullptr);
        }

    }
    EXPECT_EQ(programs[2]->Operations().size(), 1);

    programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJsonNew.dump(), programJson.dump());
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1) {
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t *)nullptr, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, nullptr, "npu_val"),
                                      Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", "DISABLE_PASS", true);
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)}) {
            output = TopK(input_a, k, -1, isLargest);
        }
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedule", "DISABLE_PASS", false);

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
    #endif
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1_cov) {
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t *)nullptr, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, nullptr, "npu_val"),
                                      Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)}) {
            output = TopK(input_a, k, -1, isLargest);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_2) {
    IfaTileShapeConfig tileConfig {
        256, // block size
        32,  // nTile
        {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {32, 32, 64, 64, 256, 256}, // c1 tile for S1D@S2D
        {32, 256}, // v1 tile for S1S2
        {32, 32, 64, 64, 256, 256}, // c2 tile for S1S2@S2D
        {32, 256}, // v2 tile for S1D
    };

    const int b = 4;
    const int nq = 32;
    const int s2 = 256;
    const int blockSize = tileConfig.blockSize;

    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = 0.8f;

    // 输出size
    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    PROGRAM("PageAttentionStatic") {
        Tensor qNope(DT_BF16, {b * sq * nq, dn}, (uint8_t *)nullptr, "qNope");
        Tensor qRope(DT_BF16, {b * sq * nq, dr}, (uint8_t *)nullptr, "qRope");
        Tensor kNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t *)nullptr, "kNopeCache");
        Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, (uint8_t *)nullptr, "kRope");
        Tensor vNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t *)nullptr, "vNopeCache");

        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));

        Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, nullptr, "attentionOut");

        // 计算流程开始
        config::SetBuildStatic(true);
        FUNCTION("IfaStatic",
            {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, attentionOut}) {
            IncreFlashAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
                attentionOut, tileConfig);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
    #endif
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

/*
* input -> view1(01) -> view1_out -/
*                                  add(03) -> add_out -> abc(04) -> final_out
* input -> View2(01) -> view2_out -/
*/
void InitGraphBuilder (ComputationalGraphBuilder &G, std::vector<int64_t> tileShape) {
    // 1. 定义张量和操作
    std::vector<std::string> tensorNames = {"input", "view1_out", "view2_out", "add_out", "final_out"};
    std::vector<Opcode> opCodes = {
        Opcode::OP_VIEW, 
        Opcode::OP_VIEW,
        Opcode::OP_ADD,
        Opcode::OP_ABS
    };
    std::vector<std::vector<std::string>> ioperands = {
        {"input"},          // view1
        {"input"},          // view2 (确保与view1不同输出)
        {"view1_out", "view2_out"}, // add (两个不同输入)
        {"add_out"}         // abs
    };
    std::vector<std::vector<std::string>> ooperands = {
        {"view1_out"},
        {"view2_out"},
        {"add_out"},
        {"final_out"}
    };
    std::vector<std::string> opNames = {"view1", "view2", "add", "abs_final"};

    // 2. 添加张量和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 3. 设置内存类型和边界张量
    auto input_tensor = G.GetTensor("input");
    auto final_out_tensor = G.GetTensor("final_out");
    input_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    final_out_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    input_tensor->isSubGraphBoundary = true;
    final_out_tensor->isSubGraphBoundary = true;

    // 4. 设置输入输出转换
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    // 5. 设置子图ID（所有操作在同一个子图）
    for (const auto& opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(0);
    }
}

TEST_F(SubgraphToFunctionTest, TestBasicSubgraphConversion) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    // 初始化
    InitGraphBuilder(G, tileShape);

    // 获取Function并执行子图转换Pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(1);  // 总子图数=1

    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    status = pass.PostCheck(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证结果
    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetGraphType(), GraphType::EXECUTE_GRAPH);

    // 检查子图调用信息
    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 1);  // 应有一个子图调用

    // 检查子图内部操作（应保留VIEW+VIEW+ADD+ABS）
    auto leafFunc = rootFunc->programs_.begin()->second;
    EXPECT_EQ(leafFunc->Operations().size(), 4);
}

TEST_F(SubgraphToFunctionTest, MultiSubgraphDependencyWithMixedOps) {
    // 1. 构建包含3个子图的依赖链：AIC -> AIV -> AICPU
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames = {
        "input", "aic_out", "aiv_out", "final_out"
    };

    // 定义操作类型（AIC/AIV/AICPU）
    std::vector<Opcode> opCodes = {
        Opcode::OP_A_MUL_B,  // AIC 子图 (0)
        Opcode::OP_ADD,     // AIV 子图 (1)
        Opcode::OP_EXP      // AICPU 子图 (2)
    };

    // 输入输出张量关系（形成依赖链）
    std::vector<std::vector<std::string>> ioperands = {
        {"input"},          // MATMUL (子图0)
        {"aic_out"},        // ADD (子图1)
        {"aiv_out"}         // EXP (子图2)
    };

    std::vector<std::vector<std::string>> ooperands = {
        {"aic_out"},
        {"aiv_out"},
        {"final_out"}
    };

    std::vector<std::string> opNames = {
        "matmul_aic", "add_aiv", "exp_aicpu"
    };

    // 2. 添加张量和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 3. 显式设置子图ID和核心类型
    G.GetOp("matmul_aic")->UpdateSubgraphID(0);
    G.GetOp("matmul_aic")->SetCoreType(CoreType::AIC);  // 标记为AIC操作
    G.GetOp("matmul_aic")->SetAttribute(OpAttributeKey::isCube, true);

    G.GetOp("add_aiv")->UpdateSubgraphID(1);
    G.GetOp("add_aiv")->SetCoreType(CoreType::AIV);     // 标记为AIV操作

    G.GetOp("exp_aicpu")->UpdateSubgraphID(2);
    G.GetOp("exp_aicpu")->SetCoreType(CoreType::AICPU); // 标记为AICPU操作

    // 4. 设置内存类型和边界张量
    auto input_tensor = G.GetTensor("input");
    auto final_out_tensor = G.GetTensor("final_out");

    // 输入输出为DDR内存
    input_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    final_out_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    // 标记边界张量
    input_tensor->isSubGraphBoundary = true;
    final_out_tensor->isSubGraphBoundary = true;

    // 中间张量作为子图边界
    G.GetTensor("aic_out")->isSubGraphBoundary = true;
    G.GetTensor("aiv_out")->isSubGraphBoundary = true;

    // 5. 设置输入输出转换
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    // 6. 获取Function并设置总子图数
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(3);  // 共3个子图

    // 7. 执行子图转换Pass
    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // 8. 验证结果
    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);

    // 8.1 验证子图数量
    EXPECT_EQ(rootFunc->programs_.size(), 3);  // 应生成3个子图程序

    // 8.2 验证拓扑依赖关系
    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 3);  // 应有3个子图调用

    // 检查依赖链：0 -> 1 -> 2
    EXPECT_EQ(topoInfo.topology_[0].outGraph, std::unordered_set<int>{1});  // 子图0依赖子图1
    EXPECT_EQ(topoInfo.topology_[1].outGraph, std::unordered_set<int>{2});  // 子图1依赖子图2
    EXPECT_TRUE(topoInfo.topology_[2].outGraph.empty());                // 子图2无后继

    // 8.3 验证readyState计算
    EXPECT_EQ(topoInfo.topology_[0].readyState, 0);   // 子图0无前驱，初始ready
    EXPECT_EQ(topoInfo.topology_[1].readyState, -1);  // 子图1依赖子图0
    EXPECT_EQ(topoInfo.topology_[2].readyState, -1);  // 子图2依赖子图1

    // 8.4 验证子图类型分类
    const auto& callOps = rootFunc->Operations();
    ASSERT_EQ(callOps.size(), 3);

    // 检查CallOpAttribute中的graphType
    auto check_graph_type = [&callOps](size_t idx, CoreType expected) {
        auto attr = dynamic_cast<CallOpAttribute*>(callOps[idx].GetOpAttribute().get());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->invokeInfo_->GetGraphType(), expected);
    };

    check_graph_type(0, CoreType::AIC);    // 子图0应为AIC
    check_graph_type(1, CoreType::AIV);    // 子图1应为AIV
    check_graph_type(2, CoreType::AICPU);  // 子图2应为AICPU

    // 8.5 验证ready子图列表
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIC), 1);    // 子图0应在AIC就绪列表
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIV), 0);    // 子图1未就绪
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AICPU), 0);  // 子图2未就绪
}

TEST_F(SubgraphToFunctionTest, EliminateRedundantEdges) {
    ComputationalGraphBuilder G;

    // 定义张量（需要更多张量来创建冗余路径）
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6"};

    // 定义操作和子图分配 - 创建两条路径到MAX_SG3
    std::vector<Opcode> opCodes{
        Opcode::OP_ADD,    // 子图0
        Opcode::OP_CONV,   // 子图1
        Opcode::OP_ABS,    // 子图2
        Opcode::OP_ADD,    // 子图3
        Opcode::OP_MAXIMUM // 子图4
    };

    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t1"},  // ADD1_SG0
        {"t2"},        // CONV_SG1
        {"t2"},        // ABS_SG2
        {"t3", "t4"},  // ADD2_SG3
        {"t4", "t5"}   // MAX_SG4 (接收来自ADD和ABS的输入)
    };

    std::vector<std::vector<std::string>> ooperands{
        {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}
    };

    std::vector<std::string> opNames{
        "ADD1_SG0", "CONV_SG1", "ABS_SG2", "ADD2_SG3", "MAX_SG4"
    };

    // 创建图和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 设置子图ID - 创建跨子图冗余
    G.GetOp("ADD1_SG0")->UpdateSubgraphID(0);
    G.GetOp("CONV_SG1")->UpdateSubgraphID(1);
    G.GetOp("ABS_SG2")->UpdateSubgraphID(2);
    G.GetOp("ADD2_SG3")->UpdateSubgraphID(3);
    G.GetOp("MAX_SG4")->UpdateSubgraphID(4);

    // 设置边界张量
    EXPECT_TRUE(G.SetInCast({"t0", "t1"}));
    EXPECT_TRUE(G.SetOutCast({"t6"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(5);  // 共5个子图
    // 2. 运行SubgraphToFunction pass
    SubgraphToFunction pass;
    pass.SetupStaticProcessor();
    // 构建基础图结构
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);

    // 3. 验证初始边关系（通过消费者关系）
    auto* abs_op = G.GetOp("ABS_SG2");
    auto* add2_op = G.GetOp("ADD2_SG3");
    auto* max_op = G.GetOp("MAX_SG4");

    // 验证ABS_SG2的消费者包含ADD2_SG3、MAX_SG4
    auto abs_consumers1 = abs_op->ConsumerOps();
    EXPECT_TRUE(abs_consumers1.find(add2_op) != abs_consumers1.end());
    EXPECT_TRUE(abs_consumers1.find(max_op) != abs_consumers1.end());

    // 4. 构建颜色图并消除冗余边
    pass.staticProcessor_.BuildColorGraph(*function);
    pass.staticProcessor_.EraseRedundantColorEdges(*function);

    // 5. 验证冗余边已被移除
    const auto& colorOutGraph = pass.staticProcessor_.colorOutGraph;
    const int abs_sgid = G.GetOp("ABS_SG2")->GetSubgraphID();
    const int max_sgid = G.GetOp("MAX_SG4")->GetSubgraphID();
    const auto& abs_out_edges = colorOutGraph[abs_sgid];
    bool found = std::find(abs_out_edges.begin(), abs_out_edges.end(), max_sgid) != abs_out_edges.end();
    EXPECT_FALSE(found) << "Redundant edge not removed!";
}

TEST_F(SubgraphToFunctionTest, ReshapeDependencyHandling) {
    ComputationalGraphBuilder G;

    // 1. 构建测试图：包含一个RESHAPE操作和其消费者
    std::vector<std::string> tensorNames{"t0", "t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands{
        {"t0"},        // RESHAPE_SG0 (无输入子图)
        {"t1"}         // ABS_SG1 (输入来自RESHAPE)
    };
    std::vector<std::vector<std::string>> ooperands{
        {"t1"}, {"t2"}
    };
    std::vector<std::string> opNames{"RESHAPE_SG0", "ABS_SG1"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 2. 设置子图ID（确保RESHAPE是独立子图）
    auto set_subgraph_id = [&G](const std::string& op_name, int id) {
        auto* op = G.GetOp(op_name);
        ASSERT_NE(op, nullptr) << "Operation " << op_name << " not found!";
        op->UpdateSubgraphID(id);
    };
    set_subgraph_id("RESHAPE_SG0", 0);  // RESHAPE单独子图且无输入子图
    set_subgraph_id("ABS_SG1", 1);

    // 3. 构建函数并运行pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);  // 2个子图

    SubgraphToFunction pass;
    pass.SetupStaticProcessor(); // 初始化静态处理器
    // 通过静态处理器调用BuildGraph
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);
    pass.ConstructParamMap(*function);

    // 4. 验证RESHAPE子图的特殊处理
    auto* reshape_op = G.GetOp("RESHAPE_SG0");
    ASSERT_NE(reshape_op, nullptr);
    const int reshape_sgid = reshape_op->GetSubgraphID();

    // 4.1 验证RESHAPE子图被正确标记
    EXPECT_TRUE(pass.staticProcessor_.isReshape[reshape_sgid])
        << "RESHAPE subgraph should be marked when it has no input subgraph and single reshape op";

    EXPECT_TRUE(function->topoInfo_.GetSuccs(reshape_sgid).empty())
    << "RESHAPE subgraph should have empty successors set";

    int expected_out_degree = 0; // 根据实际图结构调整这个值
    bool found = false;
    for (const auto& entry : function->topoInfo_.GetTopology()) {
        if (entry.esgId == 1) { // ABS_SG1的子图ID
            EXPECT_EQ(entry.readyState, -1 * expected_out_degree)
                << "Consumer subgraph's readyOrNot should exclude RESHAPE inputs";
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "ABS_SG1 subgraph entry not found in topology";
}
} // namespace tile_fwk
} // namespace npu
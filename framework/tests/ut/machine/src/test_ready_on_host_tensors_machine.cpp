/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ready_on_host_tensors_machine.cpp
 * \brief Test GetReadyOnHostTensorsSet with list[list[string]] form.
 */
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "machine/host/backend.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

class TestReadyOnHostTensorsMachine : public testing::Test {
public:
    void SetUp() override { config::Reset(); }
    void TearDown() override
    {
        Program::GetInstance().SetCurrentDynamicFunction(nullptr);
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestReadyOnHostTensorsMachine, GetReadyOnHostTensorsSetListListForm)
{
    // 1. Setup jit scope and config with list[list[string]] form.
    ConfigManagerNg::JitScopeGuard guard("jit_scope", std::map<std::string, std::any>{});
    auto scope = ConfigManagerNg::CurrentScope();
    scope->UpdateValueWithAny(std::string("runtime.") + READY_ON_HOST_TENSORS,
                              std::vector<std::pair<std::string, std::string>>{{"a", "a_cpu"}, {"b", "b_cpu"}});

    // 2. Create Function + DyndevFunctionAttribute and set as current dynamic function.
    Program::GetInstance().Reset();
    auto dynFunc = std::make_shared<Function>(Program::GetInstance(), "test_magic", "test", nullptr);
    dynFunc->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    auto dynAttr = std::make_shared<DyndevFunctionAttribute>();
    dynFunc->SetDyndevAttribute(dynAttr);
    Program::GetInstance().InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
    Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());

    // 3. Create LogicalTensors and add to startArgsInputLogicalTensorList.
    //    Indices: 0="a", 1="a_cpu", 2="b", 3="b_cpu"
    Shape shape = {32, 32};
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "a"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "a_cpu"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "b"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "b_cpu"));

    // 4. Call GetReadyOnHostTensorsSet.
    std::unordered_set<int> readyOnHostTensorsSet;
    GetReadyOnHostTensorsSet(readyOnHostTensorsSet);

    // 5. Verify results.
    EXPECT_EQ(readyOnHostTensorsSet.size(), 2);
    EXPECT_TRUE(readyOnHostTensorsSet.count(0) > 0);
    EXPECT_TRUE(readyOnHostTensorsSet.count(2) > 0);
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs.size(), 2);
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[0].first, "a");
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[0].second, "a_cpu");
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[1].first, "b");
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[1].second, "b_cpu");
}

TEST_F(TestReadyOnHostTensorsMachine, GetReadyOnHostTensorsSetLegacyListForm)
{
    // 1. Setup jit scope and config with legacy list[string] form.
    ConfigManagerNg::JitScopeGuard guard("jit_scope", std::map<std::string, std::any>{});
    auto scope = ConfigManagerNg::CurrentScope();
    scope->UpdateValueWithAny(std::string("runtime.") + READY_ON_HOST_TENSORS, std::vector<std::string>{"a", "b"});

    // 2. Create Function + DyndevFunctionAttribute and set as current dynamic function.
    Program::GetInstance().Reset();
    auto dynFunc = std::make_shared<Function>(Program::GetInstance(), "test_magic", "test", nullptr);
    dynFunc->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    auto dynAttr = std::make_shared<DyndevFunctionAttribute>();
    dynFunc->SetDyndevAttribute(dynAttr);
    Program::GetInstance().InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
    Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());

    // 3. Create LogicalTensors and add to startArgsInputLogicalTensorList.
    //    Indices: 0="a", 1="b"
    Shape shape = {32, 32};
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "a"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "b"));

    // 4. Call GetReadyOnHostTensorsSet.
    std::unordered_set<int> readyOnHostTensorsSet;
    GetReadyOnHostTensorsSet(readyOnHostTensorsSet);

    // 5. Verify results: set populated, pairs empty (legacy form has no cpu versions).
    EXPECT_EQ(readyOnHostTensorsSet.size(), 2);
    EXPECT_TRUE(readyOnHostTensorsSet.count(0) > 0);
    EXPECT_TRUE(readyOnHostTensorsSet.count(1) > 0);
    EXPECT_TRUE(dynAttr->readyOnHostCpuPairs.empty());
}

TEST_F(TestReadyOnHostTensorsMachine, GetReadyOnHostTensorsSetListListWithEmptyCpuForm)
{
    // 1. Setup jit scope and config with list[list[string]] form, where "a" has an empty cpu name.
    ConfigManagerNg::JitScopeGuard guard("jit_scope", std::map<std::string, std::any>{});
    auto scope = ConfigManagerNg::CurrentScope();
    scope->UpdateValueWithAny(std::string("runtime.") + READY_ON_HOST_TENSORS,
                              std::vector<std::pair<std::string, std::string>>{{"a", ""}, {"b", "b_cpu"}});

    // 2. Create Function + DyndevFunctionAttribute and set as current dynamic function.
    Program::GetInstance().Reset();
    auto dynFunc = std::make_shared<Function>(Program::GetInstance(), "test_magic", "test", nullptr);
    dynFunc->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    auto dynAttr = std::make_shared<DyndevFunctionAttribute>();
    dynFunc->SetDyndevAttribute(dynAttr);
    Program::GetInstance().InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
    Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());

    // 3. Create LogicalTensors and add to startArgsInputLogicalTensorList.
    //    Indices: 0="a", 1="b", 2="b_cpu"
    Shape shape = {32, 32};
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "a"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "b"));
    dynAttr->startArgsInputLogicalTensorList.push_back(
        std::make_shared<LogicalTensor>(*dynFunc, DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "b_cpu"));

    // 4. Call GetReadyOnHostTensorsSet.
    std::unordered_set<int> readyOnHostTensorsSet;
    GetReadyOnHostTensorsSet(readyOnHostTensorsSet);

    // 5. Verify results: both in set, but only "b" has cpu pair.
    EXPECT_EQ(readyOnHostTensorsSet.size(), 2);
    EXPECT_TRUE(readyOnHostTensorsSet.count(0) > 0);
    EXPECT_TRUE(readyOnHostTensorsSet.count(1) > 0);
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs.size(), 1);
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[0].first, "b");
    EXPECT_EQ(dynAttr->readyOnHostCpuPairs[0].second, "b_cpu");
    EXPECT_EQ(dynAttr->valueDependInputIndices.size(), 1);
    EXPECT_EQ(dynAttr->valueDependInputIndices[0], 2);
}

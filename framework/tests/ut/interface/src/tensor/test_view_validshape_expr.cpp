/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_view_validshape_expr.cpp
 * \brief View 场景表：LogicalTensor dynValidShape 表达式 Dump 黄金值校验。
 *
 * GetViewValidShapeDim 在识别上一层 GVS(v,o1,s1) 时合并为单层 GVS：
 *   mergedOffset = o1 + o2，mergedViewShape = min(s2, s1 - o2)
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {

constexpr int64_t kBaseDim0 = 100;
constexpr int64_t kBaseDim1 = 100;

struct ViewChainStep {
    std::vector<int64_t> offset;
    std::vector<int64_t> shape;
};

struct ViewScenarioSpec {
    std::string name;
    std::vector<ViewChainStep> viewChain; // 空表示不 View，保留基 tensor dynValidShape
    std::vector<std::string> expectedDynValidShapeDump;
};

SymbolicScalar CreateTestScalarVar(const std::string& sym)
{
    static std::unordered_map<std::string, SymbolicScalar> cache;
    auto iter = cache.find(sym);
    if (iter != cache.end()) {
        return iter->second;
    }
    return cache.emplace(sym, IRBuilder().CreateScalarVar(sym)).first->second;
}

std::shared_ptr<LogicalTensor> BuildTensorForViewScenario(const ViewScenarioSpec& spec)
{
    const std::vector<int64_t> baseShape = {kBaseDim0, kBaseDim1};
    const std::vector<int64_t> baseOffset = {0, 0};
    const std::vector<SymbolicScalar> dynValidShape = {
        CreateTestScalarVar(spec.name + "_H"), CreateTestScalarVar(spec.name + "_W")};

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, baseShape);
    auto tensor = IRBuilder().CreateTensorVar(rawTensor, baseOffset, baseShape, dynValidShape);
    auto func = std::make_shared<Function>(
        Program::GetInstance(), "ViewScenario_" + spec.name, "ViewScenario_" + spec.name, nullptr);

    std::shared_ptr<LogicalTensor> current = tensor;
    for (const auto& step : spec.viewChain) {
        current = current->View(*func, step.shape, step.offset);
    }
    return current;
}

std::shared_ptr<LogicalTensor> BuildTensorForConcreteViewScenario(
    const std::vector<int64_t>& baseValidShape, const std::vector<ViewChainStep>& viewChain)
{
    const std::vector<int64_t> baseShape = {kBaseDim0, kBaseDim1};
    const std::vector<int64_t> baseOffset = {0, 0};
    const std::vector<SymbolicScalar> dynValidShape = SymbolicScalar::FromConcrete(baseValidShape);

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, baseShape);
    auto tensor = IRBuilder().CreateTensorVar(rawTensor, baseOffset, baseShape, dynValidShape);
    auto func = std::make_shared<Function>(
        Program::GetInstance(), "ConcreteViewScenario", "ConcreteViewScenario", nullptr);

    std::shared_ptr<LogicalTensor> current = tensor;
    for (const auto& step : viewChain) {
        current = current->View(*func, step.shape, step.offset);
    }
    return current;
}

void ExpectSingleLayerGvsDump(const std::string& dump, const std::string& context)
{
    EXPECT_EQ(dump.find("RUNTIME_GetViewValidShapeDim(RUNTIME_GetViewValidShapeDim"), std::string::npos)
        << context << ": nested GVS not allowed";
    EXPECT_EQ(dump.find("RUNTIME_Min"), std::string::npos) << context << ": Min expansion not allowed";
    EXPECT_EQ(dump.find("RUNTIME_Max"), std::string::npos) << context << ": Max expansion not allowed";
}

std::shared_ptr<LogicalTensor> BuildTensorViaOpView(
    const std::string& name, const std::vector<int64_t>& baseShape, const std::vector<int64_t>& viewShape,
    const std::vector<int64_t>& viewOffset, const std::vector<SymbolicScalar>& baseDynValidShape)
{
    std::shared_ptr<LogicalTensor> resultStorage;
    FUNCTION("OpView_" + name, {}, {})
    {
        Tensor base(DT_FP32, baseShape, "OpViewBase_" + name);
        base.GetStorage()->UpdateDynValidShape(baseDynValidShape);
        Tensor viewed = View(base, viewShape, viewOffset);
        resultStorage = viewed.GetStorage();
    }
    return resultStorage;
}

void RunViewScenarioDumpExpectation(const ViewScenarioSpec& spec)
{
    SCOPED_TRACE(spec.name);
    const auto tensor = BuildTensorForViewScenario(spec);
    const auto& dynValid = tensor->GetDynValidShape();
    ASSERT_EQ(dynValid.size(), spec.expectedDynValidShapeDump.size())
        << spec.name << ": dynValidShape rank mismatch";

    for (size_t dim = 0; dim < dynValid.size(); ++dim) {
        const std::string dump = dynValid[dim].Dump();
        SCOPED_TRACE("dim " + std::to_string(dim));
        EXPECT_EQ(dump, spec.expectedDynValidShapeDump[dim]) << spec.name;
        if (dump.find("RUNTIME_GetViewValidShapeDim") != std::string::npos) {
            ExpectSingleLayerGvsDump(dump, spec.name + " dim" + std::to_string(dim));
        }
    }
}

} // namespace

class TestViewValidShapeExpr : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestViewValidShapeExpr, ViewScenarios_DynValidShapeDumpMatchesExpected)
{
    // 基 tensor：physical shape {100,100}，dynValidShape 符号 {Scenario*_H, Scenario*_W}。
    // 逐场景构造 View 链，断言 GetDynValidShape()[i].Dump()；含 GVS 时须为单层 CALL。
    const std::vector<ViewScenarioSpec> scenarios = {
        // ScenarioA：无 View。
        // 预期 dim0=ScenarioA_H，dim1=ScenarioA_W（原始符号，非 GVS）。
        {"ScenarioA",
            {},
            {"ScenarioA_H", "ScenarioA_W"}},

        // ScenarioB：一次 View，offset {10,20} → shape {30,40}。
        // 预期 dim0=GVS(H,10,30)，dim1=GVS(W,20,40)。
        {"ScenarioB",
            {{{10, 20}, {30, 40}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioB_H,10,30)",
             "RUNTIME_GetViewValidShapeDim(ScenarioB_W,20,40)"}},

        // ScenarioC：一次 View，offset {0,5} → shape {100,25}；dim0 无 offset。
        // 预期 dim0=GVS(H,0,100)，dim1=GVS(W,5,25)。
        {"ScenarioC",
            {{{0, 5}, {100, 25}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioC_H,0,100)",
             "RUNTIME_GetViewValidShapeDim(ScenarioC_W,5,25)"}},

        // ScenarioD：两次 View，step offset 均为 {5,10}；
        //   {5,10}→{50,50}，再 {5,10}→{30,40}；累计 offset {10,20}。
        // 预期与 ScenarioB 相同：GVS(H,10,30)，GVS(W,20,40)（GVS 合并后单层）。
        {"ScenarioD",
            {{{5, 10}, {50, 50}}, {{5, 10}, {30, 40}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioD_H,10,30)",
             "RUNTIME_GetViewValidShapeDim(ScenarioD_W,20,40)"}},

        // ScenarioE：两次 View，step offset 不同；
        //   {3,10}→{50,50}，再 {7,10}→{30,40}；dim0 合并 3+7=10，dim1 合并 10+10=20。
        // 预期与 ScenarioB 相同：GVS(H,10,30)，GVS(W,20,40)。
        {"ScenarioE",
            {{{3, 10}, {50, 50}}, {{7, 10}, {30, 40}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioE_H,10,30)",
             "RUNTIME_GetViewValidShapeDim(ScenarioE_W,20,40)"}},

        // ScenarioF：两次 View；
        //   {0,0}→{100,50}，再 {20,10}→{40,30}；累计 offset {20,10}。
        // 预期 dim0=GVS(H,20,40)，dim1=GVS(W,10,30)。
        {"ScenarioF",
            {{{0, 0}, {100, 50}}, {{20, 10}, {40, 30}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioF_H,20,40)",
             "RUNTIME_GetViewValidShapeDim(ScenarioF_W,10,30)"}},

        // ScenarioG：三次 View；
        //   {3,5}→{90,90}，{4,5}→{60,60}，{3,10}→{30,40}；累计 offset {10,20}。
        // 预期与 ScenarioB 相同：GVS(H,10,30)，GVS(W,20,40)（三次合并仍为单层 GVS）。
        {"ScenarioG",
            {{{3, 5}, {90, 90}}, {{4, 5}, {60, 60}}, {{3, 10}, {30, 40}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioG_H,10,30)",
             "RUNTIME_GetViewValidShapeDim(ScenarioG_W,20,40)"}},

        // ScenarioH：一次全图 View，offset {0,0} → shape {100,100}。
        // 预期 dim0=GVS(H,0,100)，dim1=GVS(W,0,100)。
        {"ScenarioH",
            {{{0, 0}, {100, 100}}},
            {"RUNTIME_GetViewValidShapeDim(ScenarioH_H,0,100)",
             "RUNTIME_GetViewValidShapeDim(ScenarioH_W,0,100)"}},
    };

    for (const auto& scenario : scenarios) {
        RunViewScenarioDumpExpectation(scenario);
    }
}

TEST_F(TestViewValidShapeExpr, OpView_OutOfBounds_Offset60Shape50_DynValidShapeDump)
{
    // 基 tensor physical shape {100,100}，dynValidShape 符号 {OpViewJ_H, OpViewJ_W}。
    // op View：offset {60,50} → shape {50,50}；dim0 物理越界 (60+50=110>100)，dim1 恰达边界 (50+50=100)。
    // op View 不做 LogicalTensor::View 的 shape[i] >= newShape[i]+newOffset[i] 边界断言。
    // GVS 表达式仍按 (validShape, offset, viewShape) 构造，runtime clamp 有效长度。
    const std::vector<SymbolicScalar> baseDynValidShape = {
        CreateTestScalarVar("OpViewJ_H"), CreateTestScalarVar("OpViewJ_W")};
    const auto tensor = BuildTensorViaOpView("J", {kBaseDim0, kBaseDim1}, {50, 50}, {60, 50}, baseDynValidShape);
    ASSERT_NE(tensor, nullptr);

    const std::vector<std::string> expected = {
        "RUNTIME_GetViewValidShapeDim(OpViewJ_H,60,50)", "RUNTIME_GetViewValidShapeDim(OpViewJ_W,50,50)"};
    const auto& dynValid = tensor->GetDynValidShape();
    ASSERT_EQ(dynValid.size(), expected.size());
    for (size_t dim = 0; dim < dynValid.size(); ++dim) {
        const std::string dump = dynValid[dim].Dump();
        SCOPED_TRACE("dim " + std::to_string(dim));
        EXPECT_EQ(dump, expected[dim]);
        ExpectSingleLayerGvsDump(dump, "OpViewJ dim" + std::to_string(dim));
    }
}

TEST_F(TestViewValidShapeExpr, AllImmediateViewChain_FoldsToConcreteDump)
{
    // 基 tensor dynValidShape 为立即数 {100,100}，两次 View 参数也全为立即数。
    // 与 ScenarioD 几何相同：{5,10}→{50,50}，再 {5,10}→{30,40}；累计 offset {10,20}。
    // 预期直接折叠为立即数 dim0=30，dim1=40，不出现 GVS。
    const std::vector<ViewChainStep> viewChain = {{{5, 10}, {50, 50}}, {{5, 10}, {30, 40}}};
    const auto tensor = BuildTensorForConcreteViewScenario({kBaseDim0, kBaseDim1}, viewChain);
    ASSERT_NE(tensor, nullptr);

    const std::vector<int64_t> expectedValues = {30, 40};
    const auto& dynValid = tensor->GetDynValidShape();
    ASSERT_EQ(dynValid.size(), expectedValues.size());
    for (size_t dim = 0; dim < dynValid.size(); ++dim) {
        SCOPED_TRACE("dim " + std::to_string(dim));
        EXPECT_TRUE(dynValid[dim].ConcreteValid()) << "dim" << dim << " should be concrete immediate";
        EXPECT_EQ(dynValid[dim].Concrete(), expectedValues[dim]);
        EXPECT_EQ(dynValid[dim].Dump(), std::to_string(expectedValues[dim]));
        EXPECT_EQ(dynValid[dim].Dump().find("RUNTIME_GetViewValidShapeDim"), std::string::npos)
            << "concrete fold should not emit GVS";
    }
}

TEST_F(TestViewValidShapeExpr, ViewShapeMinusOne_SymbolicValid_EmitsMaxNotGvs)
{
    // viewShape=-1 且 valid 含符号：runtime GVS 不支持第三参 -1，应展开为 Max(valid-offset, 0)。
    const SymbolicScalar h = CreateTestScalarVar("MinusOne_H");
    const auto result = GetViewValidShapeDim(h, SymbolicScalar(0), SymbolicScalar(-1));
    EXPECT_EQ(result.Dump().find("RUNTIME_GetViewValidShapeDim"), std::string::npos);
    EXPECT_NE(result.Dump().find("RUNTIME_Max"), std::string::npos);
    EXPECT_NE(result.Dump().find("MinusOne_H"), std::string::npos);
}

TEST_F(TestViewValidShapeExpr, ViewShapeMinusOne_SecondBoundedView_NoGvsCapMinusOne)
{
    // 第一次 viewShape=-1 → Max；第二次有界 view 走 GVS，第三参为 30 而非 -1。
    const SymbolicScalar h = CreateTestScalarVar("MinusOne2_H");
    const auto afterFirst = GetViewValidShapeDim(h, SymbolicScalar(0), SymbolicScalar(-1));
    const auto afterSecond = GetViewValidShapeDim(afterFirst, SymbolicScalar(5), SymbolicScalar(30));
    EXPECT_EQ(afterSecond.Dump().find(",-1)"), std::string::npos);
    EXPECT_NE(afterSecond.Dump().find("RUNTIME_GetViewValidShapeDim"), std::string::npos);
    EXPECT_NE(afterSecond.Dump().find(",30)"), std::string::npos);
}

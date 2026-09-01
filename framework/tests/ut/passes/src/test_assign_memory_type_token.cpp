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
 * \file test_assign_memory_type_token.cpp
 * \brief Unit test for assign_memory_type token (RAW scenario) adaptation.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"

using namespace npu::tile_fwk;

class AssignMemoryTypeTokenTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

TEST_F(AssignMemoryTypeTokenTest, GetLogicalTensorsByRawTensorBasic)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenBasic")
    {
        Tensor a = Exp(input);
        output = Add(a, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenBasic");
    ASSERT_NE(func, nullptr);

    AssignMemoryType amt;
    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto tensors = amt.GetLogicalTensorsByRawTensor(*func, opList[0]->oOperand.front());
    EXPECT_FALSE(tensors.empty());

    auto inputTensors = amt.GetLogicalTensorsByRawTensor(*func, opList[0]->iOperand.front());
    EXPECT_FALSE(inputTensors.empty());
}

TEST_F(AssignMemoryTypeTokenTest, GetLogicalTensorsByRawTensorNullInput)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenNull") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenNull");
    ASSERT_NE(func, nullptr);

    AssignMemoryType amt;
    auto result = amt.GetLogicalTensorsByRawTensor(*func, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST_F(AssignMemoryTypeTokenTest, ResolveInconsistentRawTensorMemoryTypesFallback)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenInconsistent") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenInconsistent");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, std::vector<int64_t>{32, 32}, TileOpFormat::TILEOP_ND,
                                                 "sharedRaw");

    IRBuilder builder;
    auto srcTile = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 32}, TileOpFormat::TILEOP_ND, "srcTile");
    auto lt1 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{16, 32}, {});
    auto lt2 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 16}, std::vector<int64_t>{16, 32}, {});

    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt1});
    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt2});

    lt1->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
    lt2->SetMemoryTypeOriginal(MemoryType::MEM_L0A, true);

    AssignMemoryType amt;
    EXPECT_EQ(amt.ResolveInconsistentRawTensorMemoryTypes(*func), SUCCESS);

    EXPECT_EQ(lt1->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(lt2->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(AssignMemoryTypeTokenTest, ResolveInconsistentRawTensorMemoryTypesConsistentNoop)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenConsistent") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenConsistent");
    ASSERT_NE(func, nullptr);

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, std::vector<int64_t>{32, 32}, TileOpFormat::TILEOP_ND,
                                                 "sharedRaw2");

    IRBuilder builder;
    auto srcTile = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 32}, TileOpFormat::TILEOP_ND, "srcTile2");
    auto lt1 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{16, 32}, {});
    auto lt2 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 16}, std::vector<int64_t>{16, 32}, {});

    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt1});
    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt2});

    lt1->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
    lt2->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);

    AssignMemoryType amt;
    EXPECT_EQ(amt.ResolveInconsistentRawTensorMemoryTypes(*func), SUCCESS);

    EXPECT_EQ(lt1->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(lt2->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
}

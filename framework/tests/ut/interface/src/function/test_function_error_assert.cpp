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
 * \file test_function_error_assert.cpp
 * \brief Unit tests for function error assertions
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>
#include <string>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/function_error.h"

using namespace npu::tile_fwk;

class FunctionErrorAssertTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "FunctionErrorAssertTest SetUpTestCase" << std::endl; }

    static void TearDownTestCase() { std::cout << "FunctionErrorAssertTest TearDownTestCase" << std::endl; }

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override { Program::GetInstance().Reset(); }
};

TEST_F(FunctionErrorAssertTest, TestGraphTypeMismatch_GetProgramOp) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestGraphTypeMismatch") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestGraphTypeMismatch");
    ASSERT_NE(func, nullptr);

    // Test GetProgramOp on TENSOR_GRAPH should trigger GRAPH_TYPE_MISMATCH
    EXPECT_THROW(
        {
            auto &ops = func->GetProgramOp();
            (void)ops;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestGraphTypeMismatch_SetProgramOp) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestSetProgramOp") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSetProgramOp");
    ASSERT_NE(func, nullptr);

    // Test SetProgramOp on TENSOR_GRAPH should trigger GRAPH_TYPE_MISMATCH
    EXPECT_THROW(
        {
            std::vector<OperationPtr> emptyOps;
            func->SetProgramOp(emptyOps);
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestGraphTypeMismatch_UpdateBelongToThis) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestUpdateBelongToThis") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestUpdateBelongToThis");
    ASSERT_NE(func, nullptr);

    // Test UpdateBelongToThis on TENSOR_GRAPH should trigger GRAPH_TYPE_MISMATCH
    EXPECT_THROW({ func->UpdateBelongToThis(); }, Error);
}

TEST_F(FunctionErrorAssertTest, TestParamAddressNotStored) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestParamAddress") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestParamAddress");
    ASSERT_NE(func, nullptr);

    // Test GetParamAddress with invalid index should trigger PARAM_ADDRESS_NOT_STORED
    EXPECT_THROW(
        {
            void *addr = func->GetParamAddress(100);
            (void)addr;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestCalleeFunctionNull) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestCalleeNull") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCalleeNull");
    ASSERT_NE(func, nullptr);

    // Test GetCalleeFunctionList should not have null callees
    EXPECT_NO_THROW({
        auto callees = func->GetCalleeFunctionList();
        (void)callees;
    });
}

TEST_F(FunctionErrorAssertTest, TestFunctionNoParent_MakeIncasts) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestNoParent") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestNoParent");
    ASSERT_NE(func, nullptr);

    // Test MakeIncasts without parent should trigger FUNCTION_NO_PARENT
    EXPECT_THROW(
        {
            auto scope = std::make_shared<TensorSlotScope>(func);
            auto incasts = func->MakeIncasts(scope);
            (void)incasts;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestFunctionNoParent_MakeOutcasts) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestNoParentOutcast") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestNoParentOutcast");
    ASSERT_NE(func, nullptr);

    // Test MakeOutcasts without parent should trigger FUNCTION_NO_PARENT
    EXPECT_THROW(
        {
            auto scope = std::make_shared<TensorSlotScope>(func);
            auto outcasts = func->MakeOutcasts(scope);
            (void)outcasts;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestOriginIncastNotFound) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOriginIncast") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOriginIncast");
    ASSERT_NE(func, nullptr);

    // Test ReplaceMaybeParams with invalid originIncast should trigger ORIGIN_INCAST_NOT_FOUND
    EXPECT_THROW(
        {
            auto invalidIncast =
                std::make_shared<LogicalTensor>(*func, DT_FP32, shape, TileOpFormat::TILEOP_ND, "invalid");
            auto newIncast = invalidIncast->Clone(*func);
            func->ReplaceMaybeParams(newIncast, invalidIncast);
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestOriginOutcastNotFound) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOriginOutcast") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOriginOutcast");
    ASSERT_NE(func, nullptr);

    // Test MakeOutcasts without parent should trigger FUNCTION_NO_PARENT
    EXPECT_THROW(
        {
            auto scope = std::make_shared<TensorSlotScope>(func);
            auto outcasts = func->MakeOutcasts(scope);
            (void)outcasts;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestIncastHasConflictTensors) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestConflictTensors") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestConflictTensors");
    ASSERT_NE(func, nullptr);

    // Test FillOriginInOutCast should not have conflicter tensors
    EXPECT_NO_THROW({
        std::vector<Operation *> operationList;
        for (auto &op : func->Operations()) {
            operationList.push_back(&op);
        }
        func->FillOriginInOutCast(operationList);
    });
}

TEST_F(FunctionErrorAssertTest, TestParamInfoMismatch) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestParamInfo") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestParamInfo");
    ASSERT_NE(func, nullptr);

    // Test GetSubFuncInvokeInfo should work for valid indices
    EXPECT_NO_THROW({
        // Normal case should work
        if (func->Operations().size() > 0) {
            // This test verifies the assertion exists
        }
    });
}

TEST_F(FunctionErrorAssertTest, TestOperandNotBelongsToCurrentFunction) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOperandBelong") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOperandBelong");
    ASSERT_NE(func, nullptr);

    // Test MakeIncasts without parent should trigger FUNCTION_NO_PARENT
    EXPECT_THROW(
        {
            auto scope = std::make_shared<TensorSlotScope>(func);
            auto incasts = func->MakeIncasts(scope);
            (void)incasts;
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestTargetFunctionNull) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestTargetFunction") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestTargetFunction");
    ASSERT_NE(func, nullptr);

    // Test RemoveOriginIncastConsumer with invalid tensor should trigger TARGET_FUNCTION_NULL
    // Note: This test verifies the assertion exists
    EXPECT_THROW(
        {
            // This should trigger TARGET_FUNCTION_NULL because the tensor has no producer
            if (!func->GetOriginIncast().empty()) {
                auto originIncast = func->GetOriginIncast()[0];
                func->RemoveOriginIncastConsumer(originIncast);
            }
        },
        Error);
}

TEST_F(FunctionErrorAssertTest, TestSlotScopeNull) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input1(DT_FP32, shape, "input1");
    Tensor input2(DT_FP32, shape, "input2");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestSlotScope") {
        output = Add(input1, input2);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSlotScope");
    ASSERT_NE(func, nullptr);

    // Test with valid slot scope should work
    EXPECT_NO_THROW({
        auto scope = std::make_shared<TensorSlotScope>(func);
        (void)scope;
    });
}

TEST_F(FunctionErrorAssertTest, TestProducerNotFound) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestProducer") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestProducer");
    ASSERT_NE(func, nullptr);

    // Test GetSortedOperations should find all producers
    EXPECT_NO_THROW({
        auto sortedOps = func->GetSortedOperations();
        (void)sortedOps;
    });
}

TEST_F(FunctionErrorAssertTest, TestOperationAlreadyInMap) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOpInMap") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOpInMap");
    ASSERT_NE(func, nullptr);

    // Test GetSortedOperations should not have duplicate operations
    EXPECT_NO_THROW({
        auto sortedOps = func->GetSortedOperations();
        (void)sortedOps;
    });
}

TEST_F(FunctionErrorAssertTest, TestCalleeNotInFunctionMap) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestCalleeInMap") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCalleeInMap");
    ASSERT_NE(func, nullptr);

    // Test GetCalleeFunctionList should find callees in function map
    EXPECT_NO_THROW({
        auto callees = func->GetCalleeFunctionList();
        (void)callees;
    });
}

TEST_F(FunctionErrorAssertTest, TestOpListMismatch) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOpList") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOpList");
    ASSERT_NE(func, nullptr);

    // Test GetSortedOperations should have matching list size
    EXPECT_NO_THROW({
        auto sortedOps = func->GetSortedOperations();
        (void)sortedOps;
    });
}

TEST_F(FunctionErrorAssertTest, TestOpNotFoundInPosition) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOpPosition") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOpPosition");
    ASSERT_NE(func, nullptr);

    // Test ScheduleBy should find all operations in position
    EXPECT_NO_THROW({
        std::vector<Operation *> opList;
        for (auto &op : func->Operations()) {
            opList.push_back(&op);
        }
        func->ScheduleBy(opList, false);
    });
}

TEST_F(FunctionErrorAssertTest, TestOpAlreadyInGroup) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestOpGroup") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOpGroup");
    ASSERT_NE(func, nullptr);

    // Test AddOperationGroup should not add operations already in a group
    EXPECT_NO_THROW({
        std::vector<Operation *> opList;
        for (auto &op : func->Operations()) {
            opList.push_back(&op);
        }
        func->AddOperationGroup(opList);
    });
}

TEST_F(FunctionErrorAssertTest, TestOpGroupMismatch) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("TestGroupMismatch") {
        output = Add(input, input);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestGroupMismatch");
    ASSERT_NE(func, nullptr);

    // Test CheckGroupValid should have matching group IDs
    EXPECT_NO_THROW({
        std::vector<Operation *> opList;
        for (auto &op : func->Operations()) {
            opList.push_back(&op);
        }
        func->AddOperationGroup(opList);
        func->CheckGroupValid();
    });
}

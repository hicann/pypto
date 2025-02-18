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
 * \file test_subfunction.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include "passes/pass_utils/pass_utils.h"

using namespace npu::tile_fwk;

class SubFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "SubFunctionTest SetUpTestCase" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SubFunctionTest TearDownTestCase" << std::endl;
    }

    void SetUp() override {
        std::cout << "SubFunctionTest SetUp" << std::endl;
    }

    void TearDown() override {
        std::cout << "SubFunctionTest TearDown" << std::endl;
    }
};

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_PrintInvokeInfo) {
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    subfuncInvokeInfo.PrintInvokeInfo("extra_info");
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_PrettyPrintInvokeInfo1) {
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    subfuncInvokeInfo.PrettyPrintInvokeInfo(123);
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_DumpInvokeInfo1) {
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    std::vector<int64_t> invokeParamVec(10, 10);
    subfuncInvokeInfo.DumpInvokeInfo(0, invokeParamVec.data());
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_LookupInvokeArgs1) {
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.DoFinishRecord();
    subfuncInvokeInfo.ConstructActualInvokeParam(123);
    subfuncInvokeInfo.UpdateProgramSubgraphId(456);

    std::tuple<int, int, int> tp1(123, 0, 0);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0), tp1);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0x10000000), tp1);
    EXPECT_EQ(subfuncInvokeInfo.LookupInvokeArgs(0x20000000), tp1);
}

TEST_F(SubFunctionTest, SubfuncInvokeInfoTy_Print1) {
    SubfuncInvokeInfoTy subfuncInvokeInfo;

    subfuncInvokeInfo.RecordTensorArg(0, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, false, nullptr, 10);
    subfuncInvokeInfo.RecordTensorArg(1, 456, {0, 0}, {128, 128}, {128, 128}, DataType::DT_FP32, false, nullptr, 20);

    subfuncInvokeInfo.RecordConnection(2, 2, 2, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 30);
    subfuncInvokeInfo.RecordConnection(3, 3, 3, 123, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 40);

    std::vector<SubfuncInvokeInfoTy::SuccessorIncastRecTy> inCasts;
    subfuncInvokeInfo.RecordOutcast(4, 0, 4, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 50);
    subfuncInvokeInfo.RecordOutcast(5, 0, 5, 123, inCasts, {0, 0}, {64, 64}, {64, 64}, DataType::DT_FP32, nullptr, 60);

    subfuncInvokeInfo.Print("extra_info");

    subfuncInvokeInfo.SetGraphType(CoreType::AIC);
    EXPECT_EQ(subfuncInvokeInfo.GetGraphType(), CoreType::AIC);
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_TopoSort) {
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    subfuncTopoInfo.TopoSort();

    EXPECT_EQ(subfuncTopoInfo.IsEsgReady(1), true);
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_Print) {
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    subfuncTopoInfo.SetMaxM(10);
    subfuncTopoInfo.Print();
}

TEST_F(SubFunctionTest, SubfuncTopologyInfoTy_DumpEachEntryInfo) {
    SubfuncTopologyInfoTy subfuncTopoInfo;
    int esgId = 0;
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {esgId++});
    subfuncTopoInfo.AddEntry(esgId, 0, {});

    std::vector<int64_t> entryParam(10, 10);
    std::vector<int32_t> readyState(10, 10);
    subfuncTopoInfo.DumpEachEntryInfo(1, CoreType::AIC, 0, entryParam.data(), readyState.data());
}

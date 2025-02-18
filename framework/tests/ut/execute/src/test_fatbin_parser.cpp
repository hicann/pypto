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
 * \file test_fatbin_parser.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "execute/fatbin_parser.h"
#include "tilefwk/op_registry.h"
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
class FatbinParserUnitTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

void DynamicAdd(uint64_t configKey) {
    std::unordered_set<uint64_t> regKeySet = {0, 1, 2};
    if (regKeySet.count(configKey) == 0) {
        return;
    }
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor t1(DT_FP32, {32, 32}, "t1");
    Tensor t2(DT_FP32, {32, 32}, "t2");
    Tensor t3(DT_FP32, {32, 32}, "t3");

    FUNCTION("main", {t0, t1}, {t3}, {{t2, t0}}) {
        LOOP("l0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            t3 = Add(t0, t1);
            Assemble(t3, {0, 0}, t2);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(t3, 0.0f),
    });
}
REGISTER_OP(Add).ImplFunc({{0, DynamicAdd}, {1, DynamicAdd}, {2, DynamicAdd}});

TEST_F(FatbinParserUnitTest, test_tile_fwk_parse_fatbin) {
    bool ret = TileFwkCompileFatbin("Add", "Ascend910B1", "./dump_path", "ast_op_add");
    EXPECT_EQ(ret, true);
    std::string bin_file_path = "./dump_path/ast_op_add.o";
    size_t subkernl_index;
    std::vector<uint8_t> op_binary_bin;
    std::vector<uint8_t> kernel_bin;
    ret = FatbinParser::ParseFatbin(bin_file_path, 0, subkernl_index, op_binary_bin, kernel_bin);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(subkernl_index, 0);
    EXPECT_EQ(op_binary_bin.empty(), false);
    EXPECT_EQ(kernel_bin.empty(), false);
    ret = FatbinParser::ParseFatbin(bin_file_path, 1, subkernl_index, op_binary_bin, kernel_bin);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(subkernl_index, 1);
    EXPECT_EQ(op_binary_bin.empty(), false);
    EXPECT_EQ(kernel_bin.empty(), false);
    ret = FatbinParser::ParseFatbin(bin_file_path, 2, subkernl_index, op_binary_bin, kernel_bin);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(subkernl_index, 2);
    EXPECT_EQ(op_binary_bin.empty(), false);
    EXPECT_EQ(kernel_bin.empty(), false);
}
}

/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_indexput.cpp
 * \brief Unit test for codegen.
 */

#include <iostream>

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "passes/pass_mgr/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class TestCodegenDynIndexPut : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynIndexPut, DynIndexPutUnaligned) {
    int h = 8;
    std::vector<int64_t> vecTileShape = {h};
    std::vector<int64_t> shape1 = {h, h};
    std::vector<int64_t> shape2 = {h, h};
    std::vector<int64_t> shape3 = {h};
    TileShape::Current().SetVecTile(vecTileShape);
    Tensor input(DataType::DT_FP32, shape1, "input");
    Tensor values(DataType::DT_FP32, shape2, "values");
    Tensor indices1(DataType::DT_INT32, shape3, "indices1");
    Tensor output(DataType::DT_FP32, shape1, "output");
    std::string funcName = "IndexPut";
    FUNCTION(funcName, {input, values, indices1}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            IndexPut_(input, {indices1}, values);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + "_hiddenfunc0");
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk

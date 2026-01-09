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
 * \file test_codegen_dyn_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenGatherInUB : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        const constexpr int DummyFuncMagic = 1;
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};
// ----------------- 配置结构体（含类型） -----------------
// IndexT  : topk_indices / page_table 的整数类型
// DataT   : buffer / golden_result 的数据类型
template <typename IndexT, typename DataT>
struct PageAttentionTestConfig {
    using IndexType = IndexT;
    using DataType = DataT;

    int topk_count;         // topk 的 k 值：选出的 token 个数
    int num_logical_blocks; // 逻辑块个数（page_table 长度）
    int num_buffer_tokens;  // buffer 第一维长度：物理 token 容量
    int hidden_dim;         // buffer 第二维长度：隐藏维度大小
    int block_size;         // 每个块里有多少个 token
};
template <typename Config>
void GatherInUBUT(Config &cfg) {
    Shape srcShapes{cfg.num_buffer_tokens, cfg.hidden_dim}; // 网络中，kvcache对应的内存
    Shape offsetsShapes{1, cfg.topk_count};                 // topk的结果
    Shape pageTableShapes{1, cfg.num_logical_blocks};       // page attention 对应的页表
    Shape dstShapes{cfg.topk_count, cfg.hidden_dim};        // 结果，将topk个数据拿出来

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor pageTable(DT_INT32, pageTableShapes, "pageTable");
    Tensor dst(DT_FP16, dstShapes, "dst");
    const std::string funName = "GatherInUB";
    FUNCTION(funName, {src, offsets, pageTable}, {dst}) {
        TileShape::Current().SetVecTile({32, 64});
        std::vector<SymbolicScalar> srcValidShape = {src.GetShape()[0], src.GetShape()[1]};
        Tensor dynSrc = View(src, src.GetShape(), srcValidShape, {0, 0});

        std::vector<SymbolicScalar> offsetsValidShape = {offsets.GetShape()[0], offsets.GetShape()[1]};
        Tensor dynOffsets = View(offsets, offsets.GetShape(), offsetsValidShape, {0, 0});

        dst = experimental::GatherInUB(dynSrc, dynOffsets, pageTable, cfg.block_size, -2);
    }

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_TENSOR_" + funName + "_loop_Unroll1_PATH0_hiddenfunc0");
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funName);
#endif
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
TEST_F(TestCodegenGatherInUB, gather_in_a_) {
    using Config = PageAttentionTestConfig<int32_t, float16>;
    Config cfg;
    cfg.topk_count = 8;         // topk结果
    cfg.num_logical_blocks = 3; // 逻辑块个数
    cfg.num_buffer_tokens = 32; // buffer token 维度（物理 token 容量）
    cfg.hidden_dim = 4;         // 隐藏维度大小
    cfg.block_size = 4;         // 每个块的 token 数
    GatherInUBUT(cfg);
}
} // namespace npu::tile_fwk
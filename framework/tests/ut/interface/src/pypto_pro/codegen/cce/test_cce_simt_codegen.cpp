/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_cce_simt_codegen.cpp
 * \brief End-to-end CCE codegen tests for SIMT functions, callees, context operations, and launch.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codegen/cce/cce_codegen.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

ir::Span Sp() { return ir::Span::Unknown(); }

ir::ExprPtr ConstInt(int64_t value) { return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, Sp()); }

ir::TypePtr Scalar(ir::DataType dtype) { return std::make_shared<const ir::ScalarType>(dtype); }

ir::VarPtr Var(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, Sp());
}

ir::TileTypePtr MakeSimtTileType()
{
    auto memref = std::make_shared<const ir::MemRef>(ir::MemorySpace::Vec, ConstInt(0), 1024);
    return std::make_shared<const ir::TileType>(
        std::vector<int64_t>{1, 256}, ir::DataType::FP32, std::optional<ir::MemRefPtr>(memref),
        std::optional<ir::TileView>(std::nullopt), std::optional<ir::HardwareInfo>(ir::HardwareInfo{}));
}

ir::TensorTypePtr MakeSimtTensorType(const std::string& name)
{
    auto ptr = Var(name + "_base", std::make_shared<const ir::PtrType>(ir::DataType::FP32));
    ir::TensorView view({}, ir::TensorLayout::ND, ptr);
    return std::make_shared<const ir::TensorType>(std::vector<int64_t>{256}, ir::DataType::FP32,
                                                  std::optional<ir::MemRefPtr>(std::nullopt),
                                                  std::optional<ir::TensorView>(view));
}

ir::CallPtr Call(const std::string& name, std::vector<ir::ExprPtr> args, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Call>(name, std::move(args), type, Sp());
}

ir::CallPtr Call(const std::string& name, std::vector<ir::ExprPtr> args, Kwargs kwargs, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Call>(name, std::move(args), std::move(kwargs), type, Sp());
}

ir::StmtPtr Eval(const ir::ExprPtr& expr) { return std::make_shared<const ir::EvalStmt>(expr, Sp()); }

ir::StmtPtr Assign(const ir::VarPtr& var, const ir::ExprPtr& value)
{
    return std::make_shared<const ir::AssignStmt>(var, value, Sp());
}

ir::StmtPtr Seq(std::vector<ir::StmtPtr> stmts) { return std::make_shared<const ir::SeqStmts>(std::move(stmts), Sp()); }

std::string FindLineContaining(const std::string& text, const std::string& needle)
{
    auto pos = text.find(needle);
    if (pos == std::string::npos) {
        return "";
    }
    auto line_begin = text.rfind('\n', pos);
    line_begin = line_begin == std::string::npos ? 0 : line_begin + 1;
    auto line_end = text.find('\n', pos);
    return text.substr(line_begin, line_end - line_begin);
}

std::size_t CountOccurrences(const std::string& text, const std::string& needle)
{
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

ir::FunctionPtr MakeScalarAddCallee(const ir::TypePtr& fp32)
{
    auto value = Var("value", fp32);
    auto delta = Var("delta", fp32);
    auto sum = std::make_shared<const ir::Add>(value, delta, ir::DataType::FP32, Sp());
    auto body = std::make_shared<const ir::ReturnStmt>(std::vector<ir::ExprPtr>{sum}, Sp());
    return std::make_shared<const ir::Function>("add_scalar", std::vector<ir::VarPtr>{value, delta},
                                                std::vector<ir::TypePtr>{fp32}, body, Sp(),
                                                ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeStoreCallee(const ir::TileTypePtr& tile_type, const ir::TypePtr& u32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto index = Var("index", u32);
    auto value = Var("value", fp32);
    auto store = Call("block.setval", {tile, index, value}, ir::GetUnknownType());
    auto body = Seq({Eval(store), std::make_shared<const ir::ReturnStmt>(Sp())});
    return std::make_shared<const ir::Function>("store_scalar", std::vector<ir::VarPtr>{tile, index, value},
                                                std::vector<ir::TypePtr>{}, body, Sp(), ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeApplyCallee(const ir::TileTypePtr& tile_type, const ir::TypePtr& u32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto index = Var("index", u32);
    auto delta = Var("delta", fp32);
    auto value = Var("value", fp32);
    auto updated = Var("updated", fp32);
    auto load = Call("block.getval", {tile, index}, fp32);
    auto add = Call("add_scalar", {value, delta}, fp32);
    auto store = Call("store_scalar", {tile, index, updated}, ir::GetUnknownType());
    auto body = Seq({Assign(value, load), Assign(updated, add), Eval(store),
                     std::make_shared<const ir::ReturnStmt>(std::vector<ir::ExprPtr>{updated}, Sp())});
    return std::make_shared<const ir::Function>("apply_scalar", std::vector<ir::VarPtr>{tile, index, delta},
                                                std::vector<ir::TypePtr>{fp32}, body, Sp(),
                                                ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeSimtEntry(const ir::TileTypePtr& tile_type, const ir::TensorTypePtr& tensor_type,
                              const ir::TypePtr& u32, const ir::TypePtr& i32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto tensor = Var("tensor", tensor_type);
    auto count = Var("count", u32);
    auto delta = Var("delta", fp32);
    auto thread_x = Var("thread_x", u32);
    auto block_y = Var("block_y", u32);
    auto block_z = Var("block_z", u32);
    auto grid_x = Var("grid_x", u32);
    auto tid = Var("tid", u32);
    auto warp = Var("warp", i32);
    auto rows = Var("rows", u32);
    auto cols = Var("cols", u32);
    auto updated = Var("updated", fp32);

    auto context_call = [&](const std::string& name, int axis) { return Call(name, {}, Kwargs{{"axis", axis}}, u32); };
    auto body = Seq({
        Assign(thread_x, context_call("simt.thread_idx", 0)),
        Assign(block_y, context_call("simt.block_dim", 1)),
        Assign(block_z, context_call("simt.block_idx", 2)),
        Assign(grid_x, context_call("simt.grid_dim", 0)),
        Assign(tid, Call("simt.linear_thread_idx", {}, u32)),
        Assign(warp, Call("simt.warp_size", {}, i32)),
        Assign(rows, Call("block.tile_valid_shape", {tile}, Kwargs{{"axis", 0}}, u32)),
        Assign(cols, Call("block.tile_valid_shape", {tile}, Kwargs{{"axis", 1}}, u32)),
        Assign(updated, Call("apply_scalar", {tile, tid, delta}, fp32)),
        Eval(Call("block.setval", {tensor, tid, updated}, ir::GetUnknownType())),
    });
    return std::make_shared<const ir::Function>(
        "simt_entry", std::vector<ir::VarPtr>{tile, tensor, count, delta}, std::vector<ir::TypePtr>{}, body, Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 256}});
}

ir::ProgramPtr MakeSimtProgram()
{
    auto u32 = Scalar(ir::DataType::UINT32);
    auto i32 = Scalar(ir::DataType::INT32);
    auto fp32 = Scalar(ir::DataType::FP32);
    auto tile_type = MakeSimtTileType();
    auto tensor_type = MakeSimtTensorType("tensor");

    auto tensor = Var("tensor", tensor_type);
    auto count = Var("count", u32);
    auto delta = Var("delta", fp32);
    auto tile = Var("tile", tile_type);
    auto make_tile = Call("block.make_tile", {}, tile_type);
    auto launch = Call("simt.launch", {ConstInt(8), ConstInt(4), ConstInt(8), tile, tensor, count, delta},
                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 256}}, ir::GetUnknownType());
    auto kernel_body = Seq({Assign(tile, make_tile), Eval(launch)});
    auto kernel = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{tensor, count, delta},
                                                       std::vector<ir::TypePtr>{}, kernel_body, Sp(),
                                                       ir::FunctionType::IN_CORE, true);

    std::vector<ir::FunctionPtr> functions = {MakeScalarAddCallee(fp32), MakeStoreCallee(tile_type, u32, fp32),
                                              MakeApplyCallee(tile_type, u32, fp32),
                                              MakeSimtEntry(tile_type, tensor_type, u32, i32, fp32), kernel};
    return std::make_shared<const ir::Program>(std::move(functions), "simt_program", Sp());
}

ir::ProgramPtr MakeSimtCalleeNameCollisionProgram()
{
    auto i32 = Scalar(ir::DataType::INT32);
    auto callee_body = std::make_shared<const ir::ReturnStmt>(Sp());
    auto callee = std::make_shared<const ir::Function>("get_subblock_idx", std::vector<ir::VarPtr>{},
                                                       std::vector<ir::TypePtr>{}, callee_body, Sp(),
                                                       ir::FunctionType::SIMT_CALLEE);

    auto simt_body = Eval(Call("get_subblock_idx", {}, ir::GetUnknownType()));
    auto simt_entry = std::make_shared<const ir::Function>(
        "collision_entry", std::vector<ir::VarPtr>{}, std::vector<ir::TypePtr>{}, simt_body, Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 32}});

    auto subblock_idx = Var("subblock_idx", i32);
    auto backend_call = Call("get_subblock_idx", {}, i32);
    auto launch = Call("simt.launch", {ConstInt(32), ConstInt(1), ConstInt(1)},
                       Kwargs{{"callee", std::string("collision_entry")}, {"max_threads", 32}}, ir::GetUnknownType());
    auto kernel_body = Seq({Assign(subblock_idx, backend_call), Eval(backend_call), Eval(launch)});
    auto kernel = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{}, std::vector<ir::TypePtr>{},
                                                       kernel_body, Sp(), ir::FunctionType::IN_CORE, true);

    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{callee, simt_entry, kernel},
                                               "simt_callee_name_collision", Sp());
}

} // namespace

TEST(CCESimtCodegenTest, GeneratesContextCalleesMemoryAccessAndLaunch)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtProgram(), "a5");

    auto add_pos = generated.find("__simt_callee__ inline float add_scalar(");
    auto store_pos = generated.find("__simt_callee__ inline void store_scalar(");
    auto apply_pos = generated.find("__simt_callee__ inline float apply_scalar(");
    auto entry_pos = generated.find("__simt_vf__ __launch_bounds__(256) inline void simt_entry(");
    ASSERT_NE(add_pos, std::string::npos);
    ASSERT_NE(store_pos, std::string::npos);
    ASSERT_NE(apply_pos, std::string::npos);
    ASSERT_NE(entry_pos, std::string::npos);
    EXPECT_LT(add_pos, apply_pos);
    EXPECT_LT(store_pos, apply_pos);
    EXPECT_LT(apply_pos, entry_pos);

    auto store_signature = FindLineContaining(generated, "__simt_callee__ inline void store_scalar(");
    auto apply_signature = FindLineContaining(generated, "__simt_callee__ inline float apply_scalar(");
    auto entry_signature = FindLineContaining(generated, "__simt_vf__ __launch_bounds__(256) inline void simt_entry(");
    ASSERT_FALSE(store_signature.empty());
    ASSERT_FALSE(apply_signature.empty());
    ASSERT_FALSE(entry_signature.empty());
    EXPECT_NE(store_signature.find("__ubuf__ float*"), std::string::npos);
    EXPECT_NE(store_signature.find("__valid_row"), std::string::npos);
    EXPECT_NE(store_signature.find("__valid_col"), std::string::npos);
    EXPECT_NE(apply_signature.find("__ubuf__ float*"), std::string::npos);
    EXPECT_NE(entry_signature.find("__gm__ float*"), std::string::npos);

    EXPECT_NE(generated.find("threadIdx.x"), std::string::npos);
    EXPECT_NE(generated.find("blockDim.y"), std::string::npos);
    EXPECT_NE(generated.find("blockIdx.z"), std::string::npos);
    EXPECT_NE(generated.find("gridDim.x"), std::string::npos);
    EXPECT_NE(generated.find("threadIdx.y * blockDim.x"), std::string::npos);
    EXPECT_NE(generated.find("warpSize"), std::string::npos);

    auto store_function = generated.substr(store_pos, apply_pos - store_pos);
    auto apply_function = generated.substr(apply_pos, entry_pos - apply_pos);
    auto entry_function_end = generated.find("__aicore__ inline void kernel_impl_vector(", entry_pos);
    ASSERT_NE(entry_function_end, std::string::npos);
    auto entry_function = generated.substr(entry_pos, entry_function_end - entry_pos);
    EXPECT_NE(store_function.find("] ="), std::string::npos);
    EXPECT_EQ(store_function.find(".SetValue("), std::string::npos);
    EXPECT_NE(apply_function.find("["), std::string::npos);
    EXPECT_EQ(apply_function.find(".GetValue("), std::string::npos);
    EXPECT_NE(apply_function.find("store_scalar("), std::string::npos);
    EXPECT_NE(entry_function.find("apply_scalar("), std::string::npos);
    EXPECT_NE(entry_function.find("*((__gm__ float*)"), std::string::npos);

    auto launch_line = FindLineContaining(generated, "cce::async_invoke<simt_entry>");
    ASSERT_FALSE(launch_line.empty());
    EXPECT_NE(launch_line.find("cce::dim3{8, 4, 8}"), std::string::npos);
    EXPECT_EQ(CountOccurrences(launch_line, "(__ubuf__ float*)"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".data()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".GetValidRow()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".GetValidCol()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, "(__gm__ float*)"), 1u);
    EXPECT_EQ(generated.find("pipe_barrier(PIPE_ALL);"), std::string::npos);
    EXPECT_NE(generated.find("__aicore__ inline void kernel_impl_vector("), std::string::npos);
}

TEST(CCESimtCodegenTest, RejectsUnsupportedTargetAndArchitecture)
{
    CCECodegen cube_codegen(ir::SectionKind::Cube);
    EXPECT_THROW((void)cube_codegen.GenerateSingle(MakeSimtProgram(), "a5"), npu::tile_fwk::Error);

    CCECodegen vector_codegen(ir::SectionKind::Vector);
    EXPECT_THROW((void)vector_codegen.GenerateSingle(MakeSimtProgram(), "a3"), npu::tile_fwk::Error);
}

TEST(CCESimtCodegenTest, DoesNotInterceptOrdinaryKernelCallsWithMatchingCalleeName)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtCalleeNameCollisionProgram(), "a5");

    auto entry_pos = generated.find("__simt_vf__ __launch_bounds__(32) inline void collision_entry(");
    auto kernel_pos = generated.find("__aicore__ inline void kernel_impl_vector(");
    ASSERT_NE(entry_pos, std::string::npos);
    ASSERT_NE(kernel_pos, std::string::npos);

    auto entry_function = generated.substr(entry_pos, kernel_pos - entry_pos);
    auto kernel_function = generated.substr(kernel_pos);
    EXPECT_EQ(CountOccurrences(entry_function, "get_subblock_idx();"), 1u);
    EXPECT_EQ(CountOccurrences(kernel_function, "(int32_t)(get_subblockid())"), 1u);
}

} // namespace codegen
} // namespace pypto

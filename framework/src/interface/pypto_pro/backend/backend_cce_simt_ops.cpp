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
 * @file backend_cce_simt_ops.cpp
 * \brief Direct CCE lowering for the A5 SIMT context and launch operations.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace backend {

namespace {

const char* GetSimtAxisName(int axis)
{
    constexpr const char* axis_names[] = {"x", "y", "z"};
    return axis_names[axis];
}

std::string MakeSimtContextComponentCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               const char* op_name, const char* context_name)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(codegen.IsInSimtContext()) << op_name << " reached CCE codegen outside a SIMT function";
    return std::string(context_name) + "." + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtThreadIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.thread_idx", "threadIdx");
}

std::string MakeSimtBlockDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.block_dim", "blockDim");
}

std::string MakeSimtBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.block_idx", "blockIdx");
}

std::string MakeSimtGridDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.grid_dim", "gridDim");
}

std::string MakeSimtLinearThreadIdxCodegenCCE([[maybe_unused]] const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(codegen.IsInSimtContext()) << "simt.linear_thread_idx reached CCE codegen outside a SIMT function";
    return "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
}

std::string MakeSimtWarpSizeCodegenCCE([[maybe_unused]] const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(codegen.IsInSimtContext()) << "simt.warp_size reached CCE codegen outside a SIMT function";
    return "warpSize";
}

std::string MakeSimtLaunchCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(!codegen.IsInSimtContext()) << "Nested simt.launch is not supported";
    CHECK(codegen.GetTarget() == ir::SectionKind::Vector) << "simt.launch requires the Vector target";
    CHECK(codegen.GetArch() == "a5") << "simt.launch currently requires arch='a5'";
    int64_t thread_dims[3] = {};
    for (size_t i = 0; i < 3; ++i) {
        auto dim = ir::As<ir::ConstInt>(op->args_[i]);
        thread_dims[i] = dim->value_;
    }

    std::ostringstream call;
    call << "cce::async_invoke<" << op->GetKwarg<std::string>("callee") << ">(cce::dim3{" << thread_dims[0] << ", "
         << thread_dims[1] << ", " << thread_dims[2] << "}";
    for (size_t i = 3; i < op->args_.size(); ++i) {
        auto arg_type = op->args_[i]->GetType();
        auto tile_type = ir::As<ir::TileType>(arg_type);
        if (tile_type != nullptr) {
            std::string tile = codegen.GetExprAsCode(op->args_[i]);
            call << ", (__ubuf__ " << tile_type->dtype_.ToCTypeString() << "*)" << tile << ".data()";
            call << ", (uint32_t)" << tile << ".GetValidRow(), (uint32_t)" << tile << ".GetValidCol()";
        } else if (auto tensor_type = ir::As<ir::TensorType>(arg_type)) {
            auto tensor_var = ir::As<ir::Var>(op->args_[i]);
            std::string tensor_name = codegen.GetVarName(tensor_var);
            call << ", (__gm__ " << tensor_type->dtype_.ToCTypeString() << "*)" << codegen.GetPointer(tensor_name);
        } else {
            call << ", " << codegen.GetExprAsCode(op->args_[i]);
        }
    }
    call << ");";
    codegen.Emit(call.str());
    return "";
}

} // namespace

REGISTER_BACKEND_OP(BackendCCE, "simt.thread_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.grid_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtGridDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.linear_thread_idx")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtLinearThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.warp_size").set_pipe(ir::PipeType::S).f_codegen(MakeSimtWarpSizeCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.launch").set_pipe(ir::PipeType::V).f_codegen(MakeSimtLaunchCodegenCCE);

} // namespace backend
} // namespace pypto

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
 * @file simt_ops.cpp
 * \brief SIMT context and launch operations used by PyPTO Pro's A5 direct-CCE path.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceSimtContextComponentType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "SIMT context operations do not accept positional arguments";
    int axis = GetOpKwarg<int>(kwargs, "axis", 0);
    CHECK(axis >= 0 && axis <= 2) << "SIMT context axis must be in [0, 2]";
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtLinearThreadIdxType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "SIMT builtin scalar operations do not accept positional arguments";
    CHECK(kwargs.empty()) << "SIMT builtin scalar operations do not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtWarpSizeType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "simt.warp_size does not accept positional arguments";
    CHECK(kwargs.empty()) << "simt.warp_size does not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::INT32));
}

TypePtr DeduceSimtLaunchType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.size() >= 3) << "simt.launch requires three launch dimensions";
    int64_t total_threads = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        auto dim_type = As<ScalarType>(args[i]->GetType());
        CHECK(dim_type && dim_type->dtype_.IsInt() && dim_type->dtype_ != DataType(DataType::BOOL))
            << "simt.launch dimensions must be non-bool integer scalars";
        auto dim = As<ConstInt>(args[i]);
        CHECK(dim && dim->value_ > 0 && dim->value_ <= 2048)
            << "simt.launch dimensions must be compile-time integers in [1, 2048]";
        total_threads *= dim->value_;
    }
    CHECK(total_threads <= 2048) << "simt.launch total thread count must not exceed 2048";
    int max_threads = GetOpKwarg<int>(kwargs, "max_threads");
    CHECK(max_threads >= 1 && max_threads <= 2048) << "simt.launch max_threads must be in [1, 2048]";
    CHECK(total_threads <= max_threads) << "simt.launch threads " << total_threads << " exceed callee max_threads "
                                        << max_threads;
    for (std::size_t i = 3; i < args.size(); ++i) {
        auto arg_type = args[i]->GetType();
        if (auto tile_type = As<TileType>(arg_type)) {
            CHECK(tile_type->shape_.size() == 2) << "simt.launch Tile argument must have a two-dimensional shape";
            CHECK(As<Var>(args[i]) != nullptr) << "simt.launch Tile argument must be a Var";
            for (const auto& dim : tile_type->shape_) {
                CHECK(As<ConstInt>(dim) != nullptr) << "simt.launch Tile argument must have a static shape";
            }
            CHECK(tile_type->dtype_.GetBit() >= 8) << "simt.launch Tile argument has a sub-byte element dtype";
            CHECK(tile_type->memref_.has_value()) << "simt.launch Tile argument has no memory reference";
            CHECK((*tile_type->memref_)->memorySpace_ == MemorySpace::Vec)
                << "simt.launch Tile argument must be a Vec-memory Tile";
            CHECK(tile_type->hardwareInfo_.has_value() && tile_type->hardwareInfo_->blayout == TileLayout::row_major &&
                  tile_type->hardwareInfo_->slayout == TileLayout::none_box)
                << "simt.launch Tile argument must be an ND Vec Tile";
        } else if (auto tensor_type = As<TensorType>(arg_type)) {
            CHECK(!tensor_type->shape_.empty()) << "simt.launch Tensor argument must have a non-empty shape";
            for (const auto& dim : tensor_type->shape_) {
                CHECK(As<ConstInt>(dim) != nullptr) << "simt.launch Tensor argument must have a static shape";
            }
            CHECK(tensor_type->dtype_.GetBit() >= 8) << "simt.launch Tensor argument has a sub-byte element dtype";
            CHECK(tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->layout == TensorLayout::ND)
                << "simt.launch Tensor argument must use ND layout";
            CHECK(As<Var>(args[i]) != nullptr) << "simt.launch Tensor argument must be a Var";
        } else {
            CHECK(As<ScalarType>(arg_type)) << "simt.launch supports only Tile, Tensor, and scalar arguments";
        }
    }
    return GetUnknownType();
}

} // namespace

REGISTER_OP("simt.thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT thread index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.grid_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT grid dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.linear_thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read the x-major flattened thread index within the current block")
    .no_argument()
    .f_deduce_type(DeduceSimtLinearThreadIdxType);

REGISTER_OP("simt.warp_size")
    .set_op_category("SimtOp")
    .set_description("Read the native SIMT warp size")
    .no_argument()
    .f_deduce_type(DeduceSimtWarpSizeType);

REGISTER_OP("simt.launch")
    .set_op_category("SimtOp")
    .set_description("Launch a SIMT vector function from an AIV kernel")
    .add_argument("threads_x", "Compile-time X dimension")
    .add_argument("threads_y", "Compile-time Y dimension")
    .add_argument("threads_z", "Compile-time Z dimension")
    .set_attr<std::string>("callee")
    .set_attr<int>("max_threads")
    .f_deduce_type(DeduceSimtLaunchType);

} // namespace ir
} // namespace pypto

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
 * @file vf_ops.cpp
 * \brief VF API IR op registration.
 *
 * Registers VF ops with type deduction for the direct VF instruction path.
 * These ops are part of the merged shared IR surface and remain consumed by
 * the block VF frontend/backend path.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceVFUnknownType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return GetUnknownType();
}

TypePtr DeduceVFScalarType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {     
            return std::make_shared<ScalarType>(AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    return std::make_shared<ScalarType>(DataType::FP32);
}

TypePtr DeduceVFMaskType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return std::make_shared<ScalarType>(DataType::UINT16);
}

TypePtr DeduceVFFromDstArg(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    if (!args.empty()) {
        return args[0]->GetType();
    }
    return std::make_shared<ScalarType>(DataType::FP32);
}
} // namespace

// Register declaration
REGISTER_OP("vf.RegTensor")
    .set_op_category("VFOp")
    .set_description("Declare a vector register")
    .no_argument()
    .f_deduce_type(DeduceVFScalarType);

// Mask
REGISTER_OP("vf.CreateMask")
    .set_op_category("VFOp")
    .set_description("Declare and initialize a mask register")
    .no_argument()
    .f_deduce_type(DeduceVFMaskType);

// Initialization
REGISTER_OP("vf.Duplicate")
    .set_op_category("VFOp")
    .set_description("Scalar broadcast to register")
    .add_argument("dst", "Destination register")
    .add_argument("scalar", "Scalar value to broadcast")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

// Data transfer
REGISTER_OP("vf.LoadAlign")
    .set_op_category("VFOp")
    .set_description("Load aligned data from UB to register")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source UB pointer")
    .add_argument("offset", "Element offset into UB")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.StoreAlign")
    .set_op_category("VFOp")
    .set_description("Store aligned data from register to UB")
    .add_argument("dst", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFUnknownType);

// Compute
REGISTER_OP("vf.Max")
    .set_op_category("VFOp")
    .set_description("Vector maximum")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Add")
    .set_op_category("VFOp")
    .set_description("Vector addition")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Sub")
    .set_op_category("VFOp")
    .set_description("Vector subtraction")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.And")
    .set_op_category("VFOp")
    .set_description("Bitwise AND")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Xor")
    .set_op_category("VFOp")
    .set_description("Bitwise XOR")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Mul")
    .set_op_category("VFOp")
    .set_description("Vector multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.MulAddDst")
    .set_op_category("VFOp")
    .set_description("Fused multiply-add (FMA): dst = src0 * src1 + dst (hardware vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0 (multiplicand)")
    .add_argument("src1", "Source register 1 (multiplicand)")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Div")
    .set_op_category("VFOp")
    .set_description("Vector division")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.ShiftRights")
    .set_op_category("VFOp")
    .set_description("Right shift")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("shift_bits", "Number of bits to shift")
    .add_argument("mask", "Mask register")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      if (!args.empty()) return args[0]->GetType();
      return std::make_shared<ScalarType>(DataType::UINT32);
    });

REGISTER_OP("vf.Muls")
    .set_op_category("VFOp")
    .set_description("Scalar multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Ln")
    .set_op_category("VFOp")
    .set_description("Natural logarithm")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.FusedExpSub")
    .set_op_category("VFOp")
    .set_description("Fused exp(src - max)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("max", "Max register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Cast")
    .set_op_category("VFOp")
    .set_description("Type conversion")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.DeInterleave")
    .set_op_category("VFOp")
    .set_description("De-interleave")
    .add_argument("dst0", "Destination register 0")
    .add_argument("dst1", "Destination register 1")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Select")
    .set_op_category("VFOp")
    .set_description("Conditional select")
    .add_argument("dst", "Destination register")
    .add_argument("src_true", "True branch register")
    .add_argument("src_false", "False branch register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.UpdateMask")
    .set_op_category("VFOp")
    .set_description("Update mask with scalar value")
    .add_argument("scalar", "Scalar value for mask")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.MemBar")
    .set_op_category("VFOp")
    .set_description("Memory barrier")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

// TopK VF ops
REGISTER_OP("vf.Histograms")
    .set_op_category("VFOp")
    .set_description("Histogram accumulation")
    .add_argument("dst", "Destination histogram register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Compare")
    .set_op_category("VFOp")
    .set_description("Compare and generate mask")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      return std::make_shared<ScalarType>(DataType::UINT16);
    });

REGISTER_OP("vf.Squeeze")
    .set_op_category("VFOp")
    .set_description("Squeeze mask to indices")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Arange")
    .set_op_category("VFOp")
    .set_description("Generate incrementing index sequence")
    .add_argument("dst", "Destination register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.Gather")
    .set_op_category("VFOp")
    .set_description("Gather elements by indices")
    .add_argument("dst", "Destination register")
    .add_argument("src_ub", "Source UB pointer")
    .add_argument("indices", "Index register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.StoreUnAlign")
    .set_op_category("VFOp")
    .set_description("Store unaligned data with post update")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("align_reg", "Alignment register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.StoreUnAlignPost")
    .set_op_category("VFOp")
    .set_description("Complete unaligned store")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("align_reg", "Alignment register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.UnalignRegForStore")
    .set_op_category("VFOp")
    .set_description("Declare unaligned register for store")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.ClearSpr")
    .set_op_category("VFOp")
    .set_description("Clear special purpose register AR")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

}  // namespace ir
}  // namespace pypto

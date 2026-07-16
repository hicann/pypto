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
 * @file backend_950_cce_vf_ops.cpp
 * \brief CCE backend op registration for VF API operations (A5 target).
 *
 * VF ops directly emit VF instructions (vlds, vmax, vdup, etc.) without
 * going through the PTO-ISA intermediate layer. API naming references AscendC.
 */

#include <string>

#include "backend/910B_CCE/backend_910b_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;

// ============================================================================
// Scope markers
// ============================================================================

// ============================================================================
// RegTensor declaration
// ============================================================================

static std::string EmitVFRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto dtype = op->GetKwarg<DataType>("dtype");
    std::string reg_name = codegen.GetCurrentResultTarget();

    codegen.Emit("RegTensor<" + dtype.ToCTypeString() + "> " + reg_name + ";");
    return "";
}

// Best-effort dtype extraction from an Expr's deduced type. Handles both
// ScalarType (RegTensor outputs) and ShapedType (Tile / Tensor expressions).
static DataType GetExprDtype(const ir::ExprPtr& expr, DataType fallback = DataType::UINT32)
{
    auto type = expr->GetType();
    if (auto st = ir::As<ir::ScalarType>(type))
        return st->dtype_;
    if (auto sh = ir::As<ir::ShapedType>(type))
        return sh->dtype_;
    return fallback;
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pattern = op->GetKwarg<std::string>("pattern");
    auto dtype = op->GetKwarg<DataType>("dtype");
    std::string reg_name = codegen.GetCurrentResultTarget();

    codegen.Emit("MaskReg " + reg_name + ";");

    // Select pset instruction based on data element size (not mask type)
    // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
    if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
        codegen.Emit(reg_name + " = pset_b8(PAT_ALL);");
    } else if (dtype == DataType::FP32 || dtype == DataType::INT32 || dtype == DataType::UINT32) {
        codegen.Emit(reg_name + " = pset_b32(PAT_ALL);");
    } else {
        // FP16, BF16, UINT16, INT16 etc. (2 bytes)
        codegen.Emit(reg_name + " = pset_b16(PAT_ALL);");
    }

    return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, scalar, (optional) mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[1]);

    if (op->args_.size() >= 3) {
        // With mask: vdup(dst, scalar, preg, MODE_ZEROING)
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("vdup(" + dst + ", " + scalar_str + ", " + mask + ", MODE_ZEROING);");
    } else {
        // Without mask: vbr(dst, scalar)
        codegen.Emit("vbr(" + dst + ", " + scalar_str + ");");
    }

    return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or tile-flavored GetItemExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& cast_type = "float")
{
    std::string ptr = codegen.GetOrCreateVFTilePtr(expr, /*is_post_update=*/false);
    std::string tile_ctype = GetExprDtype(expr, DataType::FP32).ToCTypeString();
    if (cast_type == tile_ctype)
        return ptr;
    return "(__ubuf__ " + cast_type + " *)" + ptr;
}

// ============================================================================
// LoadAlign — vlds / plds
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string dst_reg = codegen.GetExprAsCode(op->args_[0]);

    // Get dist mode with default NORM
    std::string dist = "NORM";
    if (op->HasKwarg("dist")) {
        dist = op->GetKwarg<std::string>("dist");
    }

    // vlds requires dst and ptr type match — derive ptr type from dst RegTensor's dtype (args[0]).
    std::string ptr_type = "float";
    if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType())) {
        DataType dst_dtype = scalar_type->dtype_;
        if (dst_dtype == DataType::FP16 || dst_dtype == DataType::BF16) {
            ptr_type = "half";
        } else if (dst_dtype == DataType::UINT16 || dst_dtype == DataType::INT16) {
            ptr_type = "uint16_t";
        } else if (dst_dtype == DataType::UINT8 || dst_dtype == DataType::INT8) {
            ptr_type = "uint8_t";
        } else if (dst_dtype == DataType::UINT32 || dst_dtype == DataType::INT32) {
            ptr_type = "uint32_t";
        }
    }

    if (dist == "DINTLV_B8") {
        // args: [dst, dst2, ub_ptr, offset] — base ptr element type must match dst's
        // RegTensor dtype (vlds DINTLV_B8 overload is keyed by dst type).
        std::string dst2_reg = codegen.GetExprAsCode(op->args_[1]);
        std::string ptr_c_type = GetExprDtype(op->args_[0], DataType::UINT8).ToCTypeString();
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[2], ptr_c_type);
        std::string offset_str = codegen.GetExprAsCode(op->args_[3]);
        codegen.Emit("vlds(" + dst_reg + ", " + dst2_reg + ", " + ub_ptr + ", " + offset_str + ", DINTLV_B8);");
    } else if (dist == "DINTLV_B16") {
        std::string dst2_reg = codegen.GetExprAsCode(op->args_[1]);
        std::string ptr_c_type = GetExprDtype(op->args_[0], DataType::UINT16).ToCTypeString();
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[2], ptr_c_type);
        std::string offset_str = codegen.GetExprAsCode(op->args_[3]);
        codegen.Emit("vlds(" + dst_reg + ", " + dst2_reg + ", " + ub_ptr + ", " + offset_str + ", DINTLV_B16);");
    } else if (dist == "DS") {
        // args: [dst, ub_ptr, offset]
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
        std::string offset_str = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("plds(" + dst_reg + ", " + ub_ptr + ", " + offset_str + ", DS);");
    } else {
        // args: [dst, ub_ptr, offset]
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
        std::string offset_str = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("vlds(" + dst_reg + ", " + ub_ptr + ", " + offset_str + ", " + dist + ");");
    }
    return "";
}

// ============================================================================
// StoreAlign — vsts
// ============================================================================

static std::string EmitVFStoreAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst_ptr, src_reg, mask, (optional) block_stride, (optional) repeat_stride]
    std::string src_reg = codegen.GetExprAsCode(op->args_[1]);

    // Get kwargs with defaults
    std::string dist = "NORM_B32";
    if (op->HasKwarg("dist")) {
        dist = op->GetKwarg<std::string>("dist");
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = op->GetKwarg<std::string>("data_copy_mode");
    }

    // Determine pointer cast type from tile dtype, prefer src reg dtype if available
    std::string ptr_type = "float";
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (tile_type) {
        if (tile_type->dtype_ == DataType::FP16 || tile_type->dtype_ == DataType::BF16) {
            ptr_type = "half";
        } else if (tile_type->dtype_ == DataType::UINT8 || tile_type->dtype_ == DataType::INT8) {
            ptr_type = "uint8_t";
        }
    }
    if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        DataType src_dtype = scalar_type->dtype_;
        if (src_dtype == DataType::FP16 || src_dtype == DataType::BF16) {
            ptr_type = "half";
        } else if (src_dtype == DataType::UINT16 || src_dtype == DataType::INT16) {
            ptr_type = "uint16_t";
        } else if (src_dtype == DataType::UINT8 || src_dtype == DataType::INT8) {
            ptr_type = "uint8_t";
        } else if (src_dtype == DataType::UINT32 || src_dtype == DataType::INT32) {
            ptr_type = "uint32_t";
        }
    }

    // INTLV modes need two src registers: args = [dst_ptr, src_reg, src1, mask]
    bool is_intlv = (dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32");
    if (is_intlv) {
        CHECK(op->args_.size() == 4) << "vf.StoreAlign INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask_reg = codegen.GetExprAsCode(op->args_[3]);
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
        codegen.Emit("vsts(" + src_reg + ", " + src1 + ", " + dst_ptr + ", 0, " + dist + ", " + mask_reg + ");");
    } else if (data_copy_mode == "DATA_BLOCK_COPY") {
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->args_.size() >= 4) {
            block_stride = codegen.GetExprAsCode(op->args_[3]);
        } else if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->args_.size() >= 5) {
            repeat_stride = codegen.GetExprAsCode(op->args_[4]);
        } else if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
            codegen.Emit("vsstb(" + src_reg + ", " + ptr_var + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            codegen.Emit("vsstb(" + src_reg + ", " + dst_ptr + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ");");
        }
    } else if (post_update) {
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string stride = (op->args_.size() >= 4) ? codegen.GetExprAsCode(op->args_[3]) : "0";
        std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        codegen.Emit("vsts(" + src_reg + ", " + ptr_var + ", " + stride + ", " + dist + ", " + mask_reg +
                     ", POST_UPDATE);");
    } else {
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
        codegen.Emit("vsts(" + src_reg + ", " + dst_ptr + ", 0, " + dist + ", " + mask_reg + ");");
    }
    return "";
}

// ============================================================================
// MemBar — mem_bar
// ============================================================================

static std::string EmitVFMemBar(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string mode = "VST_VLD";
    if (op->HasKwarg("mode")) {
        mode = op->GetKwarg<std::string>("mode");
    }
    codegen.Emit("mem_bar(" + mode + ");");
    return "";
}

// ============================================================================
// Max — vmax
// ============================================================================

static std::string EmitVFMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src0, src1, mask]
    CHECK(op->args_.size() == 4) << "vf.Max requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.Add requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.Sub requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// And — vand
// ============================================================================

static std::string EmitVFAnd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.And requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // vand requires src1 to share dst's element type; reinterpret if needed
    // (mirrors the `(RegTensor<uint32_t>&)idxTmp` pattern in vf_topk_16_gather.h:213).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : ("(RegTensor<" + dst_dt.ToCTypeString() + "> &)" + src1);

    codegen.Emit("vand(" + dst + ", " + src0 + ", " + s1_expr + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Xor — vxor
// ============================================================================

static std::string EmitVFXor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.Xor requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);

    codegen.Emit("vxor(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.Mul requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// MulAddDst — vmula (hardware FMA: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulAddDst(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.MulAddDst requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vmula(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Div — vdiv
// ============================================================================

static std::string EmitVFDiv(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.Div requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vdiv(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
    return "";
}

static std::string EmitVFShiftRights(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, shift_bits, mask]
    CHECK(op->args_.size() == 4) << "vf.ShiftRights requires 4 args (dst, src, shift_bits, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string shift_bits = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vshrs(" + dst + ", " + src + ", (int16_t)(" + shift_bits + "), " + mask + ");");
    return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, scalar, mask]
    CHECK(op->args_.size() == 4) << "vf.Muls requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ");");
    return "";
}

// ============================================================================
// Ln — vln
// ============================================================================

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.Ln requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);

    codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// FusedExpSub — vexpdif
// ============================================================================

static std::string EmitVFFusedExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, max, mask]
    CHECK(op->args_.size() == 4) << "vf.FusedExpSub requires 4 args (dst, src, max, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string max_reg = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    codegen.Emit("vexpdif(" + dst + ", " + src + ", " + max_reg + ", " + mask + ", PART_EVEN);");
    return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.Cast requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);

    // Get layout and round_mode with defaults
    std::string layout = "ZERO";
    if (op->HasKwarg("layout")) {
        layout = op->GetKwarg<std::string>("layout");
    }
    std::string round_mode = "CAST_ROUND";
    if (op->HasKwarg("round_mode")) {
        round_mode = op->GetKwarg<std::string>("round_mode");
    }

    // A5 vcvt only supports MODE_ZEROING at the instruction level.
    // merge_mode kwarg accepted for API compatibility but always emits ZEROING.
    // For MERGING semantics, use two ZEROING casts + vor/vxor to combine.
    std::string merge_mode = "ZEROING";
    if (op->HasKwarg("merge_mode")) {
        merge_mode = op->GetKwarg<std::string>("merge_mode");
    }
    (void)merge_mode;
    std::string mode_value = "MODE_ZEROING";

    DataType src_dtype = DataType::FP32;
    DataType dst_dtype = DataType::FP32;

    // args: [dst, src, mask]
    if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType())) {
        dst_dtype = scalar_type->dtype_;
    }
    if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        src_dtype = scalar_type->dtype_;
    }

    std::string part;
    if (layout == "ZERO")
        part = "PART_EVEN";
    else if (layout == "ONE")
        part = "PART_ODD";
    else if (layout == "TWO")
        part = "PART_TWO";
    else if (layout == "THREE")
        part = "PART_THREE";
    else
        part = "PART_EVEN";

    // Map round_mode: CAST_ROUND→ROUND_R, CAST_RINT→ROUND_N
    std::string round;
    if (round_mode == "CAST_RINT")
        round = "ROUND_N";
    else
        round = "ROUND_R";
    // FP32 → FP16: vcvt(dst, src, mask, ROUND, SAT, PART, MODE_ZEROING)
    // FP16 → FP32: vcvt(dst, src, mask, PART, MODE_ZEROING)
    // UINT16 → UINT32: vcvt(dst, src, mask, PART, MODE_ZEROING)
    if (src_dtype == DataType::UINT16 && dst_dtype == DataType::UINT32) {
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
    } else if (src_dtype == DataType::FP32 && dst_dtype == DataType::FP16) {
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", RS_DISABLE, " + part + ", " +
                     mode_value + ");");
    } else if (src_dtype == DataType::FP16 && dst_dtype == DataType::FP32) {
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
    } else {
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", RS_DISABLE, " + part + ", " +
                     mode_value + ");");
    }
    return "";
}

// ============================================================================
// DeInterleave — vdintlv
// ============================================================================

static std::string EmitVFDeInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst0, dst1, src0, src1]
    CHECK(op->args_.size() == 4) << "vf.DeInterleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);

    // vdintlv overloads are keyed on the dst element type; if src dtype differs
    // (e.g. u16 reg reinterpreted as u8 to split into byte streams, mirroring
    // `(RegTensor<uint8_t>&)vreg0U16` in vf_topk.h:60), emit a reinterpret cast.
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[2]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);

    codegen.Emit("vdintlv(" + dst0 + ", " + dst1 + ", " + s0_expr + ", " + s1_expr + ");");
    return "";
}

// ============================================================================
// Select — vsel
// ============================================================================

static std::string EmitVFSelect(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src_true, src_false, mask]
    CHECK(op->args_.size() == 4) << "vf.Select requires 4 args (dst, src_true, src_false, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_true = codegen.GetExprAsCode(op->args_[1]);
    std::string src_false = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // vsel requires dst/src_true/src_false to share the same element type.
    // Use dst's dtype as canonical; reinterpret mismatched sources.
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType st_dt = GetExprDtype(op->args_[1]);
    DataType sf_dt = GetExprDtype(op->args_[2]);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string st_expr = (st_dt == dst_dt) ? src_true : (cast_prefix + src_true);
    std::string sf_expr = (sf_dt == dst_dt) ? src_false : (cast_prefix + src_false);

    codegen.Emit("vsel(" + dst + ", " + st_expr + ", " + sf_expr + ", " + mask + ");");
    return "";
}

// ============================================================================
// UpdateMask — plt_b32/plt_b16
// ============================================================================

static std::string EmitVFUpdateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string scalar = codegen.GetExprAsCode(op->args_[0]);
    std::string reg_name = codegen.GetCurrentResultTarget();

    // Default to b32 (float), use dtype kwarg to select b16
    bool use_b16 = false;
    if (op->HasKwarg("dtype")) {
        auto dtype = op->GetKwarg<DataType>("dtype");
        use_b16 = (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::UINT16 ||
                   dtype == DataType::INT16);
    }

    // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
    std::string scalar_var = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("uint32_t " + scalar_var + " = (uint32_t)" + scalar + ";");
    codegen.Emit("MaskReg " + reg_name + ";");
    if (use_b16) {
        codegen.Emit(reg_name + " = plt_b16(" + scalar_var + ", POST_UPDATE);");
    } else {
        codegen.Emit(reg_name + " = plt_b32(" + scalar_var + ", POST_UPDATE);");
    }
    return "";
}

// ============================================================================
// TopK VF API
// ============================================================================

static std::string EmitVFHistograms(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);

    std::string bin_type = op->GetKwarg<std::string>("bin_type");
    std::string bin_const = (bin_type == "BIN1") ? "Bin_N1" : "Bin_N0";

    // chistv2 strictly takes (vector_u16 &dst, vector_u8 src, vector_bool, BinN).
    // Reinterpret src as RegTensor<uint8_t>& if its dtype isn't u8 (mirrors the
    // vf_topk.h pattern of `(RegTensor<uint8_t>&)vreg0`).
    DataType src_dt = GetExprDtype(op->args_[1]);
    std::string src_expr = (src_dt == DataType::UINT8) ? src : ("(RegTensor<uint8_t> &)" + src);

    codegen.Emit("chistv2(" + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    return "";
}

static std::string EmitVFCompare(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string src0 = codegen.GetExprAsCode(op->args_[0]);
    std::string src1 = codegen.GetExprAsCode(op->args_[1]);
    std::string mask_src = codegen.GetExprAsCode(op->args_[2]);
    std::string mask_dst = codegen.GetCurrentResultTarget();

    std::string cmp_mode = op->GetKwarg<std::string>("cmp_mode");
    std::string suffix = "eq";
    if (cmp_mode == "EQ")
        suffix = "eq";
    else if (cmp_mode == "NE")
        suffix = "ne";
    else if (cmp_mode == "LT")
        suffix = "lt";
    else if (cmp_mode == "GT")
        suffix = "gt";
    else if (cmp_mode == "GE")
        suffix = "ge";
    else if (cmp_mode == "LE")
        suffix = "le";

    // vcmp_xx requires src0/src1 share the same vector element type. The caller
    // can pass an explicit `cmp_dtype` kwarg to pin the compare width (mirrors
    // `Compare<uint8_t>` in vf_topk_16_gather.h, where u16 regs are cast to u8 so
    // chistv2 sees a 256-lane byte-compare mask). When absent, fall back to src0.
    DataType canonical = GetExprDtype(op->args_[0]);
    if (op->HasKwarg("cmp_dtype")) {
        canonical = op->GetKwarg<DataType>("cmp_dtype");
    }
    DataType s0_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[1]);
    std::string cast_prefix = "(RegTensor<" + canonical.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == canonical) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == canonical) ? src1 : (cast_prefix + src1);

    codegen.Emit("MaskReg " + mask_dst + ";");
    codegen.Emit("vcmp_" + suffix + "(" + mask_dst + ", " + s0_expr + ", " + s1_expr + ", " + mask_src + ");");
    return "";
}

static std::string EmitVFSqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.Squeeze requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);

    // vsqz requires dst & src to share the same vector element type. Reinterpret
    // src as RegTensor<dst-dtype>& if necessary (mirrors vf_topk.h pattern of
    // `(RegTensor<u32>&)idxC` before passing to Squeeze).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    std::string src_expr = src;
    if (dst_dt != src_dt) {
        src_expr = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)" + src;
    }

    codegen.Emit("vsqz(" + dst + ", " + src_expr + ", " + mask + ", MODE_STORED);");
    return "";
}

static std::string EmitVFArange(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, start]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string start = codegen.GetExprAsCode(op->args_[1]);

    codegen.Emit("vci(" + dst + ", " + start + ", INC_ORDER);");
    return "";
}

static std::string EmitVFGather(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src_ub, indices, mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    // vgather2 demands the base pointer element type to match dst's vector type;
    // derive both from the dst RegTensor's dtype.
    DataType dst_dt = GetExprDtype(op->args_[0]);
    std::string base_c_type = dst_dt.ToCTypeString();
    std::string src_ub = GetUBufPtr(codegen, op->args_[1], base_c_type);
    std::string indices = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // Index reg type mirrors how vgather2 overloads are wired up:
    //   - 4-byte load (u32/s32/f32) -> u32 index
    //   - 2-byte load (u16/s16/f16/bf16) -> u16 index
    //   - 1-byte load (u8/s8/...)        -> u16 index (matches SDK macros)
    std::string idx_c_type = (dst_dt == DataType::UINT32 || dst_dt == DataType::INT32 || dst_dt == DataType::FP32) ?
                                 "uint32_t" :
                                 "uint16_t";
    codegen.Emit("vgather2(" + dst + ", " + src_ub + ", (RegTensor<" + idx_c_type + "> &)" + indices + ", " + mask +
                 ");");
    return "";
}

static std::string EmitVFStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst_ptr, src, align_reg]
    // vstur on A5 (dav_3510) only supports signed types for 4/8-byte data
    // (see asc-devkit dav_3510/kernel_reg_compute_datacopy_store_impl.h:495-498).
    // Cast unsigned 32/64-bit src + dst ptr to signed int32_t/int64_t.
    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType cast_dt = src_dt;
    if (src_dt == DataType::UINT32) {
        cast_dt = DataType::INT32;
    } else if (src_dt == DataType::UINT16) {
        cast_dt = DataType::INT16;
    } else if (src_dt == DataType::UINT8) {
        cast_dt = DataType::INT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string align_reg = codegen.GetExprAsCode(op->args_[2]);
    // Reinterpret src reg to signed type when needed.
    std::string src_expr = (cast_dt == src_dt) ? src : ("(RegTensor<" + base_c_type + "> &)" + src);

    codegen.Emit("vstur(" + align_reg + ", " + src_expr + ", " + dst_ptr + ", POST_UPDATE);");
    return "";
}

static std::string EmitVFStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst_ptr, align_reg]
    // Match the paired vstur's pointer type — A5 vstur only supports signed types.
    DataType tile_dt = GetExprDtype(op->args_[0]);
    DataType cast_dt = tile_dt;
    if (tile_dt == DataType::UINT32) {
        cast_dt = DataType::INT32;
    } else if (tile_dt == DataType::UINT16) {
        cast_dt = DataType::INT16;
    } else if (tile_dt == DataType::UINT8) {
        cast_dt = DataType::INT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type);
    std::string align_reg = codegen.GetExprAsCode(op->args_[1]);

    codegen.Emit("vstar(" + align_reg + ", " + dst_ptr + ");");
    return "";
}

static std::string EmitVFUnalignRegForStore(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();

    codegen.Emit("UnalignReg " + reg_name + ";");
    return "";
}

static std::string EmitVFClearSpr(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    codegen.Emit("sprclr(SPR_AR);");
    return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.RegTensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRegTensor(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.CreateMask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateMask(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Duplicate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDuplicate(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.LoadAlign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadAlign(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.StoreAlign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreAlign(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMax(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdd(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSub(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.And")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAnd(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFXor(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMul(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.MulAddDst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulAddDst(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDiv(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.ShiftRights")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftRights(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMuls(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLn(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.FusedExpSub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFFusedExpSub(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCast(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.DeInterleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDeInterleave(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSelect(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.UpdateMask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUpdateMask(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.MemBar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMemBar(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Histograms")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFHistograms(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Compare")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCompare(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Squeeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqueeze(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Arange")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFArange(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGather(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.StoreUnAlign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreUnAlign(op, codegen); });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.StoreUnAlignPost")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFStoreUnAlignPost(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.UnalignRegForStore")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForStore(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.ClearSpr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFClearSpr(op, codegen); });

} // namespace backend
} // namespace pypto

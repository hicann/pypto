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
#include <unordered_map>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/op_attr_types.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;

static std::string VFEnumValueName(const char* full_name)
{
    const char* sep = std::strrchr(full_name, ':');
    return sep ? std::string(sep + 1) : std::string(full_name);
}

// Returns true when the dst argument (args_[0], or args_[0]/args_[1] for 2-dst
// ops) is a MaskReg variable. Used by unified emitters to dispatch between
// v* (RegTensor) and p* (MaskReg) CCE intrinsics.
static bool IsDstMaskReg(const ir::CallPtr& op, codegen::CCECodegen& codegen, size_t idx = 0)
{
    if (idx >= op->args_.size())
        return false;
    auto dst_var = ir::As<ir::Var>(op->args_[idx]);
    if (dst_var) {
        return codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    return false;
}

// ============================================================================
// RegTensor declaration
// ============================================================================

static std::string EmitVFRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto dtype = op->GetKwarg<DataType>("dtype");
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string decl;
    if (dtype == DataType::INT4) {
        decl = "vector_s4x2 " + reg_name + ";";
    } else if (dtype == DataType::UINT4) {
        decl = "vector_u4x2 " + reg_name + ";";
    } else {
        decl = "RegTensor<" + dtype.ToCTypeString() + "> " + reg_name + ";";
    }
    codegen.HoistRegTensorDecl(decl);
    codegen.RegisterRegTensorVar(reg_name);
    return "";
}

// ============================================================================
// MaskReg declaration (no initialization — unlike create_mask which emits pset)
// ============================================================================

static std::string EmitVFMaskReg(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.HoistRegTensorDecl("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
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

static std::string GetVFMergeMode(const ir::CallPtr& op)
{
    if (!op->HasKwarg("mode"))
        return "MODE_ZEROING";
    auto mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
    return mode == ir::MergeMode::MERGING ? "MODE_MERGING" : "MODE_ZEROING";
}

// Map a DataType to the correct C pointer type for __ubuf__ load/store.
// VF load/store intrinsics (vlds/vsts/vld/vst/vsldb/vsstb) accept both
// signed and unsigned pointer types, so we use the native C type to match
// the RegTensor element type and avoid type-mismatch errors.
// B64 types (INT64/UINT64) are stored as pairs of 32-bit halves.
static std::string DtypeToPtrType(DataType dt)
{
    if (dt == DataType::INT64)
        return "int32_t";
    if (dt == DataType::UINT64)
        return "uint32_t";
    return dt.ToCTypeString();
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // pattern defaults to ALL, dtype defaults to FP32 — either kwarg may be omitted.
    auto pattern = op->HasKwarg("pattern") ? static_cast<ir::MaskPattern>(op->GetKwarg<int>("pattern")) :
                                             ir::MaskPattern::ALL;
    auto dtype = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : DataType::FP32;
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    // Map pypto pattern enum to CCE PAT_* constant
    std::string pat;
    switch (pattern) {
        case ir::MaskPattern::ALL:
            pat = "PAT_ALL";
            break;
        case ir::MaskPattern::ALLF:
            pat = "PAT_ALLF";
            break;
        case ir::MaskPattern::VL1:
            pat = "PAT_VL1";
            break;
        case ir::MaskPattern::VL2:
            pat = "PAT_VL2";
            break;
        case ir::MaskPattern::VL3:
            pat = "PAT_VL3";
            break;
        case ir::MaskPattern::VL4:
            pat = "PAT_VL4";
            break;
        case ir::MaskPattern::VL8:
            pat = "PAT_VL8";
            break;
        case ir::MaskPattern::VL16:
            pat = "PAT_VL16";
            break;
        case ir::MaskPattern::VL32:
            pat = "PAT_VL32";
            break;
        case ir::MaskPattern::VL64:
            pat = "PAT_VL64";
            break;
        case ir::MaskPattern::VL128:
            pat = "PAT_VL128";
            break;
        case ir::MaskPattern::M3:
            pat = "PAT_M3";
            break;
        case ir::MaskPattern::M4:
            pat = "PAT_M4";
            break;
        case ir::MaskPattern::H:
            pat = "PAT_H";
            break;
        case ir::MaskPattern::Q:
            pat = "PAT_Q";
            break;
        default:
            pat = "PAT_ALL";
            break;
    }
    // Select pset instruction based on data element size (not mask type)
    // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
    if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
        codegen.Emit(reg_name + " = pset_b8(" + pat + ");");
    } else if (dtype == DataType::FP32 || dtype == DataType::INT32 || dtype == DataType::UINT32) {
        codegen.Emit(reg_name + " = pset_b32(" + pat + ");");
    } else {
        // FP16, BF16, UINT16, INT16 etc. (2 bytes)
        codegen.Emit(reg_name + " = pset_b16(" + pat + ");");
    }
    return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, (optional) mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_str = codegen.GetExprAsCode(op->args_[1]);
    // Detect vector-source broadcast: either explicit pos kwarg, or src is a RegTensor variable.
    std::string pos = "";
    if (op->HasKwarg("pos")) {
        pos = VFEnumValueName(ir::EnumToString(static_cast<ir::DuplicatePos>(op->GetKwarg<int>("pos"))));
    }
    bool is_vector_src = !pos.empty();
    if (!is_vector_src) {
        auto src_var = ir::As<ir::Var>(op->args_[1]);
        if (src_var) {
            std::string src_name = codegen.GetVarName(src_var);
            is_vector_src = codegen.IsRegTensorVar(src_name);
        }
    }
    if (is_vector_src) {
        // Vector-source broadcast (Tensor mode): vdup(dst, src_vec, mask, POS_xxx, MODE)
        // pos kwarg: "LOWEST" -> POS_LOWEST, "HIGHEST" -> POS_HIGHEST
        if (pos.empty() || pos == "LOWEST")
            pos = "POS_LOWEST";
        else if (pos == "HIGHEST")
            pos = "POS_HIGHEST";
        std::string mode = GetVFMergeMode(op);
        // Tensor mode always requires a mask (AscendC Duplicate(dstReg, srcReg, mask))
        if (op->args_.size() >= 3) {
            std::string mask = codegen.GetExprAsCode(op->args_[2]);
            codegen.Emit("vdup(" + dst + ", " + src_str + ", " + mask + ", " + pos + ", " + mode + ");");
        } else {
            // No mask provided — create an ALL mask inline for Tensor mode
            static int dup_mask_counter = 0;
            std::string mask_var = "__dup_mask_" + std::to_string(dup_mask_counter++);
            DataType src_dt = GetExprDtype(op->args_[1], DataType::FP32);
            std::string pat = "PAT_ALL";
            std::string pset_fn = "pset_b32";
            if (src_dt == DataType::UINT8 || src_dt == DataType::INT8)
                pset_fn = "pset_b8";
            else if (src_dt == DataType::UINT16 || src_dt == DataType::INT16 || src_dt == DataType::FP16 ||
                     src_dt == DataType::BF16)
                pset_fn = "pset_b16";
            codegen.Emit("MaskReg " + mask_var + " = " + pset_fn + "(" + pat + ");");
            codegen.RegisterMaskRegVar(mask_var);
            codegen.Emit("vdup(" + dst + ", " + src_str + ", " + mask_var + ", " + pos + ", " + mode + ");");
        }
    } else if (op->args_.size() >= 3) {
        // Scalar broadcast with mask: vdup(dst, scalar, preg, MODE_ZEROING/MERGING)
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = GetVFMergeMode(op);
        codegen.Emit("vdup(" + dst + ", " + src_str + ", " + mask + ", " + mode + ");");
    } else {
        // Scalar broadcast without mask: vbr(dst, scalar)
        codegen.Emit("vbr(" + dst + ", " + src_str + ");");
    }
    return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or tile-flavored GetItemExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& cast_type = "float", bool is_post_update = false)
{
    std::string ptr = codegen.GetOrCreateVFTilePtr(expr, is_post_update);
    std::string tile_ctype = GetExprDtype(expr, DataType::FP32).ToCTypeString();
    if (cast_type == tile_ctype)
        return ptr;
    return "(__ubuf__ " + cast_type + " *)" + ptr;
}

// ============================================================================
// LoadAlign (unified) — vlds / plds with dist & post_update kwargs
// Replaces: LoadAlign, LoadAlignMode, LoadAlignPostUpdate, LoadAlignPostupdate, LoadAlignUnpackV2
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // 4-arg form: deinterleave load_align(dst0, dst1, ptr, offset, dist="DINTLV_Bxx")
    if (op->args_.size() == 4) {
        std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
        std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
        std::string offset_str = codegen.GetExprAsCode(op->args_[3]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        std::string ptr_type = DtypeToPtrType(dst_dt);
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        std::string dintlv_mode;
        if (op->HasKwarg("dist")) {
            dintlv_mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
        } else {
            if (dst_dt == DataType::UINT8 || dst_dt == DataType::INT8)
                dintlv_mode = "DINTLV_B8";
            else if (dst_dt == DataType::UINT16 || dst_dt == DataType::INT16 || dst_dt == DataType::FP16 ||
                     dst_dt == DataType::BF16)
                dintlv_mode = "DINTLV_B16";
            else
                dintlv_mode = "DINTLV_B32";
        }
        bool post_update = false;
        if (op->HasKwarg("post_update")) {
            post_update = op->GetKwarg<bool>("post_update");
        }
        if (post_update) {
            codegen.Emit("vlds(" + dst0 + ", " + dst1 + ", " + ub_ptr + ", " + offset_str + ", " + dintlv_mode +
                         ", POST_UPDATE);");
        } else {
            codegen.Emit("vlds(" + dst0 + ", " + dst1 + ", " + ub_ptr + ", " + offset_str + ", " + dintlv_mode + ");");
        }
        return "";
    }
    // 2-arg form: load_align(dst, ptr) — MaskReg dst → plds, RegTensor dst → vlds
    if (op->args_.size() == 2) {
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        bool dst_is_mask = false;
        if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
            dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
        }
        if (dst_is_mask) {
            std::string mode = "NORM";
            if (op->HasKwarg("dist"))
                mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
            std::string plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", 0, " + mode + ");");
        } else {
            std::string ptr_type = DtypeToPtrType(dst_dt);
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            codegen.Emit("vlds(" + dst + ", " + ub_ptr + ", 0, NORM);");
        }
        return "";
    }
    CHECK(op->args_.size() == 3)
        << "vf.load_align requires 2, 3, or 4 args (dst, src_ptr[, offset]) or (dst0, dst1, src_ptr, offset)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string offset_str = codegen.GetExprAsCode(op->args_[2]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    std::string ptr_type = DtypeToPtrType(dst_dt);
    // Is the destination a MaskReg? (routes to pld/plds instead of vld/vlds)
    bool dst_is_mask = false;
    if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
        dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
    }
    // Determine mode from kwargs (dist kwarg is legacy alias for mode).
    // MaskReg (plds) and RegTensor (vlds) paths share the same LoadDist enum:
    // LoadDist includes NORM/US/DS/BRC/... and EnumToString yields the bare
    // name (e.g. "DS"), which plds accepts directly and vlds maps to Bxx suffix.
    std::string mode = "NORM";
    if (op->HasKwarg("dist"))
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
    // AddrReg offset path: MaskReg dst -> pld, RegTensor dst -> vld
    if (codegen.IsAddrRegVar(offset_str)) {
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], dst_is_mask ? "uint32_t" : ptr_type);
        if (dst_is_mask) {
            codegen.Emit("pld(" + dst + ", " + ub_ptr + ", " + offset_str + ", " + mode + ");");
        } else {
            codegen.Emit("vld(" + dst + ", " + ub_ptr + ", " + offset_str + ", " + mode + ");");
        }
        return "";
    }
    // Check for DataBlock load path (vsldb)
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    // vsldb path (non-contiguous datablock load). Accept both DATA_BLOCK_LOAD
    // (pypto legacy name) and DATA_BLOCK_COPY (AscendC's name for this mode) so
    // code written against AscendC semantics does not silently fall back to vlds.
    if (data_copy_mode == "DATA_BLOCK_LOAD" || data_copy_mode == "DATA_BLOCK_COPY") {
        // vsldb path: load_align(dst, ptr, mask, data_copy_mode=..., block_stride=N, ...)
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u) | (" + repeat_stride +
                         " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u), " + mask_reg + ");");
        }
        return "";
    }
    // Route by dst variable type: MaskReg → plds, RegTensor → vlds
    if (dst_is_mask) {
        // plds path: dst is MaskReg, pointer is always uint32_t*
        // Supported modes: NORM / US / DS (matches asc-devkit MaskDist enum)
        std::string plds_mode = mode; // pass through NORM/US/DS directly
        std::string plds_ptr;
        if (post_update) {
            plds_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ", POST_UPDATE);");
        } else {
            plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ");");
        }
        return "";
    }
    // vlds path: dst is RegTensor
    // Get UB pointer (post_update uses reference-based ptr)
    std::string ub_ptr;
    if (post_update) {
        ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
    } else {
        ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    }
    // Determine vlds mode string
    std::string vlds_mode;
    if (mode == "NORM") {
        vlds_mode = "NORM";
    } else if (mode == "BRC") {
        if (dst_dt == DataType::UINT8 || dst_dt == DataType::INT8)
            vlds_mode = "BRC_B8";
        else if (dst_dt == DataType::UINT16 || dst_dt == DataType::INT16 || dst_dt == DataType::FP16 ||
                 dst_dt == DataType::BF16)
            vlds_mode = "BRC_B16";
        else
            vlds_mode = "BRC_B32";
    } else if (mode == "US") {
        if (dst_dt == DataType::UINT8 || dst_dt == DataType::INT8)
            vlds_mode = "US_B8";
        else
            vlds_mode = "US_B16";
    } else if (mode == "DS") {
        if (dst_dt == DataType::UINT8 || dst_dt == DataType::INT8)
            vlds_mode = "DS_B8";
        else
            vlds_mode = "DS_B16";
    } else if (mode == "UNPK") {
        if (dst_dt == DataType::UINT8 || dst_dt == DataType::INT8)
            vlds_mode = "UNPK_B8";
        else if (dst_dt == DataType::UINT16 || dst_dt == DataType::INT16 || dst_dt == DataType::FP16 ||
                 dst_dt == DataType::BF16)
            vlds_mode = "UNPK_B16";
        else
            vlds_mode = "UNPK_B32";
    } else if (mode == "UNPK4") {
        vlds_mode = "UNPK4_B8";
    } else if (mode == "BLK") {
        vlds_mode = "BLK";
    } else if (mode == "E2B") {
        if (dst_dt == DataType::UINT16 || dst_dt == DataType::INT16 || dst_dt == DataType::FP16 ||
            dst_dt == DataType::BF16)
            vlds_mode = "E2B_B16";
        else
            vlds_mode = "E2B_B32";
    } else {
        // Fallback: pass through directly (e.g. BRC_B32, DS_B16, E2B_B32, etc.)
        vlds_mode = mode;
    }
    if (post_update) {
        // B64 types need stride doubled (AscendC postUpdateStride * 2 for 8-byte elements)
        std::string effective_offset = offset_str;
        if (dst_dt.GetBit() == 64) {
            effective_offset = "(" + offset_str + ") * 2";
        }
        codegen.Emit("vlds(" + dst + ", " + ub_ptr + ", " + effective_offset + ", " + vlds_mode + ", POST_UPDATE);");
    } else {
        codegen.Emit("vlds(" + dst + ", " + ub_ptr + ", " + offset_str + ", " + vlds_mode + ");");
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
    // MaskReg src path: when args[1] is a MaskReg, dispatch to psts/pst (mask store)
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    if (src_is_mask) {
        std::string dist = "NORM";
        if (op->HasKwarg("dist")) {
            dist = VFEnumValueName(ir::EnumToString(static_cast<ir::StoreDist>(op->GetKwarg<int>("dist"))));
            // psts/pst use PK (not PACK) for packed mode
            if (dist == "PACK")
                dist = "PK";
        }
        // AddrReg offset path: 3rd arg is AddrReg -> pst(mask, ptr, areg, dist)
        if (op->args_.size() >= 3) {
            std::string third_arg = codegen.GetExprAsCode(op->args_[2]);
            if (codegen.IsAddrRegVar(third_arg)) {
                std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
                codegen.Emit("pst(" + src_reg + ", " + ub_ptr + ", " + third_arg + ", " + dist + ");");
                return "";
            }
            // Post-update path: int offset with post_update kwarg
            bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
            if (post_update) {
                std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
                codegen.Emit("psts(" + src_reg + ", " + ptr_var + ", " + third_arg + ", " + dist + ", POST_UPDATE);");
                return "";
            }
        }
        // Default: psts with offset=0
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("psts(" + src_reg + ", " + ub_ptr + ", 0, " + dist + ");");
        return "";
    }
    // AddrReg offset path: when 4th arg is an AddrReg variable,
    // emit vst(src, ptr, areg, dist, mask) — 5 args (note: vst, not vsts)
    if (op->args_.size() >= 4) {
        std::string addr_reg = codegen.GetExprAsCode(op->args_[3]);
        if (codegen.IsAddrRegVar(addr_reg)) {
            std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
            std::string ptr_type = "float";
            if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
                ptr_type = DtypeToPtrType(scalar_type->dtype_);
            }
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            // Auto-select dist based on src dtype
            DataType src_dt = GetExprDtype(op->args_[1]);
            std::string dist = "NORM_B32";
            if (src_dt == DataType::UINT8 || src_dt == DataType::INT8)
                dist = "NORM_B8";
            else if (src_dt == DataType::FP16 || src_dt == DataType::BF16 || src_dt == DataType::UINT16 ||
                     src_dt == DataType::INT16)
                dist = "NORM_B16";
            codegen.Emit("vst(" + src_reg + ", " + dst_ptr + ", " + addr_reg + ", " + dist + ", " + mask_reg + ");");
            return "";
        }
    }
    // Get kwargs with defaults
    std::string dist = "";
    if (op->HasKwarg("dist")) {
        dist = VFEnumValueName(ir::EnumToString(static_cast<ir::StoreDist>(op->GetKwarg<int>("dist"))));
    }
    if (dist.empty()) {
        // Auto-select default dist based on src dtype (AscendC uses NORM_B8/B16/B32 by element width)
        DataType src_dtype = DataType::FP32;
        if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
            src_dtype = scalar_type->dtype_;
        } else if (auto tile_type_tmp = ir::As<ir::TileType>(op->args_[0]->GetType())) {
            src_dtype = tile_type_tmp->dtype_;
        }
        if (src_dtype == DataType::UINT8 || src_dtype == DataType::INT8)
            dist = "NORM_B8";
        else if (src_dtype == DataType::FP16 || src_dtype == DataType::BF16 || src_dtype == DataType::UINT16 ||
                 src_dtype == DataType::INT16)
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    }
    // Auto-expand shorthand dist names to element-width-qualified CCE constants
    if (dist == "FIRST_ELEMENT" || dist == "FIRST_ELE") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (sd == DataType::UINT8 || sd == DataType::INT8)
            dist = "ONEPT_B8";
        else if (sd == DataType::FP16 || sd == DataType::BF16 || sd == DataType::UINT16 || sd == DataType::INT16)
            dist = "ONEPT_B16";
        else
            dist = "ONEPT_B32";
    } else if (dist == "PACK") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (sd == DataType::UINT8 || sd == DataType::INT8 || sd == DataType::FP16 || sd == DataType::BF16 ||
            sd == DataType::UINT16 || sd == DataType::INT16)
            dist = "PK_B16";
        else if (sd == DataType::UINT32 || sd == DataType::INT32 || sd == DataType::FP32)
            dist = "PK_B32";
        else
            dist = "PK_B64";
    } else if (dist == "PACK4") {
        dist = "PK4_B32";
    } else if (dist == "INTLV") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (sd == DataType::UINT8 || sd == DataType::INT8)
            dist = "INTLV_B8";
        else if (sd == DataType::FP16 || sd == DataType::BF16 || sd == DataType::UINT16 || sd == DataType::INT16)
            dist = "INTLV_B16";
        else
            dist = "INTLV_B32";
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    // Determine pointer cast type from tile dtype, prefer src reg dtype if available
    std::string ptr_type = "float";
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (tile_type) {
        ptr_type = DtypeToPtrType(tile_type->dtype_);
    }
    if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        ptr_type = DtypeToPtrType(scalar_type->dtype_);
    }
    // INTLV modes need two src registers: args = [dst_ptr, src_reg, src1, mask]
    bool is_intlv = (dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32");
    if (is_intlv) {
        CHECK(op->args_.size() == 4) << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask_reg = codegen.GetExprAsCode(op->args_[3]);
        // vsts 2-source overload (__VF_VSTSX2) does not exist for FP32;
        // cast to UINT32 (same 32-bit width) to use the UINT32 overload.
        DataType src_dt = GetExprDtype(op->args_[1]);
        std::string intlv_ptr_type = ptr_type;
        if (src_dt == DataType::FP32) {
            src_reg = "(RegTensor<uint32_t> &)" + src_reg;
            src1 = "(RegTensor<uint32_t> &)" + src1;
            intlv_ptr_type = "uint32_t";
        }
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], intlv_ptr_type);
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
        // B64 types need stride doubled (AscendC postUpdateStride * 2 for 8-byte elements)
        DataType src_dt = GetExprDtype(op->args_[1]);
        std::string effective_stride = stride;
        if (src_dt.GetBit() == 64) {
            effective_stride = "(" + stride + ") * 2";
        }
        std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        codegen.Emit("vsts(" + src_reg + ", " + ptr_var + ", " + effective_stride + ", " + dist + ", " + mask_reg +
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
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::MemBarMode>(op->GetKwarg<int>("mode"))));
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
    CHECK(op->args_.size() == 4) << "vf.max requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    if (op->args_.size() == 5) {
        // 5-arg form: Add(carry_out, dst, src0, src1, mask) -> vaddc
        DataType dst_dt = GetExprDtype(op->args_[1]);
        CHECK(dst_dt == DataType::INT32 || dst_dt == DataType::UINT32)
            << "vf.add (carry form, vaddc) only supports INT32/UINT32, got " << dst_dt.ToString();
        std::string carry_out = codegen.GetExprAsCode(op->args_[0]);
        std::string dst = codegen.GetExprAsCode(op->args_[1]);
        std::string src0 = codegen.GetExprAsCode(op->args_[2]);
        std::string src1 = codegen.GetExprAsCode(op->args_[3]);
        std::string mask = codegen.GetExprAsCode(op->args_[4]);
        codegen.Emit("vaddc(" + carry_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
    } else {
        CHECK(op->args_.size() == 4)
            << "vf.add requires 4 args (dst, src0, src1, mask) or 5 args (carry_out, dst, src0, src1, mask)";
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src0 = codegen.GetExprAsCode(op->args_[1]);
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask = codegen.GetExprAsCode(op->args_[3]);
        std::string mode = GetVFMergeMode(op);
        codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    if (op->args_.size() == 5) {
        // 5-arg form: Sub(borrow_out, dst, src0, src1, mask) -> vsubc
        DataType dst_dt = GetExprDtype(op->args_[1]);
        CHECK(dst_dt == DataType::INT32 || dst_dt == DataType::UINT32)
            << "vf.sub (carry form, vsubc) only supports INT32/UINT32, got " << dst_dt.ToString();
        std::string borrow_out = codegen.GetExprAsCode(op->args_[0]);
        std::string dst = codegen.GetExprAsCode(op->args_[1]);
        std::string src0 = codegen.GetExprAsCode(op->args_[2]);
        std::string src1 = codegen.GetExprAsCode(op->args_[3]);
        std::string mask = codegen.GetExprAsCode(op->args_[4]);
        codegen.Emit("vsubc(" + borrow_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
    } else {
        CHECK(op->args_.size() == 4)
            << "vf.sub requires 4 args (dst, src0, src1, mask) or 5 args (borrow_out, dst, src0, src1, mask)";
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src0 = codegen.GetExprAsCode(op->args_[1]);
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask = codegen.GetExprAsCode(op->args_[3]);
        std::string mode = GetVFMergeMode(op);
        codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// And — vand
// ============================================================================

static std::string EmitVFAnd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.and_ requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pand(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    // vand requires src1 to share dst's element type; reinterpret if needed
    // (mirrors the `(RegTensor<uint32_t>&)idxTmp` pattern in vf_topk_16_gather.h:213).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : ("(RegTensor<" + dst_dt.ToCTypeString() + "> &)" + src1);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vand(" + dst + ", " + src0 + ", " + s1_expr + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Xor — vxor
// ============================================================================

static std::string EmitVFBinaryBitwise(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                       const std::string& op_name, const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << op_name << " requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        std::string p_instr = "p" + instruction.substr(1);
        codegen.Emit(p_instr + "(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    codegen.Emit(instruction + "(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", " + GetVFMergeMode(op) +
                 ");");
    return "";
}

static std::string EmitVFXor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.xor", "vxor");
}

// ============================================================================
// Or — vor
// ============================================================================

static std::string EmitVFOr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.or_", "vor");
}

// ============================================================================
// Reduce — vcadd/vcmax/vcmin + vcgadd/vcgmax/vcgmin (unified)
// Supports both new-style (mode=SUM/MAX/MIN, datablock) and legacy (reduce_type=ADD/MAX, merge_mode)
// ============================================================================

// Shared reduction emitter. `reduce_mode` is one of "SUM"/"MAX"/"MIN".
static std::string EmitVFReduceImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                    const std::string& reduce_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.reduce_* requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    bool datablock = false;
    if (op->HasKwarg("datablock")) {
        datablock = op->GetKwarg<bool>("datablock");
    }
    std::string intrinsic;
    if (reduce_mode == "SUM")
        intrinsic = datablock ? "vcgadd" : "vcadd";
    else if (reduce_mode == "MAX")
        intrinsic = datablock ? "vcgmax" : "vcmax";
    else
        intrinsic = datablock ? "vcgmin" : "vcmin";
    std::string merge = "MODE_ZEROING";
    if (op->HasKwarg("merge_mode")) {
        merge = "MODE_" +
                VFEnumValueName(ir::EnumToString(static_cast<ir::MergeMode>(op->GetKwarg<int>("merge_mode"))));
    }
    codegen.Emit(intrinsic + "(" + dst + ", " + src + ", " + mask + ", " + merge + ");");
    return "";
}

static std::string EmitVFReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "SUM");
}

static std::string EmitVFReduceMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MAX");
}

static std::string EmitVFReduceMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MIN");
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MulAddDst — vmula (hardware FMA: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulAddDst(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul_add_dst requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmula(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Div — vdiv
// ============================================================================

static std::string EmitVFDiv(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.div requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vdiv(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, scalar, mask]
    CHECK(op->args_.size() == 4) << "vf.muls requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Ln — vln (natural logarithm, basic precision)
// ============================================================================

static std::string EmitVFUnary(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << op_name << " requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit(instruction + "(" + dst + ", " + src + ", " + mask + ", " + GetVFMergeMode(op) + ");");
    return "";
}

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFUnary(op, codegen_base, "vf.ln", "vln");
}

// ============================================================================
// Log — vln (natural logarithm, same as Ln on A5/dav_3510)
// ============================================================================

static std::string EmitVFLog(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFUnary(op, codegen_base, "vf.log", "vln");
}

// ============================================================================
// Min — vmin
// ============================================================================

static std::string EmitVFMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.min requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmin(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Exp — vexp
// ============================================================================

static std::string EmitVFExp(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.exp requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Abs — vabs
// ============================================================================

static std::string EmitVFAbs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.abs requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vabs(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Not — vnot
// ============================================================================

static std::string EmitVFNot(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.not_ requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pnot(" + dst + ", " + src + ", " + mask + ");");
        return "";
    }
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vnot(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Sqrt — vsqrt
// ============================================================================

static std::string EmitVFSqrt(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.sqrt requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vsqrt(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Relu — vrelu
// ============================================================================

static std::string EmitVFRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.relu requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vrelu(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Neg — vneg
// ============================================================================

static std::string EmitVFNeg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.neg requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vneg(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Adds — vadds (scalar addition)
// ============================================================================

static std::string EmitVFAdds(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.adds requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vadds(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Subs — vsubs (scalar subtraction: dst = src - scalar)
// ============================================================================

static std::string EmitVFSubs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.subs requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vadds(" + dst + ", " + src + ", -(" + scalar_str + "), " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Mins — vmins (scalar minimum)
// ============================================================================

static std::string EmitVFMins(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mins requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmins(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Maxs — vmaxs (scalar maximum)
// ============================================================================

static std::string EmitVFMaxs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.maxs requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmaxs(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// LeakyRelu — vlrelu
// ============================================================================

static std::string EmitVFLeakyRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.leaky_relu requires 4 args (dst, src, alpha, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string alpha = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vlrelu(" + dst + ", " + src + ", " + alpha + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Interleave — vintlv
// ============================================================================

static std::string EmitVFInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        std::string pintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pintlv_op = "pintlv_b8";
        } else if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::UINT16 ||
                   dtype == DataType::INT16) {
            pintlv_op = "pintlv_b16";
        } else {
            pintlv_op = "pintlv_b32";
        }
        codegen.Emit(pintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    codegen.Emit("vintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    return "";
}

// ============================================================================
// PairReduceSum — vcpadd
// ============================================================================

static std::string EmitVFPairReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.pair_reduce_sum requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vcpadd(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// AbsSub — vabsdif (absolute difference: |src0 - src1|)
// ============================================================================

static std::string EmitVFAbsSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.abs_sub requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vabsdif(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Axpy — vaxpy (accumulate: dst = src * scalar + dst)
// ============================================================================

static std::string EmitVFAxpy(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.axpy requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vaxpy(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Copy — vmov (register copy with MODE_MERGING)
// ============================================================================

static std::string EmitVFCopy(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.copy requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit("vmov(" + dst + ", " + src + ", " + mask + ", MODE_MERGING);");
    return "";
}

// ============================================================================
// ============================================================================
// Madd — vmadd (multiply-accumulate: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulDstAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul_dst_add requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vmadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Pack — vpack (narrow data type)
// ============================================================================

static std::string EmitVFPack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.pack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(op->GetKwarg<int>("part"))));
    }
    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("ppack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType src_dt = GetExprDtype(op->args_[1]);

    if (src_dt.GetBit() == 64) {
        // 64-bit source → 32-bit dst: use DeInterleave with a zero register,
        // mirroring AscendC PackImpl<..., part>(dst, src) for 8-byte src.
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string zero_var = dst + "_pack_zero_";
        std::string dump_var = dst + "_pack_dump_";
        std::string mask_var = dst + "_pack_mask_";
        codegen.Emit("RegTensor<uint32_t> " + zero_var + ";");
        codegen.Emit("RegTensor<uint32_t> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        codegen.Emit("vdup(" + zero_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        if (part_check == "LOWEST") {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + "(RegTensor<uint32_t>&)" +
                         src + ", " + zero_var + ");");
        } else {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + zero_var +
                         ", (RegTensor<uint32_t>&)" + src + ");");
        }
    } else {
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vpack(" + dst + ", " + src + ", " + cce_part + ", MODE_UNKNOWN);");
    }
    return "";
}

// ============================================================================
// Unpack — vunpack (widen data type)
// ============================================================================

static std::string EmitVFUnpack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.unpack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(op->GetKwarg<int>("part"))));
    }

    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("punpack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);

    if (dst_dt.GetBit() == 64) {
        std::string src_ctype = src_dt.ToCTypeString();
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string pad_var = dst + "_unpack_pad_";
        std::string dump_var = dst + "_unpack_dump_";
        std::string mask_var = dst + "_unpack_mask_";
        codegen.Emit("RegTensor<" + src_ctype + "> " + pad_var + ";");
        codegen.Emit("RegTensor<" + src_ctype + "> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        if (src_dt == DataType::INT32) {
            codegen.Emit("vshrs(" + pad_var + ", " + src + ", 31, " + mask_var + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + pad_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        }
        if (part_check == "LOWEST") {
            codegen.Emit("vintlv((RegTensor<" + src_ctype + ">&)" + dst + ", " + dump_var + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        } else {
            codegen.Emit("vintlv(" + dump_var + ", (RegTensor<" + src_ctype + ">&)" + dst + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        }
    } else {
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vunpack(" + dst + ", " + src + ", " + cce_part + ");");
    }
    return "";
}

// ============================================================================
// PRelu — vprelu (parametric ReLU with per-element slope)
// ============================================================================

static std::string EmitVFPRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.prelu requires 4 args (dst, src, slope, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string slope = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vprelu(" + dst + ", " + src + ", " + slope + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// ShiftLeft — unified left shift: vshl (per-lane, shift is a RegTensor) or
// vshls (uniform, shift is a scalar). The former standalone vf.shift_lefts
// (scalar) op is merged in here; the register-vs-scalar decision is made from
// the codegen RegTensor registry (mirrors EmitVFDuplicate's IsRegTensorVar
// dispatch), not from the op name.
// ============================================================================

// Returns true when the shift-amount arg (op->args_[2]) is a per-lane vector
// register; false for a uniform scalar shift (integer literal or plain scalar).
static bool ShiftAmountIsRegister(const ir::CallPtr& op, codegen::CCECodegen& codegen)
{
    // Integer literals are always scalar shifts — no need to consult the registry.
    if (ir::As<ir::ConstInt>(op->args_[2]))
        return false;
    auto shift_var = ir::As<ir::Var>(op->args_[2]);
    if (!shift_var)
        return false;
    return codegen.IsRegTensorVar(codegen.GetVarName(shift_var));
}

static std::string EmitVFShift(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& vector_instruction, const std::string& scalar_instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << op_name << " requires 4 args (dst, src, shift, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string shift = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = GetVFMergeMode(op);
    if (ShiftAmountIsRegister(op, codegen)) {
        codegen.Emit(vector_instruction + "(" + dst + ", " + src + ", *(vector_s32*)&(" + shift + "), " + mask + ", " +
                     mode + ");");
    } else {
        codegen.Emit(scalar_instruction + "(" + dst + ", " + src + ", (int16_t)(" + shift + "), " + mask + ", " + mode +
                     ");");
    }
    return "";
}

static std::string EmitVFShiftLeft(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_left", "vshl", "vshls");
}

// ============================================================================
// ShiftRight — unified right shift: vshr (per-lane, shift is a RegTensor) or
// vshrs (uniform, shift is a scalar). The former standalone vf.shift_rights
// (scalar) op is merged in here; dispatch uses ShiftAmountIsRegister (codegen
// RegTensor registry), not the op name.
// ============================================================================

static std::string EmitVFShiftRight(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_right", "vshr", "vshrs");
}

// ============================================================================
// Mull — vmull (long multiply: 32x32->64, lo/hi split)
// ============================================================================

static std::string EmitVFMull(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 5) << "vf.mull requires 5 args (dst_lo, dst_hi, src0, src1, mask)";
    std::string dst_lo = codegen.GetExprAsCode(op->args_[0]);
    std::string dst_hi = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string mask = codegen.GetExprAsCode(op->args_[4]);
    codegen.Emit("vmull(" + dst_lo + ", " + dst_hi + ", " + src0 + ", " + src1 + ", " + mask + ");");
    return "";
}

// ============================================================================
// Addc — vaddcs (add with carry)
// ============================================================================

static std::string EmitVFAddc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 6) << "vf.addc requires 6 args (carry_out, dst, src0, src1, carry_in, mask)";
    DataType dst_dt = GetExprDtype(op->args_[1]);
    CHECK(dst_dt == DataType::INT32 || dst_dt == DataType::UINT32)
        << "vf.addc (vaddcs) only supports INT32/UINT32, got " << dst_dt.ToString();
    std::string carry_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string carry_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vaddcs(" + carry_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + carry_in + ", " + mask + ");");
    return "";
}

// ============================================================================
// Subc — vsubcs (subtract with borrow)
// ============================================================================

static std::string EmitVFSubc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 6) << "vf.subc requires 6 args (borrow_out, dst, src0, src1, borrow_in, mask)";
    DataType dst_dt = GetExprDtype(op->args_[1]);
    CHECK(dst_dt == DataType::INT32 || dst_dt == DataType::UINT32)
        << "vf.subc (vsubcs) only supports INT32/UINT32, got " << dst_dt.ToString();
    std::string borrow_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string borrow_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vsubcs(" + borrow_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + borrow_in + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// ExpSub — vexpdif
// ============================================================================

static std::string EmitVFExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, max, mask]
    CHECK(op->args_.size() == 4) << "vf.exp_sub requires 4 args (dst, src, max, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string max_reg = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vexpdif(" + dst + ", " + src + ", " + max_reg + ", " + mask + ", " + part + ");");
    return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.astype requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // Get layout and round_mode with defaults
    std::string layout = "ZERO";
    if (op->HasKwarg("layout")) {
        layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
    }
    std::string round_mode = "CAST_ROUND";
    if (op->HasKwarg("round_mode")) {
        round_mode = VFEnumValueName(ir::EnumToString(static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"))));
    }
    // A5 vcvt only supports MODE_ZEROING at the instruction level.
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
    std::string part_pp;
    if (layout == "ZERO")
        part_pp = "PART_P0";
    else if (layout == "ONE")
        part_pp = "PART_P1";
    else
        part_pp = "PART_P0";
    // Map round_mode: CAST_ROUND→ROUND_R, CAST_RINT→ROUND_N
    std::string round;
    if (round_mode == "CAST_RINT")
        round = "ROUND_N";
    else if (round_mode == "CAST_FLOOR")
        round = "ROUND_F";
    else if (round_mode == "CAST_CEIL")
        round = "ROUND_C";
    else if (round_mode == "CAST_TRUNC")
        round = "ROUND_Z";
    else if (round_mode == "CAST_RNA")
        round = "ROUND_A";
    else if (round_mode == "CAST_ODD")
        round = "ROUND_R"; // ROUND_O not supported by vcvt on 3510; fall back to nearest
    else if (round_mode == "CAST_HYBRID")
        round = "ROUND_H"; // hybrid rounding, 3510 only (AscendC CAST_HYBRID -> ::ROUND::H)
    else
        round = "ROUND_R";
    // Get saturation kwarg (default: disabled)
    std::string sat = "RS_DISABLE";
    if (op->HasKwarg("saturate")) {
        std::string sat_val = VFEnumValueName(
            ir::EnumToString(static_cast<ir::SaturateMode>(op->GetKwarg<int>("saturate"))));
        if (sat_val == "ON" || sat_val == "ENABLE")
            sat = "RS_ENABLE";
    }
    // Determine if narrowing or widening conversion based on src/dst dtype.
    // Widening: dst wider than src — no ROUND/RS needed, just PART + MODE
    // Narrowing: dst narrower — needs ROUND + RS + PART + MODE
    // Float→same-width-int (FP32→S32): ROUND + RS + MODE (no PART)
    // INT→FLOAT same-width (S32→FP32): ROUND + MODE (no RS, no PART)
    // FLOAT→FLOAT same-width (FP16→BF16): ROUND + MODE (no RS, no PART)
    // INT→INT narrowing: RS + PART + MODE (no ROUND)
    // Cross-width INT↔FLOAT (S16→FP32, S32→FP16, FP32→S64, S64→FP32): ROUND + PART + MODE
    bool is_widening = false;
    bool is_int_to_float = false;
    bool is_float_to_same_int = false;  // FP32→S32, FP16→S16 (same-width float→int)
    bool is_float_to_wider_int = false; // FP16→S32, BF16→S32 (widening float→int)
    bool is_int_narrowing = false;      // int→int narrowing (no ROUND, just RS + PART)
    bool is_cross_width = false;        // cross-width INT↔FLOAT with ROUND + PART + MODE
    bool is_s4_widening = false;        // INT4→FP16/BF16/INT16 (vcvt_s42f16/bf16/s16)
    bool is_s4_narrowing = false;       // FP16→INT4 (vcvt_f162s4)
    // S4 (INT4) special instructions: vcvt_s42f16, vcvt_s42bf16, vcvt_s42s16, vcvt_f162s4
    if (src_dtype == DataType::INT4 &&
        (dst_dtype == DataType::FP16 || dst_dtype == DataType::BF16 || dst_dtype == DataType::INT16)) {
        is_s4_widening = true;
    } else if (dst_dtype == DataType::INT4 && src_dtype == DataType::FP16) {
        is_s4_narrowing = true;
    }
    // Widening cases: vcvt(dst, src, mask, PART, MODE) — no ROUND/RS
    // FP16→FP32, BF16→FP32, UINT16→UINT32, INT16→INT32, INT8→INT16, UINT8→UINT16,
    // INT8/UINT8→FP16, S32→S64, U32→S64, INT16/UINT16→FP32 (__VF_VCVTIF_PART)
    else if ((src_dtype == DataType::FP16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::BF16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::UINT32) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::INT32) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::INT16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::UINT16) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::INT32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::UINT32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::FP32)) {
        is_widening = true;
    }
    // INT→FLOAT same-width or FLOAT→FLOAT same-width: vcvt(dst, src, mask, ROUND, MODE_ZEROING)
    // S32/U32→FP32, S16/U16→FP16, FP16→BF16
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP32) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP16 && dst_dtype == DataType::BF16)) {
        is_int_to_float = true;
    }
    // FLOAT→same-width-INT: vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
    // FP32→S32/U32, FP16→S16/U16 (same element width)
    else if ((src_dtype == DataType::FP32 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::FP16 && (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16))) {
        is_float_to_same_int = true;
    }
    // FLOAT→wider-INT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
    // FP16→S32/U32, BF16→S32/U32 (half-width float to full-width int)
    else if ((src_dtype == DataType::FP16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::BF16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_float_to_wider_int = true;
    }
    // Cross-width INT↔FLOAT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING)
    // S32/U32→FP16, FP32→S64, S64→FP32
    // (S16/U16→FP32 moved to is_widening — uses __VF_VCVTIF_PART with PART+MODE only)
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        is_cross_width = true;
    }
    // INT→INT narrowing: vcvt(dst, src, mask, RS, PART, MODE_ZEROING)
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) &&
              (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16 || dst_dtype == DataType::INT8 ||
               dst_dtype == DataType::UINT8)) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) &&
              (dst_dtype == DataType::INT8 || dst_dtype == DataType::UINT8)) ||
             (src_dtype == DataType::INT64 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_int_narrowing = true;
    }
    if (is_s4_widening) {
        // INT4→FP16/BF16/INT16: specialized vcvt_s42*
        if (dst_dtype == DataType::FP16) {
            codegen.Emit("vcvt_s42f16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else if (dst_dtype == DataType::BF16) {
            codegen.Emit("vcvt_s42bf16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else {
            codegen.Emit("vcvt_s42s16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        }
    } else if (is_s4_narrowing) {
        // FP16→INT4: vcvt_f162s4
        codegen.Emit("vcvt_f162s4(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part_pp +
                     ", " + mode_value + ");");
    } else if (is_widening) {
        // vcvt(dst, src, mask, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
    } else if (is_int_to_float) {
        // vcvt(dst, src, mask, ROUND, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + mode_value + ");");
    } else if (is_float_to_same_int) {
        // vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + mode_value + ");");
    } else if (is_float_to_wider_int || is_cross_width) {
        // vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part + ", " + mode_value + ");");
    } else if (is_int_narrowing) {
        // vcvt(dst, src, mask, RS, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part + ", " + mode_value + ");");
    } else {
        // Float narrowing (FP32→FP16, FP32→BF16, FP16→INT8, etc.)
        // vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
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
    CHECK(op->args_.size() == 4) << "vf.de_interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        std::string pdintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pdintlv_op = "pdintlv_b8";
        } else if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::UINT16 ||
                   dtype == DataType::INT16) {
            pdintlv_op = "pdintlv_b16";
        } else {
            pdintlv_op = "pdintlv_b32";
        }
        codegen.Emit(pdintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
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
    CHECK(op->args_.size() == 4) << "vf.select requires 4 args (dst, src_true, src_false, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_true = codegen.GetExprAsCode(op->args_[1]);
    std::string src_false = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("psel(" + dst + ", " + src_true + ", " + src_false + ", " + mask + ");");
        return "";
    }
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
    // Default to b32 (float), use dtype kwarg to select b16 or b8
    bool use_b8 = false;
    bool use_b16 = false;
    if (op->HasKwarg("dtype")) {
        auto dtype = op->GetKwarg<DataType>("dtype");
        use_b8 = (dtype == DataType::UINT8 || dtype == DataType::INT8);
        use_b16 = (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::UINT16 ||
                   dtype == DataType::INT16);
    }
    // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
    std::string scalar_var = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("uint32_t " + scalar_var + " = (uint32_t)" + scalar + ";");
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    if (use_b8) {
        codegen.Emit(reg_name + " = plt_b8(" + scalar_var + ", POST_UPDATE);");
    } else if (use_b16) {
        codegen.Emit(reg_name + " = plt_b16(" + scalar_var + ", POST_UPDATE);");
    } else {
        codegen.Emit(reg_name + " = plt_b32(" + scalar_var + ", POST_UPDATE);");
    }
    return "";
}

static std::string EmitVFHistograms(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string bin_type = VFEnumValueName(ir::EnumToString(static_cast<ir::BinType>(op->GetKwarg<int>("bin_type"))));
    std::string bin_const = (bin_type == "BIN1") ? "Bin_N1" : "Bin_N0";
    // Reinterpret src as RegTensor<uint8_t>& if its dtype isn't u8
    DataType src_dt = GetExprDtype(op->args_[1]);
    std::string src_expr = (src_dt == DataType::UINT8) ? src : ("(RegTensor<uint8_t> &)" + src);
    // hist_type: ACCUMULATE (chistv2, default) or FREQUENCY (dhistv2)
    std::string hist_type = "ACCUMULATE";
    if (op->HasKwarg("hist_type")) {
        hist_type = VFEnumValueName(ir::EnumToString(static_cast<ir::HistType>(op->GetKwarg<int>("hist_type"))));
    }
    if (hist_type == "FREQUENCY") {
        codegen.Emit("dhistv2(" + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    } else {
        codegen.Emit("chistv2(" + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    }
    return "";
}

static std::string EmitVFCompareImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                     const std::string& cmp_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src0, src1, mask_src]
    std::string mask_dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask_src = codegen.GetExprAsCode(op->args_[3]);
    std::string suffix = "eq";
    if (cmp_mode == "NE")
        suffix = "ne";
    else if (cmp_mode == "LT")
        suffix = "lt";
    else if (cmp_mode == "GT")
        suffix = "gt";
    else if (cmp_mode == "GE")
        suffix = "ge";
    else if (cmp_mode == "LE")
        suffix = "le";
    // Detect scalar compare: check if src1 is a RegTensor variable (mirrors EmitVFDuplicate).
    bool is_scalar_src = true;
    auto src1_var = ir::As<ir::Var>(op->args_[2]);
    if (src1_var) {
        std::string src1_name = codegen.GetVarName(src1_var);
        is_scalar_src = !codegen.IsRegTensorVar(src1_name);
    }
    if (is_scalar_src) {
        // Vector-scalar compare: vcmps_xx
        codegen.Emit("vcmps_" + suffix + "(" + mask_dst + ", " + src0 + ", " + src1 + ", " + mask_src + ");");
    } else {
        // Vector-vector compare: vcmp_xx
        // vcmp_xx requires src0/src1 share the same vector element type. The caller
        // can pass an explicit `cmp_dtype` kwarg to pin the compare width (mirrors
        // `Compare<uint8_t>` in vf_topk_16_gather.h, where u16 regs are cast to u8 so
        // chistv2 sees a 256-lane byte-compare mask). When absent, fall back to src0.
        DataType canonical = GetExprDtype(op->args_[1]);
        if (op->HasKwarg("cmp_dtype")) {
            canonical = op->GetKwarg<DataType>("cmp_dtype");
        }
        DataType s0_dt = GetExprDtype(op->args_[1]);
        DataType s1_dt = GetExprDtype(op->args_[2]);
        std::string cast_prefix = "(RegTensor<" + canonical.ToCTypeString() + "> &)";
        std::string s0_expr = (s0_dt == canonical) ? src0 : (cast_prefix + src0);
        std::string s1_expr = (s1_dt == canonical) ? src1 : (cast_prefix + src1);
        codegen.Emit("vcmp_" + suffix + "(" + mask_dst + ", " + s0_expr + ", " + s1_expr + ", " + mask_src + ");");
    }
    return "";
}

static std::string EmitVFSqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.squeeze requires 3 args (dst, src, mask)";
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
    // mode kwarg: "STORED" (default) or "NO_STORED"
    // backward compat: gather_mode="STORE_REG" / "NO_STORE_REG"
    std::string mode = "MODE_STORED";
    if (op->HasKwarg("gather_mode")) {
        auto gm = static_cast<ir::SqueezeMode>(op->GetKwarg<int>("gather_mode"));
        if (gm == ir::SqueezeMode::NO_STORE_REG)
            mode = "MODE_NO_STORED";
        else
            mode = "MODE_STORED";
    }
    codegen.Emit("vsqz(" + dst + ", " + src_expr + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFArange(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, start]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string start = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // index_order kwarg selects the direction: INCREASE_ORDER (default) ->
    // dst[i] = start + i; DECREASE_ORDER -> dst[i] = start - i.
    bool is_decrease = false;
    if (op->HasKwarg("index_order")) {
        auto o = static_cast<ir::IndexOrder>(op->GetKwarg<int>("index_order"));
        if (o == ir::IndexOrder::DECREASE_ORDER)
            is_decrease = true;
    }
    // vci only accepts signed integer types (vector_s32/s16/s8) and float types (vector_f16/f32).
    // Cast dst to the matching signed type to ensure overload resolution succeeds.
    std::string signed_type;
    if (dst_dt == DataType::UINT32 || dst_dt == DataType::INT32 || dst_dt == DataType::FP32)
        signed_type = "int32_t";
    else if (dst_dt == DataType::UINT16 || dst_dt == DataType::INT16 || dst_dt == DataType::FP16 ||
             dst_dt == DataType::BF16)
        signed_type = "int16_t";
    else
        signed_type = "int8_t";
    // b64 (INT64/UINT64): single vci does not support 8-byte elements.
    // Replicate AscendC ArangeB64Impl using pure bisheng intrinsics:
    //   1. vci int32 low-half (0,1,2,...) into a temp RegTensor<int32_t>
    //   2. vneg if DECREASE_ORDER (produces 0,-1,-2,...)
    //   3. vdup int32 high-half = 0
    //   4. vintlv to interleave low/high into dst (RegTensor<int64_t>)
    //   5. vadds to add the scalar start offset (as b64)
    if (dst_dt == DataType::INT64 || dst_dt == DataType::UINT64) {
        std::string lo = dst + "_b64_lo_";
        std::string hi = dst + "_b64_hi_";
        std::string dump = dst + "_b64_dump_";
        std::string m = dst + "_b64_m_";
        codegen.Emit("RegTensor<int32_t> " + lo + ";");
        codegen.Emit("RegTensor<int32_t> " + hi + ";");
        codegen.Emit("RegTensor<int32_t> " + dump + ";");
        codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
        codegen.Emit("vci(" + lo + ", 0, INC_ORDER);");
        if (is_decrease) {
            codegen.Emit("vneg(" + lo + ", " + lo + ", " + m + ", MODE_ZEROING);");
        }
        codegen.Emit("vdup(" + hi + ", 0, " + m + ", MODE_ZEROING);");
        codegen.Emit("vintlv((RegTensor<uint32_t> &)" + dst + ", (RegTensor<uint32_t> &)" + dump +
                     ", (RegTensor<uint32_t> &)" + lo + ", (RegTensor<uint32_t> &)" + hi + ");");
        codegen.Emit("vadds(" + dst + ", " + dst + ", (int64_t)(" + start + "), " + m + ", MODE_ZEROING);");
        return "";
    }
    if (is_decrease) {
        std::string m = dst + "_arange_m_";
        codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
        codegen.Emit("vci((RegTensor<" + signed_type + "> &)" + dst + ", 0, INC_ORDER);");
        codegen.Emit("vneg((RegTensor<" + signed_type + "> &)" + dst + ", (RegTensor<" + signed_type + "> &)" + dst +
                     ", " + m + ", MODE_ZEROING);");
        codegen.Emit("vadds((RegTensor<" + signed_type + "> &)" + dst + ", (RegTensor<" + signed_type + "> &)" + dst +
                     ", " + start + ", " + m + ", MODE_ZEROING);");
    } else {
        codegen.Emit("vci((RegTensor<" + signed_type + "> &)" + dst + ", " + start + ", INC_ORDER);");
    }
    return "";
}

static std::string EmitVFGather(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src_ub, indices, mask]
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    std::string base_c_type = dst_dt.ToCTypeString();
    std::string src_ub = GetUBufPtr(codegen, op->args_[1], base_c_type);
    std::string indices = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // Check mode: DATA_BLOCK_LOAD -> vgatherb, otherwise -> vgather2
    bool is_datablock = false;
    if (op->HasKwarg("data_copy_mode")) {
        auto mode = static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"));
        is_datablock = (mode == ir::DataCopyMode::DATA_BLOCK_LOAD);
    }

    if (is_datablock) {
        codegen.Emit("vgatherb(" + dst + ", " + src_ub + ", (RegTensor<uint32_t> &)" + indices + ", " + mask + ");");
    } else {
        // Index reg type mirrors how vgather2 overloads are wired up:
        //   - 4-byte load (u32/s32/f32) -> u32 index
        //   - 2-byte load (u16/s16/f16/bf16) -> u16 index
        //   - 1-byte load (u8/s8/...)        -> u16 index (matches SDK macros)
        std::string idx_c_type = (dst_dt == DataType::UINT32) ? "uint32_t" :
                                 (dst_dt == DataType::INT32)  ? "int32_t" :
                                 (dst_dt == DataType::FP32)   ? "uint32_t" :
                                                                "uint16_t";
        codegen.Emit("vgather2(" + dst + ", " + src_ub + ", (RegTensor<" + idx_c_type + "> &)" + indices + ", " + mask +
                     ");");
    }
    return "";
}

static std::string EmitVFStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to pstu
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    if (src_is_mask) {
        CHECK(op->args_.size() == 3) << "vf.store_unalign mask path requires 3 args (ptr, mask, ureg)";
        std::string vreg = codegen.GetExprAsCode(op->args_[1]);
        DataType mask_dt = GetExprDtype(op->args_[1], DataType::UINT16);
        int elem_bytes = static_cast<int>(mask_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        // pstu only accepts uint16_t* or uint32_t* (AscendC DataCopyUnAlignImpl
        // casts to unsigned regardless of template T). b16→uint16_t, b32→uint32_t.
        std::string ptr_type = (elem_bytes <= 2) ? "uint16_t" : "uint32_t";
        // pstu modifies the pointer in-place (*&), so use post-update ref.
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type, /*is_post_update=*/true);
        std::string ureg = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("pstu(" + ureg + ", " + vreg + ", " + ub_ptr + ");");
        return "";
    }
    // Two calling conventions, distinguished by arg count:
    //   3 args [dst, src, align_reg]                    -> vstur (strideless, legacy)
    //   4 args [dst, vreg, ureg, stride] (+post_update) -> vstus (strided)
    // Store-unalign on A5 (dav_3510) only supports signed types for 4/8-byte data
    // (see asc-devkit dav_3510/kernel_reg_compute_datacopy_store_impl.h:495-498).
    // Cast unsigned 32/64-bit src reg + dst ptr to signed int32_t/int64_t.
    CHECK(op->args_.size() >= 3) << "vf.store_unalign requires >=3 args (dst, vreg, ureg[, stride])";
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
    // Unalign-store cursor: keep separate from a base-pointer load of the same tile.
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type, /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    // Reinterpret src reg to signed type when needed.
    std::string vreg_expr = (cast_dt == src_dt) ? vreg : ("(RegTensor<" + base_c_type + "> &)" + vreg);

    if (op->args_.size() >= 4) {
        // Strided form -> vstus(ureg, stride, vreg, dst, POST_UPDATE|NORM)
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        std::string pu = post_update ? "POST_UPDATE" : "NORM";
        codegen.Emit("vstus(" + ureg + ", " + stride + ", " + vreg_expr + ", " + dst_ptr + ", " + pu + ");");
    } else {
        // Strideless legacy form -> vstur(align_reg, src, dst, POST_UPDATE)
        codegen.Emit("vstur(" + ureg + ", " + vreg_expr + ", " + dst_ptr + ", POST_UPDATE);");
    }
    return "";
}

static std::string EmitVFStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Two calling conventions, distinguished by arg count:
    //   2 args [dst, align_reg]                  -> vstar (strideless, legacy)
    //   3 args [dst, ureg, stride] (+post_update) -> vstas (strided)
    // Match the paired store's pointer type — A5 only supports signed types.
    CHECK(op->args_.size() >= 2) << "vf.store_unalign_post requires >=2 args (dst, ureg[, stride])";
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
    // Share the same cursor as the paired store_unalign (is_post_update=true).
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type, /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);

    if (op->args_.size() >= 3) {
        // Strided form -> vstas(ureg, dst, stride, POST_UPDATE|NORM)
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        std::string pu = post_update ? "POST_UPDATE" : "NORM";
        codegen.Emit("vstas(" + ureg + ", " + dst_ptr + ", " + stride + ", " + pu + ");");
    } else {
        // Strideless legacy form -> vstar(align_reg, dst)
        codegen.Emit("vstar(" + ureg + ", " + dst_ptr + ");");
    }
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
// UnalignRegForLoad — declare unaligned load register
// ============================================================================

static std::string EmitVFUnalignRegForLoad(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + reg_name + ";");
    return "";
}

// ============================================================================
// LoadUnalignPre — vldas (setup unaligned load)
// ============================================================================

static std::string EmitVFLoadUnalignPre(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.load_unalign_pre requires 2 args (ureg, src_ptr)";
    std::string ureg = codegen.GetExprAsCode(op->args_[0]);
    DataType dt = GetExprDtype(op->args_[1], DataType::FP32);
    int elem_bytes = static_cast<int>(dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    std::string ptr_type;
    if (elem_bytes == 1) {
        ptr_type = "uint8_t";
    } else if (elem_bytes == 8) {
        ptr_type = "uint32_t";
    } else if (elem_bytes == 4) {
        ptr_type = "int32_t";
    } else {
        if (dt == DataType::FP16 || dt == DataType::BF16)
            ptr_type = "half";
        else
            ptr_type = "uint16_t";
    }
    std::string src_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    codegen.Emit("vldas(" + ureg + ", " + src_ptr + ");");
    return "";
}

// ============================================================================
// LoadUnalign — vldus (unaligned load body, supports 3-arg and 4-arg strided)
// ============================================================================

static std::string EmitVFLoadUnalign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 3) << "vf.load_unalign requires 3-4 args (dst, ureg, src_ptr [, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    std::string ptr_type = dst_dt.ToCTypeString();
    if (op->args_.size() >= 4) {
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        std::string src_ptr = codegen.GetOrCreateVFTilePtr(op->args_[2], /*is_post_update=*/true);
        if (ptr_type != "float") {
            src_ptr = "(__ubuf__ " + ptr_type + " *&)" + src_ptr;
        }
        int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        std::string effective_stride = (elem_bytes == 8) ? ("(" + stride + ") * 2") : stride;
        codegen.Emit("vldus(" + dst + ", " + ureg + ", " + src_ptr + ", " + effective_stride + ", POST_UPDATE);");
    } else {
        std::string src_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        codegen.Emit("vldus(" + dst + ", " + ureg + ", " + src_ptr + ");");
    }
    return "";
}

// ============================================================================
// Scatter — vscatter (scatter store by indices)
// ============================================================================

static std::string EmitVFScatter(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.scatter requires 4 args (base_ptr, src, index, mask)";
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string index = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    std::string base_c_type = src_dt.ToCTypeString();
    std::string base_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type);
    std::string idx_c_type = (src_dt == DataType::UINT32) ? "uint32_t" :
                             (src_dt == DataType::INT32)  ? "int32_t" :
                             (src_dt == DataType::FP32)   ? "uint32_t" :
                                                            "uint16_t";
    codegen.Emit("vscatter(" + src + ", " + base_ptr + ", (RegTensor<" + idx_c_type + "> &)" + index + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// Unsqueeze — vusqz (expand mask bits into register)
// ============================================================================

static std::string EmitVFUnsqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.unsqueeze requires 2 args (dst, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string mask = codegen.GetExprAsCode(op->args_[1]);
    codegen.Emit("vusqz(" + dst + ", " + mask + ");");
    return "";
}

// ============================================================================
// Truncate — vtrc with ROUND_Z (alias for Round with round_mode=TRUNC)
// ============================================================================

static std::string EmitVFTruncate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.truncate requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string round_const = "ROUND_Z";
    if (op->HasKwarg("round_mode")) {
        auto rm = static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"));
        if (rm == ir::VFRoundMode::CAST_RINT)
            round_const = "ROUND_R";
        else if (rm == ir::VFRoundMode::CAST_CEIL)
            round_const = "ROUND_C";
        else if (rm == ir::VFRoundMode::CAST_FLOOR)
            round_const = "ROUND_F";
        else
            round_const = "ROUND_Z";
    }
    std::string mode = GetVFMergeMode(op);
    codegen.Emit("vtrc(" + dst + ", " + src + ", " + round_const + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MaskGenWithRegTensor — movvp (generate MaskReg from RegTensor bit offset)
// ============================================================================

static std::string EmitVFMaskGenWithRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "vf.mask_gen_with_reg_tensor requires 1 arg (src)";
    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string mask_dst = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + mask_dst + ";");
    codegen.RegisterMaskRegVar(mask_dst);
    std::string offset = "0";
    if (op->HasKwarg("offset")) {
        offset = std::to_string(op->GetKwarg<int>("offset"));
    }
    DataType src_dt = GetExprDtype(op->args_[0]);
    if (src_dt == DataType::FP16 || src_dt == DataType::BF16 || src_dt == DataType::UINT16 ||
        src_dt == DataType::INT16) {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint16_t> &)" + src + ", " + offset + ");");
    } else {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint32_t> &)" + src + ", " + offset + ");");
    }
    return "";
}

// ============================================================================
// GetMaskSpr (unified) — movp_b32/movp_b16 with width kwarg
// Replaces: GetMaskSprB32, GetMaskSprB16
// ============================================================================

static std::string EmitVFGetMaskSpr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string width = "B32";
    if (op->HasKwarg("width"))
        width = VFEnumValueName(ir::EnumToString(static_cast<ir::MaskWidth>(op->GetKwarg<int>("width"))));
    if (width == "B16")
        codegen.Emit("MaskReg " + reg_name + " = movp_b16();");
    else
        codegen.Emit("MaskReg " + reg_name + " = movp_b32();");
    codegen.RegisterMaskRegVar(reg_name);
    return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "vf.reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRegTensor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_reg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaskReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.create_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDuplicate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.and_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAnd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFXor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.or_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFOr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMul(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulAddDst(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDiv(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMuls(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLn(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExp(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.not_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNot(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqrt(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNeg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdds(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSubs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMins(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaxs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.leaky_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLeakyRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pair_reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPairReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbsSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.axpy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAxpy(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.copy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCopy(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_dst_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulDstAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unpack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnpack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_left")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftLeft(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_right")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftRight(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mull")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMull(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAddc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSubc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExpSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.astype")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCast(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.de_interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDeInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSelect(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.update_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUpdateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mem_bar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMemBar(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.histograms")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFHistograms(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.eq")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "EQ");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ne")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "NE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.lt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.gt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.le")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ge")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.arange")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFArange(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGather(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreUnAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFStoreUnAlignPost(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unalign_reg_for_store")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForStore(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.clear_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFClearSpr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_init")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForLoad(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_pre")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalignPre(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.scatter")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFScatter(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unsqueeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnsqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.truncate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFTruncate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_gen_with_reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFMaskGenWithRegTensor(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.get_mask_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGetMaskSpr(op, codegen); });

// ============================================================================
// Log2 — composite: vln + vmuls(1/ln2) = ln(x) * 1.4426950408889634
// ============================================================================

static std::string EmitVFLog2(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.log2 requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", MODE_ZEROING);");
    codegen.Emit("vmuls(" + dst + ", " + dst + ", 1.4426950408889634f, " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// Log10 — composite: vln + vmuls(1/ln10) = ln(x) * 0.4342944819032518
// ============================================================================

static std::string EmitVFLog10(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.log10 requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", MODE_ZEROING);");
    codegen.Emit("vmuls(" + dst + ", " + dst + ", 0.4342944819032518f, " + mask + ", MODE_ZEROING);");
    return "";
}

// ============================================================================
// MulsCast — vmulscvt (fused multiply-scalar-cast: dst(fp16) = cast(src(fp32) * scalar))
// ============================================================================

static std::string EmitVFMulsCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.muls_cast requires 4 args (dst, src, scalar, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vmulscvt(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + part + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.log2")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog2(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log10")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog10(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulsCast(op, codegen); });

// ============================================================================
// Load (unified) — AscendC Load: vldas + vldus all-in-one unaligned load
// ============================================================================

static std::string EmitVFLoad(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 3) << "vf.load requires 2-3 args (dst, src_ptr[, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // vldas requires integer pointer types; vldus requires dst reg type == ptr type.
    // Strategy: use int-type ptr for vldas, native-type ptr for vldus, cast dst for vldus if needed.
    std::string vldas_ptr_type;
    if (elem_bytes == 1) {
        vldas_ptr_type = "uint8_t";
    } else if (elem_bytes == 8) {
        vldas_ptr_type = "uint32_t";
    } else if (elem_bytes == 4) {
        vldas_ptr_type = "int32_t";
    } else {
        if (dst_dt == DataType::FP16 || dst_dt == DataType::BF16)
            vldas_ptr_type = "half";
        else
            vldas_ptr_type = "uint16_t";
    }
    // vldus needs matching dst/ptr types; for float data use float ptr directly.
    std::string vldus_ptr_type = dst_dt.ToCTypeString();
    static int load_counter = 0;
    std::string ureg_name = "__ureg_ld_" + std::to_string(load_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    if (op->args_.size() == 3) {
        std::string vldas_ptr = GetUBufPtr(codegen, op->args_[1], vldas_ptr_type, /*is_post_update=*/true);
        std::string vldus_ptr = GetUBufPtr(codegen, op->args_[1], vldus_ptr_type, /*is_post_update=*/true);
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        std::string effective_stride = (elem_bytes == 8) ? ("(" + stride + ") * 2") : stride;
        std::string post_mode = "POST_UPDATE";
        if (op->HasKwarg("post_mode")) {
            post_mode = op->GetKwarg<std::string>("post_mode");
        }
        codegen.Emit("vldas(" + ureg_name + ", " + vldas_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + vldus_ptr + ", " + effective_stride + ", " + post_mode +
                     ");");
    } else {
        std::string vldas_ptr = GetUBufPtr(codegen, op->args_[1], vldas_ptr_type);
        std::string vldus_ptr = GetUBufPtr(codegen, op->args_[1], vldus_ptr_type);
        codegen.Emit("vldas(" + ureg_name + ", " + vldas_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + vldus_ptr + ");");
    }
    return "";
}

// ============================================================================
// Store (unified) — AscendC Store: vstus + vstas all-in-one unaligned store
// ============================================================================

static std::string EmitVFStore(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 3) << "vf.store requires 2-3 args (dst_ptr, src[, count])";
    DataType src_dt = GetExprDtype(op->args_[1]);
    int elem_bytes = static_cast<int>(src_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // vstus requires matching src reg type and dst pointer type (same as vldus).
    // Use native type for the pointer; no cast on the src register.
    std::string ptr_type = src_dt.ToCTypeString();
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type, /*is_post_update=*/true);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string count;
    if (op->args_.size() == 3) {
        count = codegen.GetExprAsCode(op->args_[2]);
    } else if (op->HasKwarg("count")) {
        count = std::to_string(op->GetKwarg<int>("count"));
    } else {
        count = std::to_string(256 / elem_bytes);
    }
    std::string effective_count = (elem_bytes == 8) ? ("(" + count + ") * 2") : count;
    std::string post_mode = "POST_UPDATE";
    if (op->HasKwarg("post_mode")) {
        post_mode = op->GetKwarg<std::string>("post_mode");
    }
    static int store_counter = 0;
    std::string ureg_name = "__ureg_st_" + std::to_string(store_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    codegen.Emit("vstus(" + ureg_name + ", " + effective_count + ", " + src + ", " + dst_ptr + ", " + post_mode + ");");
    codegen.Emit("vstas(" + ureg_name + ", " + dst_ptr + ", 0, " + post_mode + ");");
    return "";
}

// EmitVFMaskLoad/Store/StoreUnalign have been removed — their logic is now
// unified into EmitVFLoadAlign/EmitVFStoreAlign/EmitVFStoreUnAlign via
// IsMaskRegVar dispatch, matching AscendC's function-overloading model.

REGISTER_BACKEND_OP(BackendCCE, "vf.load")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoad(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStore(op, codegen); });

// mask_load/mask_store/mask_store_unalign backend registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch via IsMaskRegVar.

// ============================================================================
// CreateAddrReg — AddrReg declaration + vag_b8/b16/b32 intrinsic
// ============================================================================

static std::string EmitVFCreateAddrReg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 8 && op->args_.size() % 2 == 0)
        << "vf.create_addr_reg requires 2-8 args (1-4 index/stride pairs)";
    std::string reg_name = codegen.GetCurrentResultTarget();
    // Determine element width from dtype kwarg (default b32)
    DataType dt = DataType::FP32;
    if (op->HasKwarg("dtype")) {
        dt = op->GetKwarg<DataType>("dtype");
    }
    std::string vag_fn;
    if (dt == DataType::UINT8 || dt == DataType::INT8)
        vag_fn = "vag_b8";
    else if (dt == DataType::UINT16 || dt == DataType::INT16 || dt == DataType::FP16 || dt == DataType::BF16)
        vag_fn = "vag_b16";
    else
        vag_fn = "vag_b32";
    // Collect stride args (every 2nd arg). For b64, each stride is doubled.
    std::string stride_args;
    for (size_t i = 1; i < op->args_.size(); i += 2) {
        std::string stride = codegen.GetExprAsCode(op->args_[i]);
        if (dt == DataType::UINT64 || dt == DataType::INT64) {
            stride = "(" + stride + ") * 2";
        }
        if (!stride_args.empty())
            stride_args += ", ";
        stride_args += stride;
    }
    // AddrReg (vector_address) must be declared and initialized in a single
    // statement (bisheng rejects a separate declaration + assignment). Emit the
    // declaration and vag_* initializer together, matching AscendC's
    // `AddrReg x = CreateAddrReg<T>(...)` usage. The vag_* must sit inside the
    // physical loop it is bound to (HardwareLoop); the single-iteration
    // optimization must not collapse that loop — see BodyContainsAddrReg.
    codegen.Emit("AddrReg " + reg_name + " = " + vag_fn + "(" + stride_args + ");");
    codegen.RegisterAddrRegVar(reg_name);
    return "";
}

// ============================================================================
// Move — vmov (RegTensor) / pmov (MaskReg), with or without mask
// ============================================================================

static std::string EmitVFMove(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2 || op->args_.size() == 3)
        << "vf.move requires 2 args (dst, src) or 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    // Detect dst type: MaskReg vs RegTensor
    bool is_mask_dst = false;
    auto dst_var = ir::As<ir::Var>(op->args_[0]);
    if (dst_var) {
        is_mask_dst = codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    if (op->args_.size() == 3) {
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ", " + mask + ");");
        } else {
            codegen.Emit("vmov(" + dst + ", " + src + ", " + mask + ", MODE_MERGING);");
        }
    } else {
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ");");
        } else {
            codegen.Emit("vmov(" + dst + ", " + src + ");");
        }
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.create_addr_reg")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateAddrReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.move")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMove(op, codegen); });

} // namespace backend
} // namespace pypto

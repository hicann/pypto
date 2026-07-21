/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "codegen/cce/cce_codegen.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

class CapturingCCECodegen final : public codegen::CCECodegen {
public:
    void SetTarget(std::string target) { target_ = std::move(target); }

    [[nodiscard]] std::string GetCurrentResultTarget() const override { return target_; }

    void Emit(const std::string& line) override
    {
        emitted_ += line;
        emitted_ += '\n';
    }

    std::string GetExprAsCode(const ir::ExprPtr& expr) override
    {
        if (auto var = ir::As<ir::Var>(expr)) {
            return var->name_;
        }
        if (auto value = ir::As<ir::ConstInt>(expr)) {
            return std::to_string(value->value_);
        }
        if (auto value = ir::As<ir::ConstFloat>(expr)) {
            return std::to_string(value->value_);
        }
        return codegen::CCECodegen::GetExprAsCode(expr);
    }

    std::string GetVarName(const ir::VarPtr& var) override { return var->name_; }

    [[nodiscard]] const std::string& Emitted() const { return emitted_; }

private:
    std::string target_{"result"};
    std::string emitted_;
};

ir::TypePtr ScalarType(ir::DataType dtype) { return std::make_shared<const ir::ScalarType>(dtype); }

ir::VarPtr MakeVar(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    return std::make_shared<const ir::Var>(name, ScalarType(dtype), ir::Span::Unknown());
}

ir::VarPtr MakeTile(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    auto type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, dtype,
                                                     std::optional<ir::MemRefPtr>(std::nullopt),
                                                     std::optional<ir::TileView>(std::nullopt));
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::ExprPtr Int(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, ir::Span::Unknown());
}

ir::ExprPtr Float(double value)
{
    return std::make_shared<const ir::ConstFloat>(value, ir::DataType::FP32, ir::Span::Unknown());
}

int EnumValue(ir::MergeMode value) { return static_cast<int>(value); }

template <typename Enum>
int EnumValue(Enum value)
{
    return static_cast<int>(value);
}

ir::CallPtr MakeCall(const std::string& name, std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {})
{
    return std::make_shared<const ir::Call>(name, std::move(args), std::move(kwargs), ir::Span::Unknown());
}

std::string Invoke(CapturingCCECodegen& codegen, const std::string& name, std::vector<ir::ExprPtr> args = {},
                   Kwargs kwargs = {}, const std::string& target = "result")
{
    const std::size_t emitted_size = codegen.Emitted().size();
    const auto* info = BackendCCE::Instance().GetOpInfo(name);
    EXPECT_NE(info, nullptr) << name;
    if (info == nullptr) {
        return "";
    }
    EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
    codegen.SetTarget(target);
    EXPECT_TRUE(info->codegen_func(MakeCall(name, std::move(args), std::move(kwargs)), codegen).empty());
    return codegen.Emitted().substr(emitted_size);
}

void ExpectContains(const std::string& generated, const std::vector<std::string>& fragments)
{
    for (const auto& fragment : fragments) {
        EXPECT_NE(generated.find(fragment), std::string::npos) << fragment << "\n" << generated;
    }
}

void ExpectInvoke(CapturingCCECodegen& codegen, const std::string& name, const std::vector<std::string>& expected,
                  std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {}, const std::string& target = "result")
{
    SCOPED_TRACE(name);
    ExpectContains(Invoke(codegen, name, std::move(args), std::move(kwargs), target), expected);
}

TEST(BackendCCEVFOpsTest, RegistersExpectedVectorFunctionOperations)
{
    const std::vector<std::string> names = {"vf.reg_tensor",
                                            "vf.create_mask",
                                            "vf.full",
                                            "vf.load_align",
                                            "vf.store_align",
                                            "vf.max",
                                            "vf.add",
                                            "vf.sub",
                                            "vf.and_",
                                            "vf.xor",
                                            "vf.or_",
                                            "vf.reduce_sum",
                                            "vf.reduce_max",
                                            "vf.reduce_min",
                                            "vf.mul",
                                            "vf.mul_add_dst",
                                            "vf.div",
                                            "vf.muls",
                                            "vf.ln",
                                            "vf.log",
                                            "vf.min",
                                            "vf.exp",
                                            "vf.abs",
                                            "vf.not_",
                                            "vf.sqrt",
                                            "vf.relu",
                                            "vf.neg",
                                            "vf.adds",
                                            "vf.subs",
                                            "vf.mins",
                                            "vf.maxs",
                                            "vf.leaky_relu",
                                            "vf.interleave",
                                            "vf.pair_reduce_sum",
                                            "vf.abs_sub",
                                            "vf.axpy",
                                            "vf.copy",
                                            "vf.mul_dst_add",
                                            "vf.pack",
                                            "vf.unpack",
                                            "vf.prelu",
                                            "vf.shift_left",
                                            "vf.shift_right",
                                            "vf.mull",
                                            "vf.addc",
                                            "vf.subc",
                                            "vf.exp_sub",
                                            "vf.astype",
                                            "vf.de_interleave",
                                            "vf.select",
                                            "vf.update_mask",
                                            "vf.mem_bar",
                                            "vf.histograms",
                                            "vf.eq",
                                            "vf.ne",
                                            "vf.lt",
                                            "vf.gt",
                                            "vf.le",
                                            "vf.ge",
                                            "vf.squeeze",
                                            "vf.arange",
                                            "vf.gather",
                                            "vf.store_unalign",
                                            "vf.store_unalign_post",
                                            "vf.unalign_reg_for_store",
                                            "vf.clear_spr",
                                            "vf.load_unalign_init",
                                            "vf.load_unalign_pre",
                                            "vf.load_unalign",
                                            "vf.scatter",
                                            "vf.unsqueeze",
                                            "vf.truncate",
                                            "vf.mask_gen_with_reg_tensor",
                                            "vf.get_mask_spr",
                                            "vf.log2",
                                            "vf.log10",
                                            "vf.muls_cast",
                                            "vf.load",
                                            "vf.store",
                                            "vf.create_addr_reg",
                                            "vf.move"};

    for (const auto& name : names) {
        const auto* info = BackendCCE::Instance().GetOpInfo(name);
        ASSERT_NE(info, nullptr) << name;
        EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
        EXPECT_TRUE(static_cast<bool>(info->codegen_func)) << name;
    }
}

TEST(BackendCCEVFOpsTest, EmitsDeclarationsMasksBroadcastsAndMoves)
{
    CapturingCCECodegen codegen;
    auto fp = MakeVar("fp");
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto u4 = MakeVar("u4", ir::DataType::UINT4);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);

    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::FP32}}, "fp");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::INT4}}, "s4");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::UINT4}}, "u4");
    EXPECT_TRUE(codegen.IsRegTensorVar("fp"));
    EXPECT_TRUE(codegen.IsRegTensorVar("s4"));
    EXPECT_TRUE(codegen.IsRegTensorVar("u4"));

    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask;", "mask = pset_b8(PAT_VL16);"}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::VL16)}, {"dtype", ir::DataType::INT8}}, "mask");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask2;", "mask2 = pset_b16(PAT_ALL);"}, {},
                 {{"dtype", ir::DataType::FP16}}, "mask2");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask_default;", "mask_default = pset_b32(PAT_ALL);"}, {}, {},
                 "mask_default");
    EXPECT_TRUE(codegen.IsMaskRegVar("mask"));

    ExpectInvoke(codegen, "vf.full", {"vbr(fp, 2.500000);"}, {fp, Float(2.5)});
    ExpectInvoke(codegen, "vf.full", {"vdup(fp, 1.000000, mask, MODE_MERGING);"}, {fp, Float(1.0), mask},
                 {{"mode", EnumValue(ir::MergeMode::MERGING)}});
    codegen.RegisterRegTensorVar("s4");
    ExpectInvoke(codegen, "vf.full", {"POS_HIGHEST", "MODE_ZEROING"}, {fp, s4},
                 {{"pos", EnumValue(ir::DuplicatePos::HIGHEST)}});

    ExpectInvoke(codegen, "vf.create_addr_reg", {"AddrReg addr = vag_b32((4) * 2, (8) * 2);"},
                 {Int(0), Int(4), Int(1), Int(8)}, {{"dtype", ir::DataType::INT64}}, "addr");
    EXPECT_TRUE(codegen.IsAddrRegVar("addr"));
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, s4);"}, {fp, s4});
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, s4, mask, MODE_MERGING);"}, {fp, s4, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask);"}, {mask2, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask, mask);"}, {mask2, mask, mask});
}

TEST(BackendCCEVFOpsTest, EmitsArithmeticIntrinsics)
{
    CapturingCCECodegen codegen;
    auto dst = MakeVar("dst");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);

    const Kwargs merging = {{"mode", EnumValue(ir::MergeMode::MERGING)}};
    const auto expect_binary = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, src1, mask}, merging);
    };
    expect_binary("vf.max", "vmax(");
    expect_binary("vf.add", "vadd(");
    expect_binary("vf.sub", "vsub(");
    expect_binary("vf.and_", "vand(");
    expect_binary("vf.xor", "vxor(");
    expect_binary("vf.or_", "vor(");
    expect_binary("vf.mul", "vmul(");
    expect_binary("vf.mul_add_dst", "vmula(");
    expect_binary("vf.div", "vdiv(");
    expect_binary("vf.min", "vmin(");
    expect_binary("vf.abs_sub", "vabsdif(");
    expect_binary("vf.mul_dst_add", "vmadd(");
    expect_binary("vf.prelu", "vprelu(");
    ExpectInvoke(codegen, "vf.add", {"vaddc("}, {carry, int_dst, int_src0, int_src1, mask});
    ExpectInvoke(codegen, "vf.sub", {"vsubc("}, {carry, int_dst, int_src0, int_src1, mask});

    const auto expect_unary = [&](const std::string& name, const std::vector<std::string>& expected) {
        ExpectInvoke(codegen, name, expected, {dst, src0, mask}, merging);
    };
    expect_unary("vf.ln", {"vln("});
    expect_unary("vf.log", {"vln("});
    expect_unary("vf.exp", {"vexp("});
    expect_unary("vf.abs", {"vabs("});
    expect_unary("vf.not_", {"vnot("});
    expect_unary("vf.sqrt", {"vsqrt("});
    expect_unary("vf.relu", {"vrelu("});
    expect_unary("vf.neg", {"vneg("});
    expect_unary("vf.log2", {"vln(", "1.4426950408889634f"});
    expect_unary("vf.log10", {"vln(", "0.4342944819032518f"});

    const auto expect_scalar = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, Float(0.5), mask}, merging);
    };
    expect_scalar("vf.muls", "vmuls(");
    expect_scalar("vf.adds", "vadds(");
    expect_scalar("vf.subs", "vadds(");
    expect_scalar("vf.mins", "vmins(");
    expect_scalar("vf.maxs", "vmaxs(");
    expect_scalar("vf.leaky_relu", "vlrelu(");
    expect_scalar("vf.muls_cast", "vmulscvt(");
}

TEST(BackendCCEVFOpsTest, EmitsReductionAndPermutationIntrinsics)
{
    CapturingCCECodegen codegen;
    auto dst = MakeVar("dst");
    auto dst2 = MakeVar("dst2");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto int_src = MakeVar("int_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("int_src");
    const Kwargs merging = {{"mode", EnumValue(ir::MergeMode::MERGING)}};

    ExpectInvoke(codegen, "vf.reduce_sum", {"vcgadd("}, {dst, src0, mask},
                 {{"datablock", true}, {"merge_mode", EnumValue(ir::MergeMode::MERGING)}});
    ExpectInvoke(codegen, "vf.reduce_max", {"vcmax("}, {dst, src0, mask});
    ExpectInvoke(codegen, "vf.reduce_min", {"vcgmin("}, {dst, src0, mask}, {{"datablock", true}});
    ExpectInvoke(codegen, "vf.interleave", {"vintlv("}, {dst, dst2, src0, src1});
    ExpectInvoke(codegen, "vf.de_interleave", {"vdintlv("}, {dst, dst2, src0, int_src});
    ExpectInvoke(codegen, "vf.pair_reduce_sum", {"vcpadd("}, {dst, src0, mask}, merging);
    ExpectInvoke(codegen, "vf.axpy", {"vaxpy("}, {dst, src0, Float(0.25), mask}, merging);
    ExpectInvoke(codegen, "vf.copy", {"vmov("}, {dst, src0, mask});
    ExpectInvoke(codegen, "vf.shift_left", {"vshls("}, {dst, src0, Int(2), mask}, merging);
    ExpectInvoke(codegen, "vf.shift_right", {"vshr("}, {dst, src0, int_src, mask}, merging);
    ExpectInvoke(codegen, "vf.mull", {"vmull("}, {dst, dst2, src0, src1, mask});
    ExpectInvoke(codegen, "vf.addc", {"vaddcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.subc", {"vsubcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.exp_sub", {"vexpdif(", "PART_ODD"}, {dst, src0, src1, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}});
    ExpectInvoke(codegen, "vf.select", {"vsel("}, {dst, src0, int_src, mask});
    ExpectInvoke(codegen, "vf.mem_bar", {"mem_bar(VV_ALL)"}, {}, {{"mode", EnumValue(ir::MemBarMode::VV_ALL)}});
}

TEST(BackendCCEVFOpsTest, EmitsPackAndCastIntrinsics)
{
    CapturingCCECodegen codegen;
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto i16 = MakeVar("i16", ir::DataType::INT16);
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var : {fp32, fp16, i64, i32, i16, s4}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.pack", {"vpack(", "HIGHER"}, {fp16, fp32}, {{"part", EnumValue(ir::PackPart::UPPER)}});
    ExpectInvoke(codegen, "vf.pack", {"vdintlv("}, {i32, i64});
    ExpectInvoke(codegen, "vf.unpack", {"vunpack(", "HIGHER"}, {fp32, fp16},
                 {{"part", EnumValue(ir::PackPart::UPPER)}});
    ExpectInvoke(codegen, "vf.unpack", {"vintlv("}, {i64, i32});

    const Kwargs cast_options = {{"layout", EnumValue(ir::CastLayout::ONE)},
                                 {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                                 {"saturate", EnumValue(ir::SaturateMode::ON)}};
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, fp16, mask, PART_ODD, MODE_ZEROING);"}, {fp32, fp16, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, i32, mask, ROUND_F, MODE_ZEROING);"}, {fp32, i32, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp32, mask, ROUND_F, RS_ENABLE, MODE_ZEROING);"}, {i32, fp32, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp16, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {i32, fp16, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp16, i32, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {fp16, i32, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, i32, mask, RS_ENABLE, PART_ODD, MODE_ZEROING);"}, {i16, i32, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt_s42f16(fp16, s4, mask, PART_P1, MODE_ZEROING);"}, {fp16, s4, mask},
                 cast_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt_f162s4(s4, fp16, mask, ROUND_F, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {s4, fp16, mask}, cast_options);
}

TEST(BackendCCEVFOpsTest, EmitsCompareHistogramAndMaskConversions)
{
    CapturingCCECodegen codegen;
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var : {fp32, fp16, i64, i32, u8}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.update_mask", {"plt_b8("}, {Int(17)}, {{"dtype", ir::DataType::UINT8}}, "mask8");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b16("}, {Int(17)}, {{"dtype", ir::DataType::FP16}}, "mask16");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b32("}, {Int(17)}, {}, "mask32");
    ExpectInvoke(codegen, "vf.histograms", {"dhistv2("}, {i32, fp32, mask},
                 {{"bin_type", EnumValue(ir::BinType::BIN1)}, {"hist_type", EnumValue(ir::HistType::FREQUENCY)}});
    ExpectInvoke(codegen, "vf.histograms", {"chistv2("}, {i32, u8, mask}, {{"bin_type", EnumValue(ir::BinType::BIN0)}});

    ExpectInvoke(codegen, "vf.eq", {"vcmps_eq("}, {mask, fp32, Float(1.0), mask});
    ExpectInvoke(codegen, "vf.ne", {"vcmp_ne("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.lt", {"vcmp_lt("}, {mask, fp32, fp16, mask});
    ExpectInvoke(codegen, "vf.gt", {"vcmp_gt("}, {mask, fp32, fp16, mask});
    ExpectInvoke(codegen, "vf.le", {"vcmp_le("}, {mask, fp32, fp16, mask});
    ExpectInvoke(codegen, "vf.ge", {"vcmp_ge("}, {mask, fp32, fp16, mask});
    ExpectInvoke(codegen, "vf.squeeze", {"vsqz(", "MODE_NO_STORED"}, {i32, fp16, mask},
                 {{"gather_mode", EnumValue(ir::SqueezeMode::NO_STORE_REG)}});
    ExpectInvoke(codegen, "vf.arange", {"vneg(", "vadds("}, {i32, Int(3)},
                 {{"index_order", EnumValue(ir::IndexOrder::DECREASE_ORDER)}});
    ExpectInvoke(codegen, "vf.arange", {"vci(i64_b64_lo_"}, {i64, Int(5)});
    ExpectInvoke(codegen, "vf.unsqueeze", {"vusqz("}, {i32, mask});
    ExpectInvoke(codegen, "vf.truncate", {"vtrc(i32, fp32, ROUND_C, mask, MODE_MERGING)"}, {i32, fp32, mask},
                 {{"round_mode", EnumValue(ir::VFRoundMode::CAST_CEIL)}, {"mode", EnumValue(ir::MergeMode::MERGING)}});
}

TEST(BackendCCEVFOpsTest, EmitsAlignedDataMovement)
{
    CapturingCCECodegen codegen;
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterMaskRegVar("mask");

    ExpectInvoke(codegen, "vf.load_align", {"vlds(fp16"}, {fp16, tile, Int(0)});
    ExpectInvoke(codegen, "vf.load_align", {"BRC_B8"}, {u8, tile8, Int(1)}, {{"dist", EnumValue(ir::LoadDist::BRC)}});
    ExpectInvoke(codegen, "vf.load_align", {"(2) * 2", "POST_UPDATE"}, {i64, tile64, Int(2)}, {{"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vld(fp16"}, {fp16, tile, addr});
    ExpectInvoke(codegen, "vf.load_align", {"plds(mask"}, {mask, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::US)}});
    ExpectInvoke(codegen, "vf.load_align", {"DINTLV_B16", "POST_UPDATE"}, {fp16, fp16b, tile, Int(3)},
                 {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vsldb(", "POST_UPDATE"}, {u8, tile8, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)},
                  {"block_stride", 2},
                  {"repeat_stride", 3},
                  {"post_update", true}});

    ExpectInvoke(codegen, "vf.store_align", {"vsts("}, {tile, fp16, mask});
    ExpectInvoke(codegen, "vf.store_align", {"ONEPT_B8"}, {tile8, u8, mask},
                 {{"dist", EnumValue(ir::StoreDist::FIRST_ELEMENT)}});
    ExpectInvoke(codegen, "vf.store_align", {"INTLV_B16"}, {tile, fp16, fp16b, mask},
                 {{"dist", EnumValue(ir::StoreDist::INTLV)}});
    ExpectInvoke(codegen, "vf.store_align", {"vsstb(", "POST_UPDATE"}, {tile, fp16, mask, Int(2), Int(3)},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.store_align", {"vst(fp16"}, {tile, fp16, mask, addr});
}

TEST(BackendCCEVFOpsTest, EmitsGatherAndUnalignedDataMovement)
{
    CapturingCCECodegen codegen;
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto index = MakeVar("index", ir::DataType::UINT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto ureg = MakeVar("ureg", ir::DataType::INT64);

    ExpectInvoke(codegen, "vf.gather", {"vgather2("}, {fp16, tile, index, mask});
    ExpectInvoke(codegen, "vf.gather", {"vgatherb("}, {fp16, tile, index, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_LOAD)}});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter("}, {tile, fp16, index, mask});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus("}, {fp16, tile});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus(", "(4) * 2", "NORM"},
                 {i64, tile64, Int(4)}, {{"post_mode", std::string("NORM")}});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas("}, {tile, fp16});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas(", "(7) * 2", "NORM"},
                 {tile64, i64, Int(7)}, {{"post_mode", std::string("NORM")}});

    ExpectInvoke(codegen, "vf.unalign_reg_for_store", {"UnalignReg store_ureg;"}, {}, {}, "store_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_init", {"UnalignReg load_ureg;"}, {}, {}, "load_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_pre", {"vldas("}, {ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus("}, {fp16, ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus(", "(4) * 2", "POST_UPDATE"}, {i64, ureg, tile64, Int(4)});
    ExpectInvoke(codegen, "vf.store_unalign", {"vstur("}, {tile8, u8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign", {"vstus(", "POST_UPDATE"}, {tile8, u8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.store_unalign_post", {"vstar("}, {tile8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign_post", {"vstas(", "POST_UPDATE"}, {tile8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.clear_spr", {"sprclr(SPR_AR)"});
}

TEST(BackendCCEVFOpsTest, EmitsMaskLogicOperations)
{
    CapturingCCECodegen codegen;
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterMaskRegVar("mask2");

    ExpectInvoke(codegen, "vf.and_", {"pand("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.or_", {"por("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.xor", {"pxor("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.not_", {"pnot("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.move", {"pmov("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.select", {"psel("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.pack", {"ppack(mask0, mask1, HIGHER)"}, {mask0, mask1},
                 {{"part", EnumValue(ir::PackPart::UPPER)}});
    ExpectInvoke(codegen, "vf.unpack", {"punpack(mask0, mask1, LOWER)"}, {mask0, mask1});
    ExpectInvoke(codegen, "vf.interleave", {"pintlv_b8("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.de_interleave", {"pdintlv_b16("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::FP16}});
}

TEST(BackendCCEVFOpsTest, EmitsMaskMemoryAndSpecialRegisterOperations)
{
    CapturingCCECodegen codegen;
    auto tile = MakeTile("tile", ir::DataType::UINT32);
    auto reg = MakeVar("reg", ir::DataType::UINT16);
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterAddrRegVar("addr");

    ExpectInvoke(codegen, "vf.load_align", {"plds(mask0"}, {mask0, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::DS)}});
    ExpectInvoke(codegen, "vf.load_align", {"pld(mask1"}, {mask1, tile, addr});
    ExpectInvoke(codegen, "vf.store_align", {"psts(mask0"}, {tile, mask0, mask1},
                 {{"dist", EnumValue(ir::StoreDist::NORM)}});
    ExpectInvoke(codegen, "vf.store_align", {"pst(mask0"}, {tile, mask0, addr});
    ExpectInvoke(codegen, "vf.store_unalign", {"pstu("}, {tile, mask0, ureg});
    ExpectInvoke(codegen, "vf.mask_gen_with_reg_tensor", {"movvp(generated_mask, (RegTensor<uint16_t> &)reg, 4)"},
                 {reg}, {{"offset", 4}}, "generated_mask");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b16()"}, {}, {{"width", EnumValue(ir::MaskWidth::B16)}},
                 "spr_mask16");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b32()"}, {}, {}, "spr_mask32");
}

} // namespace
} // namespace backend
} // namespace pypto

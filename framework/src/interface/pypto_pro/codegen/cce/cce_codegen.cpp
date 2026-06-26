/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/cce_codegen.h"

#include <cctype>
#include <cstddef>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "backend/common/backend_config.h"
#include "backend/common/backend_utils.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/kind_traits.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {
using ir::DataType;

// Header for single-file mode (no tensor.h needed - uses direct __gm__ pointers)
const char KERNEL_HEADER_SINGLE[] = R"(
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;
)";

namespace {

bool IsNZTensorType(const ir::TensorTypePtr& tensor_type)
{
    return tensor_type && tensor_type->tensor_view_.has_value() &&
           tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

bool IsIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }

bool IsIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

std::string TrimCopy(const std::string& s)
{
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

bool ConsumeIdentifier(const std::string& s, size_t& pos)
{
    if (pos >= s.size() || !IsIdentStart(s[pos])) {
        return false;
    }
    while (pos < s.size() && IsIdentChar(s[pos])) {
        ++pos;
    }
    return true;
}

bool ConsumeBracketIndex(const std::string& s, size_t& pos)
{
    if (pos >= s.size() || s[pos] != '[') {
        return false;
    }
    int depth = 1;
    ++pos;
    while (pos < s.size() && depth > 0) {
        if (s[pos] == '[') {
            ++depth;
        } else if (s[pos] == ']') {
            --depth;
        }
        ++pos;
    }
    return depth == 0;
}

bool ConsumeMemberAccess(const std::string& s, size_t& pos)
{
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        return ConsumeIdentifier(s, pos);
    }
    if (pos + 1 < s.size() && s[pos] == '-' && s[pos + 1] == '>') {
        pos += 2;
        return ConsumeIdentifier(s, pos);
    }
    return false;
}

bool ConsumeLValueSuffix(const std::string& s, size_t& pos)
{
    if (ConsumeMemberAccess(s, pos)) {
        return true;
    }
    return ConsumeBracketIndex(s, pos);
}

bool IsWritableLValueExpr(const std::string& expr)
{
    std::string s = TrimCopy(expr);

    size_t i = 0;
    if (!ConsumeIdentifier(s, i)) {
        return false;
    }

    while (i < s.size()) {
        if (!ConsumeLValueSuffix(s, i)) {
            return false;
        }
    }
    return true;
}

void ValidateStaticNZTensorShape(const ir::TensorTypePtr& tensor_type, const std::vector<int64_t>& logical_dims)
{
    CHECK(tensor_type != nullptr) << "CCE NZ tensor lowering requires a valid TensorType";
    if (logical_dims.size() != 2) {
        throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires a 2D tensor");
    }

    const int64_t rows = logical_dims[0];
    const int64_t cols = logical_dims[1];
    const int64_t c0 = backend::cce::GetNZInnerCols(tensor_type->dtype_);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    if (rows % 16 != 0 || cols % c0 != 0) {
        throw pypto::ir::ValueError(
            "CCE NZ tensor lowering requires rows divisible by 16 and cols divisible by the destination C0 size");
    }
}

std::vector<int64_t> BuildNZPhysicalShapeDims(
    const ir::TensorTypePtr& tensor_type, const std::vector<int64_t>& logical_dims)
{
    ValidateStaticNZTensorShape(tensor_type, logical_dims);
    const int64_t c0 = backend::cce::GetNZInnerCols(tensor_type->dtype_);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    return {1, logical_dims[1] / c0, logical_dims[0] / 16, 16, c0};
}

} // namespace

CCECodegen::CCECodegen() : backend_(backend::GetBackend())
{
    auto type = backend::GetBackendType();
    CHECK(type == backend::BackendType::CCE)
        << "CCECodegen requires CCE backend, but unknown is configured";
}

// ============================================================================
// Helper function inlining
// ============================================================================

namespace {

/// Renames all variable definitions in a function body by adding a prefix,
/// and substitutes parameter references with call-site argument expressions.
class BodyRenamer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitExpr_;
    using ir::IRMutator::VisitStmt_;
    BodyRenamer(const std::string& prefix, const std::vector<ir::VarPtr>& params, const std::vector<ir::ExprPtr>& args)
        : prefix_(prefix)
    {
        for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
            var_remap_[params[i].get()] = args[i];
        }
    }

private:
    std::string prefix_;

    ir::VarPtr RenameVar(const ir::VarPtr& var)
    {
        auto it = var_remap_.find(var.get());
        if (it != var_remap_.end()) {
            if (auto v = ir::As<ir::Var>(it->second))
                return v;
            return var;
        }
        auto renamed = std::make_shared<const ir::Var>(prefix_ + var->name_, var->GetType(), var->span_);
        var_remap_[var.get()] = renamed;
        return renamed;
    }

    ir::ExprPtr VisitExpr_(const ir::VarPtr& op) override
    {
        auto it = var_remap_.find(op.get());
        if (it != var_remap_.end())
            return it->second;
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto new_value = VisitExpr(op->value_);
        auto target_var = op->var_;
        auto new_var = RenameVar(target_var);
        if (new_var.get() != target_var.get() || new_value.get() != op->value_.get()) {
            return std::make_shared<const ir::AssignStmt>(std::move(new_var), std::move(new_value), op->span_);
        }
        return op;
    }

    std::vector<ir::IterArgPtr> RenameIterArgs(const std::vector<ir::IterArgPtr>& iter_args)
    {
        std::vector<ir::IterArgPtr> new_iter_args;
        new_iter_args.reserve(iter_args.size());
        for (const auto& ia : iter_args) {
            auto new_init = VisitExpr(ia->initValue_);
            // Iter args may shadow helper params, so keep them isolated from call-site remaps.
            auto renamed_var = std::make_shared<const ir::Var>(
                prefix_ + ia->iterVar_->name_, ia->iterVar_->GetType(), ia->iterVar_->span_);
            auto renamed = std::make_shared<const ir::IterArg>(renamed_var, new_init);
            var_remap_[ia->iterVar_.get()] = renamed_var;
            new_iter_args.push_back(renamed);
        }
        return new_iter_args;
    }

    ir::StmtPtr VisitStmt_(const ir::ForStmtPtr& op) override
    {
        auto new_loop_var = RenameVar(op->loopVar_);
        var_remap_[op->loopVar_.get()] = new_loop_var;
        auto new_start = VisitExpr(op->start_);
        auto new_stop = VisitExpr(op->stop_);
        auto new_step = VisitExpr(op->step_);

        auto new_iter_args = RenameIterArgs(op->iterArgs_);
        auto new_body = VisitStmt(op->body_);

        std::vector<ir::VarPtr> new_return_vars;
        new_return_vars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_return_vars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::ForStmt>(
            new_loop_var, new_start, new_stop, new_step, std::move(new_iter_args), new_body, std::move(new_return_vars),
            op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::WhileStmtPtr& op) override
    {
        auto new_iter_args = RenameIterArgs(op->iterArgs_);
        auto new_cond = VisitExpr(op->condition_);
        auto new_body = VisitStmt(op->body_);

        std::vector<ir::VarPtr> new_return_vars;
        new_return_vars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_return_vars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::WhileStmt>(
            new_cond, std::move(new_iter_args), new_body, std::move(new_return_vars), op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::IfStmtPtr& op) override
    {
        auto new_cond = VisitExpr(op->condition_);
        auto new_then = VisitStmt(op->thenBody_);
        std::optional<ir::StmtPtr> new_else;
        if (op->elseBody_.has_value()) {
            new_else = VisitStmt(*op->elseBody_);
        }
        std::vector<ir::VarPtr> new_rvs;
        new_rvs.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_rvs.push_back(RenameVar(rv));
        }
        return std::make_shared<const ir::IfStmt>(new_cond, new_then, new_else, std::move(new_rvs), op->span_);
    }
};

/// Replaces all ReturnStmt nodes in a body with an assignment to `target`.
///
/// The call site is always a single var (tuple unpacking is lowered to
/// `_tuple_tmp = helper(); a = _tuple_tmp[0]` by the parser), and a tuple
/// return is a single MakeTuple expression. So a return lowers to exactly
/// `target = value`; tuple resolution is left to backend codegen.
class ReturnReplacer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit ReturnReplacer(ir::VarPtr target) : target_(std::move(target)) {}

    ir::StmtPtr VisitStmt_(const ir::ReturnStmtPtr& op) override
    {
        if (!target_ || op->value_.empty()) {
            // Void call — replace return with empty SeqStmts
            return std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{}, op->span_);
        }
        CHECK(op->value_.size() == 1)
            << "helper return must carry a single value (tuple returns are a single MakeTuple)";
        return std::make_shared<const ir::AssignStmt>(target_, op->value_[0], op->span_);
    }

private:
    ir::VarPtr target_;
};

/// Build the inlined body for a helper call. Replaces ReturnStmt with an
/// assignment to `target` (null for void/EvalStmt calls).
ir::StmtPtr BuildInlinedBody(
    const ir::FunctionPtr& helper, const std::vector<ir::ExprPtr>& call_args, const std::string& prefix,
    const ir::VarPtr& target)
{
    BodyRenamer renamer(prefix, helper->params_, call_args);
    auto renamed_body = renamer.VisitStmt(helper->body_);

    ReturnReplacer replacer(target);
    return replacer.VisitStmt(renamed_body);
}

/// Check if a Call targets a helper function by name.
bool IsHelperCall(const ir::CallPtr& call, const std::map<std::string, ir::FunctionPtr>& helpers)
{
    if (!call || call->name_.empty())
        return false;
    return helpers.count(call->name_) > 0;
}

/// Walks the kernel body, replacing helper calls with inlined bodies.
///
/// Tuple handling is intentionally left to backend codegen: a tuple-returning
/// helper inlines to `target = MakeTuple(...)` (via ReturnReplacer), and a
/// later `target[i]` is resolved by CCE/PTO codegen's tuple_var_to_make_tuple_.
class CallInliner : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit CallInliner(const std::map<std::string, ir::FunctionPtr>& helpers) : helpers_(helpers) {}

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto call = ir::As<ir::Call>(op->value_);
        if (!IsHelperCall(call, helpers_)) {
            return IRMutator::VisitStmt_(op);
        }
        auto target_var = op->var_;
        std::string prefix = "_inl" + std::to_string(counter_++) + "_";
        const auto& helper = helpers_.at(call->name_);

        std::vector<ir::ExprPtr> args;
        args.reserve(call->args_.size());
        for (const auto& a : call->args_) {
            args.push_back(VisitExpr(a));
        }

        // Single target: tuple returns lower to `target = MakeTuple(...)`.
        auto inlined = BuildInlinedBody(helper, args, prefix, target_var);
        return VisitStmt(inlined); // recurse for nested helper calls
    }

    ir::StmtPtr VisitStmt_(const ir::EvalStmtPtr& op) override
    {
        auto call = ir::As<ir::Call>(op->expr_);
        if (!IsHelperCall(call, helpers_)) {
            return IRMutator::VisitStmt_(op);
        }
        std::string prefix = "_inl" + std::to_string(counter_++) + "_";
        const auto& helper = helpers_.at(call->name_);

        std::vector<ir::ExprPtr> args;
        args.reserve(call->args_.size());
        for (const auto& a : call->args_) {
            args.push_back(VisitExpr(a));
        }

        auto inlined = BuildInlinedBody(helper, args, prefix, /*target=*/nullptr);
        return VisitStmt(inlined);
    }

private:
    const std::map<std::string, ir::FunctionPtr>& helpers_;
    int counter_ = 0;
};

} // namespace

ir::ProgramPtr InlineHelperCalls(const ir::ProgramPtr& program)
{
    // Step 1: collect helper functions by name
    std::map<std::string, ir::FunctionPtr> helpers;
    ir::FunctionPtr kernel_func;
    std::string kernel_name;

    for (const auto& [name, func] : program->functions_) {
        if (func->funcType_ == ir::FunctionType::HELPER) {
            helpers[func->name_] = func;
        } else if (func->funcType_ != ir::FunctionType::ORCHESTRATION && !kernel_func) {
            kernel_func = func;
            kernel_name = name;
        }
    }

    if (helpers.empty() || !kernel_func) {
        return program; // nothing to inline
    }

    // Step 2: inline all helper calls in the kernel body
    CallInliner inliner(helpers);
    auto new_body = inliner.VisitStmt(kernel_func->body_);

    auto new_func = std::make_shared<const ir::Function>(
        kernel_func->name_, kernel_func->params_, kernel_func->returnTypes_, new_body, kernel_func->span_,
        kernel_func->funcType_);

    // Step 3: build a new program with only the kernel (no helpers)
    std::map<std::string, ir::FunctionPtr> new_funcs;
    for (const auto& [name, func] : program->functions_) {
        if (func->funcType_ == ir::FunctionType::HELPER) {
            continue; // drop helpers
        }
        if (name == kernel_name) {
            new_funcs[name] = new_func;
        } else {
            new_funcs[name] = func;
        }
    }

    return std::make_shared<const ir::Program>(std::move(new_funcs), program->name_, program->span_);
}

// ============================================================================
// Single-file MIX mode generation (skip ptoas)
// ============================================================================

void CCECodegen::PreScanKernel(const ir::FunctionPtr& kernel_func)
{
    var_read_names_.clear();
    CollectVarReadNames(kernel_func->body_, var_read_names_);

    cube_mutex_pipes_.clear();
    vec_mutex_pipes_.clear();
    CollectMutexPipeInfo(kernel_func->body_);
}

void CCECodegen::PrepareBodyGeneration()
{
    PreScanValidShapes();
    tuple_var_to_make_tuple_.clear();
}

std::string CCECodegen::GenerateSingle(const ir::ProgramPtr& program, const std::string& arch)
{
    CHECK(program != nullptr) << "Cannot generate code for null program";

    arch_ = arch;

    // Capture the tuple/struct field-name side table from the entry program (may be null).
    // Passes below rebuild the Program (dropping this annotation), but the TupleType pointers
    // they key on stay valid, so codegen reads names through this captured table.
    debug_info_ = program->GetDebugInfo();

    ir::ProgramPtr lowered = ir::pass::LowerBreakContinue()(program);
    ir::ProgramPtr ssa_program = ir::pass::ConvertToSSA()(lowered);
    // ir::ProgramPtr opt_program = ir::pass::ConstFoldAndSimplify()(ssa_program);
    ir::ProgramPtr inlined_program = InlineHelperCalls(ssa_program);
    ir::ProgramPtr opt_program = ir::pass::ConstFoldAndSimplify()(inlined_program);

    ir::FunctionPtr kernel_func;
    for (const auto& func_entry : opt_program->functions_) {
        const auto& func = func_entry.second;
        if (func->funcType_ == ir::FunctionType::ORCHESTRATION) {
            continue;
        }
        if (!kernel_func) {
            kernel_func = func;
        }
    }
    CHECK(kernel_func != nullptr) << "No kernel function found in program";

    emitter_.Clear();
    context_.Clear();
    tensor_to_pointer_.clear();

    PreScanKernel(kernel_func);

    // Detect cross-core sync (a5 uses hardware sync, not ffts)
    bool has_cross_sync = DetectCrossCoreSyncOps(kernel_func->body_);
    bool needs_ffts = has_cross_sync && (arch_ != "a5");

    emitter_.EmitLine(KERNEL_HEADER_SINGLE);

    tiling_headers_.clear();
    PreEmitStructTypes(kernel_func->body_);
    EmitTilingStructTypes(kernel_func);

    GenerateSinglePrologue(kernel_func, needs_ffts);
    PrepareBodyGeneration();
    GenerateBody(kernel_func);

    return emitter_.GetCode();
}

namespace {
/**
 * \brief Collect tile definitions sourced from block.make_tile calls.
 *
 * Scans AssignStmt nodes and records each block.make_tile allocation with:
 *   - The var that holds the result (or a synthetic var_N for tuple elements)
 *   - The TileType of the allocation
 *   - The SectionKind in which the make_tile appears (nullopt = shared/outside section)
 *
 * Inline tuple elements are named with a flat depth-first index (var_0, var_1, ...).
 */
class MakeTileDefCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    std::vector<TileDef> tile_defs_;

    void VisitStmt_(const ir::SectionStmtPtr& op) override
    {
        auto prev = current_section_;
        current_section_ = op->sectionKind_;
        ir::IRVisitor::VisitStmt_(op);
        current_section_ = prev;
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto target_var = op->var_;
        if (!op->value_)
            return;
        if (auto call = ir::As<ir::Call>(op->value_)) {
            // Direct block.make_tile assignment ->primary collection path.
            if (call->name_ == "block.make_tile") {
                if (auto tile_type = ir::As<ir::TileType>(call->GetType())) {
                    tile_defs_.push_back({target_var, tile_type, current_section_});
                }
            }
            return;
        }
    }

private:
    std::optional<ir::SectionKind> current_section_;
};

// Extract valid_shape constructor arguments from a TileType for CCE code generation.
//
// needs_ctor == true - template has -1 params (dynamic), Tile constructor must receive runtime values.
// needs_ctor == false - template has explicit static params, no constructor args needed.
//
// Mapping for each valid_shape element:
//   absent / ConstInt(-1) - template param = -1, ctor_arg = rows/cols  (needs_ctor = true)
//   ConstInt(N > 0)       - template param = N,  ctor_arg unused        (needs_ctor = false)
//   Var(name)             - template param = -1 (skipped by ConvertTileType), ctor_arg = var_name
struct ValidShapeInfo {
    std::string row_ctor_arg;
    std::string col_ctor_arg;
    bool needs_ctor = false; // true when template uses -1 and constructor args are required
};

inline ValidShapeInfo ExtractValidShapeInfo(
    const ir::TileTypePtr& tile_type, int64_t rows, int64_t cols,
    std::function<std::string(const ir::VarPtr&)> get_var_name)
{
    ValidShapeInfo info;
    if (!tile_type->tileView_.has_value()) {
        // No tile_view: absent valid_shape - template uses -1, ctor uses full shape.
        info.row_ctor_arg = std::to_string(rows);
        info.col_ctor_arg = std::to_string(cols);
        info.needs_ctor = true;
        return info;
    }
    const auto& tv = tile_type->tileView_.value();
    if (tv.validShape.empty()) {
        // valid_shape absent: template uses -1, ctor uses full shape.
        info.row_ctor_arg = std::to_string(rows);
        info.col_ctor_arg = std::to_string(cols);
        info.needs_ctor = true;
        return info;
    }
    // valid_shape provided: check whether any element is -1 or a runtime Var.
    auto resolve_dim = [&](const ir::ExprPtr& expr, int64_t fallback, std::string& out_arg) -> bool {
        if (auto var = ir::As<ir::Var>(expr)) {
            out_arg = get_var_name(var);
            return true; // runtime var - needs ctor
        }
        if (auto c = ir::As<ir::ConstInt>(expr)) {
            if (c->value_ == -1) {
                out_arg = std::to_string(fallback);
                return true; // -1 sentinel - template gets -1, ctor gets actual dim
            }
            out_arg = std::to_string(c->value_);
            return false; // explicit static value - no ctor needed
        }
        return false;
    };
    bool row_dynamic = false;
    bool col_dynamic = false;
    if (tv.validShape.size() >= 1)
        row_dynamic = resolve_dim(tv.validShape[0], rows, info.row_ctor_arg);
    if (tv.validShape.size() >= 2)
        col_dynamic = resolve_dim(tv.validShape[1], cols, info.col_ctor_arg);
    info.needs_ctor = row_dynamic || col_dynamic;
    return info;
}

// Build Tile constructor argument string.
// Only call when ValidShapeInfo::needs_ctor is true.
inline std::string BuildTileCtorArgs(const ValidShapeInfo& vs, int64_t rows, int64_t cols)
{
    std::string r = vs.row_ctor_arg.empty() ? std::to_string(rows) : vs.row_ctor_arg;
    std::string c = vs.col_ctor_arg.empty() ? std::to_string(cols) : vs.col_ctor_arg;
    return r + ", " + c;
}

std::vector<ir::VarPtr> CollectDynamicDimVars(const ir::FunctionPtr& func)
{
    std::vector<ir::VarPtr> dyn_dim_vars;
    std::set<std::string> seen_dyn_names;
    for (const auto& param : func->params_) {
        auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
        if (!tensor_type) {
            continue;
        }
        for (const auto& dim_expr : tensor_type->shape_) {
            auto dim_var = std::dynamic_pointer_cast<const ir::Var>(dim_expr);
            if (dim_var && seen_dyn_names.insert(dim_var->name_).second) {
                dyn_dim_vars.push_back(dim_var);
            }
        }
    }
    return dyn_dim_vars;
}

bool IsOnlyYieldStmts(const ir::StmtPtr& stmt)
{
    if (!stmt) {
        return false;
    }
    if (ir::As<ir::YieldStmt>(stmt)) {
        return true;
    }

    auto seq = ir::As<ir::SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) {
        return false;
    }
    for (const auto& st : seq->stmts_) {
        if (!ir::As<ir::YieldStmt>(st)) {
            return false;
        }
    }
    return true;
}

} // namespace

// ========================================================================
// Phase 6 helpers: GenerateSinglePrologue sub-functions
// ========================================================================

void CCECodegen::EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool has_cross_sync)
{
    // Collect dynamic dim variables from tensor shapes (first-occurrence order)
    std::vector<ir::VarPtr> dyn_dim_vars = CollectDynamicDimVars(func);

    // Build PTO-style function signature: __global__ AICORE void func_name(__gm__ type* p1, ...)
    std::ostringstream sig;
    sig << "__global__ AICORE void " << func->name_ << "(";
    bool first = true;
    for (const auto& param : func->params_) {
        if (!first)
            sig << ", ";
        first = false;
        // Each parameter is registered here while building the signature (the single place
        // that walks all params): the C++ var name, plus the raw pointer for tensor params.
        if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
            std::string element_type = tensor_type->dtype_.ToCTypeString();
            std::string param_name = context_.SanitizeName(param);
            // Tensor parameter is a raw pointer named "<name>_ptr"; the GlobalTensor wrapper
            // emitted in EmitSingleTensorDeclarations takes the tensor var name itself.
            sig << "__gm__ " << element_type << "* " << param_name << "_ptr";
            context_.RegisterVar(param, param_name);
            RegisterPointer(param_name, param_name + "_ptr");
        } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
            std::string cpp_type = scalar_type->dtype_.ToCTypeString();
            std::string param_name = context_.SanitizeName(param);
            sig << cpp_type << " " << param_name;
            context_.RegisterVar(param, param_name);
        } else if (auto ptr_type = std::dynamic_pointer_cast<const ir::PtrType>(param->GetType())) {
            std::string element_type = ptr_type->dtype_.ToCTypeString();
            std::string param_name = context_.SanitizeName(param);
            sig << "__gm__ " << element_type << "* " << param_name;
            context_.RegisterVar(param, param_name);
        } else if (std::dynamic_pointer_cast<const ir::TupleType>(param->GetType())) {
            // Tiling struct param: pass a raw GM byte pointer `<name>_ptr`; the local
            // struct `<name>` (filled by EmitTilingStructCopy) takes the bare name,
            // so `GetItemExpr` reads emit `<name>.field`.
            std::string param_name = context_.SanitizeName(param);
            sig << "__gm__ uint8_t* " << param_name << "_ptr";
            context_.RegisterVar(param, param_name);
        }
    }
    // Append dynamic dim variables as int64_t parameters
    for (const auto& dyn_var : dyn_dim_vars) {
        if (!first)
            sig << ", ";
        first = false;
        sig << "int32_t " << context_.SanitizeName(dyn_var);
    }
    if (has_cross_sync) {
        if (!first)
            sig << ", ";
        sig << "__gm__ int64_t* ffts_addr";
    }
    sig << ")";

    emitter_.EmitLine(sig.str());
    emitter_.EmitLine("{");
    emitter_.IncreaseIndent();

    // Register dynamic dim parameters directly (no _local_ copy needed)
    for (const auto& dyn_var : dyn_dim_vars) {
        std::string param_name = context_.SanitizeName(dyn_var);
        context_.RegisterVar(dyn_var, param_name);
    }

    // Emit set_ffts_base_addr if cross-core sync
    if (has_cross_sync) {
        emitter_.EmitLine("set_ffts_base_addr((unsigned long)ffts_addr);");
    }
    emitter_.EmitLine("");
}

void CCECodegen::EmitSingleTensorDeclarations(const ir::FunctionPtr& func)
{
    // Prescan all accessed tensor vars into tensor_defs_ (params + ptr.make_tensor views).
    // Parameter pointers are already registered while building the signature (see
    // EmitSingleFunctionSignature). Here we emit GlobalTensor declarations for the tensor
    // parameters only; ptr.make_tensor views are emitted in place at their op (see
    // MakeBlockMakeTensorCodegenCCE), where the source pointer's C++ code is already in scope.
    CollectTensorDefs(func);

    std::vector<TensorDef> param_defs;
    param_defs.reserve(tensor_defs_.size());
    for (const auto& [name, def] : tensor_defs_) {
        (void)name;
        auto tt = ir::As<ir::TensorType>(def.var->GetType());
        const bool is_view = tt && tt->tensor_view_.has_value() && tt->tensor_view_->ptr.has_value();
        if (!is_view) {
            param_defs.push_back(def);
        }
    }

    EmitSectionAwareTensors(param_defs);
}

void CCECodegen::EmitSectionAwareTensors(const std::vector<TensorDef>& defs)
{
    std::vector<const TensorDef*> shared;
    std::vector<const TensorDef*> cube;
    std::vector<const TensorDef*> vec;
    for (const auto& def : defs) {
        if (!def.def_section.has_value()) {
            shared.push_back(&def);
        } else if (*def.def_section == ir::SectionKind::Cube) {
            cube.push_back(&def);
        } else {
            vec.push_back(&def);
        }
    }

    if (!shared.empty()) {
        for (const auto* def : shared) {
            GenerateGlobalTensorTypeDeclaration(*def);
            emitter_.EmitLine("");
        }
    }

    if (!cube.empty()) {
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
        for (const auto* def : cube) {
            GenerateGlobalTensorTypeDeclaration(*def);
            emitter_.EmitLine("");
        }
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }

    if (!vec.empty()) {
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
        for (const auto* def : vec) {
            GenerateGlobalTensorTypeDeclaration(*def);
            emitter_.EmitLine("");
        }
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }
}

void CCECodegen::EmitSingleTileDeclarations(const ir::FunctionPtr& func)
{
    if (!func->body_)
        return;

    // Collect tile definitions from block.make_tile sources only.
    MakeTileDefCollector def_collector;
    def_collector.VisitStmt(func->body_);

    auto has_independent_runtime_metadata = [](const ir::TileTypePtr& tile_type) {
        return tile_type != nullptr && tile_type->tileView_.has_value() && !tile_type->tileView_->validShape.empty();
    };

    std::vector<TileDef> kept_defs;
    std::vector<std::pair<ir::VarPtr, ir::VarPtr>> deduped_aliases;
    std::map<std::string, ir::VarPtr> kept_tile_addr_vars;

    for (const auto& td : def_collector.tile_defs_) {
        const auto& tile_type = td.tile_type;
        if (tile_type->memref_.has_value()) {
            // Same-address tiles with explicit valid_shape metadata must stay as
            // distinct C++ objects; a later SetValidShape() on one tile would otherwise
            // narrow every deduped alias sharing the same backing storage.
            if (has_independent_runtime_metadata(tile_type)) {
                kept_defs.push_back(td);
                continue;
            }
            int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_);
            auto space = (*tile_type->memref_)->memorySpace_;
            std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
            std::string type_key = type_converter_.ConvertTileType(
                tile_type, shape_dims.size() >= 1 ? shape_dims[0] : 1, shape_dims.size() >= 2 ? shape_dims[1] : 1);
            std::string dedup_key =
                std::to_string(static_cast<int>(space)) + ":" + std::to_string(addr) + ":" + type_key;
            auto it = kept_tile_addr_vars.find(dedup_key);
            if (it != kept_tile_addr_vars.end()) {
                deduped_aliases.emplace_back(td.var, it->second);
                continue;
            }
            kept_tile_addr_vars[dedup_key] = td.var;
        }
        kept_defs.push_back(td);
    }

    if (!kept_defs.empty() || !deduped_aliases.empty()) {
        EmitSectionAwareTiles(kept_defs, deduped_aliases);
    }
}

void CCECodegen::EmitDedupedTileAliases(
    const std::vector<TileDef>& tile_defs, const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases,
    std::optional<ir::SectionKind> section)
{
    for (const auto& [dup_var, kept_var] : deduped_aliases) {
        std::optional<ir::SectionKind> kept_section;
        for (const auto& td : tile_defs) {
            if (td.var->name_ == kept_var->name_) {
                kept_section = td.def_section;
                break;
            }
        }
        if (kept_section == section) {
            std::string san_dup = context_.SanitizeName(dup_var);
            std::string san_kept = context_.SanitizeName(kept_var);
            emitter_.EmitLine("auto& " + san_dup + " = " + san_kept + ";");
            emitted_tile_aliases_.insert(san_dup);
        }
    }
}

void CCECodegen::EmitSectionAwareTiles(
    const std::vector<TileDef>& tile_defs, const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases)
{
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> cube_tiles;
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> vec_tiles;
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> shared_tiles;

    for (const auto& td : tile_defs) {
        if (!td.def_section.has_value()) {
            shared_tiles.emplace_back(td.var, td.tile_type);
        } else if (*td.def_section == ir::SectionKind::Cube) {
            cube_tiles.emplace_back(td.var, td.tile_type);
        } else {
            vec_tiles.emplace_back(td.var, td.tile_type);
        }
    }

    if (!shared_tiles.empty()) {
        for (const auto& [var, tile_type] : shared_tiles) {
            GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
        }
        EmitDedupedTileAliases(tile_defs, deduped_aliases, std::nullopt);
        emitter_.EmitLine("");
    }

    if (!cube_tiles.empty()) {
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
        for (const auto& [var, tile_type] : cube_tiles) {
            GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
        }
        EmitDedupedTileAliases(tile_defs, deduped_aliases, ir::SectionKind::Cube);
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }

    if (!vec_tiles.empty()) {
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
        for (const auto& [var, tile_type] : vec_tiles) {
            GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
        }
        EmitDedupedTileAliases(tile_defs, deduped_aliases, ir::SectionKind::Vector);
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }
}

void CCECodegen::GenerateSinglePrologue(const ir::FunctionPtr& func, bool has_cross_sync)
{
    EmitSingleFunctionSignature(func, has_cross_sync);
    // Tiling struct variable definition + GM copy must come first: the very next decls
    // (tensors/tiles with dynamic shapes) and the body read `tiling.field`.
    EmitTilingStructCopy(func);
    EmitSingleTensorDeclarations(func);
    EmitSingleTileDeclarations(func);
}

void CCECodegen::VisitStmt_(const ir::SectionStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null SectionStmt";

    // VF section: emit a __VEC_SCOPE__ { ... } block with all tile->ptr base declarations
    // hoisted to just before the scope (section_hoist), so they never sit after a mem_bar.
    if (op->sectionKind_ == ir::SectionKind::VF) {
        // After InlineHelperCalls, a VF section is always nested inside a Vector section.
        INTERNAL_CHECK(current_section_kind_.has_value() && *current_section_kind_ == ir::SectionKind::Vector)
            << "VF section must be nested inside a Vector section";
        // VF sections do not nest, so the hoist buffer must be empty on entry.
        INTERNAL_CHECK(section_hoisted_decls_.empty());

        int saved_indent = emitter_.GetIndentLevel();
        std::string pre_scope = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);

        auto prev_kind = current_section_kind_;
        current_section_kind_ = ir::SectionKind::VF;
        emitter_.EmitLine("__VEC_SCOPE__ {");
        emitter_.IncreaseIndent();
        if (op->body_) {
            VisitStmt(op->body_); // GetItemExpr / GetUBufPtr lazily push base-ptr decls
        }
        emitter_.DecreaseIndent();
        emitter_.EmitLine("}");
        current_section_kind_ = prev_kind;

        std::string scope_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);
        emitter_.EmitRaw(pre_scope);
        // vf_tile_ptr_N names are globally unique, so the hoisted decls can sit before __VEC_SCOPE__
        // at this scope level without colliding across sections.
        for (const auto& decl : section_hoisted_decls_) {
            emitter_.EmitLine(decl); // flush before __VEC_SCOPE__ (ahead of every mem_bar inside)
        }
        section_hoisted_decls_.clear();
        emitter_.EmitRaw(scope_code);

        vf_tile_ptrs_.clear();
        return;
    }

    // Emit #if defined(__DAV_CUBE__) / #if defined(__DAV_VEC__)
    if (op->sectionKind_ == ir::SectionKind::Cube) {
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    } else {
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
    }

    // Emit vector mask initialization at the start of Vec section (a3 only)
    if (op->sectionKind_ == ir::SectionKind::Vector && arch_ == "a3") {
        emitter_.EmitLine("set_mask_norm();");
        emitter_.EmitLine("set_vector_mask(-1, -1);");
        emitter_.EmitLine("");
    }

    // Visit the body
    if (op->body_) {
        auto prev_section = current_section_kind_;
        current_section_kind_ = op->sectionKind_;
        VisitStmt(op->body_);
        current_section_kind_ = prev_section;
    }

    emitter_.EmitLine("#endif");
}

bool CCECodegen::DetectCrossCoreSyncOps(const ir::StmtPtr& stmt)
{
    if (!stmt)
        return false;

    class CrossCoreSyncDetector : public ir::IRVisitor {
        using ir::IRVisitor::VisitExpr_;
        using ir::IRVisitor::VisitStmt_;

    public:
        bool found = false;
        void VisitExpr_(const ir::CallPtr& op) override
        {
            const std::string& name = op->name_;
            if (name == "system.set_cross_core" || name == "system.wait_cross_core" ||
                name == "system.set_cross_core_dyn" || name == "system.wait_cross_core_dyn") {
                found = true;
            }
            ir::IRVisitor::VisitExpr_(op);
        }
    };

    CrossCoreSyncDetector detector;
    detector.VisitStmt(stmt);
    return detector.found;
}

void CCECodegen::GenerateBody(const ir::FunctionPtr& func)
{
    if (func->body_) {
        VisitStmt(func->body_);
    }

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");
}


void CCECodegen::VisitStmt_(const ir::AssignStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";
    INTERNAL_CHECK(op->var_ != nullptr) << "Internal error: AssignStmt has null var";
    INTERNAL_CHECK(op->value_ != nullptr) << "Internal error: AssignStmt has null value";

    auto target_var = op->var_;
    std::string var_name = context_.SanitizeName(target_var);
    context_.RegisterVar(target_var, var_name);

    // Named-struct alias: `ctx_alias = arr[idx]` on a struct-tuple-array RHS.
    // Emit a real C++ reference `auto& ctx_alias = arr_code[idx_code];` instead of
    // string-substituting the var name. The reference gives bisheng a hoist barrier
    // for subsequent field accesses inside VF (vector function) scope, where deeply
    // nested `_dyn_arr[arr[idx].field]` expressions trigger "must be hoisted".
    if (auto gi = ir::As<ir::GetItemExpr>(op->value_)) {
        auto var_tt = ir::As<ir::TupleType>(target_var->GetType());
        auto outer_tt = ir::As<ir::TupleType>(gi->value_->GetType());
        ir::TupleTypePtr inner_tt;
        if (outer_tt && !outer_tt->types_.empty()) {
            inner_tt = ir::As<ir::TupleType>(outer_tt->types_[0]);
        }
        auto is_named_tuple = [this](const ir::TupleTypePtr& tt) {
            return tt && debug_info_ != nullptr && debug_info_->GetTupleFields(tt.get()) != nullptr;
        };
        bool is_struct_tuple_alias = is_named_tuple(var_tt) && is_named_tuple(inner_tt);
        if (is_struct_tuple_alias) {
            std::string rhs_code = GetExprAsCode(op->value_);
            if (!rhs_code.empty()) {
                Emit("auto& " + var_name + " = " + rhs_code + ";");
                current_tuple_ = nullptr;
                current_expr_value_ = "";
                current_target_var_ = "";
                return;
            }
        }
    }

    current_target_var_ = var_name;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    VisitExpr(op->value_);

    if (!current_expr_value_.empty()) {
        auto var_type = target_var->GetType();
        bool is_tile = ir::As<ir::TileType>(var_type) != nullptr;
        bool is_tuple = ir::As<ir::TupleType>(var_type) != nullptr;
        bool rhs_is_var = ir::As<ir::Var>(op->value_) != nullptr;
        // Tile/tuple aliases and simple var-to-var copies: register as alias (no code emitted).
        // Scalar complex expressions: emit "auto x = expr;" so codegen hoists the expression
        // into a C++ local variable rather than inlining it at every use site.
        if (is_tile || is_tuple || rhs_is_var || tile_addresses_.count(current_expr_value_)) {
            context_.RegisterVar(target_var, current_expr_value_);
        } else {
            // Scalar non-trivial expression: emit a local variable declaration so the
            // expression (e.g. min(), %, *) is computed once and not re-expanded at
            // every use site (avoids VFLoop Cond containing function calls).
            emitter_.EmitLine("auto " + var_name + " = " + current_expr_value_ + ";");
            context_.RegisterVar(target_var, var_name);
        }
    }
    if (current_tuple_) {
        tuple_var_to_make_tuple_.emplace(var_name, current_tuple_); // first-write wins

        // Emit the dynamic-index array at the MakeTuple assignment
        // node (the stable identity shared with every use site). Gate on homogeneity: an
        // array is only valid when all elements map to the same C++ type. All element
        // variables (struct instances / tiles / scalars) are already in scope here, and this
        // assignment dominates the use sites, so the array has the correct scope. The array
        // name is the tuple var name (1:1 with the MakeTuple); SSA guarantees uniqueness.
        auto tt = ir::As<ir::TupleType>(target_var->GetType());
        if (tt && IsHomogeneousTuple(tt)) {
            auto elem_names = CollectTupleElemNames(op->var_);
            if (!elem_names.empty()) {
                std::string arr_decl = BuildDynamicTupleArrayDecl(
                    tt->types_[0], elem_names, var_name, /*allow_struct_tuple=*/true);
                if (!arr_decl.empty()) {
                    emitter_.EmitLine(arr_decl);
                }
            }
        }
    }
    if (tile_addresses_.count(current_expr_value_)) {
        tile_addresses_[var_name] = tile_addresses_[current_expr_value_];
    }

    current_tuple_ = nullptr;
    current_expr_value_ = "";
    current_target_var_ = "";
}

void CCECodegen::VisitStmt_(const ir::EvalStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
    INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";

    // EvalStmt: evaluate expression for side effects (e.g., sync operations)
    // Sync ops (set_flag, wait_flag, pipe_barrier) are registered with f_codegen_cce
    // and will be invoked via VisitExpr_(Call). Cross-core sync redundancy is
    // checked inside the per-op codegen handler (Make{Set,Wait}CrossCoreCodegenCCE).
    VisitExpr(op->expr_);
}

void CCECodegen::VisitStmt_(const ir::ReturnStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ReturnStmt";
    // Helper returns are rewritten to assignments during inlining; a ReturnStmt
    // reaching codegen is a kernel-level (void) return, which emits nothing.
}

void CCECodegen::VisitStmt_(const ir::YieldStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

    if (op->value_.empty()) {
        return; // No values to yield
    }

    // Visit each yielded expression and collect values
    std::vector<std::string> yielded_values;
    for (const auto& expr : op->value_) {
        VisitExpr(expr);
        yielded_values.push_back(current_expr_value_);
    }

    // Store in temporary buffer for ForStmt to pick up
    yield_buffer_ = yielded_values;
    current_expr_value_ = "";
}

std::optional<std::string> CCECodegen::ResolveYieldName(const ir::ExprPtr& value) const
{
    if (auto var = ir::As<ir::Var>(value)) {
        return context_.SanitizeName(var);
    }
    if (auto cint = ir::As<ir::ConstInt>(value)) {
        return std::to_string(cint->value_);
    }

    auto gi = ir::As<ir::GetItemExpr>(value);
    if (!gi || !ir::As<ir::TupleType>(gi->value_->GetType())) {
        return std::nullopt;
    }

    auto const_idx = ir::As<ir::ConstInt>(gi->slice_);
    if (!const_idx) {
        return std::nullopt;
    }
    int idx_i = static_cast<int>(const_idx->value_);
    if (auto tuple_var = ir::As<ir::Var>(gi->value_)) {
        return context_.SanitizeName(tuple_var) + "_" + std::to_string(idx_i);
    }

    auto make_tuple = ir::As<ir::MakeTuple>(gi->value_);
    size_t idx = static_cast<size_t>(idx_i);
    if (!make_tuple || idx >= make_tuple->elements_.size()) {
        return std::nullopt;
    }
    return ResolveYieldName(make_tuple->elements_[idx]);
}

std::vector<std::string> CCECodegen::ExtractYieldNames(const ir::StmtPtr& body) const
{
    std::vector<std::string> yields;
    ir::YieldStmtPtr yield_stmt;
    if (auto y = ir::As<ir::YieldStmt>(body)) {
        yield_stmt = y;
    } else if (auto seq = ir::As<ir::SeqStmts>(body)) {
        if (!seq->stmts_.empty())
            yield_stmt = ir::As<ir::YieldStmt>(seq->stmts_.back());
    }
    if (!yield_stmt)
        return yields;
    for (const auto& val : yield_stmt->value_) {
        auto name = ResolveYieldName(val);
        if (!name.has_value()) {
            return {};
        }
        yields.push_back(name.value());
    }
    return yields;
}

void CCECodegen::EmitYieldAssignments(
    const std::vector<ir::VarPtr>& return_vars, const std::vector<std::string>& target_names)
{
    if (return_vars.empty() || yield_buffer_.empty())
        return;
    for (size_t i = 0; i < return_vars.size(); ++i) {
        const auto& return_var = return_vars[i];
        std::string return_var_name = target_names[i];
        std::string yielded_value = yield_buffer_[i];

        auto return_type = return_var->GetType();
        std::string resolved_yield = yielded_value;
        if ((ir::As<ir::TileType>(return_type) || ir::As<ir::TupleType>(return_type)) &&
            tile_addresses_.count(yielded_value)) {
            emitter_.EmitLine("TASSIGN(" + return_var_name + ", " + tile_addresses_[yielded_value] + ");");
            tile_addresses_[return_var_name] = tile_addresses_[yielded_value];
        } else if (return_var_name != resolved_yield) {
            emitter_.EmitLine(return_var_name + " = " + resolved_yield + ";");
        }

        if (std::dynamic_pointer_cast<const ir::TensorType>(return_type)) {
            std::string yielded_ptr = GetPointer(resolved_yield);
            RegisterPointer(return_var_name, yielded_ptr);
        }
    }
    yield_buffer_.clear();
}

// ========================================================================
// Dynamic tuple array decl: visitExpr-driven resolution + section-aware splice
// ========================================================================

std::vector<std::string> CCECodegen::CollectTupleElemNames(const ir::ExprPtr& tuple_value)
{
    // Step 1: resolve the tuple expression to its underlying MakeTuple via visitExpr.
    current_tuple_ = nullptr;
    current_expr_value_ = "";
    VisitExpr(tuple_value);
    if (!current_tuple_)
        return {};
    auto mt = current_tuple_; // snapshot, element visits below overwrite current_tuple_

    // Step 2: name each element by visiting it. For dynamic-GetItem the elements are
    // tile/scalar Var or const literals, so each visit yields a non-empty current_expr_value_.
    std::vector<std::string> names;
    names.reserve(mt->elements_.size());
    for (const auto& elem : mt->elements_) {
        current_tuple_ = nullptr;
        current_expr_value_ = "";
        VisitExpr(elem);
        if (current_expr_value_.empty())
            return {};
        names.push_back(current_expr_value_);
    }
    return names;
}

std::string CCECodegen::BuildDynamicTupleArrayDecl(
    const ir::TypePtr& elem_type, const std::vector<std::string>& elem_names, const std::string& arr_name,
    bool allow_struct_tuple) const
{
    if (elem_names.empty()) {
        return "";
    }

    std::ostringstream init;
    for (size_t i = 0; i < elem_names.size(); ++i) {
        if (i > 0)
            init << ", ";
        init << elem_names[i];
    }

    std::string elem_cpp_type;
    if (auto tile_type = ir::As<ir::TileType>(elem_type)) {
        auto shape_dims = ExtractShapeDimensions(tile_type->shape_);
        int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
        int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
        elem_cpp_type = type_converter_.ConvertTileType(tile_type, rows, cols);
    } else if (auto scalar_type = ir::As<ir::ScalarType>(elem_type)) {
        elem_cpp_type = "const " + scalar_type->dtype_.ToCTypeString();
    } else if (allow_struct_tuple) {
        auto inner_tt = ir::As<ir::TupleType>(elem_type);
        if (inner_tt && debug_info_ != nullptr && debug_info_->GetTupleFields(inner_tt.get()) != nullptr) {
            auto type_it = struct_var_to_type_name_.find(elem_names[0]);
            if (type_it != struct_var_to_type_name_.end()) {
                elem_cpp_type = type_it->second;
            }
        }
    }

    if (elem_cpp_type.empty()) {
        return "";
    }
    return elem_cpp_type + " " + arr_name + "[] = {" + init.str() + "};";
}

bool CCECodegen::IsHomogeneousTuple(const ir::TupleTypePtr& tt) const
{
    if (!tt || tt->types_.empty())
        return false;
    for (size_t i = 1; i < tt->types_.size(); ++i) {
        if (!ir::structural_equal(tt->types_[i], tt->types_[0]))
            return false;
    }
    return true;
}

void CCECodegen::EmitFullPhiIf(const ir::IfStmtPtr& op)
{
    // Declare and register return variables BEFORE the if statement
    for (const auto& return_var : op->returnVars_) {
        std::string return_var_name = context_.SanitizeName(return_var);
        context_.RegisterVar(return_var, return_var_name);

        if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(return_var->GetType())) {
            std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
            int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
            int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
            auto vs =
                ExtractValidShapeInfo(tile_type, rows, cols, [this](const ir::VarPtr& v) { return GetVarName(v); });
            std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);
            std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
            std::string type_alias_name = return_var_name + "Type";
            std::string tile_type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
            if (loop_depth_ > 0) {
                loop_hoisted_decls_.push_back("using " + type_alias_name + " = " + tile_type_str + ";");
                loop_hoisted_decls_.push_back(type_alias_name + " " + return_var_name + ctor_suffix + ";");
            } else {
                emitter_.EmitLine("using " + type_alias_name + " = " + tile_type_str + ";");
                emitter_.EmitLine(type_alias_name + " " + return_var_name + ctor_suffix + ";");
            }
        } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(return_var->GetType())) {
            std::string cpp_type = scalar_type->dtype_.ToCTypeString();
            emitter_.EmitLine(cpp_type + " " + return_var_name + ";");
        } else {
            throw ir::RuntimeError("Unsupported return_var type in IfStmt");
        }
    }

    VisitExpr(op->condition_);
    std::string condition = current_expr_value_;
    current_expr_value_ = "";

    emitter_.EmitLine("if (" + condition + ") {");
    emitter_.IncreaseIndent();
    VisitStmt(op->thenBody_);
    {
        std::vector<std::string> phi_names;
        for (const auto& rv : op->returnVars_)
            phi_names.push_back(context_.SanitizeName(rv));
        EmitYieldAssignments(op->returnVars_, phi_names);
    }
    emitter_.DecreaseIndent();

    if (op->elseBody_.has_value()) {
        emitter_.EmitLine("} else {");
        emitter_.IncreaseIndent();
        VisitStmt(*op->elseBody_);
        {
            std::vector<std::string> phi_names;
            for (const auto& rv : op->returnVars_)
                phi_names.push_back(context_.SanitizeName(rv));
            EmitYieldAssignments(op->returnVars_, phi_names);
        }
        emitter_.DecreaseIndent();
    }
    emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::IfStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
    INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
    INTERNAL_CHECK(op->thenBody_ != nullptr) << "Internal error: IfStmt has null then_body";

    // Drop phi return_vars with no downstream consumer. The SSA pass
    // conservatively inserts phi nodes whenever a variable is re-assigned
    // across control flow, but if no one reads the phi output, emitting a
    // declaration + per-branch yield-assignment is pure dead code.
    ir::IfStmtPtr effective_op = op;
    if (!op->returnVars_.empty()) {
        bool any_used = false;
        for (const auto& rv : op->returnVars_) {
            if (rv && var_read_names_.count(rv->name_)) {
                any_used = true;
                break;
            }
        }
        if (!any_used) {
            // An else-body consisting only of YieldStmts also becomes dead once
            // return_vars are dropped ->the yields have no consumer.
            std::optional<ir::StmtPtr> new_else = op->elseBody_;
            if (new_else.has_value() && IsOnlyYieldStmts(*new_else)) {
                new_else = std::nullopt;
            }
            effective_op = std::make_shared<ir::IfStmt>(
                op->condition_, op->thenBody_, new_else, std::vector<ir::VarPtr>{}, op->span_);
        }
    }

    // If-level hoisting: buffer output so array decls can be hoisted before the if
    bool is_outermost_if = (loop_depth_ == 0 && if_depth_ == 0);
    if_depth_++;

    std::string pre_if_code;
    int saved_if_indent = emitter_.GetIndentLevel();
    if (is_outermost_if) {
        pre_if_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_if_indent);
    }

    EmitFullPhiIf(effective_op);

    if_depth_--;

    // Insert hoisted declarations before the if statement
    if (is_outermost_if) {
        std::string if_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_if_indent);
        emitter_.EmitRaw(pre_if_code);

        if (!loop_hoisted_decls_.empty()) {
            for (const auto& decl : loop_hoisted_decls_) {
                emitter_.EmitLine(decl);
            }
            emitter_.EmitLine("");
            loop_hoisted_decls_.clear();
        }

        emitter_.EmitRaw(if_code);
    }
}

// ========================================================================
// Phase 5 helpers: ForStmt iter-arg registration and yield assignments
// ========================================================================

std::vector<std::string> CCECodegen::RegisterForIterArgs(const ir::ForStmtPtr& op)
{
    std::vector<std::string> iter_arg_names;
    if (op->iterArgs_.empty())
        return iter_arg_names;

    bool any_emitted = false;
    for (auto& iter_arg : op->iterArgs_) {
        const auto& var = iter_arg->iterVar_;
        std::string iter_arg_name = context_.SanitizeName(var);

        VisitExpr(iter_arg->initValue_);
        std::string init_value = current_expr_value_;
        current_expr_value_ = "";

        auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
        if (init_var && std::dynamic_pointer_cast<const ir::TensorType>(init_var->GetType())) {
            std::string init_var_name = context_.GetVarName(init_var);
            std::string init_ptr = GetPointer(init_var_name);
            RegisterPointer(iter_arg_name, init_ptr);
        }

        std::string resolved_init = init_value;
        bool is_simple_var_copy = (init_var != nullptr) &&
                                  !std::dynamic_pointer_cast<const ir::TensorType>(init_var->GetType()) &&
                                  !std::dynamic_pointer_cast<const ir::TileType>(init_var->GetType()) &&
                                  !std::dynamic_pointer_cast<const ir::TupleType>(init_var->GetType());
        bool can_reuse_writable_slot = is_simple_var_copy && IsWritableLValueExpr(resolved_init);
        if (can_reuse_writable_slot && context_.IsAutoRegistered(resolved_init)) {
            can_reuse_writable_slot = false;
        }

        if (can_reuse_writable_slot) {
            context_.RegisterVar(var, resolved_init);
            iter_arg_names.push_back(resolved_init);
        } else {
            context_.RegisterVar(var, iter_arg_name);
            iter_arg_names.push_back(iter_arg_name);
            if (!any_emitted) {
                any_emitted = true;
            }
            std::string safe_init = init_value;
            if (context_.IsAutoRegistered(init_value)) {
                safe_init = "0";
            }
            emitter_.EmitLine("auto " + iter_arg_name + " = " + safe_init + ";");
        }
    }
    if (any_emitted) {
        emitter_.EmitLine("");
    }
    return iter_arg_names;
}

void CCECodegen::EmitForYieldAssignments(const std::vector<std::string>& iter_arg_names)
{
    if (yield_buffer_.empty())
        return;

    CHECK(yield_buffer_.size() == iter_arg_names.size())
        << "Yielded " << yield_buffer_.size() << " values but expected " << iter_arg_names.size();

    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
        std::string lhs = iter_arg_names[i];
        std::string rhs = yield_buffer_[i];
        if (lhs == rhs) {
            continue; // Self-assignment after alias resolution - skip
        }
        // For tiles: use TASSIGN instead of operator= to transfer hardware address
        std::string tile_source = tile_addresses_.count(yield_buffer_[i]) ? yield_buffer_[i] : rhs;
        if (tile_addresses_.count(tile_source)) {
            emitter_.EmitLine("TASSIGN(" + lhs + ", " + tile_addresses_[tile_source] + ");");
            tile_addresses_[lhs] = tile_addresses_[tile_source];
        } else {
            emitter_.EmitLine(lhs + " = " + rhs + ";");
        }
    }
    yield_buffer_.clear();
}

void CCECodegen::RegisterForReturnVars(const ir::ForStmtPtr& op, const std::vector<std::string>& iter_arg_names)
{
    if (op->returnVars_.empty()) {
        return;
    }
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        const auto& return_var = op->returnVars_[i];
        if (i < iter_arg_names.size()) {
            context_.RegisterVar(return_var, iter_arg_names[i]);
        } else {
            throw ir::RuntimeError("ForStmt return_var has no corresponding iter_arg");
        }
    }
}

void CCECodegen::PropagateTupleIterArgs(const ir::ForStmtPtr& op)
{
    for (size_t i = 0; i < op->iterArgs_.size(); ++i) {
        const auto& ia = op->iterArgs_[i];
        if (!ia || !ir::As<ir::TupleType>(ia->iterVar_->GetType()) || !ia->initValue_)
            continue;
        std::string saved_expr = current_expr_value_;
        ir::MakeTuplePtr saved_tuple = current_tuple_;
        current_tuple_ = nullptr;
        current_expr_value_ = "";
        VisitExpr(ia->initValue_);
        if (current_tuple_) {
            std::string ia_name = context_.SanitizeName(ia->iterVar_);
            tuple_var_to_make_tuple_.emplace(ia_name, current_tuple_);
            if (i < op->returnVars_.size() && op->returnVars_[i] &&
                ir::As<ir::TupleType>(op->returnVars_[i]->GetType())) {
                std::string rv_name = context_.SanitizeName(op->returnVars_[i]);
                tuple_var_to_make_tuple_.emplace(rv_name, current_tuple_);
            }
        }
        current_expr_value_ = saved_expr;
        current_tuple_ = saved_tuple;
    }
}

void CCECodegen::VisitStmt_(const ir::ForStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
    INTERNAL_CHECK(op->loopVar_ != nullptr) << "Internal error: ForStmt has null loop_var";
    INTERNAL_CHECK(op->start_ != nullptr) << "Internal error: ForStmt has null start";
    INTERNAL_CHECK(op->stop_ != nullptr) << "Internal error: ForStmt has null stop";
    INTERNAL_CHECK(op->step_ != nullptr) << "Internal error: ForStmt has null step";
    INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

    // Check consistency: iter_args and return_vars must have same size
    CHECK(op->iterArgs_.size() == op->returnVars_.size())
        << "ForStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
        << op->returnVars_.size() << ")";

    // --- Early single-iteration detection ---
    auto start_ci = ir::As<ir::ConstInt>(op->start_);
    auto stop_ci = ir::As<ir::ConstInt>(op->stop_);
    auto step_ci = ir::As<ir::ConstInt>(op->step_);
    bool is_single_iter =
        (start_ci && stop_ci && step_ci && start_ci->value_ == 0 && stop_ci->value_ == 1 && step_ci->value_ == 1);

    // Register loop variable
    std::string loop_var_name;
    if (is_single_iter) {
        // Single-iteration: register loop var as constant "0"
        loop_var_name = "0";
        context_.RegisterVar(op->loopVar_, "0");
    } else {
        loop_var_name = context_.SanitizeName(op->loopVar_);
        context_.RegisterVar(op->loopVar_, loop_var_name);
    }

    std::vector<std::string> iter_arg_names = RegisterForIterArgs(op);
    PropagateTupleIterArgs(op);

    // Evaluate loop range
    VisitExpr(op->start_);
    std::string start = current_expr_value_;
    current_expr_value_ = "";

    VisitExpr(op->stop_);
    std::string stop = current_expr_value_;
    current_expr_value_ = "";

    VisitExpr(op->step_);
    std::string step = current_expr_value_;
    current_expr_value_ = "";

    // --- Single-iteration loop unrolling ---
    if (is_single_iter) {
        // Visit loop body directly (no loop wrapper)
        yield_buffer_.clear();
        VisitStmt(op->body_);
        yield_buffer_.clear();

        // Register return variables with same names as iter_args
        RegisterForReturnVars(op, iter_arg_names);
        return;
    }

    // --- Emit for-loop with hoisting ---
    EmitForLoopWithHoisting(op, loop_var_name, iter_arg_names, start, stop, step);
}

void CCECodegen::EmitForLoopWithHoisting(
    const ir::ForStmtPtr& op, const std::string& loop_var_name, const std::vector<std::string>& iter_arg_names,
    const std::string& start, const std::string& stop, const std::string& step)
{
    bool is_outermost_loop = (loop_depth_ == 0);
    // VF base-ptr / POST_UPDATE decls now go to section_hoisted_decls_ (hoisted to __VEC_SCOPE__),
    // so loop-level hoisting is only needed for the outermost kernel loop.
    bool should_hoist = is_outermost_loop;
    loop_depth_++;
    size_t hoist_start_idx = loop_hoisted_decls_.size();

    std::string pre_for_code;
    int saved_indent = emitter_.GetIndentLevel();
    if (should_hoist) {
        pre_for_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);
    }

    // In __VEC_SCOPE__, bisheng requires uint16_t loop variables AND the bound
    // expression must also resolve to uint16_t (otherwise -Wcce-compat -Werror
    // fails the build). Force a static_cast when we're inside a VF section.
    std::string loop_type = IsInVFSection() ? "uint16_t" : "uint64_t";
    std::string stop_expr = IsInVFSection() ? ("(uint16_t)(" + stop + ")") : stop;
    emitter_.EmitLine("for (" + loop_type + " " + loop_var_name + " = " + start + "; " + loop_var_name + " < " + stop_expr +
                        "; " + loop_var_name + " += " + step + ") {");
    emitter_.IncreaseIndent();

    yield_buffer_.clear();
    VisitStmt(op->body_);

    if (!op->iterArgs_.empty()) {
        EmitForYieldAssignments(iter_arg_names);
    }

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");

    loop_depth_--;

    // Insert hoisted declarations before the for-loop
    if (should_hoist) {
        std::string for_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);
        emitter_.EmitRaw(pre_for_code);

        if (loop_hoisted_decls_.size() > hoist_start_idx) {
            for (size_t i = hoist_start_idx; i < loop_hoisted_decls_.size(); ++i) {
                emitter_.EmitLine(loop_hoisted_decls_[i]);
            }
            emitter_.EmitLine("");
            loop_hoisted_decls_.resize(hoist_start_idx);
        }

        emitter_.EmitRaw(for_code);
    }

    // Register return variables with same names as iter_args
    RegisterForReturnVars(op, iter_arg_names);
}

void CCECodegen::VisitStmt_(const ir::WhileStmtPtr& op)
{
    (void)op;
    throw ir::RuntimeError("WhileStmt codegen not yet implemented");
}

// ========================================================================
// Expression Visitor Methods - Dual-Mode Pattern
// ========================================================================
// - Statement-Emitting Mode (Call): Uses current_target_var_, emits instructions
// - Value-Returning Mode (others): Sets current_expr_value_ with inline C++ code

// ---- Leaf Nodes ----
void CCECodegen::VisitExpr_(const ir::VarPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
    std::string name = context_.SanitizeName(op);
    auto it = tuple_var_to_make_tuple_.find(name);
    if (it == tuple_var_to_make_tuple_.end())
        it = tuple_var_to_make_tuple_.find(name + "_0");
    // Positional tuple Var: produce the underlying MakeTuple in current_tuple_
    // (with _0 SSA fallback). Empty current_expr_value_ signals "no direct identifier".
    current_tuple_ = (it != tuple_var_to_make_tuple_.end()) ? it->second : nullptr;
    current_expr_value_ = context_.GetVarName(op);
}

void CCECodegen::VisitExpr_(const ir::MakeTuplePtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null MakeTuple";
    current_tuple_ = op;
    current_expr_value_ = "";
}

void CCECodegen::VisitExpr_(const ir::ConstIntPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstInt";
    current_expr_value_ = std::to_string(op->value_);
}

void CCECodegen::VisitExpr_(const ir::ConstFloatPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstFloat";
    current_expr_value_ = std::to_string(op->value_);
}

void CCECodegen::VisitExpr_(const ir::ConstBoolPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstBool";
    current_expr_value_ = op->value_ ? "true" : "false";
}

std::string CCECodegen::GetOrCreateVFTilePtr(const ir::ExprPtr& expr, bool is_post_update)
{
    std::string code = GetExprAsCode(expr);
    auto it = vf_tile_ptrs_.find(code);
    if (it != vf_tile_ptrs_.end())
        return it->second;

    auto tile_type = ir::As<ir::TileType>(expr->GetType());
    INTERNAL_CHECK(tile_type != nullptr) << "GetOrCreateVFTilePtr expects a tile expr, got " << expr->TypeName();
    std::string elem_ctype = tile_type->dtype_.ToCTypeString();

    // tile[offset] is the rvalue `(vf_tile_ptr_N + off)`; a load uses it as-is, no new variable.
    bool is_offset = code.find("vf_tile_ptr_") != std::string::npos;
    if (is_offset && !is_post_update)
        return code;

    // Declare a vf_tile_ptr_N once, cached and hoisted before __VEC_SCOPE__: plain tile -> base from
    // .data(); post_update -> a cursor the intrinsic advances in place.
    std::string var = "vf_tile_ptr_" + std::to_string(vf_tile_ptr_counter_++);
    std::string init = is_offset ? code : ("(__ubuf__ " + elem_ctype + " *)" + code + ".data()");
    section_hoisted_decls_.push_back("__ubuf__ " + elem_ctype + " *" + var + " = " + init + ";");
    vf_tile_ptrs_[code] = var;
    return var;
}

void CCECodegen::VisitExpr_(const ir::GetItemExprPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null GetItemExpr";

    auto value_type = op->value_->GetType();

    // Tuple element access.
    if (ir::As<ir::TupleType>(value_type)) {
        auto const_idx = ir::As<ir::ConstInt>(op->slice_);
        if (const_idx != nullptr) {
            // Static (constant) index: VisitExpr drives current_tuple_, then VisitExpr the element.
            int idx = static_cast<int>(const_idx->value_);
            INTERNAL_CHECK(idx >= 0) << "GetItemExpr negative tuple index " << idx;

            current_tuple_ = nullptr;
            current_expr_value_ = "";
            VisitExpr(op->value_);
            if (current_tuple_ == nullptr) {
                // The base has no underlying MakeTuple to fold into (e.g. a tiling struct
                // param or its array field). Field names registered in IRDebugInfo mark a
                // struct -> emit `base.field`; their absence marks an array field (the nested
                // TupleType of an Array[T, N]) -> emit `base[idx]`.
                auto tuple_type = ir::As<ir::TupleType>(value_type);
                const std::vector<std::string>* fields =
                    debug_info_ != nullptr ? debug_info_->GetTupleFields(tuple_type.get()) : nullptr;
                std::string base_code = GetExprAsCode(op->value_);
                current_tuple_ = nullptr;
                if (fields != nullptr) {
                    CHECK(idx < static_cast<int>(fields->size()))
                        << "GetItemExpr struct field: no field name for tuple index " << idx;
                    current_expr_value_ = base_code + "." + (*fields)[idx];
                } else {
                    // base_code already carries the array field name (e.g. tiling.opkind),
                    // so appending the constant index yields tiling.opkind[idx].
                    current_expr_value_ = base_code + "[" + std::to_string(idx) + "]";
                }
                return;
            }
            INTERNAL_CHECK(idx < static_cast<int>(current_tuple_->elements_.size()))
                << "GetItemExpr index " << idx << " out of bounds";
            ir::ExprPtr elem = current_tuple_->elements_[idx];

            current_tuple_ = nullptr;
            current_expr_value_ = "";
            VisitExpr(elem); // result lands in current_tuple_ (sub-tuple) or current_expr_value_
            return;
        }

        // Dynamic index: resolve value_ back to its underlying MakeTuple (Var aliases,
        // iterArgs, nested static GetItem and struct attrs all land in current_tuple_), then
        // look up the array emitted at the MakeTuple assignment, which is tuple var.
        std::string index_code = GetExprAsCode(op->slice_);
        current_tuple_ = nullptr;
        current_expr_value_ = "";
        VisitExpr(op->value_);
        CHECK(current_tuple_ != nullptr)
            << "Dynamic tuple GetItem: cannot resolve underlying MakeTuple for "
            << op->value_->TypeName();

        std::string arr_name;
        if (current_expr_value_ != "") {
            arr_name = current_expr_value_;
        } else {
            ir::MakeTuplePtr resolved_mt = current_tuple_;
            if (resolved_mt && !resolved_mt->elements_.empty()) {
                auto elem_type = resolved_mt->elements_[0]->GetType();
                std::vector<std::string> elem_names;
                elem_names.reserve(resolved_mt->elements_.size());
                for (size_t i = 0; i < resolved_mt->elements_.size(); ++i) {
                    elem_names.push_back(GetExprAsCode(resolved_mt->elements_[i]));
                }
                std::string candidate_arr_name = "_dyn_arr_" + std::to_string(dyn_arr_counter_);
                std::string arr_decl =
                    BuildDynamicTupleArrayDecl(elem_type, elem_names, candidate_arr_name, /*allow_struct_tuple=*/false);
                if (!arr_decl.empty()) {
                    arr_name = candidate_arr_name;
                    ++dyn_arr_counter_;
                    emitter_.EmitLine(arr_decl);
                }
            }
        }
        current_expr_value_ = arr_name + "[" + index_code + "]";
        return;
    }

    // Tile element offset: slice_ is an integer expression (static or dynamic).
    auto tile_type = ir::As<ir::TileType>(value_type);
    INTERNAL_CHECK(tile_type != nullptr) << "GetItemExpr requires value to have TupleType or TileType, got "
                                         << value_type->TypeName();

    std::string base_tile = GetExprAsCode(op->value_);
    std::string offset_expr = GetExprAsCode(op->slice_);

    int elem_bytes = static_cast<int>(tile_type->dtype_.GetBit() / 8);
    if (elem_bytes == 0)
        elem_bytes = 1; // guard sub-byte types

    // VF section: typed pointer arithmetic off the section-hoisted base ptr. `vf_tile_ptr_N + offset`
    // advances by whole elements and lets matching loads/stores skip the cast.
    if (IsInVFSection()) {
        std::string base_ptr = GetOrCreateVFTilePtr(op->value_, /*is_post_update=*/false);
        current_expr_value_ = "(" + base_ptr + " + (" + offset_expr + "))";
        return;
    }

    // Get base tile address ->try tile_addresses_ first, then extract from TileType memref
    std::string base_addr;
    auto addr_it = tile_addresses_.find(base_tile);
    if (addr_it != tile_addresses_.end()) {
        base_addr = addr_it->second;
    } else {
        INTERNAL_CHECK(tile_type->memref_.has_value())
            << "GetItemExpr: base tile '" << base_tile << "' has no address info";
        int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_);
        base_addr = FormatAddressHex(addr);
    }

    // Sanitize base_tile into a valid C++ identifier prefix: replace non-alnum/underscore chars.
    std::string base_tile_id = base_tile;
    for (char& c : base_tile_id) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    }
    std::string temp_name = base_tile_id + "_eoff_" + std::to_string(tile_offset_counter_++);

    std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
    int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
    int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
    auto vs = ExtractValidShapeInfo(tile_type, rows, cols, [this](const ir::VarPtr& v) { return GetVarName(v); });
    std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);
    std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
    std::string type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
    std::string temp_addr = base_addr + " + (" + offset_expr + ") * " + std::to_string(elem_bytes);
    emitter_.EmitLine(
        type_str + " " + temp_name + ctor_suffix + "; " + "TASSIGN(" + temp_name + ", " + temp_addr + ");");
    tile_addresses_[temp_name] = temp_addr;

    current_expr_value_ = temp_name;
}

// ========================================================================
// CodegenBase interface and CCE-specific helper methods
// ========================================================================

std::string CCECodegen::GetExprAsCode(const ir::ExprPtr& expr)
{
    VisitExpr(expr);
    return current_expr_value_;
}

void CCECodegen::Emit(const std::string& line) { emitter_.EmitLine(line); }

std::string CCECodegen::GetTypeString(const ir::DataType& dtype) const { return dtype.ToCTypeString(); }

int64_t CCECodegen::GetConstIntValue(const ir::ExprPtr& expr) { return ExtractConstInt(expr); }

std::string CCECodegen::GetVarName(const ir::VarPtr& var) { return context_.GetVarName(var); }

void CCECodegen::RegisterPointer(const std::string& tensor_var_name, const std::string& ptr_name)
{
    CHECK(!tensor_var_name.empty()) << "Cannot register pointer with empty tensor var name";
    CHECK(!ptr_name.empty()) << "Cannot register pointer with empty pointer name";

    auto it = tensor_to_pointer_.find(tensor_var_name);
    if (it != tensor_to_pointer_.end() && it->second != ptr_name) {
        IR_LOGW() << "Pointer for tensor " << tensor_var_name << " re-registered with: " << ptr_name << " vs "
                  << it->second;
    }
    tensor_to_pointer_[tensor_var_name] = ptr_name;
}

std::string CCECodegen::GetPointer(const std::string& var_name)
{
    auto it = tensor_to_pointer_.find(var_name);
    CHECK(it != tensor_to_pointer_.end()) << "Pointer for tensor " << var_name << " not found";
    return it->second;
}

bool CCECodegen::HasPointer(const std::string& var_name) const
{
    return tensor_to_pointer_.find(var_name) != tensor_to_pointer_.end();
}

const TensorDef* CCECodegen::GetTensorDef(const std::string& name) const
{
    auto it = tensor_defs_.find(name);
    return it != tensor_defs_.end() ? &it->second : nullptr;
}

std::string CCECodegen::ComputeIRBasedOffset(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets)
{
    // Compute row-major strides from IR tensor shape and build offset expression.
    // For tensor shape [d0, d1, ..., dn-1]:
    //   stride[i] = d_{i+1} * d_{i+2} * ... * d_{n-1}
    //   stride[n-1] = 1
    //   offset = off[0]*stride[0] + off[1]*stride[1] + ...
    size_t ndim = tensor_type->shape_.size();
    CHECK(offsets->elements_.size() == ndim)
        << "Offset dimensions (" << offsets->elements_.size() << ") != tensor dimensions (" << ndim << ")";

    std::ostringstream result;
    result << "(";
    bool first = true;
    for (size_t i = 0; i < ndim; ++i) {
        std::string off_expr = GetExprAsCode(offsets->elements_[i]);

        // Build stride as product of shape[i+1..n-1]
        // For the last dimension, stride = 1, so just add the offset directly
        if (!first)
            result << " + ";
        first = false;
        result << off_expr;

        for (size_t j = i + 1; j < ndim; ++j) {
            result << " * " << GetExprAsCode(tensor_type->shape_[j]);
        }
    }
    result << ")";
    return result.str();
}

// ========================================================================
// Call Expression Visitor (uses operator registry codegen functions)
// ========================================================================

void CCECodegen::VisitExpr_(const ir::CallPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Call";

    CHECK(backend_ != nullptr) << "CCE backend must not be null";
    const auto* op_info = backend_->GetOpInfo(op->name_);
    CHECK(op_info != nullptr) << "Unknown call '" << op->name_
                              << "' reached CCE codegen; helper calls must be inlined";
    std::string result = op_info->codegen_func(op, *this);
    current_expr_value_ = result;
}

// ---- Binary Operators ----

#define IMPLEMENT_BINARY_OP(OpType, OpName, CppOp)                            \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                    \
    {                                                                         \
        INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName); \
        VisitExpr(op->left_);                                                 \
        std::string left = current_expr_value_;                               \
        VisitExpr(op->right_);                                                \
        std::string right = current_expr_value_;                              \
        current_expr_value_ = "(" + left + " " + (CppOp) + " " + right + ")"; \
    }

// Arithmetic operators
IMPLEMENT_BINARY_OP(Add, "Add", "+")
IMPLEMENT_BINARY_OP(Sub, "Sub", "-")
IMPLEMENT_BINARY_OP(Mul, "Mul", "*")
IMPLEMENT_BINARY_OP(FloorDiv, "FloorDiv", "/")
IMPLEMENT_BINARY_OP(FloorMod, "FloorMod", "%")
IMPLEMENT_BINARY_OP(FloatDiv, "FloatDiv", "/")

// Comparison operators
IMPLEMENT_BINARY_OP(Eq, "Eq", "==")
IMPLEMENT_BINARY_OP(Ne, "Ne", "!=")
IMPLEMENT_BINARY_OP(Lt, "Lt", "<")
IMPLEMENT_BINARY_OP(Le, "Le", "<=")
IMPLEMENT_BINARY_OP(Gt, "Gt", ">")
IMPLEMENT_BINARY_OP(Ge, "Ge", ">=")

// Logical operators
IMPLEMENT_BINARY_OP(And, "And", "&&")
IMPLEMENT_BINARY_OP(Or, "Or", "||")
IMPLEMENT_BINARY_OP(Xor, "Xor", "^")

// Bitwise operators
IMPLEMENT_BINARY_OP(BitAnd, "BitAnd", "&")
IMPLEMENT_BINARY_OP(BitOr, "BitOr", "|")
IMPLEMENT_BINARY_OP(BitXor, "BitXor", "^")
IMPLEMENT_BINARY_OP(BitShiftLeft, "BitShiftLeft", "<<")
IMPLEMENT_BINARY_OP(BitShiftRight, "BitShiftRight", ">>")

#undef IMPLEMENT_BINARY_OP

// Special binary operators (function calls)
void CCECodegen::VisitExpr_(const ir::MinPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Min";
    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cpp_type = scalar_type ? scalar_type->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "min((" + cpp_type + ")(" + left + "), (" + cpp_type + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::MaxPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Max";
    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cpp_type = scalar_type ? scalar_type->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "max((" + cpp_type + ")(" + left + "), (" + cpp_type + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::PowPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Pow";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "pow(" + left + ", " + right + ")";
}

// ---- Unary Operators ----

#define IMPLEMENT_UNARY_OP(OpType, OpName, CppOp)                                     \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                            \
    {                                                                                 \
        INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName);         \
        VisitExpr(op->operand_);                                                      \
        current_expr_value_ = std::string("(") + (CppOp) + current_expr_value_ + ")"; \
    }

IMPLEMENT_UNARY_OP(Neg, "Neg", "-")
IMPLEMENT_UNARY_OP(Not, "Not", "!")
IMPLEMENT_UNARY_OP(BitNot, "BitNot", "~")

#undef IMPLEMENT_UNARY_OP

// Special unary operators
void CCECodegen::VisitExpr_(const ir::AbsPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Abs";
    VisitExpr(op->operand_);
    std::string operand = current_expr_value_;
    current_expr_value_ = "abs(" + operand + ")";
}

void CCECodegen::VisitExpr_(const ir::CastPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Cast";
    VisitExpr(op->operand_);
    std::string operand = current_expr_value_;

    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    CHECK(scalar_type != nullptr) << "Cast target must be ScalarType";

    std::string cpp_type = scalar_type->dtype_.ToCTypeString();
    current_expr_value_ = "((" + cpp_type + ")" + operand + ")";
}

// ========================================================================
// End of Expression Visitor Methods
// ========================================================================

int64_t CCECodegen::ExtractConstInt(const ir::ExprPtr& expr) const
{
    auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
    CHECK(const_int != nullptr) << "Expected constant integer expression";
    return const_int->value_;
}

namespace {

/**
 * \brief Helper visitor for collecting TileType variables from IR
 *
 * Traverses the IR tree and collects all variables with TileType.
 * Uses the visitor pattern for clean, extensible traversal.
 */
class TileCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars_;

    // Extract the first TileType from a possibly nested TupleType
    static ir::TileTypePtr ExtractLeafTileType(const ir::TypePtr& type)
    {
        if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
            return tile;
        }
        if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
            for (const auto& elem : tuple->types_) {
                auto result = ExtractLeafTileType(elem);
                if (result)
                    return result;
            }
        }
        return nullptr;
    }

    // Recursively collect all TileTypes from a type (handles TupleType nesting)
    static void CollectTileTypesFromType(const ir::TypePtr& type, std::vector<ir::TileTypePtr>& result)
    {
        if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
            result.push_back(tile);
        } else if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
            for (const auto& elem : tuple->types_) {
                CollectTileTypesFromType(elem, result);
            }
        }
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto target_var = op->var_;
        auto var_type = target_var->GetType();
        // Direct TileType variable
        if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(var_type)) {
            tile_vars_.emplace_back(target_var, tile_type);
            return;
        }
        // TupleType variable (from make_tile or double-buffer patterns):
        // Expand each TileType element as a separate tile with indexed name.
        if (std::dynamic_pointer_cast<const ir::TupleType>(var_type)) {
            std::vector<ir::TileTypePtr> tile_types;
            CollectTileTypesFromType(var_type, tile_types);
            for (size_t i = 0; i < tile_types.size(); ++i) {
                // Create a synthetic Var for each tuple element: varname_0, varname_1, ...
                auto elem_var = std::make_shared<ir::Var>(
                    target_var->name_ + "_" + std::to_string(i), tile_types[i], target_var->span_);
                tile_vars_.emplace_back(elem_var, tile_types[i]);
            }
        }
    }
};

/**
 * \brief Helper visitor for collecting tensor definitions from block.load/store
 *
 * Traverses the IR tree to find block.load/block.store calls and aggregates, for
 * each tensor variable (keyed by its cce variable name), the access window shape,
 * defining section, tile_dims and DN flag into a single TensorDef. Mirrors
 * MakeTileDefCollector: one pass, one container, no parallel per-section maps. The
 * name key lets CCECodegen move defs_ straight into tensor_defs_ (no rekeying).
 *
 * A tensor accessed across both Cube and Vector sections collapses to a single
 * TensorDef with def_section == nullopt (shared region, visible to both compilation
 * units), instead of one declaration per section guard.
 */
class TensorDefCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    explicit TensorDefCollector(const CodeContext& ctx) : ctx_(ctx) {}

    std::map<std::string, TensorDef> defs_; ///< cce var name -> aggregated TensorDef

    void VisitStmt_(const ir::SectionStmtPtr& op) override
    {
        auto prev = current_section_;
        current_section_ = op->sectionKind_;
        ir::IRVisitor::VisitStmt_(op);
        current_section_ = prev;
    }

    void VisitExpr_(const ir::CallPtr& op) override
    {
        AccessArgIndices indices = ResolveAccessArgIndices(op->name_);
        if (indices.tensor_arg_idx >= 0 && static_cast<int>(op->args_.size()) > indices.tensor_arg_idx) {
            auto tensor_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[indices.tensor_arg_idx]);
            RecordTensorDef(op, tensor_var, indices);
        }

        ir::IRVisitor::VisitExpr_(op);
    }

private:
    struct AccessArgIndices {
        int tensor_arg_idx = -1;
        int tile_arg_idx = -1;
    };

    AccessArgIndices ResolveAccessArgIndices(const std::string& op_name) const
    {
        AccessArgIndices indices;
        if (op_name == "block.load") {
            indices.tensor_arg_idx = 0;
            indices.tile_arg_idx = 2;
        } else if (op_name == "block.store") {
            indices.tensor_arg_idx = 2;
            indices.tile_arg_idx = 0;
        } else if (op_name == "block.store_fp") {
            indices.tensor_arg_idx = 3;
            indices.tile_arg_idx = 0;
        }
        return indices;
    }

    // Access window shape: explicit shapes tuple if present, otherwise the tile's shape.
    std::optional<std::vector<ir::ExprPtr>> ResolveAccessShape(
        const ir::CallPtr& op, const AccessArgIndices& indices) const
    {
        if (indices.tile_arg_idx >= 0 && indices.tile_arg_idx < static_cast<int>(op->args_.size())) {
            auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(op->args_[indices.tile_arg_idx]->GetType());
            if (tile_type) {
                return tile_type->shape_;
            }
        }
        return std::nullopt;
    }

    void RecordTensorDef(
        const ir::CallPtr& op, const std::shared_ptr<const ir::Var>& tensor_var, const AccessArgIndices& indices)
    {
        if (!tensor_var) {
            return;
        }
        // A ptr.make_tensor view is self-identifying (its TensorType carries a source
        // pointer); it lands here just like a tensor parameter and becomes a TensorDef.
        auto var = std::const_pointer_cast<ir::Var>(tensor_var);
        TensorDef& def = defs_[ctx_.SanitizeName(var)];
        if (def.var == nullptr) {
            def.var = var;
            def.def_section = current_section_;
        } else if (def.def_section != current_section_) {
            def.def_section = std::nullopt; // accessed across sections -> shared region
        }
        if (def.access_shape.empty()) {
            if (auto shape = ResolveAccessShape(op, indices)) {
                def.access_shape = *shape;
            }
        }
        if (!def.tile_dims.has_value() && op->HasKwarg("tile_dims")) {
            def.tile_dims = op->GetKwarg<std::vector<int>>("tile_dims");
        }
        if (op->name_ == "block.load" && op->HasKwarg("layout") && op->GetKwarg<std::string>("layout") == "dn") {
            def.is_dn = true;
        }
    }

    const CodeContext& ctx_; ///< for SanitizeName (pure: derives the cce var-name key)
    std::optional<ir::SectionKind> current_section_;
};

} // namespace

std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CCECodegen::CollectTileVariables(const ir::StmtPtr& stmt)
{
    if (!stmt) {
        return {};
    }

    TileCollector collector;
    collector.VisitStmt(stmt);
    return collector.tile_vars_;
}

namespace {
// Collect names of Var nodes that appear in any read position ->Call args,
// Yield values, AssignStmt RHS, return values, expressions inside subscripts,
// for-loop ranges, etc. A phi return_var whose name is not in this set has
// no consumer and can be safely dropped from IfStmt codegen.
class VarReadCollector : public ir::IRVisitor {
public:
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

    std::set<std::string> var_read_names_;
    std::unordered_map<std::string, int> var_read_counts_;

    // Any time we visit a Var as a sub-expression (via the default Expr walk),
    // it is in a read position ->the write site is an AssignStmt.var_ which
    // the default Stmt walker does not route through VisitExpr.
    void VisitExpr_(const ir::VarPtr& op) override
    {
        if (op) {
            var_read_names_.insert(op->name_);
            var_read_counts_[op->name_]++;
        }
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        // Skip op->var_ (LHS is a definition, not a use). Walk the RHS.
        if (op && op->value_)
            VisitExpr(op->value_);
    }

    void VisitStmt_(const ir::IfStmtPtr& op) override
    {
        // Skip op->return_vars_ ->they are phi outputs (definitions), not reads.
        // Walk condition + branches explicitly.
        if (!op)
            return;
        if (op->condition_)
            VisitExpr(op->condition_);
        if (op->thenBody_)
            VisitStmt(op->thenBody_);
        if (op->elseBody_.has_value() && *op->elseBody_)
            VisitStmt(*op->elseBody_);
    }

    void VisitStmt_(const ir::ForStmtPtr& op) override
    {
        // Similar concern for For loops: return_vars_ (output iter args) are
        // written, not read, on each iteration's phi merge.
        if (!op)
            return;
        if (op->start_)
            VisitExpr(op->start_);
        if (op->stop_)
            VisitExpr(op->stop_);
        if (op->step_)
            VisitExpr(op->step_);
        for (const auto& ia : op->iterArgs_) {
            if (ia && ia->initValue_)
                VisitExpr(ia->initValue_);
        }
        if (op->body_)
            VisitStmt(op->body_);
    }
};

} // namespace

void CCECodegen::CollectVarReadNames(const ir::StmtPtr& stmt, std::set<std::string>& out) const
{
    if (!stmt)
        return;
    VarReadCollector collector;
    collector.VisitStmt(stmt);
    out = std::move(collector.var_read_names_);
    // Also populate read counts on the mutable codegen instance
    const_cast<CCECodegen*>(this)->var_read_counts_ = std::move(collector.var_read_counts_);
}

namespace {

class MutexPipeCollector : public ir::IRVisitor {
public:
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

    std::map<int, std::set<ir::PipeType>> cube_mutex_pipes;
    std::map<int, std::set<ir::PipeType>> vec_mutex_pipes;

    void VisitStmt_(const ir::SectionStmtPtr& op) override
    {
        auto prev = current_section_;
        current_section_ = op->sectionKind_;
        ir::IRVisitor::VisitStmt_(op);
        current_section_ = prev;
    }

    void VisitExpr_(const ir::CallPtr& op) override
    {
        if (op) {
            CollectMutexInfo(op);
        }
        ir::IRVisitor::VisitExpr_(op);
    }

    void CollectMutexInfo(const ir::CallPtr& op)
    {
        const std::string& name = op->name_;
        bool is_mutex =
            (name == "system.mutex_lock" || name == "system.mutex_unlock" || name == "system.mutex_lock_dyn" ||
             name == "system.mutex_unlock_dyn");
        if (!is_mutex) {
            return;
        }
        ir::PipeType pipe = ir::PipeType::S;
        int static_bid = -1;
        std::vector<int> dyn_bids;
        for (const auto& [key, value] : op->kwargs_) {
            if (key == "pipe")
                pipe = static_cast<ir::PipeType>(std::any_cast<int>(value));
            if (key == "mutex_id")
                static_bid = std::any_cast<int>(value);
            if (key == "mutex_ids")
                dyn_bids = std::any_cast<std::vector<int>>(value);
        }
        auto& target = (current_section_ == ir::SectionKind::Cube) ? cube_mutex_pipes : vec_mutex_pipes;
        auto record = [&](int bid) { target[bid].insert(pipe); };
        if (static_bid >= 0)
            record(static_bid);
        for (int bid : dyn_bids)
            record(bid);
        if (static_bid < 0 && dyn_bids.empty()) {
            int max_id = 2;
            for (const auto& [key, value] : op->kwargs_) {
                if (key == "max_mutex_id")
                    max_id = std::any_cast<int>(value);
            }
            for (int i = 0; i < max_id; ++i)
                record(i);
        }
    }

private:
    ir::SectionKind current_section_ = ir::SectionKind::Vector;
};

} // namespace

void CCECodegen::CollectMutexPipeInfo(const ir::StmtPtr& stmt)
{
    if (!stmt)
        return;
    MutexPipeCollector collector;
    collector.VisitStmt(stmt);
    cube_mutex_pipes_ = std::move(collector.cube_mutex_pipes);
    vec_mutex_pipes_ = std::move(collector.vec_mutex_pipes);
}

bool CCECodegen::ShouldSkipVPipeMutex(ir::PipeType pipe, const std::vector<int>& buf_ids) const
{
    if (pipe != ir::PipeType::V || arch_ != "a5")
        return false;
    const auto& section_map = (current_section_kind_ == ir::SectionKind::Cube) ? cube_mutex_pipes_ : vec_mutex_pipes_;
    for (int bid : buf_ids) {
        auto it = section_map.find(bid);
        if (it == section_map.end())
            continue;
        for (auto p : it->second) {
            if (p != ir::PipeType::V)
                return false;
        }
    }
    return true;
}

void CCECodegen::PreScanValidShapes()
{
    // set_validshape is guaranteed to precede any load/store that uses the same tile.
    // Shape registration is therefore handled at emit-time by RegisterTileEmitShape()
    // (called from MakeBlockOutSetValidShapeCodegenCCE), so no pre-scan is needed.
    // This function clears the emit-time map so each GenerateSingle
    // call starts with a clean slate.
    tile_emit_shape_.clear();
}

void CCECodegen::CollectTensorDefs(const ir::FunctionPtr& func)
{
    if (!func || !func->body_)
        return;

    TensorDefCollector collector(context_);
    collector.VisitStmt(func->body_);
    tensor_defs_ = std::move(collector.defs_);
}

std::vector<int64_t> CCECodegen::ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs) const
{
    std::vector<int64_t> dims;
    dims.reserve(shape_exprs.size());
    for (const auto& expr : shape_exprs) {
        dims.push_back(ExtractConstInt(expr));
    }
    return dims;
}

std::string CCECodegen::FormatAddressHex(int64_t addr)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << addr;
    return oss.str();
}

namespace {

// Map an IR field type to a best-effort C++ type string for struct member declaration.
// Only the common cases needed by v1 are covered; uncovered cases fall back to int64_t.
std::string CppTypeForField(const ir::TypePtr& t)
{
    if (auto scalar = ir::As<ir::ScalarType>(t)) {
        const auto& d = scalar->dtype_;
        if (d == DataType::INDEX || d == DataType::INT64)
            return "int64_t";
        if (d == DataType::INT32)
            return "int32_t";
        if (d == DataType::FP32)
            return "float";
        if (d == DataType::FP16)
            return "half";
        if (d == DataType::BOOL)
            return "bool";
        return "int64_t";
    }
    if (ir::As<ir::TileType>(t)) {
        // Tile fields are stored as opaque tile references; v1 emits as auto-deduced.
        return "auto";
    }
    // Nested named tuple ->the field stores another C++ struct; we have to look up its
    // type name. For v1 we fall back to int64_t and rely on CCE flat expansion if the
    // nested tuple type isn't already materialized; full nested support is a follow-up.
    return "int64_t";
}

// Build a C++ `struct Name { <type> <field>; ... };` definition from a struct type name,
// its field names, and per-field IR types. A TupleType field (Array[T, N]) becomes an
// array member `<T> name[N];`.
std::string BuildStructTypeDef(
    const std::string& name, const std::vector<std::string>& fields, const std::vector<ir::TypePtr>& types)
{
    std::string def = "struct " + name + " { ";
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& ft = types[i];
        if (auto arr = ir::As<ir::TupleType>(ft)) {
            def += CppTypeForField(arr->types_[0]) + " " + fields[i] + "[" +
                   std::to_string(arr->types_.size()) + "]; ";
        } else {
            def += CppTypeForField(ft) + " " + fields[i] + "; ";
        }
    }
    def += "};";
    return def;
}

struct StructDeclInfo {
    std::string struct_name; // user-provided C++ type name
    std::vector<std::string> fields;
    std::vector<ir::TypePtr> types;
};

// Scan IR for struct.create Call sites and collect each unique struct type's
// materialization info so PreEmitStructTypes can emit one C++ struct definition
// per (name, signature).
void CollectStructDeclares(const ir::StmtPtr& stmt, std::vector<StructDeclInfo>& out)
{
    if (!stmt)
        return;
    if (auto seq = ir::As<ir::SeqStmts>(stmt)) {
        for (const auto& s : seq->stmts_)
            CollectStructDeclares(s, out);
    } else if (auto assign = ir::As<ir::AssignStmt>(stmt)) {
        if (auto call = ir::As<ir::Call>(assign->value_)) {
            if (call->name_ == "struct.create" && call->HasKwarg("name")) {
                StructDeclInfo info;
                info.struct_name = call->GetKwarg<std::string>("name");
                info.fields = call->GetKwarg<std::vector<std::string>>("fields");
                info.types.reserve(call->args_.size());
                for (const auto& a : call->args_) {
                    info.types.push_back(a->GetType());
                }
                out.push_back(std::move(info));
            }
        }
    } else if (auto for_stmt = ir::As<ir::ForStmt>(stmt)) {
        CollectStructDeclares(for_stmt->body_, out);
    } else if (auto if_stmt = ir::As<ir::IfStmt>(stmt)) {
        CollectStructDeclares(if_stmt->thenBody_, out);
        if (if_stmt->elseBody_)
            CollectStructDeclares(*if_stmt->elseBody_, out);
    } else if (auto section = ir::As<ir::SectionStmt>(stmt)) {
        CollectStructDeclares(section->body_, out);
    }
}

} // namespace

void CCECodegen::PreEmitStructTypes(const ir::StmtPtr& body)
{
    std::vector<StructDeclInfo> declares;
    CollectStructDeclares(body, declares);

    std::set<std::string> emitted_names;
    for (const auto& info : declares) {
        if (!emitted_names.insert(info.struct_name).second)
            continue; // dedup by struct type name
        emitter_.EmitLine(BuildStructTypeDef(info.struct_name, info.fields, info.types));
    }
}

void CCECodegen::EmitTilingStructTypes(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tuple_type = ir::As<ir::TupleType>(param->GetType());
        if (tuple_type == nullptr)
            continue;
        const std::vector<std::string>* fields =
            debug_info_ != nullptr ? debug_info_->GetTupleFields(tuple_type.get()) : nullptr;
        CHECK(fields != nullptr) << "Tiling struct param '" << param->name_
                                 << "' has no field names registered in IRDebugInfo";
        CHECK(fields->size() == tuple_type->types_.size())
            << "Tiling struct field count mismatch for param '" << param->name_ << "'";
        const std::string* registered_name = debug_info_->GetTupleName(tuple_type.get());
        const std::string struct_name = registered_name != nullptr ? *registered_name : "Tiling";
        const std::string header_name = struct_name + "_tiling.h";
        // Emit the include into kernel.cpp once per struct; the header content (deduped via
        // tiling_headers_ keyed by filename) is written next to kernel.cpp by the caller.
        if (tiling_headers_.count(header_name) == 0) {
            std::string def = BuildStructTypeDef(struct_name, *fields, tuple_type->types_);
            tiling_headers_[header_name] = "#pragma once\n" + def + "\n";
            emitter_.EmitLine("#include \"" + header_name + "\"");
        }
    }
}

void CCECodegen::EmitTilingStructCopy(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tuple_type = ir::As<ir::TupleType>(param->GetType());
        if (tuple_type == nullptr)
            continue;
        std::string name = context_.GetVarName(param);
        const std::string* registered_name = debug_info_->GetTupleName(tuple_type.get());
        const std::string struct_name = registered_name != nullptr ? *registered_name : "Tiling";
        // Shared local struct (one of __DAV_CUBE__/__DAV_VEC__ is defined per compile).
        emitter_.EmitLine("constexpr uint32_t " + name + "_all_bytes = sizeof(" + struct_name + ");");
        emitter_.EmitLine(struct_name + " " + name + ";");
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
        emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__gm__ uint8_t *)" + name +
                          "_ptr, " + name + "_all_bytes);");
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
        emitter_.EmitLine("__ubuf__ uint8_t *" + name + "_in_ub = (__ubuf__ uint8_t *)get_imm(0);");
        emitter_.EmitLine("constexpr uint32_t " + name + "_len_burst = (" + name + "_all_bytes + 31) / 32;");
        emitter_.EmitLine("copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)" + name + "_in_ub, (__gm__ uint8_t *)" +
                          name + "_ptr, 0, 1, " + name + "_len_burst * 32, 0, 0, false, 0, 0, 0);");
        emitter_.EmitLine("set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
        emitter_.EmitLine("wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
        emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__ubuf__ uint8_t *)" + name +
                          "_in_ub, " + name + "_all_bytes);");
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }
}

void CCECodegen::GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type)
{
    INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
    INTERNAL_CHECK(tile_type != nullptr) << "Internal error: tile_type is null";

    // Extract tile shape dimensions
    std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);

    // CCE codegen only supports 1D and 2D tiles
    CHECK(shape_dims.size() <= 2) << "CCE codegen only supports 1D and 2D TileType, but got " << shape_dims.size()
                                  << " dimensions. Multi-dimensional tiles (>2D) are supported at IR level "
                                  << "but not yet in code generation.";

    // Determine tile dimensions (default to 1 if not specified)
    int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
    int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;

    // Extract valid_shape: compute runtime ctor args.
    auto vs = ExtractValidShapeInfo(tile_type, rows, cols, [this](const ir::VarPtr& v) { return GetVarName(v); });
    std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);

    // Generate Tile type alias (with dedup: reuse alias if same type string already emitted)
    std::string tile_type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
    std::string type_alias_name;
    auto dedup_it = emitted_tile_types_.find(tile_type_str);
    if (dedup_it != emitted_tile_types_.end()) {
        type_alias_name = dedup_it->second;
    } else {
        // Derive alias name from var_name base (strip trailing _N suffix for dedup readability)
        auto last_underscore = var_name.rfind('_');
        bool has_index_suffix =
            (last_underscore != std::string::npos && last_underscore + 1 < var_name.size() &&
             std::isdigit(var_name[last_underscore + 1]));
        std::string base_name = has_index_suffix ? var_name.substr(0, last_underscore) : var_name;
        type_alias_name = base_name + "_Type";
        // Ensure uniqueness against existing aliases
        if (emitted_tile_types_.count(type_alias_name + "_used") > 0 &&
            emitted_tile_types_[type_alias_name + "_used"] != tile_type_str) {
            type_alias_name = var_name + "Type";
        }
        emitted_tile_types_[tile_type_str] = type_alias_name;
        emitted_tile_types_[type_alias_name + "_used"] = tile_type_str;
        emitter_.EmitLine("using " + type_alias_name + " = " + tile_type_str + ";");
    }

    // Generate Tile instance + TASSIGN on one line (compact)
    // Only pass ctor args when template has -1 params (dynamic valid_shape).
    std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
    if (tile_type->memref_.has_value()) {
        int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_); // NOLINT(bugprone-unchecked-optional-access)
        std::string addr_str = FormatAddressHex(addr);
        emitter_.EmitLine(
            type_alias_name + " " + var_name + ctor_suffix + "; TASSIGN(" + var_name + ", " + addr_str + ");");
        tile_addresses_[var_name] = addr_str;
    } else {
        emitter_.EmitLine(type_alias_name + " " + var_name + ctor_suffix + ";");
    }
}

// ========================================================================
// Phase 7 helpers: Stride type generation and GlobalTensor instance emission
// ========================================================================

std::string CCECodegen::GenerateSingleFileStrideType(const std::vector<int64_t>& dims) const
{
    // ND (row-major) static strides: stride[i] = product(dims[i+1..n-1]), padded with
    // leading 1s to five slots. Used for the NZ physical fractal shape (5D).
    std::ostringstream oss;
    oss << "pto::Stride<";
    const size_t target_dims = 5;
    size_t n = dims.size();
    for (size_t i = 0; i < target_dims - n; ++i) {
        oss << "1, ";
    }
    for (size_t i = 0; i < n; ++i) {
        int64_t stride = 1;
        for (size_t j = i + 1; j < n; ++j) {
            stride *= dims[j];
        }
        oss << stride;
        if (i < n - 1)
            oss << ", ";
    }
    oss << ">";
    return oss.str();
}

void CCECodegen::AppendDynamicStrideGlobalTensorArgs(
    std::ostringstream& global_instance, const std::string& shape_type_name, const std::string& stride_type_name,
    const ir::TensorTypePtr& tensor_type, const std::vector<int64_t>& shape_dims,
    const std::optional<std::vector<int>>& tile_dims, bool is_dn)
{
    auto dim_expr = [this](const ir::ExprPtr& dim, int64_t fallback) -> std::string {
        if (auto ci = std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
            return std::to_string(ci->value_);
        }
        if (auto var = std::dynamic_pointer_cast<const ir::Var>(dim)) {
            return context_.GetVarName(std::const_pointer_cast<ir::Var>(var));
        }
        return std::to_string(fallback);
    };
    auto full_stride_expr = [&](int dim_idx) -> std::string {
        std::string expr = "1";
        for (size_t i = static_cast<size_t>(dim_idx) + 1; i < tensor_type->shape_.size(); ++i) {
            std::string dim = dim_expr(tensor_type->shape_[i], 1);
            expr = (expr == "1") ? dim : expr + "*" + dim;
        }
        return expr;
    };

    std::string row_stride_expr;
    std::string col_stride_expr;
    if (tile_dims.has_value() && tile_dims->size() == 2) {
        // tile_dims=[a, b] BSND view: row stride is the product below dim a (e.g. N*D),
        // not the access-window column.
        row_stride_expr = full_stride_expr((*tile_dims)[0]);
        col_stride_expr = full_stride_expr((*tile_dims)[1]);
    } else {
        row_stride_expr = dim_expr(tensor_type->shape_.back(), shape_dims.back());
        col_stride_expr = "1";
    }
    // Our tensors are 2D; the three leading stride slots are unused -> fixed to 1.
    // DN (column-major) swaps the low two stride values relative to ND.
    global_instance << ", " << shape_type_name << "(), " << stride_type_name << "(1, 1, 1, ";
    if (is_dn) {
        global_instance << col_stride_expr << ", " << row_stride_expr;
    } else {
        global_instance << row_stride_expr << ", " << col_stride_expr;
    }
    global_instance << ")";
}

void CCECodegen::EmitGlobalTensorInstance(
    const std::string& var_name, const std::string& global_type_name, const std::string& shape_type_name,
    const std::string& stride_type_name, const ir::TensorTypePtr& tensor_type, const std::vector<int64_t>& shape_dims,
    const std::optional<std::vector<int>>& tile_dims, bool is_dn, const std::string& base_pointer)
{
    std::ostringstream global_instance;
    global_instance << global_type_name << " " << var_name << "(" << base_pointer;
    AppendDynamicStrideGlobalTensorArgs(
        global_instance, shape_type_name, stride_type_name, tensor_type, shape_dims, tile_dims, is_dn);
    global_instance << ");";
    emitter_.EmitLine(global_instance.str());
}

std::string CCECodegen::BuildDynamicNZTensorDimArg(const ir::TensorTypePtr& tensor_type, size_t axis)
{
    if (auto const_dim = ir::As<ir::ConstInt>(tensor_type->shape_[axis])) {
        return std::to_string(const_dim->value_);
    }
    return GetExprAsCode(tensor_type->shape_[axis]);
}

std::string CCECodegen::BuildDynamicNZShapeArg(
    const ir::TensorTypePtr& tensor_type, size_t axis, const std::optional<std::vector<ir::ExprPtr>>& access_shape)
{
    if (!access_shape.has_value() || axis >= access_shape->size()) {
        return BuildDynamicNZTensorDimArg(tensor_type, axis);
    }

    const auto& expr = access_shape->at(axis);
    if (auto const_dim = ir::As<ir::ConstInt>(expr)) {
        return std::to_string(const_dim->value_);
    }

    auto var_dim = ir::As<ir::Var>(expr);
    auto tensor_dim_var = ir::As<ir::Var>(tensor_type->shape_[axis]);
    if (var_dim && tensor_dim_var && var_dim->name_ == tensor_dim_var->name_) {
        return BuildDynamicNZTensorDimArg(tensor_type, axis);
    }
    return GetExprAsCode(expr);
}

void CCECodegen::EmitDynamicNZGlobalTensorDeclaration(
    const std::string& var_name, const ir::TensorTypePtr& tensor_type, const std::optional<std::string>& base_pointer,
    const std::optional<std::vector<ir::ExprPtr>>& access_shape, const std::string& element_type,
    const std::string& shape_type_name, const std::string& stride_type_name, const std::string& global_type_name)
{
    if (tensor_type->shape_.size() != 2) {
        throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires a 2D tensor");
    }
    (void)backend::cce::GetNZInnerCols(tensor_type->dtype_);

    const std::string row_arg = BuildDynamicNZShapeArg(tensor_type, 0, access_shape);
    const std::string col_arg = BuildDynamicNZShapeArg(tensor_type, 1, access_shape);
    const std::string full_row_arg = BuildDynamicNZTensorDimArg(tensor_type, 0);
    const std::string full_col_arg = BuildDynamicNZTensorDimArg(tensor_type, 1);

    emitter_.EmitLine(
        "using " + shape_type_name + " = pto::TileShape2D<" + element_type +
        ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    emitter_.EmitLine(
        "using " + stride_type_name + " = pto::BaseShape2D<" + element_type +
        ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    emitter_.EmitLine(
        "using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
        stride_type_name + ", Layout::NZ>;");

    std::ostringstream nz_instance;
    nz_instance << global_type_name << " " << var_name << "(";
    if (base_pointer.has_value()) {
        nz_instance << base_pointer.value() << ", " << shape_type_name << "(" << row_arg << ", " << col_arg << "), "
                    << stride_type_name << "(" << full_row_arg << ", " << full_col_arg << ")";
    }
    nz_instance << ");";
    emitter_.EmitLine(nz_instance.str());
}

std::string CCECodegen::BuildGlobalTensorLayoutArg(
    const std::string& stride_type_name, const std::vector<int64_t>& shape_dims, bool is_dn) const
{
    if (*shape_dims.rbegin() == 1 || is_dn) {
        return stride_type_name + ", Layout::DN";
    }
    return stride_type_name;
}

void CCECodegen::EmitNZGlobalTensorDeclaration(
    const TensorDef& def, const std::string& var_name, const ir::TensorTypePtr& tensor_type)
{
    std::optional<std::string> base_pointer;
    if (HasPointer(var_name)) {
        base_pointer = GetPointer(var_name);
    }
    std::optional<std::vector<ir::ExprPtr>> access_shape =
        def.access_shape.empty() ? std::optional<std::vector<ir::ExprPtr>>() :
                                   std::optional<std::vector<ir::ExprPtr>>(def.access_shape);

    std::string element_type = tensor_type->dtype_.ToCTypeString();
    std::string shape_type_name = var_name + "ShapeDim5";
    std::string stride_type_name = var_name + "StrideDim5";
    std::string global_type_name = var_name + "Type";

    bool all_static = true;
    for (const auto& dim : tensor_type->shape_) {
        if (!std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
            all_static = false;
            break;
        }
    }

    // Dynamic NZ tensors use special TileShape2D/BaseShape2D types and a runtime ctor.
    if (!all_static) {
        EmitDynamicNZGlobalTensorDeclaration(
            var_name, tensor_type, base_pointer, access_shape, element_type, shape_type_name, stride_type_name,
            global_type_name);
        return;
    }

    // Static NZ: physical fractal shape {1, cols/c0, rows/16, 16, c0} with fully static strides.
    std::vector<int64_t> tensor_dims = ExtractShapeDimensions(tensor_type->shape_);
    std::vector<int64_t> emitted_shape_dims = BuildNZPhysicalShapeDims(tensor_type, tensor_dims);

    emitter_.EmitLine("using " + shape_type_name + " = " + type_converter_.GenerateShapeType(emitted_shape_dims) + ";");
    emitter_.EmitLine("using " + stride_type_name + " = " + GenerateSingleFileStrideType(emitted_shape_dims) + ";");
    emitter_.EmitLine(
        "using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
        stride_type_name + ", Layout::NZ>;");

    std::ostringstream nz_instance;
    nz_instance << global_type_name << " " << var_name << "(";
    if (base_pointer.has_value()) {
        nz_instance << base_pointer.value();
    }
    nz_instance << ");";
    emitter_.EmitLine(nz_instance.str());
}

void CCECodegen::GenerateGlobalTensorTypeDeclaration(const TensorDef& def)
{
    INTERNAL_CHECK(def.var != nullptr) << "Internal error: TensorDef.var is null";
    std::string var_name = context_.SanitizeName(def.var);
    auto tensor_type = ir::As<ir::TensorType>(def.var->GetType());
    INTERNAL_CHECK(tensor_type != nullptr) << "Internal error: TensorDef.var is not a tensor";

    // NZ tensors follow a distinct fractal layout; handle them entirely in one place.
    if (IsNZTensorType(tensor_type)) {
        EmitNZGlobalTensorDeclaration(def, var_name, tensor_type);
        return;
    }

    // Non-NZ invariants: every accessed tensor has a tile-derived access_shape and a base
    // pointer registered before this point (params at signature build, ptr.make_tensor
    // views at their op).
    INTERNAL_CHECK(HasPointer(var_name)) << "Internal error: tensor '" << var_name << "' has no base pointer";
    INTERNAL_CHECK(!def.access_shape.empty()) << "Internal error: tensor '" << var_name << "' has no access_shape";

    const bool is_dn = def.is_dn;
    const std::optional<std::vector<int>>& tile_dims = def.tile_dims;
    const std::string base_pointer = GetPointer(var_name);

    // shape_dims: tile/access shape, used for the last-dim==1 DN detection and as the
    // row-stride fallback. Must contain integer values (no -1).
    std::vector<int64_t> shape_dims = ExtractShapeDimensions(def.access_shape);
    std::string element_type = tensor_type->dtype_.ToCTypeString();

    std::string shape_type_name = var_name + "ShapeDim5";
    std::string stride_type_name = var_name + "StrideDim5";
    std::string global_type_name = var_name + "Type";

    // Shape and stride are always dynamic: Shape<1,1,1,-1,-1> + Stride<-1,-1,-1,-1,-1>.
    // DIM_3/DIM_4 are configured per access via SetShape; the stride values are supplied
    // through the instance ctor below.
    emitter_.EmitLine(
        "using " + shape_type_name + " = " + type_converter_.GenerateShapeType({-1, -1}) + ";");
    emitter_.EmitLine("using " + stride_type_name + " = pto::Stride<-1, -1, -1, -1, -1>;");

    // Layout: DN if last dim is 1 or if the def is flagged is_dn.
    std::string global_layout_arg = BuildGlobalTensorLayoutArg(stride_type_name, shape_dims, is_dn);
    emitter_.EmitLine(
        "using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
        global_layout_arg + ">;");

    EmitGlobalTensorInstance(
        var_name, global_type_name, shape_type_name, stride_type_name, tensor_type, shape_dims, tile_dims, is_dn,
        base_pointer);
}

} // namespace codegen

} // namespace pypto

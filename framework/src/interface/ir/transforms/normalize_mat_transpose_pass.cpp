/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeMatTransposeLayout
// ===========================
// Lets a matmul operand that lives in the L1 (Mat) buffer be declared in "clear form" -- its
// shape/layout describing the data as produced, not the physical fractal the hardware needs --
// and rewrites it to that physical form. See the pass factory doc in ir/transforms/passes.h.
//
// Background (the hardware constraint)
// ------------------------------------
// The L1(Mat)->L0(Left/Right) move (`pl.move`, lowered to the CANN TMOV tileop) requires src
// and dst to have the SAME physical [Rows, Cols] (a static_assert in the generated kernel). It
// realizes a *logical transpose* only by the NZ-vs-ZN fractal mismatch, never by changing the
// dims. And because `Mat [N,K] NZ` and `Mat [K,N] ZN` are the *same bytes*, a transposed
// operand physically has to be declared `[K,N] ZN` -- a shape that does not match the [N,K]
// data it is filled from. This pass lets you write the [N,K] shape and fixes it up here.
//
// What the frontend writes vs. what this pass emits
// -------------------------------------------------
// Example matmul:  out[M,N] = A[M,K] @ B[K,N], with the RIGHT operand B transposed on-chip.
//
// (1) ON-CHIP (insert) path -- B is produced by the vector core and inserted into L1:
//
//        # clear form: shape [N,K] matches the inserted data d, layout NZ matches the shape
//        rhs_mat = pl.make_tile(pl.TileType(shape=[N, K], layout=pl.NZ,
//                                           target_memory=pl.MemorySpace.Mat), ...)
//        pl.insert(rhs_mat, tile_nz, [off, 0])            # fill L1 from a Vec NZ tile
//        rhs_right = pl.make_tile(... shape=[K, N], Right, layout=pl.ZN)
//        pl.move(rhs_right, rhs_mat)                       # src [N,K] vs dst [K,N]  <-- reversed
//
//     This pass relabels `rhs_mat` to the physical `shape=[K,N], layout=pl.ZN` (swap dims +
//     swap NZ<->ZN, identical bytes), so the move becomes same-shape [K,N]->[K,N] and its
//     fractal mismatch performs the transpose. Emitted tile: Tile<Mat,half,K,N,ZN,...>.
//
// (2) GM normal-load path -- B^T is in GM as b_t=[N,K] and loaded straight into L1:
//
//        b_mat = pl.make_tile(pl.TileType(shape=[N, K], layout=pl.NZ, Mat), ...)   # clear form
//        pl.load(b_mat, b_t, [0, 0])                       # NORMAL load, no is_transpose
//        b_right = pl.make_tile(... shape=[K, N], Right, layout=pl.ZN)
//        pl.move(b_right, b_mat)                            # reversed, same as above
//
//     Relabeling alone is not enough here: the load's GM access window is derived from the
//     tile shape, so swapping `b_mat` to [K,N] would make the load read the [N,K] GM wrong.
//     So the pass ALSO flips this feeding load to `is_transpose=True`, which reads b_t=[N,K]
//     transposed into the [K,N] tile -- exactly the code the explicit is_transpose form below
//     would have produced.
//
// (3) GM transpose-load path -- already physical, LEFT UNTOUCHED:
//
//        b_mat = pl.make_tile(pl.TileType(shape=[K, N], layout=pl.ZN, Mat), ...)   # physical
//        pl.load(b_mat, b_t, [0, 0], is_transpose=True)    # load transposes; Mat holds B
//        pl.move(b_right, b_mat)                            # same-shape [K,N]->[K,N]
//
//     The move is same-shape, so this Mat is never a candidate; the is_transpose feed is also
//     recorded as a hard exclusion so it can never be relabeled by accident.
//
// Two phases
// ----------
//   1. Analysis (TransposeMatCollector): find every `block.move(dst=Left/Right, src=Mat)`
//      whose src 2D shape is the reverse of the dst shape and not square -> candidate Mat.
//      Record Mats fed by an is_transpose load as a hard exclusion.
//   2. Rewrite (TransposeMatRewriter): for every reference to a target allocation swap the two
//      shape dims and swap blayout<->slayout (NZ<->ZN) -- a pure byte relabeling applied to the
//      make_tile def and every reference Var, so it follows tile assignment / control flow. A
//      normal `block.load` feeding a target Mat is additionally flipped to is_transpose.
//
// The pass is a no-op when src.shape == dst.shape (the physical convention every other kernel
// uses) and for square tiles (reverse == itself is ambiguous), so existing kernels are safe.

#include <any>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memory_space.h"
#include "ir/memref.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Return the two static dimensions of a 2D TileType, or nullopt if it is not a
// rank-2 tile with compile-time-constant dims.
std::optional<std::pair<int64_t, int64_t>> Static2DShape(const TileTypePtr& tt)
{
    if (!tt || tt->shape_.size() != 2)
        return std::nullopt;
    auto d0 = As<ConstInt>(tt->shape_[0]);
    auto d1 = As<ConstInt>(tt->shape_[1]);
    if (!d0 || !d1)
        return std::nullopt;
    return std::make_pair(d0->value_, d1->value_);
}

std::optional<MemorySpace> SpaceOf(const TileTypePtr& tt)
{
    if (!tt || !tt->memref_.has_value())
        return std::nullopt;
    return (*tt->memref_)->memorySpace_;
}

// A Mat allocation is uniquely named by its MemRef (name is generated from the
// allocation id), so the name identifies the allocation across every reference.
std::optional<std::string> AllocNameOf(const TileTypePtr& tt)
{
    if (!tt || !tt->memref_.has_value())
        return std::nullopt;
    return (*tt->memref_)->name_;
}

// -------------------------------------------------------------------------------------------
// Phase 1: collect Mat allocations that are clear-form transpose operands.
//
// e.g. for `pl.move(rhs_right[K,N], rhs_mat[N,K])` this records the `rhs_mat` allocation as a
// candidate; a `pl.load(rhs_mat, d, ..., is_transpose=True)` instead records it as excluded.
// -------------------------------------------------------------------------------------------
class TransposeMatCollector : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    std::set<std::string> candidates;       // Mat allocations reached by a reversed Mat->L0 move
    std::set<std::string> transpose_loaded; // Mat allocations already filled by an is_transpose load

    // Targets are reversed-move candidates, minus any Mat that is already fed by an
    // is_transpose load. An is_transpose load fills the Mat in the physical (transposed)
    // form the frontend declared, so relabeling it would double-transpose; such a buffer is
    // used with a same-shape move and never appears here anyway, but excluding it is safe.
    //
    // A Mat fed by a NORMAL load with a reversed move IS a target: it is the clear-form GM
    // path (declare the logical shape/NZ matching the GM tensor, transpose on the way to
    // L0). For those the rewriter both relabels the Mat and flips the feeding load to
    // is_transpose, so the load reads the GM correctly into the relabeled tile -- exactly
    // the code an explicit is_transpose load would have produced.
    std::set<std::string> Targets() const
    {
        std::set<std::string> out;
        for (const auto& name : candidates) {
            if (transpose_loaded.count(name) == 0)
                out.insert(name);
        }
        return out;
    }

    // Var name -> the value it is bound to (from AssignStmts). Lets a tile-group slot access
    // (a GetItemExpr through the group's named tuple) be resolved back to the MakeTuple of slot
    // tiles. Populated while walking; a tile group is always defined before it is used.
    std::unordered_map<std::string, ExprPtr> bindings_;

    void VisitStmt_(const AssignStmtPtr& op) override
    {
        if (op->var_ && op->value_)
            bindings_[op->var_->name_] = op->value_;
        IRVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallPtr& op) override
    {
        IRVisitor::VisitExpr_(op); // keep walking nested exprs

        // A Mat already filled by an is_transpose load carries the physical form; never relabel it.
        // (block.load covers both pl.load and pl.load_tile; arg 0 is the destination tile.)
        if (op->name_ == "block.load" && !op->args_.empty()) {
            if (op->HasKwarg("is_transpose") && op->GetKwarg<bool>("is_transpose")) {
                if (auto dst = As<TileType>(op->args_[0]->GetType()); dst && SpaceOf(dst) == MemorySpace::Mat) {
                    for (const auto& name : MatSlotNames(op->args_[0]))
                        transpose_loaded.insert(name);
                }
            }
            return;
        }

        if (op->name_ != "block.move" || op->args_.size() < 2)
            return;
        auto dst = As<TileType>(op->args_[0]->GetType());
        auto src = As<TileType>(op->args_[1]->GetType());
        if (!dst || !src)
            return;

        auto dst_space = SpaceOf(dst);
        auto src_space = SpaceOf(src);
        if (src_space != MemorySpace::Mat)
            return;
        if (dst_space != MemorySpace::Left && dst_space != MemorySpace::Right)
            return;

        auto src_shape = Static2DShape(src);
        auto dst_shape = Static2DShape(dst);
        if (!src_shape || !dst_shape)
            return;

        // Clear form: src is the reverse of dst and not square. e.g. src Mat [N,K] -> dst
        // Right [K,N] is reversed (a clear-form transpose operand); src [K,N] -> dst [K,N] is
        // same-shape (the physical convention every other kernel uses); a square [D,D] tile is
        // ambiguous because reverse == itself. Only the reversed, non-square case is rewritten.
        bool reversed = src_shape->first == dst_shape->second && src_shape->second == dst_shape->first;
        bool square = src_shape->first == src_shape->second;
        if (!reversed || square)
            return;

        // A plain tile contributes one name; a tile-group slot access contributes every sibling
        // slot, since a rotating group shares one TileType and must be relabeled as a whole.
        for (const auto& name : MatSlotNames(op->args_[1]))
            candidates.insert(name);
    }

    // Follow Var bindings and constant tuple indices to the MakeTuple that `e` refers to.
    MakeTuplePtr ResolveToMakeTuple(const ExprPtr& e, int depth = 0) const
    {
        if (!e || depth > 32)
            return nullptr;
        if (auto mt = As<MakeTuple>(e))
            return mt;
        if (auto v = As<Var>(e)) {
            auto it = bindings_.find(v->name_);
            return it == bindings_.end() ? nullptr : ResolveToMakeTuple(it->second, depth + 1);
        }
        if (auto gi = As<GetItemExpr>(e)) {
            if (auto ci = As<ConstInt>(gi->slice_)) {
                auto inner = ResolveToMakeTuple(gi->value_, depth + 1);
                if (inner && ci->value_ >= 0 && ci->value_ < static_cast<int64_t>(inner->elements_.size()))
                    return ResolveToMakeTuple(inner->elements_[ci->value_], depth + 1);
            }
        }
        return nullptr;
    }

    // The Mat allocation name(s) a tile operand refers to. For a tile-group slot access
    // `GetItemExpr(tiles, idx)` this resolves `tiles` to its MakeTuple and returns every slot's
    // name; otherwise it is the single name on the operand's own TileType.
    std::set<std::string> MatSlotNames(const ExprPtr& tile_expr) const
    {
        std::set<std::string> out;
        // `b_mat = grp.next()` binds the operand to a Var, so follow the binding chain to the
        // underlying slot access before checking whether it is a tile-group GetItemExpr.
        ExprPtr e = tile_expr;
        for (int depth = 0; depth < 32; ++depth) {
            auto v = As<Var>(e);
            if (!v)
                break;
            auto it = bindings_.find(v->name_);
            if (it == bindings_.end())
                break;
            e = it->second;
        }
        if (auto gi = As<GetItemExpr>(e)) {
            if (auto tiles = ResolveToMakeTuple(gi->value_)) {
                for (const auto& elem : tiles->elements_) {
                    if (auto tt = As<TileType>(elem->GetType()))
                        if (auto name = AllocNameOf(tt))
                            out.insert(*name);
                }
                if (!out.empty())
                    return out;
            }
        }
        if (auto tt = As<TileType>(tile_expr->GetType()))
            if (auto name = AllocNameOf(tt))
                out.insert(*name);
        return out;
    }
};

// -------------------------------------------------------------------------------------------
// Phase 2: relabel every reference to a recorded allocation.
//
// For a target `rhs_mat` declared `TileType(shape=[N,K], layout=pl.NZ)` this rewrites the
// make_tile def, and every Var that refers to it, to `shape=[K,N], layout=pl.ZN` (same bytes),
// and flips any normal `pl.load` into it to is_transpose. Downstream ops (`pl.insert`,
// `pl.move`, `pl.matmul`) then see a Mat whose physical shape matches its L0 destination.
// -------------------------------------------------------------------------------------------
class TransposeMatRewriter : public IRMutator {
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

public:
    explicit TransposeMatRewriter(std::set<std::string> targets) : targets_(std::move(targets)) {}

    ExprPtr VisitExpr_(const VarPtr& op) override
    {
        auto swapped = SwapContainedTargets(op->GetType());
        if (swapped.get() != op->GetType().get()) {
            auto it = var_cache_.find(op.get());
            if (it != var_cache_.end())
                return it->second;
            auto rebuilt = std::make_shared<const Var>(op->name_, swapped, op->span_);
            var_cache_[op.get()] = rebuilt;
            return rebuilt;
        }
        return IRMutator::VisitExpr_(op);
    }

    // Swap target TileTypes appearing directly in a type: either the type itself (a slot Var) or
    // the direct elements of a positional tuple. A tile group's `tiles` tuple is such a tuple, and
    // codegen sizes the double-buffer array from that Var's declared element types, so it must be
    // swapped too. Nested/named tuples (the group's {tiles, mutex_ids, cursor}) are left intact so
    // their pointer-keyed field-name side table stays valid; their `tiles` access resolves to the
    // swapped tiles Var directly.
    TypePtr SwapContainedTargets(const TypePtr& type) const
    {
        if (auto tt = As<TileType>(type))
            return IsTarget(tt) ? std::static_pointer_cast<const Type>(SwapTileType(tt)) : type;
        if (auto tup = As<TupleType>(type)) {
            std::vector<TypePtr> elems;
            elems.reserve(tup->types_.size());
            bool changed = false;
            for (const auto& e : tup->types_) {
                if (auto et = As<TileType>(e); et && IsTarget(et)) {
                    elems.push_back(SwapTileType(et));
                    changed = true;
                } else {
                    elems.push_back(e);
                }
            }
            if (!changed)
                return type;
            return std::make_shared<const TupleType>(std::move(elems));
        }
        return type;
    }

    ExprPtr VisitExpr_(const CallPtr& op) override
    {
        auto base = IRMutator::VisitExpr_(op);
        auto call = As<Call>(base);
        if (!call)
            return base;

        if (call->name_ == "block.make_tile") {
            auto tt = As<TileType>(call->GetType());
            // Re-validate 2D-ness here rather than trusting the name alone: SwapTileType indexes
            // shape_[0]/[1], so a target must be a rank-2 tile with static dims.
            if (!tt || !IsTarget(tt) || !Static2DShape(tt))
                return base;

            // Rewrite the make_tile so its args stay consistent with the relabeled type:
            // swap the shape (and valid_shape) tuple dims and swap the blayout/slayout kwargs.
            std::vector<ExprPtr> new_args = call->args_;
            if (!new_args.empty())
                new_args[0] = SwapTuple(new_args[0]);
            if (new_args.size() >= 2)
                new_args[1] = SwapTuple(new_args[1]);

            std::vector<std::pair<std::string, std::any>> new_kwargs = SwapLayoutKwargs(call->kwargs_);

            return std::make_shared<const Call>(call->name_, std::move(new_args), std::move(new_kwargs),
                                                SwapTileType(tt), call->span_);
        }

        // A normal load feeding a relabeled Mat must become an is_transpose load. The make_tile
        // above swapped the Mat to the L0 orientation (e.g. [N,K] -> [K,N]), but the load's GM
        // access window is derived from the tile shape, so a plain load would now read the
        // [N,K] GM tensor as if it were [K,N]. Flipping is_transpose makes it read the GM
        // transposed into the swapped tile -- i.e. `pl.load(b_mat, b_t, [0,0])` becomes the
        // equivalent of `pl.load(b_mat, b_t, [0,0], is_transpose=True)`.
        if (call->name_ == "block.load" && !call->args_.empty()) {
            auto dst = As<TileType>(call->args_[0]->GetType());
            bool already_transpose = call->HasKwarg("is_transpose") && call->GetKwarg<bool>("is_transpose");
            if (dst && IsTarget(dst) && !already_transpose) {
                std::vector<std::pair<std::string, std::any>> new_kwargs = call->kwargs_;
                new_kwargs.emplace_back("is_transpose", std::any(true));
                return std::make_shared<const Call>(call->name_, call->args_, std::move(new_kwargs), call->GetType(),
                                                    call->span_);
            }
        }

        return base;
    }

private:
    bool IsTarget(const TileTypePtr& tt) const
    {
        auto name = AllocNameOf(tt);
        return name && targets_.count(*name) > 0;
    }

    // Swap the two dims of a rank-2 shape/valid_shape MakeTuple. Leaves anything
    // else (e.g. an empty valid_shape tuple) untouched.
    static ExprPtr SwapTuple(const ExprPtr& expr)
    {
        auto tuple = As<MakeTuple>(expr);
        if (!tuple || tuple->elements_.size() != 2)
            return expr;
        std::vector<ExprPtr> swapped = {tuple->elements_[1], tuple->elements_[0]};
        return std::make_shared<const MakeTuple>(std::move(swapped), tuple->span_);
    }

    // NZ is (blayout, slayout) = (col_major, row_major); ZN is the swap. Exchanging the
    // two kwargs flips NZ<->ZN, matching the TileType relabeling.
    static std::vector<std::pair<std::string, std::any>> SwapLayoutKwargs(
        const std::vector<std::pair<std::string, std::any>>& kwargs)
    {
        std::any blayout;
        std::any slayout;
        bool has_b = false;
        bool has_s = false;
        for (const auto& [k, v] : kwargs) {
            if (k == "blayout") {
                blayout = v;
                has_b = true;
            } else if (k == "slayout") {
                slayout = v;
                has_s = true;
            }
        }
        if (!has_b || !has_s)
            return kwargs;
        std::vector<std::pair<std::string, std::any>> out = kwargs;
        for (auto& [k, v] : out) {
            if (k == "blayout")
                v = slayout;
            else if (k == "slayout")
                v = blayout;
        }
        return out;
    }

    // Relabel a tile to its byte-identical transpose: swap the shape dims, swap the
    // valid_shape dims, and swap blayout<->slayout (flips NZ<->ZN). The MemRef (address/size/
    // space/id) is preserved, so this is a pure relabeling of the same L1 bytes -- e.g.
    // `Mat [N,K] NZ` -> `Mat [K,N] ZN`.
    TileTypePtr SwapTileType(const TileTypePtr& tt) const
    {
        std::vector<ExprPtr> new_shape = {tt->shape_[1], tt->shape_[0]};

        std::optional<TileView> new_view = tt->tileView_;
        if (new_view.has_value() && new_view->validShape.size() == 2)
            std::swap(new_view->validShape[0], new_view->validShape[1]);

        HardwareInfo hw = tt->hardwareInfo_.value_or(HardwareInfo{});
        std::swap(hw.blayout, hw.slayout);

        return std::make_shared<const TileType>(std::move(new_shape), tt->dtype_, tt->memref_, std::move(new_view),
                                                std::make_optional(hw));
    }

    std::set<std::string> targets_;
    std::unordered_map<const Expr*, ExprPtr> var_cache_;
};

FunctionPtr TransformNormalizeMatTranspose(const FunctionPtr& func)
{
    if (!func || !func->body_)
        return func;

    TransposeMatCollector collector;
    collector.VisitStmt(func->body_);
    auto targets = collector.Targets();
    if (targets.empty())
        return func;

    TransposeMatRewriter rewriter(std::move(targets));
    auto new_body = rewriter.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get())
        return func;
    return std::make_shared<Function>(func->name_, func->params_, func->returnTypes_, new_body, func->span_,
                                      func->funcType_);
}

} // namespace

namespace pass {

Pass NormalizeMatTransposeLayout()
{
    return CreateFunctionPass(TransformNormalizeMatTranspose, "NormalizeMatTransposeLayout");
}

} // namespace pass

} // namespace ir
} // namespace pypto

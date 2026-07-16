/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_PASSES_H_
#define PYPTO_IR_TRANSFORMS_PASSES_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ir/function.h"
#include "ir/program.h"
#include "ir/transforms/ir_property.h"
#include "ir/transforms/pass_context.h"

namespace pypto {
namespace ir {

/**
 * \brief Internal base class for pass implementations
 *
 * Most passes should use CreateFunctionPass() or CreateProgramPass() helpers.
 * Only inherit from PassImpl for complex passes with custom state.
 */
class PassImpl {
public:
    virtual ~PassImpl() = default;

    /**
     * \brief Execute the pass on a program
     */
    virtual ProgramPtr operator()(const ProgramPtr& program) = 0;

    /**
     * \brief Get the name of the pass (for debugging)
     */
    [[nodiscard]] virtual std::string GetName() const { return "UnnamedPass"; }

    /**
     * \brief Get properties required before this pass can run
     */
    [[nodiscard]] virtual IRPropertySet GetRequiredProperties() const { return {}; }

    /**
     * \brief Get properties produced (guaranteed) after this pass runs
     */
    [[nodiscard]] virtual IRPropertySet GetProducedProperties() const { return {}; }

    /**
     * \brief Get properties invalidated (broken) by this pass
     */
    [[nodiscard]] virtual IRPropertySet GetInvalidatedProperties() const { return {}; }
};

/**
 * \brief Base class for IR transformation passes
 *
 * Pass uses a pimpl pattern to hide implementation details.
 * Users should create passes using factory functions.
 */
class Pass {
public:
    Pass();
    explicit Pass(std::shared_ptr<PassImpl> impl);
    ~Pass();

    // Copy and move
    Pass(const Pass& other);
    Pass& operator=(const Pass& other);
    Pass(Pass&& other) noexcept;
    Pass& operator=(Pass&& other) noexcept;

    /**
     * \brief Execute the pass on a program (primary API)
     */
    ProgramPtr operator()(const ProgramPtr& program) const;

    /**
     * \brief Execute the pass on a program (backward compatible API)
     */
    [[nodiscard]] ProgramPtr run(const ProgramPtr& program) const;

    /**
     * \brief Get the name of the pass
     */
    [[nodiscard]] std::string GetName() const;

    /**
     * \brief Get properties required before this pass can run
     */
    [[nodiscard]] IRPropertySet GetRequiredProperties() const;

    /**
     * \brief Get properties produced (guaranteed) after this pass runs
     */
    [[nodiscard]] IRPropertySet GetProducedProperties() const;

    /**
     * \brief Get properties invalidated (broken) by this pass
     */
    [[nodiscard]] IRPropertySet GetInvalidatedProperties() const;

private:
    std::shared_ptr<PassImpl> impl_;
};

// Factory functions for built-in passes
namespace pass {

/**
 * \brief Create a pass from a function-level transform function (RECOMMENDED)
 *
 * \param transform Function that transforms a Function
 * \param name Optional name for the pass (for debugging)
 * \param properties Optional property declarations
 * \return Pass that applies the transform to each function
 */
Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform, const std::string& name = "",
                        const PassProperties& properties = {});

/**
 * \brief Create a pass from a program-level transform function
 *
 * \param transform Function that transforms a Program
 * \param name Optional name for the pass (for debugging)
 * \param properties Optional property declarations
 * \return Pass that applies the transform
 */
Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name = "",
                       const PassProperties& properties = {});

/**
 * \brief Create an init memref pass
 *
 * Initializes MemRef for all variables in functions.
 * Sets memory space to UB by default, or DDR for block.load/block.store operands.
 */
Pass InitMemRef();

/**
 * \brief Create a basic memory reuse pass
 *
 * Uses dependency analysis to identify memory reuse opportunities.
 * Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.
 */
Pass BasicMemoryReuse();

/**
 * \brief Create an allocate memory address pass
 *
 * Allocates real memory addresses for existing alloc operations.
 * Updates MemRef addresses and alloc statement arguments in place.
 */
Pass AllocateMemoryAddr();

/**
 * \brief Create an SSA conversion pass
 */
Pass ConvertToSSA();

/**
 * \brief Outline InCore scopes into separate functions
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions
 */
Pass OutlineIncoreScopes();

/**
 * \brief Convert tensor ops to block ops in InCore functions
 *
 * Inserts block.load at InCore function entry, converts tensor ops to block ops
 * using the OpConversionRegistry, inserts block.store at exit, and updates
 *
 * Requirements:
 * - Input IR must have InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass ConvertTensorToBlockOps();

/**
 * \brief Create a verifier pass with configurable rules
 *
 * \param disabled_rules Vector of rule names to disable
 * \return Pass that runs IR verification
 */
Pass RunVerifier(const std::vector<std::string>& disabled_rules = {});

/**
 * \brief Create a pass that lowers break/continue statements to structured control flow
 *
 * Transforms BreakStmt and ContinueStmt into nested scf.if blocks suitable for MLIR codegen.
 * - continue: wrapped in if(!cond) guard over remaining statements
 * - break (for): adds a _can_continue boolean iter_arg; body guarded by scf.if(_can_continue)
 * - break (while): adds a _can_continue iter_arg as sole before-region condition;
 *     original condition checked via scf.if at the start of the do-region (no And/Or/Not)
 *
 * Must run before ConvertToSSA and before codegen.
 */
Pass LowerBreakContinue();

/**
 * \brief Create a constant folding and simplification pass
 *
 * Folds constant arithmetic (ConstInt + ConstInt → ConstInt), simplifies
 * if-stmts with constant conditions or identical branches, and removes
 * dead comparisons.  Reduces scalar instruction count significantly.
 * Should run after ConvertToSSA.
 */
Pass ConstFoldAndSimplify();

/**
 * \brief Create an aggressive dead code elimination pass
 *
 * Removes statements whose defined variables are not transitively used by
 * any side-effect operation (e.g., OP_ASSEMBLE writing to input-argument
 * memory), return, yield, or control-flow statement.
 * Iterates to a fixed point so dead chains collapse fully.
 */
Pass AggressiveDCE();

/**
 * \brief Create a loop/if canonicalization pass
 *
 * Removes unused IterArgs/returnVars from ForStmt/WhileStmt and
 * unused returnVars from IfStmt. Also filters corresponding
 * YieldStmt/BreakStmt/ContinueStmt values.
 */
Pass Canonicalize();

/**
 * \brief Create a token-dependency pass
 *
 * Adds token edges (producer result_token_ -> consumer tokens_) to serialize
 * memory hazards between operations whose LogicalTensors share the same
 * RawTensor:
 *   - WAW: two writes (Assemble/Assemble_SSA) to overlapping regions.
 *   - WAR: a write that overlaps a prior read.
 * Reads are full unless the operand is read through a View, in which case the
 * partial region described by ViewOpAttribute is used. Overlap is decided with
 * the same per-dimension symbolic check as InferWriteConflict::MayOverlap.
 * Control flow is not considered; the function body is treated as a flat
 * sequence of operations.
 */
Pass TokenPass();

/**
 * \brief Create a pass that flattens nested call expressions
 */
Pass FlattenCallExpr();

/**
 * \brief Create a pass that normalizes statement structure
 */
Pass NormalizeStmtStructure();

/**
 * \brief Create a pass that recursively flattens single-statement blocks
 */
Pass FlattenSingleStmt();

Pass MergeStmtsIntoIf();

Pass CreateRootFunctions();

/**
 * \brief Finalize dynamic functions built from new IR (post create_root_functions).
 */
Pass FinalizeDynamicFunction();

/**
 * \brief Normalize "clear-form" Mat matmul operands to the physical layout pto-isa needs.
 *
 * The L1(Mat)->L0(Left/Right) move (`block.move`) requires the source and destination
 * tiles to have the SAME physical [Rows, Cols] (the CANN TMOV tileop static-asserts on it).
 * Because `Mat [N,K] NZ` and `Mat [K,N] ZN` are the *same bytes*, a matmul operand that is
 * logically transposed on-chip previously had to be declared in the physical form
 * `Mat [K,N] ZN`, whose shape does not match the source data it is filled from.
 *
 * This pass lets the frontend declare such an operand in "clear form" — shape equal to the
 * data it holds, layout matching that shape (e.g. `Mat [N,K] NZ` for data `d=[N,K]`) — and
 * rewrites it back to the physical form. It detects a `block.move(dst=Left/Right, src=Mat)`
 * whose src shape is the reverse of the dst shape (and non-square), then relabels that Mat
 * allocation: swap the two shape dims and swap blayout<->slayout (flips NZ<->ZN). The bytes
 * are unchanged, so codegen and the hardware behave exactly as with the explicit physical form.
 *
 * The rewrite follows the allocation by MemRef identity, so every reference Var is updated
 * regardless of tile-group assignment or control flow. It is a no-op when src.shape ==
 * dst.shape (the convention every other kernel uses), so existing kernels are unaffected.
 *
 * Must run before ConvertToSSA (and after helper inlining so the move and make_tile are in
 * the same function).
 */
Pass NormalizeMatTransposeLayout();
} // namespace pass

/**
 * \brief A pipeline of passes executed in sequence
 *
 * PassPipeline maintains an ordered sequence of passes and executes them in order.
 * Instrumentation (verification, logging, etc.) is handled by PassContext and its
 * PassInstruments — the pipeline itself is a simple pass list.
 *
 * Usage:
 * @code
 *   PassPipeline pipeline;
 *   pipeline.AddPass(pass::ConvertToSSA());
 *   pipeline.AddPass(pass::FlattenCallExpr());
 *   pipeline.AddPass(pass::RunVerifier());
 *   auto result = pipeline.Run(program);
 * @endcode
 */
class PassPipeline {
public:
    PassPipeline();

    /**
     * \brief Add a pass to the pipeline
     */
    void AddPass(Pass pass);

    /**
     * \brief Execute all passes in sequence
     * \param program Input program
     * \return Transformed program
     */
    [[nodiscard]] ProgramPtr Run(const ProgramPtr& program) const;

    /**
     * \brief Get the names of all passes in the pipeline
     */
    [[nodiscard]] std::vector<std::string> GetPassNames() const;

private:
    std::vector<Pass> passes_;
};

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_PASSES_H_

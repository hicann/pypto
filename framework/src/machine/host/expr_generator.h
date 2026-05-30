/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file backend_expr_generator.h
 * \brief Expression batch generator for splitting large control flow functions
 */

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "tilefwk/error.h"
#include "tilefwk/pypto_fwk_log.h"
#include "interface/tensor/symbolic_scalar.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace {
// Maximum expressions per batch/file
constexpr size_t EXPRS_PER_BATCH = 1000;
} // namespace
constexpr size_t TABSIZE = 2;

// Expression batch information
struct ExprBatchInfo {
    int devRootKey;
    size_t batchIndex;
    size_t startExprIndex;
    size_t endExprIndex;
    size_t totalExprs;
    std::string fileName;
    std::string functionName;
};

// Generator for expression batches
class ExprBatchGenerator {
public:
    ExprBatchGenerator(const std::string& outputDir, int devRootKey, size_t totalExprs)
        : outputDir_(outputDir), devRootKey_(devRootKey), totalExprs_(totalExprs)
    {
        CalculateBatches();
    }

    void HeaderFileBegin(std::ostringstream& out) const
    {
        out << "#pragma once\n"
            << "#include <cstdint>\n\n"
            << "namespace npu::tile_fwk {\n\n";
        GenerateLinkScript();
    }

    void HeaderFileEnd(std::ostringstream& out) const
    {
        std::string headerPath = outputDir_ + "/control_flow_expr_table.h";
        std::ofstream header(headerPath);
        if (!header.is_open()) {
            ASSERT(DevCommonErr::FILE_ERROR, false) << "File batch_expr.h open failed!";
            return;
        }
        out << "\n} // namespace npu::tile_fwk\n";
        header << out.str();
        header.close();
    }

    template <typename ExpressionSet>
    void GenerateBatchFile(
        SymbolicExpressionTable* exprTable, std::ostringstream& controlFlowOss, std::ostringstream& exprHeaderOss,
        const std::string& expName, const ExpressionSet& expressions, std::vector<std::string>& exprSrcFiles,
        int indent, int devRootKey)
    {
        for (auto& batch : batches_) {
            std::string filePath = outputDir_ + "/" + batch.fileName;
            std::ofstream out(filePath);
            if (!out.is_open()) {
                ASSERT(DevCommonErr::FILE_ERROR, false) << "File set_expr open failed!";
                return;
            }
            // Write file header
            out << "#define __TILE_FWK_AICPU__ 1\n"
                << "#include <stdint.h>\n\n"
                << "#include \"" << expName << "\"\n"
                << "#include \"tilefwk/aikernel_data.h\"\n"
                << "#include \"tilefwk/aicpu_runtime.h\"\n"
                << "#include \"tilefwk/aicpu_distributed.h\"\n"
                << "namespace npu::tile_fwk {\n\n"
                << "__attribute__((section(\".pypto.func\")))\n"
                << "void " << batch.functionName
                << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase "
                   "*startArgs, uint64_t *exprList) {\n";
            EmitBatchBody(out, exprTable, expressions, batch.startExprIndex, batch.endExprIndex);
            out << "}\n\n"
                << "} // namespace npu::tile_fwk\n";
            out.close();
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << batch.functionName
                           << "(ctx, symbolTable, runtimeCallList, startArgs, exprList" << devRootKey << ");\n";
            exprSrcFiles.emplace_back(filePath);
            exprHeaderOss << "void " << batch.functionName
                          << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], "
                             "DevStartArgsBase *startArgs, uint64_t *exprList);\n";
        }
        return;
    }

private:
    // 同一段内的一处差异叶子，多处差异共用单循环变量 k：值轨迹 firstImm + step * k。
    struct TrackedDiff {
        std::vector<int> path;
        int64_t firstImm;
        int64_t step;
        int64_t lastImm;
    };

    // 寻找模板相同、若干立即数同步等差的最长连续段，返回段长（≥1）。
    template <typename ExpressionSet>
    size_t DetectArithmeticRun(
        const ExpressionSet& expressions, size_t runStart, size_t batchEnd, std::vector<TrackedDiff>& tracked) const
    {
        tracked.clear();
        if (runStart + 1 >= batchEnd) { return 1; }
        std::vector<SymbolicExpressionTable::ImmediateDiff> headDiffs;
        if (!SymbolicExpressionTable::FindAllImmediateDifferences(
                expressions[runStart], expressions[runStart + 1], headDiffs)) {
            return 1;
        }
        if (headDiffs.empty()) {
            return 1;
        }
        tracked.reserve(headDiffs.size());
        for (const auto& d : headDiffs) {
            tracked.push_back({d.path, d.immLhs, d.immRhs - d.immLhs, d.immRhs});
        }
        std::sort(tracked.begin(), tracked.end(),
            [](const TrackedDiff& a, const TrackedDiff& b) { return a.path < b.path; });

        size_t runLength = 2;
        for (size_t cursor = runStart + 2; cursor < batchEnd; cursor++) {
            std::vector<SymbolicExpressionTable::ImmediateDiff> pairDiffs;
            if (!SymbolicExpressionTable::FindAllImmediateDifferences(
                    expressions[cursor - 1], expressions[cursor], pairDiffs)) {
                break;
            }
            if (pairDiffs.size() != tracked.size()) {
                break;
            }
            std::sort(pairDiffs.begin(), pairDiffs.end(),
                [](const SymbolicExpressionTable::ImmediateDiff& a,
                    const SymbolicExpressionTable::ImmediateDiff& b) { return a.path < b.path; });
            bool ok = true;
            for (size_t i = 0; i < pairDiffs.size(); i++) {
                if (pairDiffs[i].path != tracked[i].path || pairDiffs[i].immLhs != tracked[i].lastImm ||
                    pairDiffs[i].immRhs - pairDiffs[i].immLhs != tracked[i].step) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }
            for (size_t i = 0; i < pairDiffs.size(); i++) {
                tracked[i].lastImm = pairDiffs[i].immRhs;
            }
            runLength++;
        }
        return runLength;
    }

    // 构造 (firstImm + step * loopVar) 的占位 AST，step==1 / firstImm==0 会化简。
    static RawSymbolicScalarPtr BuildLoopAffinePlaceholder(
        int64_t firstImm, int64_t step, const RawSymbolicScalarPtr& loopVarNode)
    {
        RawSymbolicScalarPtr kPart = loopVarNode;
        if (step != 1) {
            auto stepImm = RawSymbolicImmediate::Create(step);
            kPart = std::make_shared<RawSymbolicExpression>(
                SymbolicOpcode::T_BOP_MUL, std::vector<RawSymbolicScalarPtr>{stepImm, loopVarNode});
        }
        if (firstImm == 0) {
            return kPart;
        }
        auto firstImmNode = RawSymbolicImmediate::Create(firstImm);
        return std::make_shared<RawSymbolicExpression>(
            SymbolicOpcode::T_BOP_ADD, std::vector<RawSymbolicScalarPtr>{firstImmNode, kPart});
    }

    // 输出折叠后的 for 循环：单差异沿用紧凑头，多差异切到计数头并展开每处 affine 子树。
    void EmitFoldedLoop(
        std::ostream& out, const RawSymbolicScalarPtr& firstExpr, const std::vector<TrackedDiff>& tracked,
        size_t startExprIdx, size_t runLength, size_t loopId) const
    {
        FE_ASSERT(!tracked.empty());
        // sym_ 前缀让 BuildSymbolName 不再附加 VALUE_。
        std::string loopVarName = "sym_expr_loop_k_" + std::to_string(loopId);
        auto loopVarNode = RawSymbolicSymbol::Create(loopVarName);

        std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>> replacements;
        replacements.reserve(tracked.size());
        const bool singleDiff = tracked.size() == 1;
        for (const auto& d : tracked) {
            if (singleDiff) {
                replacements.emplace_back(d.path, loopVarNode);
            } else {
                replacements.emplace_back(d.path, BuildLoopAffinePlaceholder(d.firstImm, d.step, loopVarNode));
            }
        }
        std::string loopBodyTemplate = SymbolicExpressionTable::BuildExpressionWithPlaceholders(firstExpr, replacements);

        out << "    {\n"
            << "        int64_t __expr_loop_idx = " << static_cast<long long>(startExprIdx) << ";\n";
        if (singleDiff) {
            int64_t firstImm = tracked.front().firstImm;
            int64_t step = tracked.front().step;
            int64_t lastImmValue = firstImm + step * static_cast<int64_t>(runLength - 1);
            out << "        for (int64_t " << loopVarName << " = " << static_cast<long long>(firstImm) << "; "
                << loopVarName << " <= " << static_cast<long long>(lastImmValue) << "; " << loopVarName
                << " += " << static_cast<long long>(step) << ") {\n";
        } else {
            out << "        for (int64_t " << loopVarName << " = 0; " << loopVarName << " < "
                << static_cast<long long>(runLength) << "; " << loopVarName << "++) {\n";
        }
        out << "            RUNTIME_SetExpr(exprList, __expr_loop_idx++, " << loopBodyTemplate << ");\n"
            << "        }\n"
            << "    }\n";
    }

    // 主驱动：扫到等差段就折叠，否则单行展开。
    template <typename ExpressionSet>
    void EmitBatchBody(
        std::ostream& out, SymbolicExpressionTable* exprTable, const ExpressionSet& expressions, size_t batchStart,
        size_t batchEnd) const
    {
        constexpr size_t MIN_LOOP_LEN = 3;
        size_t exprIdx = batchStart;
        size_t loopId = 0;
        while (exprIdx < batchEnd) {
            std::vector<TrackedDiff> tracked;
            size_t runLength = DetectArithmeticRun(expressions, exprIdx, batchEnd, tracked);
            if (runLength >= MIN_LOOP_LEN) {
                EmitFoldedLoop(out, expressions[exprIdx], tracked, exprIdx, runLength, loopId++);
                exprIdx += runLength;
            } else {
                auto exprStr = exprTable->BuildExpression(expressions[exprIdx]);
                out << "    RUNTIME_SetExpr(exprList, " << exprIdx << ", " << exprStr << ");\n";
                exprIdx++;
            }
        }
    }


    void CalculateBatches()
    {
        size_t numBatches = (totalExprs_ + EXPRS_PER_BATCH - 1) / EXPRS_PER_BATCH;

        for (size_t i = 0; i < numBatches; ++i) {
            ExprBatchInfo batch;
            batch.devRootKey = devRootKey_;
            batch.batchIndex = i;
            batch.startExprIndex = i * EXPRS_PER_BATCH;
            batch.endExprIndex = std::min(batch.startExprIndex + EXPRS_PER_BATCH, totalExprs_);
            batch.totalExprs = totalExprs_;
            batch.fileName =
                "control_flow_expr_table_" + std::to_string(devRootKey_) + "_" + std::to_string(i) + ".cpp";
            batch.functionName = "SetExprBatch_" + std::to_string(devRootKey_) + "_" + std::to_string(i);
            batches_.emplace_back(batch);
        }
    }
    void GenerateLinkScript() const
    {
        std::string scriptFile = outputDir_ + "/merge.link";
        std::ofstream file(scriptFile);
        if (!file.is_open()) {
            ASSERT(DevCommonErr::FILE_ERROR, false) << "File merge.link open failed!";
            return;
        }
        file << "SECTIONS\n{\n"
             << "    . = 0x10000;\n" // align 4K
             << "    _start = .;\n"
             << "    .pypto : { *(.pypto.entry) *(.pypto.func) *(.rodata.*) }\n}\n";
        file.close();
    }
    std::string outputDir_;
    int devRootKey_;
    size_t totalExprs_;
    std::vector<ExprBatchInfo> batches_;
};

} // namespace npu::tile_fwk

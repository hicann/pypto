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

namespace npu::tile_fwk {
namespace {
// Maximum expressions per batch/file
constexpr size_t EXPRS_PER_BATCH = 1000;
}
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
        : outputDir_(outputDir), devRootKey_(devRootKey), totalExprs_(totalExprs) {
        CalculateBatches();
    }

    void HeaderFileBegin(std::ostringstream &exprHeaderOss) const {
        exprHeaderOss << "#pragma once\n"
                       << "#include <cstdint>\n\n"
                       << "namespace npu::tile_fwk {\n\n";
        GenerateLinkScript();
    }

    void HeaderFileEnd(std::ostringstream &exprHeaderOss) const {
        std::string headerPath = outputDir_ + "/control_flow_expr_table.h";
        std::ofstream header(headerPath);
        if (!header.is_open()) {
            ASSERT(false) << "File batch_expr.h open failed!";
            return;
        }
        exprHeaderOss << "\n} // namespace npu::tile_fwk\n";
        header << exprHeaderOss.str();
        header.close();
    }

    template<typename ExpressionSet, typename BuildExpressionFunc>
    void GenerateBatchFile(std::ostringstream &controlFlowOss, std::ostringstream &exprHeaderOss, const std::string &expName,
        const ExpressionSet& expressions, std::vector<std::string> &exprSrcFiles, int indent, int devRootKey, const BuildExpressionFunc& buildExpr) {
        for (auto& batch : batches_) {
            std::string filePath = outputDir_ + "/" + batch.fileName;
            std::ofstream out(filePath);
            if (!out.is_open()) {
                ASSERT(false) << "File set_expr open failed!";
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
                << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs, uint64_t *exprList) {\n";
            for (size_t idx = batch.startExprIndex; idx < batch.endExprIndex; idx++) {
                const auto& expr = expressions[idx];
                auto exprStr = buildExpr(expr);
                out << "    RUNTIME_SetExpr(exprList, " << idx << ", " << exprStr << ");\n";
            }
            out << "}\n\n"
                << "} // namespace npu::tile_fwk\n";
            out.close();
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' 
                << batch.functionName << "(ctx, symbolTable, runtimeCallList, startArgs, exprList" << devRootKey <<");\n";
            exprSrcFiles.emplace_back(filePath);
            exprHeaderOss << "void " << batch.functionName
                << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs, uint64_t *exprList);\n";
        }
        return;
    }

private:
    void CalculateBatches() {
        size_t numBatches = (totalExprs_ + EXPRS_PER_BATCH - 1) / EXPRS_PER_BATCH;
        
        for (size_t i = 0; i < numBatches; ++i) {
            ExprBatchInfo batch;
            batch.devRootKey = devRootKey_;
            batch.batchIndex = i;
            batch.startExprIndex = i * EXPRS_PER_BATCH;
            batch.endExprIndex = std::min(batch.startExprIndex + EXPRS_PER_BATCH, totalExprs_);
            batch.totalExprs = totalExprs_;
            batch.fileName = "control_flow_expr_table_" + std::to_string(devRootKey_) + "_" + std::to_string(i) + ".cpp";
            batch.functionName = "SetExprBatch_" + std::to_string(devRootKey_) + "_" + std::to_string(i);
            batches_.emplace_back(batch);
        }
    }
    void GenerateLinkScript() const {
        std::string scriptFile = outputDir_ + "/merge.link";
        std::ofstream file(scriptFile);
        if (!file.is_open()) {
            ASSERT(false) << "File merge.link open failed!";
            return;
        }
        file << "SECTIONS\n{\n"
             << "    . = 0x10000;\n"  // align 4K
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

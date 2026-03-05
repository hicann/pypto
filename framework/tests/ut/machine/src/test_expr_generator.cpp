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
#include "machine/host/expr_generator.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace npu::tile_fwk {
namespace test {

class TestExprBatchGenerator : public testing::Test {
protected:
    void SetUp() override {
        testDir_ = "test_expr_generator_temp";
        mkdir(testDir_.c_str(), 0755);
    }

    void TearDown() override {
        std::string cmd = "rm -rf " + testDir_;
        ASSERT(system(cmd.c_str()) == 0);
    }

    std::string testDir_;
};

// Helper function to check if file exists
bool FileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// Test CalculateBatches method
TEST_F(TestExprBatchGenerator, CalculateBatches) {
    // Test with exactly EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator1(testDir_, 1, 1000);
    // Test with more than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator2(testDir_, 2, 2500);
    // Test with less than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator3(testDir_, 3, 500);
    
    // We can't directly access the private batches_ vector, but we can test the behavior
    // by checking the generated files later
}

// Test HeaderFileBegin and HeaderFileEnd methods
TEST_F(TestExprBatchGenerator, HeaderFileGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;
    
    // Test HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);
    
    // Test HeaderFileEnd
    generator.HeaderFileEnd(exprHeaderOss);
    
    // Check if header file was created
    std::string headerPath = testDir_ + "/control_flow_expr_table.h";
    ASSERT_TRUE(FileExists(headerPath));
    
    // Check header file content
    std::ifstream headerFile(headerPath);
    std::string headerContent((std::istreambuf_iterator<char>(headerFile)),
                              std::istreambuf_iterator<char>());
    ASSERT_TRUE(headerContent.find("#pragma once") != std::string::npos);
    ASSERT_TRUE(headerContent.find("namespace npu::tile_fwk") != std::string::npos);
}

// Test GenerateLinkScript method
TEST_F(TestExprBatchGenerator, LinkScriptGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;
    
    // Link script is generated in HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);
    
    // Check if link script was created
    std::string scriptPath = testDir_ + "/merge.link";
    ASSERT_TRUE(FileExists(scriptPath));
    
    // Check link script content
    std::ifstream scriptFile(scriptPath);
    std::string scriptContent((std::istreambuf_iterator<char>(scriptFile)),
                              std::istreambuf_iterator<char>());
    ASSERT_TRUE(scriptContent.find("SECTIONS") != std::string::npos);
    ASSERT_TRUE(scriptContent.find(".pypto") != std::string::npos);
}

// Test GenerateBatchFile method
TEST_F(TestExprBatchGenerator, BatchFileGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 1500); // 2 batches
    std::ostringstream controlFlowOss;
    std::ostringstream exprHeaderOss;
    std::vector<std::string> exprSrcFiles;
    
    // Create a mock expression set
    struct MockExpr {
        int value;
    };
    std::vector<MockExpr> expressions;
    for (int i = 0; i < 1500; ++i) {
        expressions.push_back({i});
    }
    
    // Mock buildExpr function
    auto buildExpr = [](const MockExpr& expr) {
        return std::to_string(expr.value);
    };
    
    // Generate batch files
    generator.GenerateBatchFile(controlFlowOss, exprHeaderOss, "test_exp.h", expressions, exprSrcFiles, 1, 1, buildExpr);
    
    // Check if batch files were created
    ASSERT_EQ(exprSrcFiles.size(), 2);
    for (const auto& filePath : exprSrcFiles) {
        ASSERT_TRUE(FileExists(filePath));
        
        // Check file content
        std::ifstream batchFile(filePath);
        std::string fileContent((std::istreambuf_iterator<char>(batchFile)),
                               std::istreambuf_iterator<char>());
        ASSERT_TRUE(fileContent.find("RUNTIME_SetExpr") != std::string::npos);
    }
    
    // Check control flow content
    std::string controlFlowContent = controlFlowOss.str();
    ASSERT_TRUE(controlFlowContent.find("SetExprBatch_1_0") != std::string::npos);
    ASSERT_TRUE(controlFlowContent.find("SetExprBatch_1_1") != std::string::npos);
    
    // Check header content
    std::string headerContent = exprHeaderOss.str();
    ASSERT_TRUE(headerContent.find("SetExprBatch_1_0") != std::string::npos);
    ASSERT_TRUE(headerContent.find("SetExprBatch_1_1") != std::string::npos);
}

} // namespace test
} // namespace npu::tile_fwk
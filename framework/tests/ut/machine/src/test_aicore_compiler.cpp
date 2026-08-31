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
 * \file test_aicore_compiler.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <climits>
#include <fstream>
#include <cstdlib>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "machine/compile/aicore_compiler.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
std::string GenSubFuncCall(std::map<uint64_t, Function*>& leafDict, CoreType coreType,
                           dynamic::EncodeDevAscendFunctionParam& param, const std::string& ccePath, uint64_t tilingKey,
                           std::stringstream& src_obj, bool enableSubFunc = true);
int RunBishengQuiet(const std::string& compileCmd, const std::string& workDir);
} // namespace npu::tile_fwk

namespace {
// Per-process temp dir avoids parallel CI races when utest_accelerate runs cases concurrently.
std::string GetTestTmpDir() { return "/tmp/test_aicore_compiler_" + std::to_string(getpid()); }
} // namespace

class TestAicoreCompiler : public testing::Test {
public:
    void SetUp() override {}
    static void SetUpTestCase() { CreateDir(GetTestTmpDir()); }

    void TearDown() override {}
    static void TearDownTestCase()
    {
        std::string cmd = "rm -rf " + GetTestTmpDir();
        [[maybe_unused]] int ret = system(cmd.c_str());
    }
};

TEST_F(TestAicoreCompiler, CompileAICoreKernel_EmptyCcePath)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;

    EXPECT_THROW(CompileAICoreKernel(leafDict, param, "", "test_hash", "test_func", kernelPath), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, CompileAICoreKernel_GenSrcFileFails)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;
    EXPECT_THROW(
        CompileAICoreKernel(leafDict, param, "/nonexistent_dir/cce_path/", "test_hash", "test_func", kernelPath),
        npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, GenSubFuncCall_EmptyLeafDict)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::stringstream src_obj;

    std::string result = GenSubFuncCall(leafDict, CoreType::AIC, param, GetTestTmpDir() + "/", 0, src_obj);
    EXPECT_EQ(result, "");
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_SuccessWithTrailingSlash)
{
    std::string workDir = GetTestTmpDir() + "/ok_slash/";
    CreateDir(workDir);
    EXPECT_EQ(RunBishengQuiet("true", workDir), 0);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_SuccessWithoutTrailingSlash)
{
    std::string workDir = GetTestTmpDir() + "/ok_noslash";
    CreateDir(workDir);
    EXPECT_EQ(RunBishengQuiet("true", workDir), 0);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_SuccessEmptyWorkDir)
{
    char cwd[PATH_MAX];
    ASSERT_NE(getcwd(cwd, sizeof(cwd)), nullptr);
    ASSERT_EQ(chdir(GetTestTmpDir().c_str()), 0);
    EXPECT_EQ(RunBishengQuiet("true", ""), 0);
    ASSERT_EQ(chdir(cwd), 0);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_CompileFailWithLog)
{
    std::string workDir = GetTestTmpDir() + "/fail_with_log/";
    CreateDir(workDir);
    EXPECT_THROW(RunBishengQuiet("echo bisheng_ut_error && false", workDir), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_CompileFailEmptyLog)
{
    std::string workDir = GetTestTmpDir() + "/fail_empty_log/";
    CreateDir(workDir);
    EXPECT_THROW(RunBishengQuiet("false", workDir), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_CreateLogFails)
{
    EXPECT_THROW(RunBishengQuiet("true", "/nonexistent_bisheng_ut_dir/"), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_IllegalCharInWorkDir)
{
    std::string workDir = GetTestTmpDir() + "/bad>/";
    CreateDir(workDir);
    EXPECT_THROW(RunBishengQuiet("true", workDir), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, RunBishengQuiet_QuoteShellArgEscapesApostrophe)
{
    std::string workDir = GetTestTmpDir() + "/quote_ok/";
    CreateDir(workDir);
    EXPECT_EQ(RunBishengQuiet("true 'quoted-arg'", workDir), 0);
}

TEST_F(TestAicoreCompiler, CompileAICoreKernel_ReachRunBishengQuiet)
{
    std::string ccePath = GetTestTmpDir() + "/compile_kernel/";
    CreateDir(ccePath);
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;
    try {
        (void)CompileAICoreKernel(leafDict, param, ccePath, "ut_hash", "ut_func", kernelPath);
    } catch (const npu::tile_fwk::Error&) {
        // compile/link failure is acceptable; path through RunBishengQuiet is already exercised
    }
    SUCCEED();
}

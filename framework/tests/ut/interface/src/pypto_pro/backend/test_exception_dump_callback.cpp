/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <sys/stat.h>
#include <vector>

extern "C" {
int32_t pro_register_exception_dump_callback();
void pro_set_dump_info(const char* kernelName, int32_t numTensors, const int32_t* types, const size_t* tensorSizes,
                       const int32_t* dataTypes, const void** tensorAddrs, const int64_t* flatShapes,
                       const int32_t* shapeCounts, int32_t maxDims);
void pro_clear_dump_info();
void pro_set_debug_cmd(const char* cmd);
void pro_set_launch_meta(const char* json);
int32_t pro_test_exception_dump_callback(uint32_t dumpSize, uint32_t* realSize, uint32_t* mode, char* outKernelName,
                                         uint32_t kernelNameBufSize, uint32_t* outExtraTensorNum);
int32_t pro_test_exception_dump_callback_nullptr();
}

class ExceptionDumpCallbackTest : public testing::Test {
protected:
    void SetUp() override { pro_clear_dump_info(); }
    void TearDown() override { pro_clear_dump_info(); }
};

TEST_F(ExceptionDumpCallbackTest, ClearDumpInfoClearsKernelName)
{
    pro_set_dump_info("test_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);
    pro_clear_dump_info();
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithZeroTensors)
{
    pro_set_dump_info("empty_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithSingleTensor)
{
    const char* kernelName = "single_tensor_kernel";
    int32_t types[] = {0};
    size_t sizes[] = {8192};
    int32_t dataTypes[] = {1};
    int64_t addr = 0x1000;
    const void* addrs[] = {&addr};
    int32_t shapeCounts[] = {2};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 1, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithMultipleTensors)
{
    const char* kernelName = "multi_tensor_kernel";
    int32_t types[] = {0, 0, 2};
    size_t sizes[] = {8192, 16384, 40};
    int32_t dataTypes[] = {1, 0, 4};
    int64_t addr0 = 0x1000;
    int64_t addr1 = 0x2000;
    int64_t addr2 = 0x3000;
    const void* addrs[] = {&addr0, &addr1, &addr2};
    int32_t shapeCounts[] = {2, 2, 0};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 3, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);
}

TEST_F(ExceptionDumpCallbackTest, SetDumpInfoWithMaxDimsExceedingShapeCount)
{
    const char* kernelName = "max_dims_kernel";
    int32_t types[] = {0};
    size_t sizes[] = {1024};
    int32_t dataTypes[] = {1};
    int64_t addr = 0x4000;
    const void* addrs[] = {&addr};
    int32_t shapeCounts[] = {1};
    int64_t flatShapes[] = {16, 0, 0, 0};

    pro_set_dump_info(kernelName, 1, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 4);
}

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithNullCmd) { pro_set_debug_cmd(nullptr); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithEmptyCmd) { pro_set_debug_cmd(""); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdWithValidCmd) { pro_set_debug_cmd("echo hello"); }

TEST_F(ExceptionDumpCallbackTest, SetDebugCmdOverwritesPreviousCmd)
{
    pro_set_debug_cmd("echo first");
    pro_set_debug_cmd("echo second");
}

TEST_F(ExceptionDumpCallbackTest, SetLaunchMetaWithNullJson) { pro_set_launch_meta(nullptr); }

TEST_F(ExceptionDumpCallbackTest, SetLaunchMetaWithEmptyJson) { pro_set_launch_meta(""); }

TEST_F(ExceptionDumpCallbackTest, SetLaunchMetaWithValidJson) { pro_set_launch_meta("{\"block_dim\":16}"); }

TEST_F(ExceptionDumpCallbackTest, SetLaunchMetaOverwritesPreviousJson)
{
    pro_set_launch_meta("{\"block_dim\":1}");
    pro_set_launch_meta("{\"block_dim\":16,\"abi\":[]}");
}

TEST_F(ExceptionDumpCallbackTest, ClearDumpInfoClearsLaunchMeta)
{
    pro_set_launch_meta("{\"block_dim\":16}");
    pro_clear_dump_info();
}

TEST_F(ExceptionDumpCallbackTest, CallbackWritesLaunchMetaFile)
{
    std::string workPath = testing::TempDir();
    ASSERT_NE(::setenv("ASCEND_WORK_PATH", workPath.c_str(), 1), -1);
    ASSERT_NE(::setenv("TILE_FWK_DEVICE_ID", "0", 1), -1);

    // 测试上下文不经 CANN，目录需手动创建（逐级，容忍已存在）
    std::string dumpDir = std::string(workPath) + "/extra-info/data-dump/0";
    for (const std::string& dir :
         {std::string(workPath) + "/extra-info", std::string(workPath) + "/extra-info/data-dump", dumpDir}) {
        ASSERT_TRUE(::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) << "mkdir failed: " << dir;
    }

    pro_set_dump_info("meta_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);
    pro_set_launch_meta(
        "{\"kernel_name\":\"meta_kernel\",\"block_dim\":16,\"abi\":[{\"kind\":\"scalar\",\"value\":256}]}");

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;
    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);

    std::string metaPath = std::string(workPath) + "/extra-info/data-dump/0/meta_kernel_launch_args.json";
    FILE* file = std::fopen(metaPath.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    char buf[512] = {0};
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, file);
    std::fclose(file);
    EXPECT_EQ(std::string(buf, n),
              "{\"kernel_name\":\"meta_kernel\",\"block_dim\":16,\"abi\":[{\"kind\":\"scalar\",\"value\":256}]}");
    ::unsetenv("ASCEND_WORK_PATH");
    ::unsetenv("TILE_FWK_DEVICE_ID");
}

TEST_F(ExceptionDumpCallbackTest, RegisterCallbackReturnsResultCode)
{
    int32_t ret = pro_register_exception_dump_callback();
    EXPECT_TRUE(ret == 0 || ret == -1);
}

TEST_F(ExceptionDumpCallbackTest, CallbackNullptrDumpInfoReturnsError)
{
    int32_t ret = pro_test_exception_dump_callback_nullptr();
    EXPECT_EQ(ret, 1);
}

TEST_F(ExceptionDumpCallbackTest, CallbackFillsKernelNameFromCache)
{
    pro_set_dump_info("cached_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, 1);
    EXPECT_EQ(realSize, 1);
    EXPECT_STREQ(kernelName, "cached_kernel");
    EXPECT_EQ(extraTensorNum, 0);
}

TEST_F(ExceptionDumpCallbackTest, CallbackFillsTensorsFromCache)
{
    const char* kernelName = "tensor_kernel";
    int32_t types[] = {0, 0};
    size_t sizes[] = {8192, 16384};
    int32_t dataTypes[] = {1, 0};
    int64_t addr0 = 0x1000;
    int64_t addr1 = 0x2000;
    const void* addrs[] = {&addr0, &addr1};
    int32_t shapeCounts[] = {2, 2};
    int64_t flatShapes[] = {64, 64, 0, 0, 0, 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0};

    pro_set_dump_info(kernelName, 2, types, sizes, dataTypes, addrs, flatShapes, shapeCounts, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char outKernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, outKernelName, sizeof(outKernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mode, 1);
    EXPECT_EQ(realSize, 1);
    EXPECT_STREQ(outKernelName, kernelName);
    EXPECT_EQ(extraTensorNum, 2);
}

TEST_F(ExceptionDumpCallbackTest, CallbackExecutesDebugCmdOnce)
{
    pro_set_debug_cmd("echo callback_test");
    pro_set_dump_info("debug_cmd_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);

    ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName), &extraTensorNum);
    EXPECT_EQ(ret, 0);
}

TEST_F(ExceptionDumpCallbackTest, CallbackSkipsDebugCmdWhenEmpty)
{
    pro_set_debug_cmd("");
    pro_set_dump_info("no_debug_cmd_kernel", 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8);

    uint32_t realSize = 0;
    uint32_t mode = 0;
    char kernelName[1024] = {0};
    uint32_t extraTensorNum = 0;

    int32_t ret = pro_test_exception_dump_callback(1, &realSize, &mode, kernelName, sizeof(kernelName),
                                                   &extraTensorNum);
    EXPECT_EQ(ret, 0);
    EXPECT_STREQ(kernelName, "no_debug_cmd_kernel");
}

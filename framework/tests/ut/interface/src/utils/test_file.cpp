/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_file.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>

#include "interface/utils/file_utils.h"

using namespace npu::tile_fwk;

#define PATH_MAX 4096

class FileTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST(FileTest, NullptrTest) {
    uint8_t a = 255;
    std::vector<uint8_t> data{a};
    SaveFile("", data);

    bool ret = SaveFile("", &a, 1);
    EXPECT_EQ(ret, false);

    EXPECT_FALSE(CopyFile("", ""));
    EXPECT_EQ(LockAndOpenFile(""), nullptr);
}

constexpr const char* TEST_LOG_PATH = "/tmp/test_file.log";
TEST(FileTest, ReadBytesFromFileTest) {
    std::vector<char> data;

    bool ret = ReadBytesFromFile("", data);
    EXPECT_EQ(ret, false);

    SaveFile(TEST_LOG_PATH, std::vector<uint8_t>({}));
    ret = ReadBytesFromFile(TEST_LOG_PATH, data);
    EXPECT_EQ(ret, false);
    DeleteFile(TEST_LOG_PATH);

    // for code coverage
    DeleteFile("");
    DeleteFile("/nonexistent/file");
}

TEST(FileTest, LoadFileTest) {
    uint8_t a = 255;
    std::vector<uint8_t> data{a};

    data = LoadFile("");
    EXPECT_EQ(data.size(), 0);

    nlohmann::json jsonObj;
    EXPECT_FALSE(ReadJsonFile("", jsonObj));
    EXPECT_FALSE(ReadJsonFile("/tmp/", jsonObj));
}

TEST(FileTest, RealPathTest) {
    EXPECT_TRUE(RealPath("").empty());
    EXPECT_TRUE(RealPath("/nonexitent/file").empty());
    EXPECT_FALSE(RealPath("/tmp").empty());
    EXPECT_TRUE(RealPath(std::string(PATH_MAX, 'a')).empty());

    EXPECT_FALSE(CreateMultiLevelDir(std::string(PATH_MAX, 'a')));
    EXPECT_FALSE(DeleteDir(""));
}
TEST(FileTest, CreateDirFailedWhenParentNotExist) {
    // 父目录不存在 -> mkdir 返回 -1，errno=ENOENT -> CreateDir 返回 false
    std::string path = "/tmp/no_such_parent_dir" + std::to_string(::getpid()) + "/child";
    EXPECT_FALSE(CreateDir(path));
}

TEST(FileTest, CreateMultiLevelDirFailedWhenParentIsFile) {
    const std::string parent = "/tmp/parent_is_file_" + std::to_string(::getpid());
    const std::string target = parent + "/child";
    DeleteDir(target);
    DeleteFile(parent);
    SaveFile(parent, std::vector<uint8_t>{0x01});
    EXPECT_FALSE(CreateMultiLevelDir(target));

    // 清理
    DeleteDir(target);
    DeleteFile(parent);
}
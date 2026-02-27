/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_log_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <dirent.h>
#include <sys/syscall.h>
#define private public
#include "utils/host_log/log_manager.h"
#undef private
#include "tilefwk/tilefwk_log.h"
#include "utils/host_log/dlog_handler.h"

namespace npu::tile_fwk {
class TestHostLog : public testing::Test {
public:
    void SetUp() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_GLOBAL_EVENT_ENABLE");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
    }
    void TearDown() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_GLOBAL_EVENT_ENABLE");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
    }
    uint64_t GetTestThreadId() {
        thread_local uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
        return tid;
    }
    size_t GetLogFileSizeOfSpecifiedDir(const std::string &dirPath, const std::string &filePrefix) {
        DIR *dir = opendir(dirPath.c_str());
        if (dir == nullptr) {
            return 0;
        }
        size_t fileSize = 0;
        struct dirent *dirp = nullptr;
        while ((dirp = readdir(dir)) != nullptr) {
            if (dirp->d_name[0] == '.') {
                continue;
            }
            std::string fileName = dirp->d_name;
            if (fileName.find(filePrefix) != 0) {
                continue;
            }
            if (fileName.find(".log") == std::string::npos) {
                continue;
            }
            fileSize++;
        }
        closedir(dir);
        return fileSize;
    }
    void RecoreLog(LogManager &log_manager, const LogLevel logLevel, const char *fmt, ...) {
        va_list list;
        va_start(list, fmt);
        log_manager.Record(logLevel, fmt, list);
        va_end(list);
    }
    void CheckLogContent(LogManager &log_manager, const LogLevel logLevel, const std::string &expectedStr, const char *fmt, ...) {
        va_list list;
        va_start(list, fmt);
        LogMsg logMsg{};
        log_manager.ConstructMessage(logLevel, fmt, list, logMsg);
        va_end(list);
        std::string retStr(logMsg.msg);
        EXPECT_EQ(retStr.find(expectedStr), retStr.size() - expectedStr.size() - 1);
    }
};

TEST_F(TestHostLog, test_tilefwk_log) {
    PYPTO_HOST_LOG(DLOG_ERROR, "TEST", "I'm a space-bound %s and your heart's the moon", "rocketship");
    PYPTO_HOST_LOG(DLOG_ERROR, "TEST", "And I aiming it right at you, right at you %f", 3.14f);
    PYPTO_HOST_LOG(DLOG_ERROR, "TEST", "%d miles on a clear night in %s", 250000, "June");
    PYPTO_HOST_LOG(DLOG_ERROR, "TEST", "And I'm so lost without you, without you %x", 626);
}
TEST_F(TestHostLog, test_log_manager_case0) {
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(LogManager::Instance().CheckLevel(LogLevel::EVENT), false);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
    RecoreLog(log_manager, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
    RecoreLog(log_manager, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
    RecoreLog(log_manager, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
    RecoreLog(log_manager, LogLevel::INFO, "And I'm so lost without you, without you %x", 626);
}

TEST_F(TestHostLog, test_log_manager_case1) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case2) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_MODULE_LOG_LEVEL", "PYPTO=2", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case3) {
    setenv("ASCEND_GLOBAL_EVENT_ENABLE", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), true);
}

TEST_F(TestHostLog, test_log_manager_case4) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "abc", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestHostLog, test_log_manager_case5) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_PROCESS_LOG_PATH", "./temp_pypto_log", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.enableStdOut_, false);
    for (size_t i = 0; i < 100000; ++i) {
        if (i%2 == 0) {
            log_manager.EnableHostLog();
        } else {
            log_manager.EnableDeviceLog();
        }
        RecoreLog(log_manager, LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
        RecoreLog(log_manager, LogLevel::INFO, "And I aiming it right at you, right at you %f", 3.14f);
        RecoreLog(log_manager, LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
        RecoreLog(log_manager, LogLevel::INFO, "And I'm so lost without you, without you %x", 626);
    }
    std::string hostLogFilePrefix = "pypto-log-" + std::to_string(GetTestThreadId());
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager.hostLogDir_, hostLogFilePrefix), 2);
    std::string devLogFilePrefix = "pypto-simulation-" + std::to_string(GetTestThreadId());
    EXPECT_EQ(GetLogFileSizeOfSpecifiedDir(log_manager.deviceLogDir_, devLogFilePrefix), 2);
}

TEST_F(TestHostLog, test_log_construct_case0) {
    LogManager log_manager;
    int32_t int32_val = -234;
    uint32_t uint32_val = 432;
    std::ostringstream oss1;
    oss1 << "[-234][432][" << std::hex << &int32_val << "][" << &uint32_val << "]";
    CheckLogContent(log_manager, LogLevel::INFO, oss1.str(), "[%d][%u][%p][%p]", int32_val, uint32_val, &int32_val, &uint32_val);

    int64_t int64_val = -789;
    uint64_t uint64_val = 987;
    CheckLogContent(log_manager, LogLevel::INFO, "[-789][987][fffffceb][3DB][0xfffffceb][0X3DB]", "[%ld][%lu][%x][%X][%#x][%#X]", int64_val, uint64_val, int64_val, uint64_val, int64_val, uint64_val);

    float float_val = 123.456f;
    std::ostringstream oss2;
    oss2 << "[123.456001][123.46][" << std::hex << &float_val << "]";
    CheckLogContent(log_manager, LogLevel::INFO, oss2.str(), "[%f][%.2f][%p]", float_val, float_val, &float_val);

    CheckLogContent(log_manager, LogLevel::INFO, "[Hello world]", "[%c%s]", 'H', "ello world");
}

TEST_F(TestHostLog, test_log_construct_case1) {
    LogManager log_manager;
    CheckLogContent(log_manager, LogLevel::INFO, "Hello world!", "Hello world!", 123, "morgan");
    CheckLogContent(log_manager, LogLevel::INFO, "[4294967173][18446744073709551160]", "[%u][%lu]", -123, -456);
    CheckLogContent(log_manager, LogLevel::INFO, "[3.140000][4294967173]", "[%f][%lu]", -123, 3.14f);

    std::ostringstream oss;
    for (size_t i = 0; i < 100; i++) {
        oss << "0123456789";
    }
    RecoreLog(log_manager, LogLevel::INFO, oss.str().c_str());
}

TEST_F(TestHostLog, test_dlog_handler_case0) {
    DLogHandler log_handler;
    EXPECT_EQ(log_handler.IsAvailable(), true);
    EXPECT_NE(log_handler.checkLevelFunc_, nullptr);
    EXPECT_NE(log_handler.logRecordFunc_, nullptr);

    EXPECT_EQ(DLogHandler::Instance().IsAvailable(), true);
    EXPECT_NE(DLogHandler::Instance().checkLevelFunc_, nullptr);
    EXPECT_NE(DLogHandler::Instance().logRecordFunc_, nullptr);
}
}

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
#define private public
#include "utils/host_log/log_manager.h"
#undef private
#include "tilefwk/tilefwk_log.h"
#include "utils/host_log/dlog_handler.h"

namespace npu::tile_fwk {
class TestLogManager : public testing::Test {
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
    void RecoreLog(LogManager &log_manager, const LogLevel logLevel, const char *fmt, ...) {
        va_list list;
        va_start(list, fmt);
        log_manager.Record(logLevel, fmt, list);
        va_end(list);
    }
};

TEST_F(TestLogManager, test_log_manager_case0) {
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

TEST_F(TestLogManager, test_log_manager_case1) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestLogManager, test_log_manager_case2) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_MODULE_LOG_LEVEL", "PYPTO=2", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestLogManager, test_log_manager_case3) {
    setenv("ASCEND_GLOBAL_EVENT_ENABLE", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), true);
}

TEST_F(TestLogManager, test_log_manager_case4) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "abc", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::WARN), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::EVENT), false);
}

TEST_F(TestLogManager, test_log_construct_case0) {
    LogManager log_manager;
    int32_t int32_val = -234;
    uint32_t uint32_val = 432;
    RecoreLog(log_manager, LogLevel::INFO, "[%d][%u][%x][%p]", int32_val, uint32_val, &int32_val, &uint32_val);

    int64_t int64_val = -789;
    uint64_t uint64_val = 987;
    RecoreLog(log_manager, LogLevel::INFO, "[%ld][%lu][%x][%X]", int64_val, uint64_val, int64_val, uint64_val);

    float float_val = 123.456f;
    RecoreLog(log_manager, LogLevel::INFO, "[%f][%.2f][%x][%p]", float_val, float_val, float_val, &float_val);

    RecoreLog(log_manager, LogLevel::INFO, "[%c%s]", 'H', "ello world");
}

TEST_F(TestLogManager, test_log_construct_case1) {
    LogManager log_manager;
    RecoreLog(log_manager, LogLevel::INFO, "Hello world!", 123, "morgan");
    RecoreLog(log_manager, LogLevel::INFO, "[%u][%lu]", -123, -456);
    RecoreLog(log_manager, LogLevel::INFO, "[%f][%lu]", -123, 3.14f);
    RecoreLog(log_manager, LogLevel::INFO, "[%x][%p]", -123, 3.14);

    std::ostringstream oss;
    for (size_t i = 0; i < 100; i++) {
        oss << "0123456789";
    }
    RecoreLog(log_manager, LogLevel::INFO, oss.str().c_str());
}

TEST_F(TestLogManager, test_dlog_handler_case0) {
    DLogHandler log_handler;
    EXPECT_EQ(log_handler.IsAvailable(), true);
    EXPECT_NE(log_handler.checkLevelFunc_, nullptr);
    EXPECT_NE(log_handler.logRecordFunc_, nullptr);

    EXPECT_EQ(DLogHandler::Instance().IsAvailable(), true);
    EXPECT_NE(DLogHandler::Instance().checkLevelFunc_, nullptr);
    EXPECT_NE(DLogHandler::Instance().logRecordFunc_, nullptr);
}
}

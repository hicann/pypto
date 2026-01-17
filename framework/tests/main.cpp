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
 * \file main.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "utils/test_cost_macro.h"

#if defined(BUILD_WITH_CANN) && defined(ENABLE_STEST)
#include "runtime/dev.h"

bool CheckDeviceConsistency() {
    /* 获取实际生效的 DeviceId */
    int32_t rtDevId = -1;  // -1 表示无效 DeviceId
    int32_t getDeviceResult = rtGetDevice(&rtDevId);
    if (getDeviceResult != RT_ERROR_NONE) {
        std::cout << "Error: Can't get deviceId" << std::endl;
        return false;
    }

    /* 获取环境变量中设置的 DeviceId */
    int32_t envDevId = 0;
    const char *devIdPtr = getenv("TILE_FWK_DEVICE_ID");
    if (devIdPtr != nullptr) {
        envDevId = std::stoi(devIdPtr);
    }

    if (rtDevId != envDevId) {
        std::cout << "Error: rtDevId(" << rtDevId << ") != envDevId(" << envDevId << ")" << std::endl;
        return false;
    }
    return true;
}

#else

bool CheckDeviceConsistency() {
    return true;
}

#endif

class TestExecutionCounter : public testing::EmptyTestEventListener {
public:
    uint64_t executed_count = 0;

    void OnTestStart(const testing::TestInfo&) override {
        executed_count++;
    }
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    auto isMetaParam = [](const std::string& arg) {
        return arg == "--gtest_list_tests_with_meta";
    };
    if (std::find_if(argv + 1, argv + argc, isMetaParam) != argv + argc) {
        ListTestsWithMetadata();
        return 0;
    }
    // 创建并注册监听器
    TestExecutionCounter counter;
    testing::UnitTest::GetInstance()->listeners().Append(&counter);

    auto ret = RUN_ALL_TESTS();

    // 移除监听器（避免析构时访问已释放内存）
    testing::UnitTest::GetInstance()->listeners().Release(&counter);
    if (counter.executed_count == 0) {
        std::cout << "Error: Can't get any case to run when using " << testing::GTEST_FLAG(filter)
                  << " to filter." << std::endl;
        ret = ret == 0 ? 1 : ret;
    }
    ret = CheckDeviceConsistency() ? ret : 1;

    return ret;
}

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
 * \file test_config_manager.cpp
 * \brief
 */
#include <climits>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk;

class TestConfigManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestConfigManager, PassGloablConfig) {
    {
        auto ret = config::GetPassGlobalConfig("pass_thread_num", 0);
        EXPECT_EQ(ret, 1);
        config::SetPassGlobalConfig("pass_thread_num", 0);
        ret = config::GetPassGlobalConfig("pass_thread_num", 1);
        EXPECT_EQ(ret, 0);
    }

    {
        auto ret = config::GetPassGlobalConfig("enable_cv_fuse", true);
        EXPECT_EQ(ret, false);
        config::SetPassGlobalConfig("enable_cv_fuse", true);
        ret = config::GetPassGlobalConfig("enable_cv_fuse", false);
        EXPECT_EQ(ret, true);
    }
}

TEST_F(TestConfigManager, PassDefaultConfig) {
        auto ret = config::GetPassDefaultConfig(KEY_PRINT_GRAPH, true);
        EXPECT_EQ(ret, false);
        config::SetPassDefaultConfig(KEY_PRINT_GRAPH, true);
        ret = config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false);
        EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, PassStrategies2) {
    {
        auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpGraph, false);

        // set default config useful
        config::SetPassDefaultConfig(npu::tile_fwk::KEY_DUMP_GRAPH, true);
        ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpGraph, true);
    }
}

TEST_F(TestConfigManager, PassStrategies3) {
    // set default config useless
    auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, false);

    config::SetPassDefaultConfig(KEY_EXPECTED_VALUE_CHECK, true);
    ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, true);
}

TEST_F(TestConfigManager, Dump) {
    auto &cm = ConfigManagerNg::GetInstance();

    cm.BeginScope("scope1", {{"debug.print.edgeitems", 10L}});
    auto scope1 = cm.CurrentScope();
    cm.EndScope();

    cm.BeginScope("scope2", {{"debug.print.edgeitems", 20L}});
    {
        cm.BeginScope("scope2.1", {{"debug.print.linewidth", 120L}});
        auto scope2 = cm.CurrentScope();
        auto linewidth = AnyCast<int64_t>(scope2->GetConfig("debug.print.linewidth"));
        EXPECT_EQ(linewidth, 120);
        auto edgeitems = AnyCast<int64_t>(scope2->GetConfig("debug.print.edgeitems"));
        EXPECT_EQ(edgeitems, 20);
        cm.EndScope();
    }

    auto scope = cm.CurrentScope();
    auto linewidth = AnyCast<int64_t>(scope->GetConfig("debug.print.linewidth"));
    EXPECT_EQ(linewidth, 80);
    auto edgeitems = AnyCast<int64_t>(scope->GetConfig("debug.print.edgeitems"));
    EXPECT_EQ(edgeitems, 20);
    cm.EndScope();

    cm.BeginScope("scope3", {{"debug.print.edgeitems", 30L}});
    auto scope3 = cm.CurrentScope();
    cm.SetScope({{"debug.print.edgeitems", 35L}});
    auto scope4 = cm.CurrentScope();
    cm.EndScope();

    std::cout << cm.GetOptionsTree() << std::endl;
    std::cout << "-- scope3 -- " << std::endl;
    std::cout << scope3->ToString() << std::endl;
}


constexpr const char *ERROR_KEY_WORD = "its value doesn't within the value range";
template <typename T>
bool RangeTest(
    const std::unordered_map<std::string, std::vector<T>> &input,
    void (*SetFunc)(const std::string &, T &&),
    std::string group) {
    for (auto &[key, val] : input) {
        for (auto it : val) {
            T rlv = it;
            try {
                SetFunc(group + "." + key, std::move(rlv));
            } catch (const std::runtime_error &e) {
                std::stringstream ss;
                ss << e.what();
                std::string errStr(ss.str());
                if (errStr.find(ERROR_KEY_WORD) == std::string::npos) {
                    std::cerr << "error exception: " << errStr << std::endl;
                    return false;
                } else {
                    continue;
                }
            }
        }
    }
    return true;
}

TEST_F(TestConfigManager, NormalRuntimeTest) {
    std::unordered_map<std::string, std::vector<int64_t>> input = {
        {DEVICE_SCHED_MODE, {0, 1, 2, 3}},
        {STITCH_FUNCTION_INNER_MEMORY, {1, INT_MAX}},
        {STITCH_FUNCTION_OUTCAST_MEMORY, {1, INT_MAX}},
        {STITCH_FUNCTION_NUM_INITIAL, {1, 128}},
        {STITCH_FUNCTION_NUM_STEP, {0, 128}},
        {CFGCACHE_DEVICE_TASK_NUM, {0, 100}},
        {CFGCACHE_ROOT_TASK_NUM, {0, 1000}},
        {CFGCACHE_LEAF_TASK_NUM, {0, 10000}},
        {STITCH_FUNCTION_SIZE, {1, 65535}},
        {CFG_RUN_MODE, {0, 1}},
    };
    bool ret = RangeTest<int64_t>(input, &(config::SetOption), "runtime");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, AbnormalRuntimeTest) {
    int64_t outVal = INT_MAX;
    ++outVal;
    std::unordered_map<std::string, std::vector<int64_t>> input = {
        {DEVICE_SCHED_MODE, {-1, 4}},
        {STITCH_FUNCTION_INNER_MEMORY, {0, outVal}},
        {STITCH_FUNCTION_OUTCAST_MEMORY, {0, outVal}},
        {STITCH_FUNCTION_NUM_INITIAL, {0, 129}},
        {STITCH_FUNCTION_NUM_STEP, {-1, 129}},
        {CFGCACHE_DEVICE_TASK_NUM, {-1, 101}},
        {CFGCACHE_ROOT_TASK_NUM, {-1, 1001}},
        {CFGCACHE_LEAF_TASK_NUM, {-1, 10001}},
        {STITCH_FUNCTION_SIZE, {0, 65536}},
        {CFG_RUN_MODE, {-1, 2}},
    };
    bool ret = RangeTest<int64_t>(input, &(config::SetOption), "runtime");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, NormalPassTest) {
    std::unordered_map<std::string, std::vector<int64_t>> input = {
        {SG_PARALLEL_NUM, {0, INT_MAX}},
        {SG_PG_UPPER_BOUND, {0, INT_MAX}},
        {SG_PG_LOWER_BOUND, {0, INT_MAX}},
        {CUBE_L1_REUSE_MODE, {0, INT_MAX}},
        {CUBE_NBUFFER_MODE, {0, 2}},
        {MG_COPYIN_UPPER_BOUND, {0, INT_MAX}},
        {VEC_NBUFFER_MODE, {0, 2}},
        {MG_VEC_PARALLEL_LB, {1, 48}},
        {SG_CUBE_PARALLEL_NUM, {1, 24}},
        {COPYOUT_RESOLVE_COALESCING, {0, 1000000}}
    };
    bool ret = RangeTest<int64_t>(input, &(config::SetOption), "pass");
    EXPECT_EQ(ret, true);

    std::unordered_map<std::string, std::vector<std::map<int64_t, int64_t>>> input2 = {
        {CUBE_L1_REUSE_SETTING, {{{0, 0}}, {{INT_MAX, INT_MAX}}}},
        {CUBE_NBUFFER_SETTING, {{{-1, 1}}, {{INT_MAX, INT_MAX}}}},
        {VEC_NBUFFER_SETTING, {{{-1, 1}}, {{INT_MAX, INT_MAX}}}}
    };
    ret = RangeTest<std::map<int64_t, int64_t>>(input2, &(config::SetOption), "pass");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, AbnormalPassTest) {
    int64_t outVal = INT_MAX;
    ++outVal;
    std::unordered_map<std::string, std::vector<int64_t>> input = {
        {SG_PARALLEL_NUM, {-1, outVal}},
        {SG_PG_UPPER_BOUND, {-1, outVal}},
        {SG_PG_LOWER_BOUND, {-1, outVal}},
        {CUBE_L1_REUSE_MODE, {-1, outVal}},
        {CUBE_NBUFFER_MODE, {-1, 3}},
        {MG_COPYIN_UPPER_BOUND, {-1, outVal}},
        {VEC_NBUFFER_MODE, {-1, 3}},
        {MG_VEC_PARALLEL_LB, {0, 49}},
        {SG_CUBE_PARALLEL_NUM, {0, 25}},
        {COPYOUT_RESOLVE_COALESCING, {-1, 1000001}}
    };
    bool ret = RangeTest<int64_t>(input, &(config::SetOption), "pass");
    EXPECT_EQ(ret, true);

    std::unordered_map<std::string, std::vector<std::map<int64_t, int64_t>>> input2 = {
        {CUBE_L1_REUSE_SETTING, {{{-1, 0}}, {{outVal, INT_MAX}}, {{0, -1}}, {{INT_MAX, outVal}}}},
        {CUBE_NBUFFER_SETTING, {{{-2, 1}}, {{INT_MAX, outVal}}, {{-1, 0}}, {{outVal, INT_MAX}}}},
        {VEC_NBUFFER_SETTING, {{{-2, 1}}, {{INT_MAX, outVal}}, {{-1, 0}}, {{outVal, INT_MAX}}}}
    };
    ret = RangeTest<std::map<int64_t, int64_t>>(input2, &(config::SetOption), "pass");
    EXPECT_EQ(ret, true);
}

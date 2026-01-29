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
 * \file test_prof.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "securec.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "machine/runtime/runtime.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "interface/utils/common.h"
#include "machine/device/tilefwk/aicpu_common.h"

#include <iostream>

using namespace npu::tile_fwk::dynamic;

class TestPro : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestPro, test_ini) {
    std::unique_ptr<AicpuTaskManager> aicpuTaskPtr = std::make_unique<AicpuTaskManager>();
    std::unique_ptr<AiCoreManager> AiCoreManagerPtr = std::make_unique<AiCoreManager>(*aicpuTaskPtr);
    AiCoreManagerPtr->aicNum_ = 0;
    AiCoreManagerPtr->aivNum_ = 1;
    AiCoreManagerPtr->aivEnd_ = 1;
    AiCoreManagerPtr->aicEnd_ = 0;
    AiCoreManagerPtr->aicpuIdx_ = 0;
    AiCoreProf prof(*AiCoreManagerPtr);

    int64_t *oriRegAddrs_ = (int64_t *)malloc(sizeof(int64_t) * 1024 * 2);
    int64_t *regAddrs_ = oriRegAddrs_ + 1024;
    regAddrs_[0] = (int64_t)&regAddrs_[0];
    std::cout << "oriRegAddrs_ " << oriRegAddrs_ << std::endl;
    std::cout << "regAddrs_    " << regAddrs_ << std::endl;
    ProfConfig profConfig;
    prof.ProfInit(regAddrs_, regAddrs_, profConfig);
    prof.ProfStart();

    int32_t aicoreId = 0;
    int32_t subgraphId = 0;
    int32_t taskId = 0;
    TaskStat *taskStat = new TaskStat();
    taskStat->taskId = 0;
    taskStat->execEnd = 1;
    taskStat->execStart = 0;
    taskStat->subGraphId = 0;

    prof.ProInitHandShake();
    prof.ProInitAiCpuTaskStat();
    int threadIdx = 0;
    AiCpuTaskStat *aiCpuStat = new AiCpuTaskStat();
    AiCpuHandShakeSta handShakeSta;
    aiCpuStat->taskId = 0;
    aiCpuStat->execEnd = 1;
    aiCpuStat->execStart = 0;
    aiCpuStat->coreId = 0;
    aiCpuStat->taskGetStart = 0;

    for (int i = 0; i < 8; i++) {
        prof.ProfGet(aicoreId, subgraphId, taskId, taskStat);
        prof.ProfGetAiCpuTaskStat(threadIdx, aiCpuStat);
        prof.ProGetHandShake(threadIdx, &handShakeSta);
    }
    prof.ProfStopHandShake();
    prof.ProfStopAiCpuTaskStat();
    int64_t flag = 0;
    prof.ProfGetSwitch(flag);

    prof.ProfStop();
    prof.GetAiCpuTaskStat(taskId);
    delete aiCpuStat;
    delete taskStat;
    free(oriRegAddrs_);
}

static void *AllocAligned(size_t alignment, size_t size)
{
    void *ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        return nullptr;
    }
    // init as zero to avoid random register values
    (void)memset_s(ptr, size, 0, size);
    return ptr;
}

TEST_F(TestPro, test_prof_start_pmu_dav2201) {
    // Setup manager: keep one aicore managed
    std::unique_ptr<AicpuTaskManager> aicpuTaskPtr = std::make_unique<AicpuTaskManager>();
    std::unique_ptr<AiCoreManager> aicoreMng = std::make_unique<AiCoreManager>(*aicpuTaskPtr);
    aicoreMng->aicNum_ = 1;
    aicoreMng->aivNum_ = 0;
    aicoreMng->aicStart_ = 0;
    aicoreMng->aicEnd_ = 1;
    aicoreMng->aicpuIdx_ = 0;
    AiCoreProf prof(*aicoreMng);

    const uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
    // cover PMU register offsets up to about 0x2000+ for 2201
    const size_t regBufSize = 0x6000;
    uint8_t *regBuf = reinterpret_cast<uint8_t *>(AllocAligned(pageSize, regBufSize));
    ASSERT_NE(regBuf, nullptr);

    // choose an address within the first page so mapBase aligns to regBuf
    void *addr = reinterpret_cast<void *>(regBuf + 0x100);

    int64_t regAddrsArr[1024] = {0};
    regAddrsArr[0] = reinterpret_cast<int64_t>(addr);

    int64_t pmuEventAddrsArr[10] = {0};
    for (int i = 0; i < 8; ++i) {
        pmuEventAddrsArr[i] = i + 1;
    }

    ProfConfig profConfig;
    profConfig.Add(ProfConfig::AICORE_PMU);
    prof.ProfInit(regAddrsArr, pmuEventAddrsArr, profConfig, ArchInfo::DAV_2201);
    prof.ProfInitPmu(regAddrsArr, pmuEventAddrsArr);
    prof.ProfStartPmu();
    TaskStat taskStat;
    taskStat.seqNo = 1;
    prof.ProfGetPmu(0, 0, 0, &taskStat);
    prof.ProfStop();

    free(regBuf);
}

TEST_F(TestPro, test_prof_start_pmu_dav3510) {
    std::unique_ptr<AicpuTaskManager> aicpuTaskPtr = std::make_unique<AicpuTaskManager>();
    std::unique_ptr<AiCoreManager> aicoreMng = std::make_unique<AiCoreManager>(*aicpuTaskPtr);
    aicoreMng->aicNum_ = 1;
    aicoreMng->aivNum_ = 0;
    aicoreMng->aicStart_ = 0;
    aicoreMng->aicEnd_ = 1;
    aicoreMng->aicpuIdx_ = 0;
    AiCoreProf prof(*aicoreMng);

    const uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
    // cover DAV_3510 PMU register offsets up to ~0x4300
    const size_t regBufSize = 0x9000;
    uint8_t *regBuf = reinterpret_cast<uint8_t *>(AllocAligned(pageSize, regBufSize));
    ASSERT_NE(regBuf, nullptr);

    void *addr = reinterpret_cast<void *>(regBuf + 0x100);

    int64_t regAddrsArr[1024] = {0};
    regAddrsArr[0] = reinterpret_cast<int64_t>(addr);

    int64_t pmuEventAddrsArr[10] = {0};
    for (int i = 0; i < 10; ++i) {
        pmuEventAddrsArr[i] = i + 1;
    }

    ProfConfig profConfig;
    profConfig.Add(ProfConfig::AICORE_PMU);
    prof.ProfInit(regAddrsArr, pmuEventAddrsArr, profConfig, ArchInfo::DAV_3510);
    prof.ProfInitPmu(regAddrsArr, pmuEventAddrsArr);
    prof.ProfStartPmu();
    TaskStat taskStat;
    taskStat.seqNo = 1;
    prof.ProfGetPmu(0, 0, 0, &taskStat);
    prof.ProfStop();

    free(regBuf);
}

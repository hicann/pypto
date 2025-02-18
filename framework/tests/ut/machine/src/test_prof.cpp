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
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "machine/runtime/runtime.h"
#include "machine/device/aicore_prof.h"
#include "machine/device/aicpu_task_manager.h"
#include "machine/device/aicore_manager.h"
#include "interface/utils/common.h"

#include <iostream>

using namespace npu::tile_fwk;

class TestPro : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestPro, test_ini) {
    std::unique_ptr<npu::tile_fwk::AicpuTaskManager> aicpuTaskPtr = std::make_unique<npu::tile_fwk::AicpuTaskManager>();
    std::unique_ptr<npu::tile_fwk::AiCoreManager> AiCoreManagerPtr = std::make_unique<npu::tile_fwk::AiCoreManager>(*aicpuTaskPtr);
    AiCoreManagerPtr->aicNum_ = 0;
    AiCoreManagerPtr->aivNum_ = 1;
    AiCoreManagerPtr->aivStart_ = 0;
    AiCoreManagerPtr->aivEnd_ = 1;
    AiCoreManagerPtr->aicStart_ = 0;
    AiCoreManagerPtr->aicEnd_ = 0;
    AiCoreManagerPtr->aicpuIdx_ = 0;
    AiCoreManagerPtr->blockIdToPhyCoreId_[0] = 0;
    npu::tile_fwk::AiCoreProf prof(*AiCoreManagerPtr);

    int64_t *oriRegAddrs_ = (int64_t *)malloc(sizeof(int64_t) * 1024 * 2);
    int64_t *regAddrs_ = oriRegAddrs_ + 1024;
    regAddrs_[0] = (int64_t)&regAddrs_[0];
    std::cout << "oriRegAddrs_ " << oriRegAddrs_ << std::endl;
    std::cout << "regAddrs_    " << regAddrs_ << std::endl;
    prof.ProfInit(regAddrs_, regAddrs_);
    prof.ProfStart();

    int32_t aicoreId = 0;
    int32_t subgraphId = 0;
    int32_t taskId = 0;
    TaskStat *taskStat = new TaskStat();
    taskStat->taskId = 0;
    taskStat->execEnd = 1;
    taskStat->execStart = 0;
    taskStat->subGraphId = 0;

    uint32_t pmuCnt0 = 0;
    prof.pmuCnt0Plain_[0] = &pmuCnt0;
    prof.pmuCnt1Plain_[0] = &pmuCnt0;
    prof.pmuCnt2Plain_[0] = &pmuCnt0;
    prof.pmuCnt3Plain_[0] = &pmuCnt0;
    prof.pmuCnt4Plain_[0] = &pmuCnt0;
    prof.pmuCnt5Plain_[0] = &pmuCnt0;
    prof.pmuCnt6Plain_[0] = &pmuCnt0;
    prof.pmuCnt7Plain_[0] = &pmuCnt0;
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

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
 * \file device_machine.h
 * \brief
 */

#pragma once

#include "device_common.h"
#include "aicore_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "tilefwk/aicore_print.h"

namespace npu::tile_fwk::dynamic {
struct AicoreLogManager {
    AicoreLogManager() {
        data_ = aligned_alloc(PAGE_SIZE, MAX_AICORE_NUM * PRINT_BUFFER_SIZE);
        uint8_t *buf = (uint8_t *)data_;
        for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
            logger[i].Init(buf, PRINT_BUFFER_SIZE);
            buf += PRINT_BUFFER_SIZE;
        }
    }
    ~AicoreLogManager() { free(data_); }

    void *data_;
    AicoreLogger logger[MAX_AICORE_NUM];
};

class DeviceMachine {
public:
    DeviceMachine() {
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; ++i) {
            aicoreManager_[i] = std::make_unique<AiCoreManager>(aicpuTaskManager_);
        }
        validCore_.fill(false);
    }

    bool CheckAndResetReg(){
        return aicoreManager_[0]->CheckAndResetReg();
    }
    
    void init(uint32_t schNum) {
        schAicpuNum_ = schNum;
    }

    int Run(int threadIdx, DeviceArgs *args, bool handShakeByGm = true) {
        int ret = 0;
        if (args->nrAic == 0 || args->nrValidAic == 0 || args->nrAicpu < NEED_LAUNCH_AICPU_MINNUM) {
            DEV_ERROR("Device machinr run invalid args aicnum:%u, blockdim:%u, launchAicpu num:%u",
                args->nrAic, args->nrValidAic, args->nrAicpu);
            return DEVICE_MACHINE_ERROR;
        }

        DEV_INFO("thread %d start .", threadIdx);
        if (static_cast<uint32_t>(threadIdx) >= MAX_SCHEDULE_AICPU_NUM) {
            DEV_INFO("thread start ignore ");
            return DEVICE_MACHINE_OK;
        }
#if ENABLE_AICORE_PRINT
        aicoreManager_[threadIdx]->InitLogger(logManager.logger);
#endif
        ret = aicoreManager_[threadIdx]->Run(threadIdx, args, handShakeByGm);
        DEV_INFO("thread  %d end , ret = %d", threadIdx, ret);
        return ret;
    }

    void CacheValidCore() {
        DEV_DEBUG("begin cache valid core.");
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_[i]->SetValidCore(&validCore_);
        }
    }

    void ResetRegAll() {
      sleep(1);
      DEV_ERROR("ResetRegAll");
      for (uint32_t i = 0; i < schAicpuNum_; ++i) {
        aicoreManager_[i]->ResetRegAll();
      }
      sleep(1);
      aicoreManager_[0]->CheckAndResetReg();
      DEV_ERROR("Exception reset reg finish.");
    }

    inline void DumpAicorePerfTrace(std::string file = "") {
        (void)file;
#if ENABLE_PERF_TRACE
        std::ostringstream oss;
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_[i]->DumpAicorePerfTrace(oss);
            oss << (i == schAicpuNum_ - 1 ? "" : ",");
        }

        const std::string& str = oss.str();
        uint32_t totalLength = str.length();
        uint32_t startPos = 0;
        uint32_t batchSize = 600;
        while (startPos < totalLength) {
            uint32_t endPos = std::min(startPos + batchSize, totalLength);
            std::string batch = str.substr(startPos, endPos - startPos);
            DEV_ERROR("tile_fwk aicore prof:%s", batch.c_str());
            startPos = endPos;
        }

        if (file != "") {
            std::ofstream os(file);
            os << "[";
            os << oss.str();
            os << "]";
        }
#endif
    }

private:
    AicpuTaskManager aicpuTaskManager_;
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    std::unique_ptr<AiCoreManager> aicoreManager_[MAX_SCHEDULE_AICPU_NUM];
    std::array<bool, MAX_AICORE_NUM> validCore_;
#if ENABLE_AICORE_PRINT
    AicoreLogManager logManager;
#endif
};
} // namespace npu::tile_fwk

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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/scheduler.h"
#include "passes/statistics/ooo_schedule_statistic.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/schedule_ooo/optimize_sort.h"
#include "passes/block_graph_pass/schedule_ooo/estimate_latency.h"

namespace npu::tile_fwk {
class OoOSchedule : public Pass {
public:
    OoOSchedule() : Pass("OoOSchedule") {}
    ~OoOSchedule() override {}

private:
    Status RunOnFunction(Function &function) override;
    bool IsAicpuProgram(std::vector<Operation *> opList);
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    void DoHealthCheckAfter(Function &function, const std::string &folderPath) override;
    void SortTaskList(std::vector<Operation*> &operations, std::vector<Operation*> &taskList);
    Status SortAndLatencyEstimate(std::vector<Operation*> &opList, std::vector<Operation*> &taskOpList,
        int &latency);
    void OoOHealthCheck(OoOScheduler &oooSchedule, Function &function, std::pair<uint64_t, Function*> &program);
    Status A23Schedule(std::vector<Operation*> &opList, Function &function, std::pair<uint64_t, Function*> &program, int &maxWorkeSpaceSize);
    Status A5Schedule(std::vector<Operation*> &opList, Function &function, std::pair<uint64_t, Function*> &program, int &maxWorkeSpaceSize);
    std::vector<Function *> oriFunctions;
    std::map<uint64_t, OoOScheduler> schedulerMap;
    OoOScheduleChecker checker;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H
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
 * \file depend_on.h
 * \brief
 */

#ifndef DEPEND_ON_H
#define DEPEND_ON_H

#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"
#include <cstdint>
#include <vector>
#include <deque>
#include <map>

namespace npu::tile_fwk {
namespace Distributed {
class DependOn {
public:
    DependOn() {};
    ~DependOn() {};

    inline void Init(DeviceTask *deviceTask) { (void)deviceTask; }

    inline void EnqueueOp(uint64_t subGraphId, uint64_t *paramList, uint32_t paramSize) {
        (void)paramList;
        (void)paramSize;
        subGraphId_.push_back(subGraphId);
    }

    inline void PollCompleted(std::vector<uint64_t> &completed) {
        completed.insert(completed.end(), subGraphId_.begin(), subGraphId_.end());
        subGraphId_.clear();
    }

private:
    std::vector<uint64_t> subGraphId_;
};

} // namespace Distributed
} // namespace npu::tile_fwk
#endif // DEPEND_ON_H

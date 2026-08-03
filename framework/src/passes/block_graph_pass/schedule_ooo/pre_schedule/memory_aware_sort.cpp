/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_aware_sort.h"
#include "memory_aware_topo_sort.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

Status MemoryAwareSort::DoSortOps()
{
    APASS_LOG_INFO_F(Elements::Operation, "Using MemoryAwareTopoSort for scheduling.");
    MemoryAwareTopoSort sorter(operations, function_);
    if (sorter.SortOps() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "MemoryAwareTopoSort failed.");
        return FAILED;
    }
    operations = sorter.operations;
    return SUCCESS;
}

} // namespace npu::tile_fwk

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file graph_partition_token_dependency.h
 * \brief Token dependency handling for graph partition finalization.
 */

#ifndef PASS_GRAPH_PARTITION_TOKEN_DEPENDENCY_H
#define PASS_GRAPH_PARTITION_TOKEN_DEPENDENCY_H

#include "interface/utils/common.h"

namespace npu::tile_fwk {

class Function;

Status FinalizePartitionWithTokenDependency(Function& function, bool splitPostLoweringMixedCoreSubgraphs = false,
                                            bool* postLoweringSplitOccurred = nullptr);

} // namespace npu::tile_fwk

#endif // PASS_GRAPH_PARTITION_TOKEN_DEPENDENCY_H

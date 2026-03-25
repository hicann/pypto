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
 * \file index_outcast_checker.cpp
 * \brief
 */

#include <utility>
#include "index_outcast_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "IndexOutcastChecker"

namespace npu {
namespace tile_fwk {

constexpr size_t DST_TILE_INPUT_PARAM_INDEX = 2;

/*
    pypto.scatter_update()约束需要原地写法x = pypto.scatter_update(x, -2, y, z)，多个scatter_update()操作同一个x时计算图中对应的INDEX_OUTCAST应该有先后顺序
    否则将会因操作顺序丢失导致第二次的scatter_update没有基于第一次的计算结果继续处理，而是基于原始结果处理，导致第一次的操作结果丢失。
*/
Status IndexOutcastChecker::CheckIndexOutcastDisorderedCoverage(Function &function) {
    for (const auto &tMap : function.GetTensorMap().tensorMap_) {
        for (const auto &tensor : tMap.second) {
            std::unordered_set<Operation *> indexOutcastConsumers;
            for (const auto &consumerOp : tensor->GetConsumers()) {
                if (consumerOp->GetOpcode() != Opcode::OP_INDEX_OUTCAST) {
                    continue;
                }
                if (consumerOp->GetIOperands().size() <= DST_TILE_INPUT_PARAM_INDEX || consumerOp->GetIOperands()[DST_TILE_INPUT_PARAM_INDEX]->GetMagic() != tensor->GetMagic()) {
                    continue;
                }
                indexOutcastConsumers.insert(consumerOp);
                if (indexOutcastConsumers.size() > 1) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] has multiple OP_INDEX_OUTCAST consumers.", tensor->GetMagic());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
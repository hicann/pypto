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
 * \file distributed_expand.h
 * \brief
 */

#ifndef DISTRIBUTED_EXPAND_H
#define DISTRIBUTED_EXPAND_H

#include <cstdint>
#include <vector>
#include <memory>
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
namespace Distributed {
inline void CreateTileOp(const TileShape& tileShape,
    const std::function<void(int32_t, int32_t, int32_t, int32_t, int32_t)>& callback)
{
    const auto& tileRow = tileShape.GetDistTileRow();
    const auto& tileCol = tileShape.GetDistTileCol();
    int32_t rowCount = tileRow[1] + (tileRow[2] == 0 ? 0 : 1);
    int32_t colCount = tileCol[1] + (tileCol[2] == 0 ? 0 : 1);
    CHECK(tileRow[0] > 0) << "Invalid tiling strategy of the row axis: the first number must be greater than 0, but "
        << "got " << tileRow[0];
    CHECK(tileCol[0] > 0) << "Invalid tiling strategy of the col axis: the first number must be greater than 0, but "
        << "got " << tileCol[0];

    int32_t tileIndex = 0;
    for (int32_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
        int32_t rowShape = ((tileRow[2] != 0) && (rowIndex == rowCount - 1)) ? tileRow[2] : tileRow[0];
        for (int32_t colIndex = 0; colIndex < colCount; colIndex++) {
            int32_t colShape = ((tileCol[2] != 0) && (colIndex == colCount - 1)) ? tileCol[2] : tileCol[0];
            callback(tileIndex, rowIndex * tileRow[0], colIndex * tileCol[0], rowShape, colShape);
            tileIndex++;
        }
    }
}

Tensor SendToRoutingExpert(const Tensor &shmemData, const Tensor &tokenTensor, const Tensor &tokenExpertTable,
    const char *group, const MoeConfig &moeConfig);
void SendToSharedExpert(const Tensor &shmemData, const Tensor &tokenTensor, 
    const Tensor &syncTensor, const char *group);
Tensor CopyToLocalExpert(const Tensor &tokenTensor, const Tensor &syncTensor);
Tensor DispatchSetFlag(Tensor &shmemFlag, const Tensor &tokenExpertTable, 
    const Tensor &syncTensor, const char *group, const MoeConfig &moeConfig);
void TiledSendToRoutingExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledSendToSharedExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledCopyToLocalExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchSetFlag(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNSched(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNBatching(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNCombineInfo(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNValidCnt(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemPut(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemPutUB2GM(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemGet(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemGetGM2UB(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemSignal(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemWaitUntil(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemSet(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemReduce(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemBindTensor(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledMoeDistributedCombineSend(
    Function &function,
    const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand,
    const Operation &op);
void TiledMoeDistributedCombineReceive(
    Function &function,
    const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand,
    const Operation &op);

} // namespace npu::tile_fwk
} // namespace Distributed

#endif // DISTRIBUTED_EXPAND_H
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
 * \file assign_memory_type.h
 * \brief
 */

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_H

#include <queue>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}
    void SpecialCallInterfaceToBeDeleted(Function &function) {
        RunOnFunction(function);
    }
private:
    Status PreCheck(Function &function) override;    
    Status RunOnFunction(Function &function) override;
    void AssignMoveOp(Operation &operation);
    void AssignMoveOpForAssemble(Operation &operation);
    void AssignMoveOpForView(Operation &operation);
    void RunOnOperation(Operation &operation);
    void AssignMemUnknown(Function &function);
    void ProcessAmulBInput(Operation &operation,LogicalTensorPtr &tensor);
    void ProcessViewwithSpecificMem(Operation &operation);
    void AssignSpecialOpMemtype(Operation &op, bool &infoBufferSize);
    void AssignOpReshapeMemtype(Operation &op);
    void AssignOpViewTypeMemtype(Operation &op);
    void AssignOpNopMemtype(Operation &op);
    void AssignMemtypeForSplitReshape(Operation &op, const LogicalTensorPtr &input, const LogicalTensorPtr &output);
    void UpdateOverSizedLocalBuffer(Operation &operation);
    void ProcesSmallTileToLargeTile(Function &function);
    void ProcessLargeTileToSamllTile(Function &function);
    std::string PrintTensorMem(std::shared_ptr<LogicalTensor>& tensor) const;
    ConvertInserter inserter;
};
} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_H

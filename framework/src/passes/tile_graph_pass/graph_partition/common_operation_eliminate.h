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
 * \file common_operation_eliminate.h
 * \brief
 */

#ifndef PASS_COMMON_OPERATION_ELIMINATE_H_
#define PASS_COMMON_OPERATION_ELIMINATE_H_

#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {
class CommonOperationEliminate : public Pass {
public:
    CommonOperationEliminate() : Pass("CommonOperationEliminate")
    {}
    ~CommonOperationEliminate() override = default;
    Status PreCheck(Function &function) override;

private:
    Status RunOnFunction(Function &function) override;
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> GetTensorProducers(Function &function);
    void UpdateConnection(LogicalTensorPtr oldtensors,  LogicalTensorPtr newtensors);
    std::pair<LogicalTensorPtr, std::vector<Operation*>> TensorHashExist(const std::pair<LogicalTensorPtr, std::vector<Operation*>>& tensorProducersPair, std::unordered_set<Operation*>& cacheProducers);
    bool TensorProducersMerge(const std::pair<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerPair, std::unordered_set<Operation*>& cacheProducers);
    void UpdateView(ViewOpAttribute *viewOpAttribute, const std::shared_ptr<LogicalTensor> oldtensors,
                    const std::shared_ptr<LogicalTensor> newtensors) const;
    void UpdateCopy(CopyOpAttribute *copyOpAttribute, const std::shared_ptr<LogicalTensor> oldtensors,
                    const std::shared_ptr<LogicalTensor> newtensors) const;
};
}  // namespace npu::tile_fwk
#endif  // PASS_COMMON_OPERATION_ELIMINATE_H_
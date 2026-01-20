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
 * \file pad_local_buffer.h
 * \brief
 */

#ifndef PAD_LOCAL_BUFFER_H
#define PAD_LOCAL_BUFFER_H
#include <unordered_map>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
/*
默认对尾轴做32B对齐，但是存在特殊情况不允许padding，不做padding的规则如下：
遇到尾轴reduce的op，目前仅支持ROWSUM_COMBINE_AXIS_SINGLE, ROWMAX_COMBINE_AXIS_SINGLE且倒数第二根轴为32B对齐，有这么几个规则
 2.1 遇到elementwise和assemble默认不padding
 2.2 遇到copyout，默认不padding
 2.3 遇到copyin, 如果消费者都是elementwise或者broadcast不做padding(遇到其他特殊类型例如reshape如果不做padding精度可能会有问题？需要确认实际场景)
 2.4 遇到尾轴expand和除copyout之外的moveout类型为不padding的终止条件，即后续的算子需要继续做padding
 2.5 遇到broadcast类型的计算节点，需要判断双输入是否都不padding，如果是的话就继续遍历，否则终止遍历，后续节点可以继续padding
*/
class PadLocalBuffer : public Pass {
public:
    explicit PadLocalBuffer(std::string name = "PadLocalBuffer", bool processTranspose = false) : Pass(name), processTranspose_(processTranspose) {}
    ~PadLocalBuffer() override = default;
private:
    Status RunOnFunction(Function &function) override;
    void PadMatmul(Operation &op, LogicalTensorPtr &in);
    size_t GetPaddingValue(LogicalTensorPtr &in);
    void PadVector(Operation &op, LogicalTensorPtr &in, std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw, bool noPadding);
    bool IsExpandLastDim(const Operation &op);
    void TraverseCopyInConsumers(Function &function, Operation &consumer, std::unordered_set<LogicalTensorPtr> &visitedTensors);
    void TraverseBroadcast(Function &function, Operation &consumer, LogicalTensorPtr output, std::unordered_set<LogicalTensorPtr> &visitedTensors);
    void TraverseAndSetAttr(LogicalTensorPtr &output, Function &function, std::unordered_set<LogicalTensorPtr> &visitedTensors);
    bool IsReduceLastDim(const Operation &op);
    void ProcessReduce(Function &function, Operation &op);
    void ProcessBroadcast(Operation &op, size_t blockPadding);
    void ProcessCopyIn(Function &function, Operation &op);
    Status ProcessTranspose(Function &function);
    void PadVectorForAxisCombine(Operation &op, LogicalTensorPtr &in, std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw);
    int64_t ProcessBroadcastForAxisCombine(LogicalTensorPtr &inTensor);
    bool IsMatmul(const LogicalTensorPtr &tensor) const;
    bool IsVector(const LogicalTensorPtr &tensor);
    void DoPadding(Function &function);
    bool IsInputInt8(const Operation &op, const LogicalTensorPtr &in) const;
    bool processTranspose_;
    std::unordered_map<int64_t, int64_t> broadcastLastAxis_;
    bool combineAxis{false};
    bool forceCombineAxis{false};
};
} // namespace
#endif  // PAD_LOCAL_BUFFER_H
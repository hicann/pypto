/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file axis_combine_marker.h
 * \brief
 */

#ifndef AXIS_COMBINE_MARKER_H
#define AXIS_COMBINE_MARKER_H

#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
namespace npu::tile_fwk {
enum class AxisReorderStatus {
    ENABLE = 0,  // 明确可以支持合轴优化
    DISABLE,  // 尾轴为1，但是不支持合轴优化的场景
    UNKNOWN   // 不涉及合轴优化
};
class AxisCombineMarker
{
  public:
    AxisCombineMarker() = default;
    ~AxisCombineMarker() = default;
    bool IsTensorEnableAxisCombine(LogicalTensorPtr tensor);
    void Run(Function &function);
  private:
    void Init(Function &function);
    std::vector<Operation *> opList_;
    std::vector<std::vector<uint16_t>> opInGraph_;
    std::vector<std::vector<uint16_t>> opOutGraph_;
    void UpdateOpACEnableForward(uint16_t opIdx);
    void UpdateOpACEnableBackward(uint16_t opIdx);
    void ForwardVisit();
    void BackwardVisit();
    std::unordered_map<LogicalTensorPtr, AxisReorderStatus> tensorStatus_;
};
}
#endif
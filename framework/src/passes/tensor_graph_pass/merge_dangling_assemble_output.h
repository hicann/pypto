/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <vector>

#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class MergeDanglingAssembleOutput : public Pass {
public:
    MergeDanglingAssembleOutput() : Pass("MergeDanglingAssembleOutput") {}
    ~MergeDanglingAssembleOutput() override = default;

private:
    struct AssembleVersion {
        Operation* producer;
        LogicalTensorPtr output;
        std::string signature;
    };

    using VersionGroup = std::vector<AssembleVersion>;
    using VersionGroups = std::vector<VersionGroup>;

    Status RunOnFunction(Function& function) override;
    std::string BuildLogicalTensorSignature(const LogicalTensor& tensor);
    VersionGroups BuildVersionGroups(Function& function);
    const AssembleVersion* FindMergeTarget(const VersionGroup& group, size_t index);
    bool TokenHasConsumer(Function& function, const ir::VarPtr& token);
    void PruneRedundantTokens(Function& function, Operation& producer, const LogicalTensorPtr& target);
    LogicalTensorPtr MergeVersion(Function& function, Operation& producer, const LogicalTensorPtr& target);
    void ReclaimDetachedTensors(Function& function, const std::vector<LogicalTensorPtr>& tensors);
};

} // namespace npu::tile_fwk

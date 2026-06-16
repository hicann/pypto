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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ir/expr.h"
#include "ir/stmt.h"

namespace npu::tile_fwk {

class Function;
class LogicalTensor;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;
using LogicalTensors = std::vector<LogicalTensorPtr>;

bool IsPureTensorOpSeq(const pypto::ir::SeqStmtsPtr& seq);

std::shared_ptr<Function> CreatePathFuncFromSeq(
    const pypto::ir::SeqStmtsPtr& seq, Function& dynFunc,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames);

pypto::ir::StmtPtr TransformAndBuildStmts(
    pypto::ir::StmtPtr stmt, Function& dynFunc,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames);

pypto::ir::StmtPtr CreateFunctionByStmt(
    pypto::ir::StmtPtr stmt, Function& dynFunc, const std::vector<std::string>& externalVarNames);

void BuildDynFuncSlotScope(std::shared_ptr<Function> dynFunc, const LogicalTensors& params);

} // namespace npu::tile_fwk
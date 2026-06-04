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
 * \file cell_match_dynamic.h
 * \brief Per-launch evaluation and refresh of dynamic CellMatchTableDesc.stride on host.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

namespace npu::tile_fwk::dynamic {

struct Evaluator;

std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(
    const DyndevFunctionAttribute& dynAttr, Evaluator& eval);

void PatchHostDynamicCellMatchTableDesc(
    DevAscendProgram* hostDevProg, const std::vector<DevDynamicCellMatchStridePatch>& patches);

void WriteDynamicCellMatchStridePatchesToLaunchArgs(
    int64_t* launchInputs, const std::vector<DevDynamicCellMatchStridePatch>& patches);

void ValidateDynamicCellMatchTableMemBudget(
    const DyndevFunctionAttribute& dynAttr, DevAscendProgram* hostDevProg);

void RefillDynamicMemBudgets(
    DevAscendProgram* hostDevProg, DyndevFunctionAttribute& dynAttr, Evaluator& eval);

std::vector<DevDynamicCellMatchStridePatch> PrepareHostDynamicCellMatchForLaunch(
    DyndevFunctionAttribute& dynAttr, Evaluator& eval, DevAscendProgram* hostDevProg);

} // namespace npu::tile_fwk::dynamic

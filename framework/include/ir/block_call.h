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
 * \file block_call.h
 * \brief
 */

#pragma once
#include "ir/program.h"
#include "ir/function.h"
#include "ir/value.h"

#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace pto {
using BlockFunctionType = std::function<std::shared_ptr<pto::Function>(
    const std::vector<std::shared_ptr<pto::TensorValue>>&,
    const std::vector<std::shared_ptr<pto::TensorValue>>&,
    const std::vector<std::shared_ptr<pto::ScalarValue>>&)>;

std::vector<npu::tile_fwk::Tensor> CallBlock(const pto::FunctionPtr &blockFuncPtr,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &outputTensorArgs,
    const std::vector<npu::tile_fwk::SymbolicScalar> &indices);

std::vector<npu::tile_fwk::Tensor> CallBlock(const BlockFunctionType &blockFunc,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &outputTensorArgs,
    const std::vector<npu::tile_fwk::SymbolicScalar> &indices);
}
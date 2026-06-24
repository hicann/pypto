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
 * \file fake_trans_operation_impl.cpp
 * \brief FakeTrans operation - connects input to output without processing.
 *        This op is a placeholder that will be replaced by a real op in the pass stage.
 */

#include "interface/inner/pre_def.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/error_code.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace FakeTrans {

void ConstructTileGraph(
    Function& function, const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr)
{
    function.AddOperation(Opcode::OP_FAKE_TRANS, {operandVec[0]}, {cTensorPtr});
}

Tensor FakeTrans(const Tensor& input, const Tensor& output)
{
    DECLARE_TRACER();
    Function* functionPtr = Program::GetInstance().GetCurrentFunction();
    CHECK_OP(functionPtr != nullptr) << "No current function context.";
    functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {input.GetStorage()}, {output.GetStorage()});
    return Tensor(output.GetStorage());
}

} // namespace FakeTrans
} // namespace tile_fwk
} // namespace npu
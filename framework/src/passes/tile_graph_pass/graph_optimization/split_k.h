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
 * \file split_k.h
 * \brief
 */

#ifndef CUBE_PROCESS_H
#define CUBE_PROCESS_H

#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";

class SplitK : public Pass {
public:
    SplitK() : Pass("SplitK") {}
    ~SplitK() override = default;

    Status PreCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status EliminateReduceAcc(Function &function);
};
}
#endif // CUBE_PROCESS_H
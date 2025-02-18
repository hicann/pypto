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
 * \file dynamic_mla.h
 * \brief
 */

#pragma once
#ifndef DYNAMIC_NSA
#define DYNAMIC_NSA

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "operator/models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {
void GenSlc(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int l_prime = 64, int d = 16, int front = 1,
    int near = 2, int topk = 16);

void GenSlcV2(const Tensor &x, Tensor &out, int validSize, int l_prime = 64, int d = 16, int front = 1,
    int near = 2, int topk = 16);

void GenTopkIndicesFun(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int front = 1, int near = 2);

std::vector<Tensor> GenTopkIndices(const Tensor &tmpOut, int s_slc, int actualTopk, SymbolicScalar validSize, bool isDyn);


} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA

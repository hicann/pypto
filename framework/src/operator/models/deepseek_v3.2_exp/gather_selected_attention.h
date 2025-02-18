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
 * \file gather_selected_attention.h
 * \brief
 */

#pragma once
#ifndef GATHER_SELECTED_ATTENTION_H
#define GATHER_SELECTED_ATTENTION_H

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"

#include "dsia_common.h"

namespace npu::tile_fwk {
void SelectedAttentionComputeV2(
    const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D, const Tensor &kNopeScales, 
    const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &kvSlcActSeqs, const int nQ, const int nKv, const float softmaxScale, 
    const int topk, const int blockSize, const int maxBlockNumPerBatch, Tensor &attentionOut, SaTileShapeConfig tileConfig={}
);

void SelectedAttentionV2(
    const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D, const Tensor &kNopeScales, 
    const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &kvSlcActSeqs, const int nQ, const int nKv, const float softmaxScale, 
    const int topk, const int blockSize, const int maxBlockNumPerBatch, Tensor &attentionOut, SaTileShapeConfig tileConfig={}
);
} // namespace npu::tile_fwk

#endif // GATHER_SELECTED_ATTENTION_H

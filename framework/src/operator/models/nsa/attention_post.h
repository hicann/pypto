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
 * \file attention_post.h
 * \brief
 */

#pragma once
#ifndef ATTENTION_POST_DYNAMIC
#define ATTENTION_POST_DYNAMIC

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {

struct PostTileConfig {
    int tileB = 8;  // tileB is 8
    int tileS = 1;
};

struct PostTensors {
    Tensor weightUV;
    Tensor weightO;
    Tensor weightUvScale;
    Tensor smoothScalesWUv;
    Tensor weightOScale;
    Tensor smoothScalesWo;
};

void PostCompute(Tensor &input, PostTensors &postTensors, const PostTileConfig &tileConfig, Tensor &postOut);

void AttentionPostStandalone(Tensor &input, PostTensors &postTensors, const PostTileConfig &tileConfig, Tensor &postOut);

} // namespace npu::tile_fwk

#endif // ATTENTION_POST_DYNAMIC

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
 * \file mla_prolog.h
 * \brief
 */

#pragma once
#ifndef MLA_PROLOG
#define MLA_PROLOG

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

struct MlaQuantInputs {
    Tensor dequantScaleX;
    Tensor dequantScaleWDq;
    Tensor dequantScaleWUqQr;
    Tensor dequantScaleWDkvKr;
    Tensor quantScaleCkv;
    Tensor quantScaleCkr;
    Tensor smoothScalesCq;
};

std::vector<Tensor> QkvPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, MlaQuantInputs quantInputs,bool splitReduceLastDim=true, bool splitK=false);

void MlaProlog(Tensor tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, MlaQuantInputs quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq = 1e-5f,
    float epsilonCkv = 1e-5f, std::string cacheMode = "BNSD",bool splitReduceLastDim=true, bool splitK=false);

} // namespace npu::tile_fwk

#endif // MLA_PROLOG
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
 * \file decode_indexer_attention.h
 * \brief
 */
 
#pragma once
#ifndef DECODE_INDEXER_ATTENTION
#define DECODE_INDEXER_ATTENTION
 
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "dynamic_mla_v32.h"
#include "gather_after_prolog.h"
#include "sparse_flash_attention.h"
#include "dsia_common.h"
#include "lightning_indexer.h"
#include "lightning_indexer_prolog.h"
 
namespace npu::tile_fwk {
void DecodeIndexerAttention(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, Tensor &blockTable, Tensor &actSeqs,
    const Tensor &qW, const Tensor &kW, const Tensor &projW, const Tensor &lnW, const Tensor &lnBias,
    const Tensor &indexKCache, Tensor &attentionOut,
    Tensor &gatherResTmp, Tensor &tmpTopkInput, Tensor &tmpIndexerTopkRes, Tensor &tmpRowSumOut,  Tensor &rmsResOut,
    Tensor &queryOut,Tensor &weightsOut, Tensor &qNopeOut,Tensor &qRopeOut, const DSIASimpleParams &params);
 
} // namespace npu::tile_fwk
 
#endif // DECODE_INDEXER_ATTENTION
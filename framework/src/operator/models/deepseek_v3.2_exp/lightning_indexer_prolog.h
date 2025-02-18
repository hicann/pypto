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
 * \file lightning_indexer_prolog.h
 * \brief
 */

#pragma once
#ifndef LIGHTNING_INDEXER_PROLOG_H
#define LIGHTNING_INDEXER_PROLOG_H

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "dsia_common.h"

namespace npu::tile_fwk {

struct IndexerShapeParams {
    int b;
    int seq;
    int dim;
    int qLoraRank;
    int headDim;
    int headNum;
    int ropeHeadDim;
    int blockSize;
    int blockNum;
    int nKV;
    int s2;
    int tileBS = 2;
    IndexerTileShapeConfig indexerTileConfigs;
    RopeTileShapeConfig ropeTileConfigs;
};

struct IndexerPrologInput {
    const Tensor &x;
    const Tensor &qr;
    const Tensor &qW;
    const Tensor &kW;
    const Tensor &projW;
    const Tensor &lnW;
    const Tensor &lnBias;
    const Tensor &cos;
    const Tensor &sin;
    const Tensor &kCache;
    const Tensor &kCacheIndex;
    const Tensor &blockTable;
};

struct IndexerPrologInputData {
    RawTensorDataPtr xData;
    RawTensorDataPtr qrData;
    RawTensorDataPtr qWData;
    RawTensorDataPtr kWData;
    RawTensorDataPtr projWData;
    RawTensorDataPtr lnWData;
    RawTensorDataPtr lnBiasData;
    RawTensorDataPtr cosData;
    RawTensorDataPtr sinData;
    RawTensorDataPtr kCacheData;
    RawTensorDataPtr kCacheIndexData;
    RawTensorDataPtr blockTableData;
};

struct IndexerPrologOutput {
    Tensor &query;
    Tensor &weight;
    Tensor &kCacheOut;
};

struct IndexerPrologOutputData {
    RawTensorDataPtr queryData;
    RawTensorDataPtr weightData;
};

template <typename T>
struct IndexerPrologOutputGolden {
    std::vector<T> queryGolden;
    std::vector<T> weightGolden;
    std::vector<T> kCacheOutGolden;
};

void LightningIndexerPrologCompute(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params);

void LightningIndexerProlog(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params);

} // namespace npu::tile_fwk

#endif
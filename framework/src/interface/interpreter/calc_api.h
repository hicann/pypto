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
 * \file calc_api.h
 * \brief Calculator API
 */

#pragma once

#include <cstdint>
#include <ostream>
#include "tilefwk/data_type.h"
#include "tilefwk/element.h"

namespace npu::tile_fwk {

struct TensorData {
    void *dataPtr = nullptr;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int64_t storageOffset;
    DataType dtype;
    bool isAxisCombine = false;
};

struct MatMulParam {
    bool aTrans = false;
    bool bTrans = false;
    int64_t kStep = 0;
    uint64_t scale = 0;
    int relu = 0;
    const TensorData *scalePtr = nullptr;
    const TensorData *biasPtr = nullptr;
};

enum class CmpOperationType {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};
enum class CmpModeType {
    BOOL,
    BIT,
};

struct CalcOps {
    void (*Random)(const TensorData &);
    bool (*AllClose)(const TensorData &, const TensorData &, double, double);

    void (*Cast)(const TensorData &, const TensorData &, CastMode);
    void (*Exp)(const TensorData &, const TensorData &);
    void (*Neg)(const TensorData &, const TensorData &);
    void (*Rsqrt)(const TensorData &, const TensorData &);
    void (*Sign)(const TensorData &, const TensorData &);
    void (*Sqrt)(const TensorData &, const TensorData &);
    void (*Ceil)(const TensorData &, const TensorData &);
    void (*Floor)(const TensorData &, const TensorData &);
    void (*Trunc)(const TensorData &, const TensorData &);
    void (*Round)(const TensorData &, const TensorData &, int);
    void (*Reciprocal)(const TensorData &, const TensorData &);
    void (*Relu)(const TensorData &, const TensorData &);
    void (*BitwiseNot)(const TensorData &, const TensorData &);
    void (*Abs)(const TensorData &, const TensorData &);
    void (*Brcb)(const TensorData &, const TensorData &);  
    void (*WhereTT)(const TensorData &, const TensorData &, const TensorData &, const TensorData &);
    void (*WhereTS)(const TensorData &, const TensorData &, const TensorData &, const Element &);
    void (*WhereST)(const TensorData &, const TensorData &, const Element &, const TensorData &);
    void (*WhereSS)(const TensorData &, const TensorData &, const Element &, const Element &);
    void (*Ln)(const TensorData &, const TensorData &);
    void (*IsFinite)(const TensorData &, const TensorData &);
    void (*LogicalNot)(const TensorData &, const TensorData &);
    void (*Range)(const TensorData &, const Element &, const Element &, const Element &);
    void (*Compare)(const TensorData &, const TensorData &, const TensorData &, CmpOperationType, CmpModeType);
    void (*Cmps)(const TensorData &, const TensorData &, const Element &, CmpOperationType, CmpModeType);
    void (*Hypot)(const TensorData &, const TensorData &, const TensorData &);
    void (*LogicalAnd)(const TensorData &, const TensorData &, const TensorData &);

    void (*AddS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*SubS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*MulS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*DivS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*FmodS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*BitwiseAndS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*BitwiseOrS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*BitwiseXorS)(const TensorData &, const TensorData &, const Element &, bool);
    void (*GcdS)(const TensorData &, const TensorData &, const Element &);

    void (*Add)(const TensorData &, const TensorData &, const TensorData &);
    void (*Sub)(const TensorData &, const TensorData &, const TensorData &);
    void (*Mul)(const TensorData &, const TensorData &, const TensorData &);
    void (*Div)(const TensorData &, const TensorData &, const TensorData &);
    void (*Fmod)(const TensorData &, const TensorData &, const TensorData &);
    void (*Pow)(const TensorData &, const TensorData &, const TensorData &);
    void (*BitwiseAnd)(const TensorData &, const TensorData &, const TensorData &);
    void (*BitwiseOr)(const TensorData &, const TensorData &, const TensorData &);
    void (*BitwiseXor)(const TensorData &, const TensorData &, const TensorData &);
    void (*CopySign)(const TensorData &, const TensorData &, const TensorData &);
    void (*Gcd)(const TensorData &, const TensorData &, const TensorData &);

    void (*PairSum)(const TensorData &, const TensorData &, const TensorData &);
    void (*PairMax)(const TensorData &, const TensorData &, const TensorData &);
    void (*PairMin)(const TensorData &, const TensorData &, const TensorData &);

    void (*Min)(const TensorData &, const TensorData &, const TensorData &);
    void (*Max)(const TensorData &, const TensorData &, const TensorData &);
    void (*MinS)(const TensorData &, const TensorData &, const Element &);
    void (*MaxS)(const TensorData &, const TensorData &, const Element &);

    void (*RowSumExpand)(const TensorData &, const TensorData &, int);
    void (*RowMinExpand)(const TensorData &, const TensorData &, int);
    void (*RowMaxExpand)(const TensorData &, const TensorData &, int);

    void (*RowSumSingle)(const TensorData &, const TensorData &, int);
    void (*RowMinSingle)(const TensorData &, const TensorData &, int);
    void (*RowMaxSingle)(const TensorData &, const TensorData &, int);

    void (*RowMinLine)(const TensorData &, const TensorData &, int);
    void (*RowMaxLine)(const TensorData &, const TensorData &, int);

    void (*OneHot)(const TensorData &, const TensorData &, int);
    void (*ExpandS)(const TensorData &, const Element &);
    void (*Expand)(const TensorData &, const TensorData &);
    void (*GatherElements)(const TensorData &, const TensorData &, const TensorData &, int);
    void (*IndexAdd)(const TensorData &, const TensorData &, const TensorData &, const TensorData &, int, const Element &);
    void (*TriU)(const TensorData &, const TensorData &, int);
    void (*TriL)(const TensorData &, const TensorData &, int);
    void (*CumSum)(const TensorData &, const TensorData &, int);
    void (*IndexPut)(const TensorData &, const TensorData &, const std::vector<TensorData> &, const TensorData &, bool);

    void (*Reshape)(const TensorData &, const TensorData &);
    void (*Permute)(const TensorData &, const TensorData &, const std::vector<int64_t> &);
    void (*Transpose)(const TensorData &, const TensorData &, int64_t, int64_t);

    void (*ReduceAcc)(const TensorData &, const std::vector<TensorData> &);
    void (*Copy)(const TensorData &, const TensorData &, bool);
    void (*ScatterUpdate)(const TensorData &, const TensorData &, const TensorData &, const TensorData &, int, std::string, int);
    void (*ScatterElement)(const TensorData &, const TensorData &, const TensorData &, const Element &, int, int);
    void (*Scatter)(const TensorData &, const TensorData &, const TensorData &, const TensorData &,
        int, int);
    void (*FormatND2NZ)(const TensorData &, const TensorData &);
    void (*FormatNZ2ND)(const TensorData &, const TensorData &);
    void (*QuantPreCompute)(const TensorData &, const TensorData &, const TensorData *, uint64_t, int);
    void (*MatMul)(const TensorData &, const TensorData &, const TensorData &, const TensorData *, MatMulParam &);

    void (*BitSort)(const TensorData &, const TensorData &, int64_t, bool, int64_t);
    void (*TiledMrgSort)(const TensorData &, const TensorData &, const TensorData &, const TensorData &, const TensorData &, int, int);
    void (*Extract)(const TensorData &, const TensorData &, int, bool);
    void (*Topk)(const TensorData &, const TensorData &, int64_t, int64_t, bool);
    void (*TopK)(const TensorData &, const TensorData &, const TensorData &, int, int, bool);
    void (*TopkSort)(const TensorData &, const TensorData &, const TensorData &, int);
    void (*TopkMerge)(const TensorData &, const TensorData &, int);
    void (*TopkExtract)(const TensorData &, const TensorData &, int, bool);
    void (*TwoTileMrgSort)(const TensorData &, const TensorData &);
    void (*Sort)(const TensorData &, const TensorData &, const TensorData &, int64_t, bool);
    void (*Gather)(const TensorData &, const TensorData &, const TensorData &, int64_t);
    void (*GatherINUB)(
        const TensorData &, const TensorData &, const TensorData &, const TensorData &, int64_t, int64_t);
    void (*GatherInL1)(
        const TensorData &, const TensorData &, const TensorData &, const TensorData &, int64_t);
    void (*BitwiseRightShift)(const TensorData &, const TensorData &, const TensorData &);
    void (*BitwiseLeftShift)(const TensorData &, const TensorData &, const TensorData &);
    void (*BitwiseRightShiftS)(const TensorData &, const TensorData &, const Element &);
    void (*BitwiseLeftShiftS)(const TensorData &, const TensorData &, const Element &);
    void (*SBitwiseRightShift)(const TensorData &, const Element &, const TensorData &);
    void (*SBitwiseLeftShift)(const TensorData &, const Element &, const TensorData &);
};

extern "C" struct CalcOps *GetCalcOps();
} // namespace npu::tile_fwk

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
#include "tilefwk/tensor.h"
#include "raw_tensor_data.h"
namespace npu::tile_fwk {
struct MatMulParam {
    bool aTrans = false;
    bool bTrans = false;
    int64_t kStep = 0;
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
    void (*Random)(LogicalTensorDataPtr);
    bool (*AllClose)(LogicalTensorDataPtr, LogicalTensorDataPtr, double, double);

    void (*Cast)(LogicalTensorDataPtr, LogicalTensorDataPtr, CastMode);
    void (*Exp)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Neg)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Rsqrt)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Sqrt)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Abs)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*WhereTT)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*WhereTS)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &);
    void (*WhereST)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, LogicalTensorDataPtr);
    void (*WhereSS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, const Element &);
    void (*Ln)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*LogicalNot)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Range)(LogicalTensorDataPtr, const Element &, const Element &, const Element &);
    void (*Compare)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, CmpOperationType, CmpModeType);
    void (*LogicalAnd)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);

    void (*AddS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool);
    void (*SubS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool);
    void (*MulS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool);
    void (*DivS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool);

    void (*Add)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Sub)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Mul)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Div)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);

    void (*PairSum)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*PairMax)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*PairMin)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);

    void (*Min)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Max)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*MinS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &);
    void (*MaxS)(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &);

    void (*RowSumExpand)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*RowMinExpand)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*RowMaxExpand)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);

    void (*RowSumSingle)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*RowMinSingle)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*RowMaxSingle)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);

    void (*OneHot)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*ExpandS)(LogicalTensorDataPtr, const Element &);
    void (*Expand)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*GatherElements)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*IndexAdd)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, int, const Element &);
    void (*CumSum)(LogicalTensorDataPtr, LogicalTensorDataPtr, int);
    void (*IndexPut)(LogicalTensorDataPtr, LogicalTensorDataPtr, std::vector<LogicalTensorDataPtr>, LogicalTensorDataPtr, bool);

    void (*Reshape)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*Permute)(LogicalTensorDataPtr, LogicalTensorDataPtr, const std::vector<int64_t> &);
    void (*Transpose)(LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t, int64_t);

    void (*ReduceAcc)(LogicalTensorDataPtr, const std::vector<LogicalTensorDataPtr> &);
    void (*Copy)(LogicalTensorDataPtr, LogicalTensorDataPtr, bool);
    void (*ScatterUpdate)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, int, std::string, int);
    void (*ScatterElement)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, int, int);
    void (*Scatter)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr,
        int, int);
    void (*FormatND2NZ)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*FormatNZ2ND)(LogicalTensorDataPtr, LogicalTensorDataPtr);
    void (*MatMul)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, MatMulParam &);

    void (*BitSort)(LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t, bool);
    void (*Extract)(LogicalTensorDataPtr, LogicalTensorDataPtr, int, bool);
    void (*Topk)(LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t, int64_t, bool);
    void (*Gather)(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t);
    void (*GatherINUB)(
        LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t, int64_t);
};

extern "C" struct CalcOps *GetCalcOps();
} // namespace npu::tile_fwk

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
 * \file calc.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>

#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "raw_tensor_data.h"
#include "calc_api.h"

namespace npu::tile_fwk::calc {

CalcOps *GetCalcOps();

inline bool IsVerifyEnabled() {
    return GetCalcOps() != nullptr;
}

inline void Random(LogicalTensorDataPtr out) {
    GetCalcOps()->Random(out);
}
inline bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol = 1e-8, double rtol = 1e-5) {
    return GetCalcOps()->AllClose(self, other, atol, rtol);
}
inline void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode = CAST_NONE) {
    GetCalcOps()->Cast(out, self, mode);
}
inline void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Exp(out, self);
}
inline void Neg(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Neg(out, self);
}
inline void Rsqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Rsqrt(out, self);
}
inline void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Sqrt(out, self);
}
inline void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Abs(out, self);
}
inline void Brcb(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Brcb(out, self);
}
inline void WhereTT(
    LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, LogicalTensorDataPtr other) {
    GetCalcOps()->WhereTT(out, condition, input, other);
}
inline void WhereTS(
    LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, const Element &other) {
    GetCalcOps()->WhereTS(out, condition, input, other);
}
inline void WhereST(
    LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, LogicalTensorDataPtr other) {
    GetCalcOps()->WhereST(out, condition, input, other);
}
inline void WhereSS(
    LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, const Element &other) {
    GetCalcOps()->WhereSS(out, condition, input, other);
}
inline void Ln(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Ln(out, self);
}
inline void LogicalNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->LogicalNot(out, self);
}
inline void Range(LogicalTensorDataPtr out, const Element &start, const Element &end, const Element &step) {
    GetCalcOps()->Range(out, start, end, step);
}
inline void Compare(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
    CmpOperationType operation, CmpModeType mode) {
    GetCalcOps()->Compare(out, self, other, operation, mode);
}
inline void LogicalAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->LogicalAnd(out, self, other);
}

inline void AddS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false) {
    GetCalcOps()->AddS(out, self, scalar, reverse);
}
inline void SubS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false) {
    GetCalcOps()->SubS(out, self, scalar, reverse);
}
inline void MulS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false) {
    GetCalcOps()->MulS(out, self, scalar, reverse);
}
inline void DivS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false) {
    GetCalcOps()->DivS(out, self, scalar, reverse);
}

inline void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Add(out, self, other);
}
inline void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Sub(out, self, other);
}
inline void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Mul(out, self, other);
}
inline void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Div(out, self, other);
}
inline void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Min(out, self, other);
}
inline void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->Max(out, self, other);
}
inline void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar) {
    GetCalcOps()->MinS(out, self, scalar);
}
inline void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar) {
    GetCalcOps()->MaxS(out, self, scalar);
}
/* used by reducc op, if shape are not same, need masked */
inline void PairSum(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->PairSum(out, self, other);
}
inline void PairMax(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->PairMax(out, self, other);
}
inline void PairMin(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    GetCalcOps()->PairMin(out, self, other);
}
inline void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowSumExpand(out, self, dim);
}
inline void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowMinExpand(out, self, dim);
}
inline void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowMaxExpand(out, self, dim);
}
inline void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowSumSingle(out, self, dim);
}
inline void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowMinSingle(out, self, dim);
}
inline void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    GetCalcOps()->RowMaxSingle(out, self, dim);
}

inline void OneHot(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int numClasses) {
    GetCalcOps()->OneHot(out, self, numClasses);
}
inline void ExpandS(LogicalTensorDataPtr out, const Element &scalar) {
    GetCalcOps()->ExpandS(out, scalar);
}
inline void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Expand(out, self);
}
inline void GatherElements(
    LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int axis) {
    GetCalcOps()->GatherElements(out, params, indices, axis);
}
inline void IndexAdd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr src,
    LogicalTensorDataPtr indices, int axis, const Element &alpha = Element(DT_FP32, 1.0)) {
    GetCalcOps()->IndexAdd(out, self, src, indices, axis, alpha);
}
inline void CumSum(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int axis) {
    GetCalcOps()->CumSum(out, in, axis);
}
inline void IndexPut(LogicalTensorDataPtr out, LogicalTensorDataPtr self, std::vector<LogicalTensorDataPtr> indices,
    LogicalTensorDataPtr values, bool accumulate = false) {
    GetCalcOps()->IndexPut(out, self, indices, values, accumulate);
}
inline void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->Reshape(out, self);
}
inline void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t> &dim) {
    GetCalcOps()->Permute(out, self, dim);
}
inline void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1) {
    GetCalcOps()->Transpose(out, self, dim0, dim1);
}

inline void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr> &tdatas) {
    GetCalcOps()->ReduceAcc(out, tdatas);
}

inline void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans = false) {
    GetCalcOps()->Copy(out, self, trans);
}
inline void ScatterUpdate(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    LogicalTensorDataPtr dst, int axis = -2, std::string cacheMode = "BSND", int blockSize = 1) {
    GetCalcOps()->ScatterUpdate(out, self, index, dst, axis, cacheMode, blockSize);
}
inline void ScatterElement(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    const Element &src, int axis, int reduce) {
    GetCalcOps()->ScatterElement(out, self, index, src, axis, reduce);
}
inline void Scatter(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    LogicalTensorDataPtr src, int axis, int reduce) {
    GetCalcOps()->Scatter(out, self, index, src, axis, reduce);
}
inline void BitSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, bool descending) {
    GetCalcOps()->BitSort(out, self, axis, descending);
}
inline void Gather(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int64_t axis) {
    GetCalcOps()->Gather(out, params, indices, axis);
}
inline void GatherINUB(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices,
    LogicalTensorDataPtr pageTable, int64_t blockSize, int64_t axis) {
    GetCalcOps()->GatherINUB(out, params, indices, pageTable, blockSize, axis);
}

inline void Extract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mod, bool descending) {
    GetCalcOps()->Extract(out, self, mod, descending);
}

inline void Topk(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, int64_t k, bool descending) {
    GetCalcOps()->Topk(out, self, axis, k, descending);
}

// matmul
inline void FormatNZ2ND(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->FormatNZ2ND(out, self);
}
inline void FormatND2NZ(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    GetCalcOps()->FormatND2NZ(out, self);
}

inline void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, 
    MatMulParam param = {false, false, 0}) {
    GetCalcOps()->MatMul(out, self, other, nullptr, param);
}

inline void AccMatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
    LogicalTensorDataPtr acc = nullptr, MatMulParam param = {false, false, 0}) {
    GetCalcOps()->MatMul(out, self, other, acc, param);
}
} // namespace npu::tile_fwk::calc

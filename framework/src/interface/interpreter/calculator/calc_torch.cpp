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
 * \file calc_torch.cpp
 * \brief
 */

#include <torch/torch.h>
#include "tilefwk/error.h"
#include "../calc_api.h"

namespace npu::tile_fwk {

#define AXIS_TO_LAST -2
#define NUM_VALUE_8 8
#define BLOCK_SIZE 32

static torch::ScalarType FromDataType(DataType t) {
    switch (t) {
        case DT_INT8: return torch::kInt8;
        case DT_INT16: return torch::kInt16;
        case DT_INT32: return torch::kInt32;
        case DT_INT64: return torch::kInt64;
        case DT_FP16: return torch::kFloat16;
        case DT_FP32: return torch::kFloat32;
        case DT_BF16: return torch::kBFloat16;
        /* unsigned int are limited supported, use signed types for temp */
        case DT_UINT8: return torch::kInt8;
        case DT_UINT16: return torch::kInt16;
        case DT_UINT32: return torch::kInt32;
        case DT_UINT64: return torch::kInt64;
        case DT_BOOL: return torch::kBool;
        case DT_DOUBLE: return torch::kDouble;
        case DT_INT4:
        case DT_FP8:
        case DT_HF4:
        case DT_HF8:
        default: assert(0);
    }
    return torch::ScalarType::Undefined;
}

static at::Scalar From(const Element &elem) {
    switch (elem.GetDataType()) {
        case DT_BOOL:
        case DT_INT4:
        case DT_INT8:
        case DT_INT16:
        case DT_INT32:
        case DT_INT64: return at::Scalar(elem.GetSignedData());
        case DT_FP16:
        case DT_FP32:
        case DT_BF16:
        case DT_DOUBLE: return at::Scalar(elem.GetFloatData());
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
            // lower version of pytorch not support uint64 type, use int64 for temp
            return at::Scalar(static_cast<int64_t>(elem.GetUnsignedData()));
        case DT_FP8:
        case DT_HF4:
        case DT_HF8:
        default: assert(0);
    }
    return at::Scalar();
}

static torch::Tensor From(LogicalTensorDataPtr data) {
    RawTensorDataPtr raw = data->GetData();
    auto tensor = torch::from_blob(raw->data(), raw->GetShape(), FromDataType(raw->GetDataType()));
    auto view = tensor.as_strided(data->GetShape(), raw->GetStride(), data->GetStorageOffset());
    if (data->IsAxisCombine())
        view = view.transpose_(-1, AXIS_TO_LAST);
    return view;
}

static torch::Tensor View(const torch::Tensor &self, const std::vector<int64_t> &shape, const std::vector<int64_t> &offset) {
    int64_t storageOffset = self.storage_offset();
    for (size_t dim = 0; dim < offset.size(); dim++) {
        storageOffset += self.stride(dim) * offset[dim];
    }
    return self.as_strided(shape, self.strides(), storageOffset);
}

static bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol, double rtol) {
    return From(self).allclose(From(other), atol, rtol);
}

static void Random(LogicalTensorDataPtr out) {
    auto tout = From(out);
    torch::rand_out(tout, tout.sizes());
}

static void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::exp_out(tout, From(self));
}

static void Neg(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::neg_out(tout, From(self));
}

static void Rsqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::rsqrt_out(tout, From(self));
}

static void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::sqrt_out(tout, From(self));
}

static void LogicalNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::logical_not_out(tout, From(self));
}

static void LogicalAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::logical_and_out(tout, From(self), From(other));
}

static void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::abs_out(tout, From(self));
}

static void WhereTT(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, LogicalTensorDataPtr other) {
    torch::Tensor tout = From(out);
    torch::Tensor tcondition = From(condition);
    torch::Tensor tinput = From(input);
    torch::Tensor tother = From(other);
    torch::where_out(tout, tcondition, tinput, tother);
}

static void WhereTS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, const Element &other) {
    torch::Tensor tout = From(out);
    torch::Tensor tcondition = From(condition);
    torch::Tensor tinput = From(input);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout, tcondition, tinput, tother);
}

static void WhereST(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, LogicalTensorDataPtr other) {
    torch::Tensor tout = From(out);
    torch::Tensor tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    torch::Tensor tother = From(other);
    torch::where_out(tout, tcondition, tinput, tother);
}

static void WhereSS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, const Element &other) {
    torch::Tensor tout = From(out);
    torch::Tensor tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout, tcondition, tinput, tother);
}

static void Ln(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::log_out(tout, From(self));
}

#define DEFINE_BINARY_S_OPS(Name, op_out)                                                                 \
    static void Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse) { \
        auto tout = From(out);                                                                            \
        if (reverse) {                                                                                    \
            torch::full_out(tout, out->GetShape(), From(scalar));                              \
            torch::op_out(tout, tout, From(self));                                                        \
        } else {                                                                                          \
            torch::op_out(tout, From(self), From(scalar));                                                \
        }                                                                                                 \
    }

DEFINE_BINARY_S_OPS(AddS, add_out)
DEFINE_BINARY_S_OPS(SubS, sub_out)
DEFINE_BINARY_S_OPS(MulS, mul_out)
DEFINE_BINARY_S_OPS(DivS, div_out)

static void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::add_out(tout, tself, tother_final);
        } else {
            torch::add_out(tout, tself, tother);
        }
    } else {
        torch::add_out(tout, tself, tother);
    }
}

static void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::sub_out(tout, tself, tother_final);
        } else {
            torch::sub_out(tout, tself, tother);
        }
    } else {
        torch::sub_out(tout, tself, tother);
    }
}

static void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::mul_out(tout, tself, tother_final);
        } else {
            torch::mul_out(tout, tself, tother);
        }
    } else {
        torch::mul_out(tout, tself, tother);
    }
}

static void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::div_out(tout, tself, tother_final);
        } else {
            torch::div_out(tout, tself, tother);
        }
    } else {
        torch::div_out(tout, tself, tother);
    }
}

static void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode) {
    if (mode == CastMode::CAST_ROUND) {
        From(out) = From(self).round();
    } else if (mode == CastMode::CAST_FLOOR) {
        From(out) = From(self).floor();
    } else if (mode == CastMode::CAST_CEIL) {
        From(out) = From(self).ceil();
    } else if (mode == CastMode::CAST_TRUNC) {
        From(out) = From(self).trunc();
    } else {
        if (IsFloat(out->GetDataType())) {
            From(out) = From(self);
        } else {
            From(out) = From(self).round();
        }
    }
}

static void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::min_out(tout, tself, tother_final);
        } else {
            torch::min_out(tout, tself, tother);
        }
    } else {
        torch::min_out(tout, tself, tother);
    }
}

static void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.sizes().vec();
    std::vector<int64_t> shape_other = tother.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::max_out(tout, tself, tother_final);
        } else {
            torch::max_out(tout, tself, tother);
        }
    } else {
        torch::max_out(tout, tself, tother);
    }
}

static void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    torch::clamp_max_out(tout, From(self), From(elem));
}

static void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    torch::clamp_min_out(tout, From(self), From(elem));
}

static void Range(LogicalTensorDataPtr out, const Element &start, const Element &end, const Element &step) {
    auto tmp = torch::arange(From(start), From(end), From(step));
    int64_t expected_numel = 1;
    for (int64_t dim : out->GetShape()) {
        expected_numel *= dim;
    }
    ASSERT(tmp.numel() == expected_numel) << "Range numel mismatch: generated " << tmp.numel() << ", expected " << expected_numel;
    auto tout = From(out);
    tout.copy_(tmp);
}

template <typename T>
static void CompareImpl(LogicalTensorDataPtr out, const torch::Tensor& tself, const T& other_op,
                        CmpOperationType operation, CmpModeType mode) {
    auto tout = From(out);
    torch::Tensor tmp_result;
    switch (operation) {
        case CmpOperationType::EQ:
            tmp_result = torch::eq(tself, other_op);
            break;
        case CmpOperationType::NE:
            tmp_result = torch::ne(tself, other_op);
            break;
        case CmpOperationType::LT:
            tmp_result = torch::lt(tself, other_op);
            break;
        case CmpOperationType::LE:
            tmp_result = torch::le(tself, other_op);
            break;
        case CmpOperationType::GT:
            tmp_result = torch::gt(tself, other_op);
            break;
        case CmpOperationType::GE:
            tmp_result = torch::ge(tself, other_op);
            break;
        default:
            ASSERT(false) << "Unsupported compare type";
            break;
    }

    if (mode == CmpModeType::BIT) {
        if (tmp_result.dim() > 0) {
            int64_t last_dim = tmp_result.size(-1);
            ASSERT(last_dim % NUM_VALUE_8 == 0) << "Last dimension must be divisible by 8 in BIT mode";
            
            auto shape = tmp_result.sizes().vec();
            shape.back() = last_dim / NUM_VALUE_8;
            
            torch::Tensor packed = torch::empty(shape, torch::kUInt8);
            auto tmp_result_contig = tmp_result.contiguous();
            auto tmp_data = tmp_result_contig.data_ptr<bool>();
            auto packed_data = packed.data_ptr<uint8_t>();
            
            const int64_t num_elements = tmp_result.numel();
            for (int64_t i = 0; i < num_elements / NUM_VALUE_8; ++i) {
                uint8_t byte = 0;
                for (int j = 0; j < NUM_VALUE_8; ++j) {
                    if (tmp_data[i * NUM_VALUE_8 + j]) {
                        byte |= (1 << j);
                    }
                }
                packed_data[i] = byte;
            }
            tout.copy_(packed);
        }
    } else {
        tout.copy_(tmp_result);
    }
}

static void Compare(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
             CmpOperationType operation, CmpModeType mode) {
    CompareImpl(out, From(self), From(other), operation, mode);
}

static void Cmps(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem,
             CmpOperationType operation, CmpModeType mode) {
    CompareImpl(out, From(self), From(elem), operation, mode);
}

#define DEFINE_BINARY_PAIR_OPS(Name, bop)                                                              \
    static void Pair##Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) { \
        auto big = self, small = other;                                                                \
        if (self->GetShape() < other->GetShape()) {                                                    \
            big = other, small = self;                                                                 \
        }                                                                                              \
        auto tout = From(out);                                                                         \
        std::vector<int64_t> offset(self->GetShape().size(), 0);                                       \
        auto tbig = View(tout, big->GetShape(), offset);                                    \
        tbig.copy_(From(big));                                                                         \
        auto tsmall = View(tout, small->GetShape(), offset);                                \
        torch::bop(tsmall, tsmall, From(small));                                                       \
    }

DEFINE_BINARY_PAIR_OPS(Sum, add_out)
DEFINE_BINARY_PAIR_OPS(Max, max_out)
DEFINE_BINARY_PAIR_OPS(Min, min_out)

std::vector<int64_t> GenAxesForTranspose(const int64_t offset, const std::vector<int64_t>& base) {
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < offset; i++) {
        axes.push_back(i);
    }
    for (auto x : base) {
        axes.push_back(x + offset);
    }
    return axes;
}

static inline int64_t alignup(int64_t x, int64_t align) {
    return (x + (align - 1)) & ~(align - 1);
}

static void FormatND2NZ(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto &shape = self->GetShape();
    ASSERT(shape.size() >= 0x2) << "Input tensor must have at least 2 dimensions";

    int64_t ndim = shape.size();
    int64_t m = shape[ndim - 0x2];
    int64_t m0 = 16; // m0 16
    int64_t padm = alignup(m, m0);
    int64_t n = shape[ndim - 1];
    int64_t n0 = BLOCK_SIZE / BytesOf(self->GetDataType());
    int64_t padn = alignup(n, n0);
    int64_t n1 = padn / n0;

    auto tself = From(self);
    tself = tself.reshape({-1, m, n}); // [b, m1*m0, n1*n0]
    if (padm != m || padn != n) {
        tself = torch::constant_pad_nd(tself, {0, padn - n, 0, padm - m}, 0); // [b, padm, padn]
    }

    tself = tself.reshape({-1, padm, n1, n0}); // [b, padm, n1, n0]
    tself = tself.permute({0, 0x2, 1, 0x3});   // [b, n1, padm, n0]

    std::vector<int64_t> nzShape(shape.begin(), shape.end() - 2); // remove last 2 dim, keep only batch dims
    nzShape.push_back(padm);
    nzShape.push_back(padn);
    tself = tself.reshape(nzShape); // [b, padm, padn]
    From(out).copy_(tself);
}

static void FormatNZ2ND(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto &shape = self->GetShape();
    ASSERT(shape.size() >= 0x2) << "Input tensor must have at least 2 dimensions";

    auto tself = From(self); // [b, m1*m0, n1*n0]
    int64_t ndim = shape.size();
    int64_t m = shape[ndim - 0x2];
    int64_t n0 = BLOCK_SIZE / BytesOf(self->GetDataType());
    int64_t n1 = shape[ndim - 1] / n0;

    tself = tself.reshape({-1, n1, m, n0});  // [b, n1, m1*m0, n0]
    tself = tself.permute({0, 0x2, 1, 0x3}); // [b, m1*m0, n1, n0]
    tself = tself.reshape(shape);            // [b, m1*m0, n1*n0]

    std::vector<int64_t> offset(ndim, 0);
    From(out).copy_(View(tself, out->GetShape(), offset));
}

static void MatmulSplitK(torch::Tensor &out, const torch::Tensor &lhs, const torch::Tensor &rhs, int64_t kstep) {
    auto shapeL = lhs.sizes().vec();
    auto shapeR = rhs.sizes().vec();
    auto offsetL = std::vector<int64_t>(shapeL.size(), 0);
    auto offsetR = std::vector<int64_t>(shapeR.size(), 0);
    int64_t kdimL = shapeL.size() - 1;
    int64_t kdimR = shapeR.size() - 0x2;
    int64_t k = shapeL[kdimL];

    for (int64_t offset = 0; offset < k; offset += kstep) {
        shapeL[kdimL] = std::min(kstep, k - offset);
        shapeR[kdimR] = std::min(kstep, k - offset);
        offsetL[kdimL] = offset;
        offsetR[kdimR] = offset;
        auto viewL = View(lhs, shapeL, offsetL);
        auto viewR = View(rhs, shapeR, offsetR);
        out.add_(torch::matmul(viewL, viewR));
    }
}

static void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, LogicalTensorDataPtr acc,
            MatMulParam &param) {
    auto tout = From(out);
    auto dtype = tout.scalar_type();
    auto calcType = dtype;
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
        calcType = torch::kFloat;
        tout = tout.to(calcType);
    }

    auto tself = From(self);
    auto tother = From(other);
    if (acc) {
        tout.copy_(From(acc));
    } else {
        tout.zero_();
    }
    if (param.aTrans) {
        tself.transpose_(-1, AXIS_TO_LAST);
    }
    if (param.bTrans) {
        tother.transpose_(-1, AXIS_TO_LAST);
    }
    if (tself.scalar_type() != calcType) {
        tself = tself.to(calcType);
    }
    if (tother.scalar_type() != calcType) {
        tother = tother.to(calcType);
    }
    if (!param.kStep || param.kStep == self->GetShape(-1)) {
        tout.add_(torch::matmul(tself, tother));
    } else {
        MatmulSplitK(tout, tself, tother, param.kStep);
    }
    if (calcType != dtype) {
        From(out) = tout.to(dtype);
    }
}

void OneHot(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int numClasses) {
    auto ret = From(out);
    auto src = From(self);
    ret.copy_(torch::nn::functional::one_hot(src.to(torch::kInt64), numClasses));
}

static void ExpandS(LogicalTensorDataPtr out, const Element &elem) {
    auto tout = From(out);
    torch::full_out(tout, out->GetShape(), From(elem));
}

static void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    if (self->GetShape(-1) != out->GetShape(-1) && self->GetShape(-1) != 1) {
        // possible block align
        tself = tself.slice(tself.dim() - 1, 0, 1);
    }
    From(out) = tself;
}
void Gather(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int64_t axis) {
    auto tout = From(out);
    auto tparams = From(params);
    auto tindices = From(indices);
    auto paramsRank = params->GetShape().size();
    if (axis < 0) {
        axis += paramsRank;
    }
    TORCH_CHECK(axis >= 0 && axis < static_cast<int64_t>(paramsRank), "axis out of range");
    auto idxFlat = tindices.to(torch::kLong).reshape({-1});
    auto gathered = tparams.index_select(/*dim=*/axis, /*index=*/idxFlat);
    std::vector<int64_t> outSize{};
    outSize.insert(outSize.end(), tparams.sizes().begin(), tparams.sizes().begin() + axis);
    outSize.insert(outSize.end(), tindices.sizes().begin(), tindices.sizes().end());
    outSize.insert(outSize.end(), tparams.sizes().begin() + axis + 1, tparams.sizes().end());
    tout = tout.view(outSize);
    tout.copy_(gathered.reshape(outSize));
}
void GatherINUBGolden(torch::Tensor &out, const torch::Tensor &params, const torch::Tensor &indices,
    const torch::Tensor &pageTable, int64_t blockSize, int64_t axis) {
    // ---- 基本约束：只做 CPU，不考虑 CUDA ----
    TORCH_CHECK(params.is_cpu() && indices.is_cpu() && pageTable.is_cpu() && out.is_cpu(),
        "CPU-only: params/indices/pageTable/out must all be on CPU.");

    // ---- axis：严格等价你 golden（token 维），只允许 axis==0 ----
    if (axis < 0)
        axis += params.dim();
    TORCH_CHECK(axis == 0, "Only axis==0 is supported to match the original golden logic.");
    TORCH_CHECK(blockSize > 0, "blockSize must be > 0.");

    // ---- 形状严格限制：indices/pageTable 只能是 [1, a] ----
    TORCH_CHECK(params.dim() == 2, "params must be [num_buffer_tokens, hidden_dim]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == 1, "indices must be [1, topk_count]");
    TORCH_CHECK(pageTable.dim() == 2 && pageTable.size(0) == 1, "pageTable must be [1, num_logical_blocks]");
    TORCH_CHECK(out.dim() == 2, "out must be [topk_count, hidden_dim]");

    const int64_t hidden_dim = params.size(1);
    const int64_t topk_count = indices.size(1);
    const int64_t num_logical_blocks = pageTable.size(1);

    TORCH_CHECK(out.size(0) == topk_count && out.size(1) == hidden_dim, "out must have shape [topk_count, hidden_dim]");

    // ---- dtype：indices/pageTable 必须是整数；统一转 int64（不转 params）----
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong, "indices must be int32 or int64");
    TORCH_CHECK(pageTable.scalar_type() == at::kInt || pageTable.scalar_type() == at::kLong,
        "pageTable must be int32 or int64");

    // out/params dtype 必须一致（index_select 不会帮你做 dtype cast）
    TORCH_CHECK(out.scalar_type() == params.scalar_type(), "out and params must have the same dtype");

    // ---- 1) logical indices: [topk] int64 ----
    at::Tensor logical = indices.reshape({-1}).to(at::kLong);

    // ---- logical 越界检查： [0, num_logical_blocks * blockSize) ----
    const int64_t total_logical_tokens = num_logical_blocks * blockSize;
    TORCH_CHECK(total_logical_tokens >= 0, "total_logical_tokens overflow?");
    TORCH_CHECK(logical.ge(0).all().item<bool>(), "logical_index < 0 exists in indices");
    TORCH_CHECK(logical.lt(total_logical_tokens).all().item<bool>(),
        "logical_index out of range: must be < num_logical_blocks * blockSize");

    // ---- 2) pageTable: [num_logical_blocks] int64 ----
    at::Tensor pt = pageTable.reshape({-1}).to(at::kLong);
    TORCH_CHECK(pt.numel() == num_logical_blocks, "pageTable numel mismatch");

    // ---- 3) compute physical indices (完全等价 golden) ----
    // logical_block = logical / blockSize
    // offset        = logical % blockSize
    // physical_blk  = pt[logical_block]
    // physical      = physical_blk * blockSize + offset
    at::Tensor logical_block = logical.floor_divide(blockSize); // trunc div for int64
    at::Tensor offset = logical.remainder(blockSize);  // same as % for non-negative

    // 逻辑块 id 范围检查（其实 logical 已经检查过，这里更保险）
    TORCH_CHECK(logical_block.ge(0).all().item<bool>(), "logical_block_id < 0 exists");
    TORCH_CHECK(logical_block.lt(num_logical_blocks).all().item<bool>(), "logical_block_id out of range for pageTable");

    at::Tensor physical_block = pt.index_select(0, logical_block);
    at::Tensor physical = physical_block.mul(blockSize).add(offset); // int64

    // ---- physical 越界检查：[0, num_buffer_tokens) ----
    TORCH_CHECK(physical.ge(0).all().item<bool>(), "physical_index < 0 exists");

    // ---- 4) index_select gather: params[physical, :] -> [topk, hidden_dim] ----
    at::Tensor selected = params.index_select(0, physical); // dtype 跟 params 一样

    // 写到 out（不要求 out contiguous；copy_ 会处理）
    out.copy_(selected);
}
static torch::Tensor From4GatherINUB(LogicalTensorDataPtr data) {
    RawTensorDataPtr raw = data->GetData();
    auto tensor = torch::from_blob(raw->data(), raw->GetShape(), FromDataType(raw->GetDataType()));
    auto view = tensor.as_strided({raw->GetShape()[0],data->GetShape()[1]}, raw->GetStride(), data->GetStorageOffset());
    if (data->IsAxisCombine())
        view = view.transpose_(-1, AXIS_TO_LAST);
    return view;
}
void GatherINUB(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices,
    LogicalTensorDataPtr pageTable, int64_t blockSize, int64_t axis) {
    auto tout = From(out);
    auto tparams = From4GatherINUB(params);
    auto tindices = From(indices);
    auto tpageTable = From(pageTable);
    GatherINUBGolden(tout, tparams, tindices, tpageTable, blockSize, axis);
}

void GatherElements(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int axis) {
    auto ret = From(out);
    auto src = From(params);
    auto index = From(indices).to(torch::kInt64);
    torch::gather_out(ret, src, axis, index);
}

void IndexAdd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr src, LogicalTensorDataPtr indices, int axis, const Element &alpha) {
    torch::Tensor output = From(out);
    torch::Tensor inputSelf = From(self);
    torch::Tensor inputSrc = From(src);
    torch::Tensor inputIndices = From(indices);
    torch::index_add_out(output, inputSelf, axis, inputIndices, inputSrc, From(alpha));
}

void CumSum(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int axis) {
    torch::Tensor output = From(out);
    torch::Tensor input = From(in);

    torch::cumsum_out(output, input, axis);
}

void IndexPut(LogicalTensorDataPtr out, LogicalTensorDataPtr self, std::vector<LogicalTensorDataPtr> indices, LogicalTensorDataPtr values, bool accumulate) {
    c10::List<c10::optional<at::Tensor>> indicesList;
    for (const auto idx : indices) {
        indicesList.push_back(From(idx));
    }
    From(out) = torch::index_put(From(self), indicesList, From(values), accumulate);
}

static void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans) {
    if (trans) {
        From(out) = From(self).transpose_(-1, AXIS_TO_LAST);
    } else {
        From(out) = From(self);
    }
}

static void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    From(out) = torch::sum(From(self), {dim}, true);
}

static void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tout = From(out);
    torch::sum_out(tout, From(self), {dim}, true);
}

static void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::min(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

static void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::max(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

static void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::min(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

static void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::max(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

static void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    From(out) = torch::reshape(From(self), out->GetShape());
}

static void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1) {
    auto tout = From(out);
    torch::transpose_copy_out(tout, From(self), dim0, dim1);
}

void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t> &dim) {
    auto tout = From(out);
    torch::permute_copy_out(tout, From(self), dim);
}

static void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr> &tdatas) {
    auto tout = From(out);
    std::vector<torch::Tensor> tensors;
    for (auto &tdata : tdatas) {
        tensors.push_back(From(tdata));
    }
    torch::sum_out(tout, torch::stack(tensors, 0), 0);
}

/**
 * @brief Perform a bitwise sort of 32 elements on the input tensor according to the specified dimension
 *        and return the output tensor
 *        e.g.,1.If the shape of the input tensor is {2,33}, a temporary tensor will be created based on the
 *             input pad to {2,64}, and an index tensor with a shape of {2,64} and values from 0 to 63 will
 *             be created along the sorting axis;
 *             2. Then stack the temporary tensor and index tensor into a new tensor with a shape of {2,64,2}
 *             3. Then, along the original sorting axis, the new tensor is transformed into a temporary tensor
 *             with 32 elements per group to {2, 32, 2, 2}. Next, the temporary tensor is sorted within the group
 *             along the axis with a size of 32 to make it ordered within the group. Finally, the sorted tensor
 *             is expanded into a tensor with a size of {2128} using reshape. Finally, the data distribution on
 *             the one-dimensional sorting axis is arranged alternately in order of value index
 *
 * @param out output tensor
 * @param self input tensor
 * @param axis Indicate on which axis of self for grouping and sorting
 * @param descending Indicate whether the sorting direction is ascending or descending
 */
static void BitSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, bool descending) {
    auto tself = From(self);
    auto tout = From(out);
    constexpr int DIM_SIZE_TWO = 2;
    axis = axis < 0?(axis + tself.dim()):axis;

    const int64_t groupSize = 32;
    auto tselfAlignShape = tself.sizes().vec();
    tselfAlignShape[axis] = (tself.size(axis) + groupSize - 1) / groupSize * groupSize;
    float padValue = descending ? (-1.0f / 0.0f) : (1.0f / 0.0f);
    auto tselfAlign = torch::full(tselfAlignShape, padValue);
    torch::Tensor tselfAlignSubview = View(tselfAlign, tself.sizes().vec(), {0, 0});
    tselfAlignSubview.copy_(tself);
    if (!descending) {
        tselfAlign.neg_();
    }

    auto indices = torch::arange(0, tselfAlign.size(axis), 1, torch::dtype(torch::kLong));
    std::vector<int64_t> indexShape(tselfAlign.dim(), 1);
    indexShape[axis] = tselfAlign.size(axis);
    indices = indices.reshape(indexShape).broadcast_to(tselfAlign.sizes());

    auto combined = torch::stack({tselfAlign, indices.to(tselfAlign.dtype())}, tselfAlign.dim());
    std::vector<int64_t> groupedShape;
    for (int64_t i = 0; i < tselfAlign.dim(); ++i) {
        if (i == axis) {
            groupedShape.push_back(tselfAlign.size(axis) / groupSize);
            groupedShape.push_back(groupSize);
        } else {
            groupedShape.push_back(tselfAlign.size(i));
        }
    }
    groupedShape.push_back(DIM_SIZE_TWO);
    auto grouped = combined.reshape(torch::IntArrayRef(groupedShape));
    torch::Tensor sortIndices;
    std::tie(std::ignore, sortIndices) = grouped.select(-1, 0).sort(axis + 1, true);

    std::vector<int64_t> expandDims(sortIndices.unsqueeze(-1).dim(), -1);
    expandDims.back() = DIM_SIZE_TWO;
    auto expandIndices = sortIndices.unsqueeze(-1).expand(torch::IntArrayRef(expandDims));
    auto sortedGroups = grouped.gather(axis + 1, expandIndices);

    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < sortedGroups.dim(); ++i) {
        if (i == axis) {
            dstShape.push_back(DIM_SIZE_TWO * tselfAlign.size(axis));
        } else if (i !=axis + 1 && i != sortedGroups.dim() - 1) {
            dstShape.push_back(sortedGroups.size(i));
        }
    }

    auto tres = sortedGroups.reshape(torch::IntArrayRef(dstShape));
    torch::Tensor expanded = torch::cat({tres, torch::zeros_like(tres)}, axis);
    torch::Tensor dstSubview = View(tout, expanded.sizes().vec(), {0, 0});
    dstSubview.copy_(expanded);
}

/**
 * @brief extract elements from the target dimension of tensors and ajust the output according to the param
 *        require the data distribution if the input tensor sorting axis to be value indexed alternately
 *        arranged in order
 *
 * @param out output tensor
 * @param self input tensor
 * @param mod used to extract elements from the target dimension of tensors, mod=0 means to obtain
 *            elements with even indices, and mod=1 means to obtain elements with odd indices
 * @param descending Indicate whether the obtained k values are the maximum or minimum k values,
 *                   and true returns the maximum k values
 */
static void Extract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mod, bool descending) {
    auto tself = From(self);
    auto tout = From(out);
    constexpr int INDICE_STEP = 2;

    int dim = tself.dim() - 1;
    auto indices = torch::arange(
        (mod == 1?1:0),
        tself.size(dim),
        INDICE_STEP,
        torch::dtype(torch::kLong)
    );
    torch::Tensor selfSubview = View(tself.index_select(dim, indices), tout.sizes().vec(), {0, 0});
    tout.copy_(selfSubview);

    if (!descending && mod == 0) {
        tout.neg_();
    }
}

/**
 * @brief Sort the input tensor according to the specified dimension and return the output tensor, requiring the
 *        data distribution of the sorting axis of the input tensor to be alternately arranged by value index
 *        e.g.,1.If the shape of the input tensor is {2,256}, then half of the sorting axis in the input tensor
 *             will be truncated as a valid tensor with a shape of {2,128}
 *             2. Then group the values and indexes along the sorting axis, dividing them into 64 value index
 *             pairs, and reshape the effective tensor to a new tensor with a shape of {2,64,2}
 *             3. Sort the new tensor along the original sorting axis in the numerical dimension, and finally
 *             use reshape expansion to sort the new tensor into output tensors of shape and size {2,128}.
 *             Finally, the data distribution of the entire tensor on the one-dimensional sorting axis is still
 *             sorted alternately by value index, and the values are ordered
 *
 * @param out output tensor
 * @param self input tensor
 * @param axis Indicate on which axis of self to obtain topk
 * @param k  Indicate the  maximum or minimum k values are obtained
 * @param descending Indicate whether the obtained k values are the maximum or minimum k values,
 *                   and true returns the maximum k values
 */
static void Topk(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, int64_t k, bool descending) {
    auto tself = From(self);
    auto tout = From(out);
    constexpr int MERGE_SORT_NUM = 4;
    constexpr int DIM_SIZE_TWO = 2;
    constexpr int ACTUAL_VALID_RATIO = 2;
    (void)descending;
    axis = axis < 0?(axis + tself.dim()):axis;
    auto sliceIndices = torch::arange(tself.size(axis) / ACTUAL_VALID_RATIO, torch::dtype(torch::kLong));
    auto tselfHalf = tself.index_select(axis, sliceIndices);

    ASSERT(axis >= 0 && axis < tselfHalf.dim()) <<
        "axis" << axis << " is out of bounds for tensor of dimension " << tselfHalf.dim();

    ASSERT(tself.size(axis) % MERGE_SORT_NUM == 0) <<
        "Expected self.size(axis) after preprocessing to be divisible by 4, but got " << tself.size(axis);

    const int64_t maxk = tself.size(axis) / MERGE_SORT_NUM;
    ASSERT(k > 0 && k <= maxk) << "Expected k to be in (0, " << maxk << "], but got " << k;

    std::vector<int64_t>newShape;
    newShape.reserve(tselfHalf.dim() + 1);
    for (int64_t i = 0; i < tselfHalf.dim(); ++i) {
        if (i == axis) {
            newShape.push_back(tselfHalf.size(axis) / ACTUAL_VALID_RATIO);
            newShape.push_back(DIM_SIZE_TWO);
        } else {
            newShape.push_back(tselfHalf.size(i));
        }
    }
    auto tselfGrouped = tselfHalf.reshape(torch::IntArrayRef(newShape));
    torch::Tensor sortedIndices;
    std::tie(std::ignore, sortedIndices) = tselfGrouped.select(-1, 0).sort(axis, true);

    std::vector<int64_t>indexShape;
    for (int64_t i = 0; i < sortedIndices.dim(); ++i) {
        indexShape.push_back(sortedIndices.size(i));
    }
    indexShape.push_back(DIM_SIZE_TWO);
    auto expanded_indices = sortedIndices.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    auto sortedGroups = tselfGrouped.gather(axis, expanded_indices);
    auto indicesk = torch::arange(k, torch::dtype(torch::kLong));
    auto topkGroups = sortedGroups.index_select(axis, indicesk);

    std::vector<int64_t> dstShape;
    dstShape.reserve(topkGroups.dim() - 1);
    for (int64_t i = 0; i < topkGroups.dim(); ++i) {
        if (i == axis) {
            dstShape.push_back(DIM_SIZE_TWO * k);
        } else if (i !=axis + 1) {
            dstShape.push_back(topkGroups.size(i));
        }
    }
    torch::Tensor dstSubview = View(tout, dstShape, {0, 0});
    dstSubview.copy_(topkGroups.reshape(torch::IntArrayRef(dstShape)));
}

static void TopkSort(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outTemp,
                     LogicalTensorDataPtr self, int startIndex) {
    auto tself = From(self);
    auto toutValue = From(outValue);
    auto toutTemp = From(outTemp);

    constexpr int GROUP_SIZE = 32;
    int axis = tself.dim() - 1;

    // 1. Generate indices starting from startIndex*len
    int64_t len = tself.size(axis);
    int64_t baseIdx = startIndex * len;
    auto indices = torch::arange(baseIdx, baseIdx + len, 1, torch::dtype(torch::kFloat));
    std::vector<int64_t> indexShape(tself.dim(), 1);
    indexShape[axis] = len;
    indices = indices.reshape(indexShape).broadcast_to(tself.sizes());

    // 2. Align to GROUP_SIZE (32)
    auto tselfAlignShape = tself.sizes().vec();
    int64_t alignedLen = (len + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
    tselfAlignShape[axis] = alignedLen;

    float padValue = -1.0f / 0.0f;  // Negative infinity for descending sort
    auto valuesAlign = torch::full(tselfAlignShape, padValue, tself.dtype());
    torch::Tensor valueView = View(valuesAlign, tself.sizes().vec(), {0, 0});
    valueView.copy_(tself);

    auto indicesAlign = torch::full(tselfAlignShape, padValue, torch::kFloat);
    torch::Tensor indexView = View(indicesAlign, indices.sizes().vec(), {0, 0});
    indexView.copy_(indices);

    // 3. Group and sort (every 32 elements)
    std::vector<int64_t> groupShape;
    for (int64_t i = 0; i < valuesAlign.dim(); ++i) {
        if (i == axis) {
            groupShape.push_back(alignedLen / GROUP_SIZE);
            groupShape.push_back(GROUP_SIZE);
        } else {
            groupShape.push_back(valuesAlign.size(i));
        }
    }

    auto valsGrouped = valuesAlign.reshape(torch::IntArrayRef(groupShape));
    auto idxsGrouped = indicesAlign.reshape(torch::IntArrayRef(groupShape));

    torch::Tensor sortIdx;
    std::tie(valsGrouped, sortIdx) = valsGrouped.sort(axis + 1, true);  // Descending
    idxsGrouped = idxsGrouped.gather(axis + 1, sortIdx);

    // 4. Flatten
    valsGrouped = valsGrouped.flatten(axis, axis + 1);
    idxsGrouped = idxsGrouped.flatten(axis, axis + 1);

    // 5. Create pack: [v0, i0, v1, i1, ...]
    auto stacked = torch::stack({valsGrouped, idxsGrouped}, -1);  // [..., len, 2]
    auto packed = stacked.flatten(axis, -1);  // [..., len*2]

    // 6. Output
    torch::Tensor tempView = View(toutTemp, packed.sizes().vec(), {0, 0});
    tempView.copy_(packed);
    torch::Tensor valView = View(toutValue, packed.sizes().vec(), {0, 0});
    valView.copy_(packed);
}

static void TopkMerge(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mergeSize) {
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // mergeSize: number of already-sorted packs

    // Extract all values (even positions)
    auto evenIndices = torch::arange(0, tself.size(axis), 2, torch::dtype(torch::kLong));
    auto values = tself.index_select(axis, evenIndices);

    // Global sort to get pack order
    torch::Tensor sortIndices;
    std::tie(std::ignore, sortIndices) = values.sort(axis, true);  // Descending

    // Build actual element indices (each pack occupies 2 positions)
    auto packIdx0 = sortIndices * 2;      // value position
    auto packIdx1 = packIdx0 + 1;         // index position
    // Stack and flatten to 1D vector for index_select
    auto allIndices = torch::stack({packIdx0.flatten(), packIdx1.flatten()}, 1).flatten();

    // Rearrange packs
    auto sorted = tself.index_select(axis, allIndices);

    torch::Tensor outView = View(tout, sorted.sizes().vec(), {0, 0});
    outView.copy_(sorted);
}

static void TopkExtract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int k, bool isIndex) {
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // isIndex=false: extract first k values (even positions: 0, 2, 4, ...)
    // isIndex=true:  extract first k indices (odd positions: 1, 3, 5, ...)

    int startOffset = isIndex ? 1 : 0;  // index starts from 1, value from 0
    int stride = 2;                      // Values and indices are interleaved in pack

    // Generate extraction indices: startOffset, startOffset+2, startOffset+4, ..., startOffset+2*(k-1)
    auto indices = torch::arange(startOffset, startOffset + k * stride, stride, torch::dtype(torch::kLong));

    // Extract
    auto extracted = tself.index_select(axis, indices);

    // If extracting indices, convert to INT32
    if (isIndex) {
        extracted = extracted.to(torch::kInt);
    }

    // Reshape to [1, k] (according to output shape in operation_impl.cpp)
    extracted = extracted.reshape({1, k});

    torch::Tensor outView = View(tout, extracted.sizes().vec(), {0, 0});
    outView.copy_(extracted);
}

bool ScatterDateCopy(const std::vector<int64_t> &loopIdx, torch::Tensor &src, torch::Tensor &indices,
    torch::Tensor &ret, int blockSize) {
    bool flag = false;
    int64_t s = indices.size(1);
    int64_t i = loopIdx[0];
    int64_t j = loopIdx[1];
    int64_t dataIdx = indices.index({i, j}).item<int64_t>();

    ASSERT(blockSize != 0);
    if (ret.dim() == 2) { // 2 dim
        int64_t srcIdx = i * s + j;
        if ((dataIdx < 0 || dataIdx >= ret.size(0)) || (srcIdx < 0 || srcIdx >= src.size(0))) {
            printf("index out of range. i:%ld, j:%ld, dst_idx:%ld, srcIdx:%ld\n", i, j, dataIdx, srcIdx);
            return flag;
        }
        ret[dataIdx] = src[srcIdx];
        flag = true;
    } else if (ret.dim() == 4) { // 4 dim
        int64_t bIdx = dataIdx / blockSize;
        int64_t sIdx = dataIdx % blockSize;
        if ((bIdx < 0 || bIdx >= ret.size(0)) || (sIdx < 0 || sIdx >= ret.size(1))) {
            printf("index out of range. i:%ld, j:%ld, dst_idx:%ld, blockSize:%d\n", i, j, dataIdx, blockSize);
            return flag;
        }
        ret[bIdx][sIdx] = src[i][j];
        flag = true;
    }

    return flag;
}

static void ScatterUpdate(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index, LogicalTensorDataPtr dst, int axis,
   std::string cacheMode, int blockSize) {
    (void)axis;
    (void)cacheMode;

    auto inplace = From(dst);
    auto ret = From(out);
    ret.copy_(inplace);
    auto src = From(self);
    auto indices = From(index);

    ASSERT(indices.dim() == 2);                   // indices should be 2 dim
    ASSERT((src.dim() == 2) || (src.dim() == 4)); // only 2, 4 dim support
    ASSERT((ret.dim() == 2) || (ret.dim() == 4)); // only 2, 4 dim support
    ASSERT(src.dim() == ret.dim());

    int64_t b = indices.size(0);
    int64_t s = indices.size(1);
    for (int64_t i = 0; i < b; i++) {
        for (int64_t j = 0; j < s; j++) {
            if (ScatterDateCopy({i, j}, src, indices, ret, blockSize) == false) {
                return;
            }
        }
    }
}

static const std::vector<std::string> scatterModeString = {"add", "multiply"};

static void ScatterElement(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    const Element &src, int axis, int reduce) {
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);

    if (reduce == 0) {
        From(out) = torch::scatter(inputSelf, axis, inputIndices, From(src));
    } else {
        From(out) = torch::scatter(inputSelf, axis, inputIndices, From(src), scatterModeString.at(reduce - 1));
    }
}

static void Brcb(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    auto tout = From(out);
    
    std::vector<int64_t> input_shape = tself.sizes().vec();
    std::vector<int64_t> output_shape = tout.sizes().vec();
    
    int64_t M = input_shape[0];
    int64_t N = output_shape[1]; 
    auto first_col = tself.index({torch::indexing::Slice(), 0}); 
    auto expanded = first_col.unsqueeze(1).expand({M, N}); 
    tout.copy_(expanded);
}
    
static void Scatter(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    LogicalTensorDataPtr src, int axis, int reduce) {
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);
    auto inputSrc = From(src);

    if (reduce == 0) {
        From(out) = torch::scatter(inputSelf, axis, inputIndices, inputSrc);
    } else {
        From(out) = torch::scatter(inputSelf, axis, inputIndices, inputSrc, scatterModeString.at(reduce - 1));
    }
}

static struct CalcOps calcOps = {
    .Random = Random,
    .AllClose = AllClose,
    .Cast = Cast,
    .Exp = Exp,
    .Neg = Neg,
    .Rsqrt = Rsqrt,
    .Sqrt = Sqrt,
    .Abs = Abs,
    .Brcb = Brcb,
    .WhereTT = WhereTT,
    .WhereTS = WhereTS,
    .WhereST = WhereST,
    .WhereSS = WhereSS,
    .Ln = Ln,
    .LogicalNot = LogicalNot,
    .Range = Range,
    .Compare = Compare,
    .Cmps = Cmps,
    .LogicalAnd = LogicalAnd,
    .AddS = AddS,
    .SubS = SubS,
    .MulS = MulS,
    .DivS = DivS,
    .Add = Add,
    .Sub = Sub,
    .Mul = Mul,
    .Div = Div,
    .PairSum = PairSum,
    .PairMax = PairMax,
    .PairMin = PairMin,
    .Min = Min,
    .Max = Max,
    .MinS = MinS,
    .MaxS = MaxS,
    .RowSumExpand = RowSumExpand,
    .RowMinExpand = RowMinExpand,
    .RowMaxExpand = RowMaxExpand,
    .RowSumSingle = RowSumSingle,
    .RowMinSingle = RowMinSingle,
    .RowMaxSingle = RowMaxSingle,
    .OneHot = OneHot,
    .ExpandS = ExpandS,
    .Expand = Expand,
    .GatherElements = GatherElements,
    .IndexAdd = IndexAdd,
    .CumSum = CumSum,
    .IndexPut = IndexPut,
    .Reshape = Reshape,
    .Permute = Permute,
    .Transpose = Transpose,
    .ReduceAcc = ReduceAcc,
    .Copy = Copy,
    .ScatterUpdate = ScatterUpdate,
    .ScatterElement = ScatterElement,
    .Scatter = Scatter,
    .FormatND2NZ = FormatND2NZ,
    .FormatNZ2ND = FormatNZ2ND,
    .MatMul = MatMul,
    .BitSort = BitSort,
    .Extract = Extract,
    .Topk = Topk,
    .TopkSort = TopkSort,
    .TopkMerge = TopkMerge,
    .TopkExtract = TopkExtract,
    .Gather = Gather,
    .GatherINUB = GatherINUB,
};

extern "C" struct CalcOps *GetCalcOps() {
    return &calcOps;
}
}

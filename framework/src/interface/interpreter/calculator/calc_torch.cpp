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
#include <limits>
#include "tilefwk/error.h"
#include "../calc_api.h"
#include "interpreter/calculator/fp8_convert.h"

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
        case DT_UINT8: return torch::kInt8;
        case DT_UINT16: return torch::kInt16;
        case DT_UINT32: return torch::kInt32;
        case DT_UINT64: return torch::kInt64;
        case DT_BOOL: return torch::kBool;
        case DT_DOUBLE: return torch::kDouble;
        case DT_INT4:
        case DT_FP8:
        case DT_FP8E5M2: return torch::kUInt8;
        case DT_FP8E4M3: return torch::kUInt8;
        case DT_FP8E8M0: return torch::kUInt8;
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
        case DT_INT32: return at::Scalar(elem.GetSignedData());
        case DT_INT64: return at::Scalar(elem.GetSignedData());
        case DT_FP16:
        case DT_BF16:
        case DT_DOUBLE:
            return at::Scalar(elem.GetFloatData());
        case DT_FP32: {
            // Clamp FP32 scalar into finite FP32 range to avoid INF
            double data = elem.GetFloatData();
            constexpr double kMaxF32 = static_cast<double>(std::numeric_limits<float>::max());
            if (data > kMaxF32) {
                data = kMaxF32;
            } else if (data < -kMaxF32) {
                data = -kMaxF32;
            }
            return at::Scalar(static_cast<float>(data));
        }
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
            // lower version of pytorch not support uint64 type, use int64 for temp
            return at::Scalar(static_cast<int64_t>(elem.GetUnsignedData()));
        case DT_FP8:
        case DT_FP8E5M2:
        case DT_FP8E4M3:
        case DT_FP8E8M0:
        case DT_HF4:
        case DT_HF8:
        default: assert(0);
    }
    return at::Scalar();
}

static void ToOperand(const torch::Tensor &src, const torch::Tensor &dst, DataType actualType) {
    if (actualType == DT_FP8E4M3 || actualType == DT_FP8E5M2 || actualType == DT_FP8E8M0) {
        dst.copy_(Float32ToFp8(src, actualType));
    } else {
        dst.copy_(src);
    }
}

static std::pair<torch::Tensor, torch::Tensor> From(LogicalTensorDataPtr data) {
    RawTensorDataPtr raw = data->GetData();
    auto ScalarDataType = FromDataType(raw->GetDataType());
    auto tensor = torch::from_blob(raw->data(), raw->GetShape(), ScalarDataType);
    auto view = tensor.as_strided(data->GetShape(), raw->GetStride(), data->GetStorageOffset());
    if (data->IsAxisCombine())
        view = view.transpose_(-1, AXIS_TO_LAST);
    auto actualView = view;
    if (ScalarDataType == torch::kUInt8) {
        actualView = Fp8ToFloat32(view, raw->GetDataType());
    }
    // view == actualView if ScalarDataType != torch::kUInt8
    return {view, actualView};
}

static torch::Tensor View(const torch::Tensor &self, const std::vector<int64_t> &shape, const std::vector<int64_t> &offset) {
    int64_t storageOffset = self.storage_offset();
    for (size_t dim = 0; dim < offset.size(); dim++) {
        storageOffset += self.stride(dim) * offset[dim];
    }
    return self.as_strided(shape, self.strides(), storageOffset);
}

static bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol, double rtol) {
    return From(self).second.allclose(From(other).second, atol, rtol);
}

static void Random(LogicalTensorDataPtr out) {
    auto tout = From(out);
    torch::rand_out(tout.second, tout.second.sizes());
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::exp_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Neg(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::neg_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Ceil(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::ceil_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Floor(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::floor_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Trunc(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::trunc_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Round(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int decimals) {
    auto tout = From(out);
    auto tself = From(self);
    tout.second.copy_(torch::round(tself.second, decimals));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Rsqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::rsqrt_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::sqrt_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Reciprocal(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::reciprocal_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_not_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void LogicalNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::logical_not_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void LogicalAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::logical_and_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::abs_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void WhereTT(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tcondition = From(condition);
    auto tinput = From(input);
    auto tother = From(other);
    torch::where_out(tout.second, tcondition.second, tinput.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void WhereTS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input, const Element &other) {
    auto tout = From(out);
    auto tcondition = From(condition);
    auto tinput = From(input);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout.second, tcondition.second, tinput.second, tother);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void WhereST(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    auto tother = From(other);
    torch::where_out(tout.second, tcondition.second, tinput, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void WhereSS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element &input, const Element &other) {
    auto tout = From(out);
    auto tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout.second, tcondition.second, tinput, tother);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Ln(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    torch::log_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

#define DEFINE_BINARY_S_OPS(Name, op_out)                                                                 \
    static void Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse) { \
        auto tout = From(out);                                                                            \
        auto tself = From(self);                                                                          \
        if (reverse) {                                                                                    \
            torch::full_out(tout.second, out->GetShape(), From(scalar));                                  \
            torch::op_out(tout.second, tout.second, tself.second);                                        \
        } else {                                                                                          \
            torch::op_out(tout.second, tself.second, From(scalar));                                       \
        }                                                                                                 \
        ToOperand(tout.second, tout.first, out->GetData()->GetDataType());                                  \
    }

DEFINE_BINARY_S_OPS(AddS, add_out)
DEFINE_BINARY_S_OPS(SubS, sub_out)
DEFINE_BINARY_S_OPS(MulS, mul_out)
DEFINE_BINARY_S_OPS(DivS, div_out)
DEFINE_BINARY_S_OPS(FmodS, fmod_out)
DEFINE_BINARY_S_OPS(BitwiseAndS, bitwise_and_out)
DEFINE_BINARY_S_OPS(BitwiseOrS, bitwise_or_out)
DEFINE_BINARY_S_OPS(BitwiseXorS, bitwise_xor_out)

static void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::add_out(tout.second, tself.second, tother_final);
        } else {
            torch::add_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::add_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::sub_out(tout.second, tself.second, tother_final);
        } else {
            torch::sub_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::sub_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::mul_out(tout.second, tself.second, tother_final);
        } else {
            torch::mul_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::mul_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::div_out(tout.second, tself.second, tother_final);
        } else {
            torch::div_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::div_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Fmod(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::fmod_out(tout.second, tself.second, tother_final);
        } else {
            torch::fmod_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::fmod_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_and_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseOr(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_or_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseXor(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_xor_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void CopySign(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::copysign_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseRightShift(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_right_shift_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseLeftShift(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_left_shift_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseRightShiftS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar) {
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_right_shift_out(tout.second, tself.second, From(scalar));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void BitwiseLeftShiftS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar) {
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_left_shift_out(tout.second, tself.second, From(scalar));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void SBitwiseRightShift(LogicalTensorDataPtr out, const Element &scalar, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tother = From(other);
    torch::bitwise_right_shift_out(tout.second, From(scalar), tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void SBitwiseLeftShift(LogicalTensorDataPtr out, const Element &scalar, LogicalTensorDataPtr other) {
    auto tout = From(out);
    auto tother = From(other);
    torch::bitwise_left_shift_out(tout.second, From(scalar), tother.second);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode) {
    auto tout = From(out);
    auto tself = From(self);
    if (mode == CastMode::CAST_ROUND) {
        ToOperand(tself.second.round(), tout.first, out->GetData()->GetDataType());
    } else if (mode == CastMode::CAST_FLOOR) {
        ToOperand(tself.second.floor(), tout.first, out->GetData()->GetDataType());
    } else if (mode == CastMode::CAST_CEIL) {
        ToOperand(tself.second.ceil(), tout.first, out->GetData()->GetDataType());
    } else if (mode == CastMode::CAST_TRUNC) {
        ToOperand(tself.second.trunc(), tout.first, out->GetData()->GetDataType());
    } else {
        if (IsFloat(out->GetDataType())) {
            ToOperand(tself.second, tout.first, out->GetData()->GetDataType());
        } else {
            ToOperand(tself.second.round(), tout.first, out->GetData()->GetDataType());
        }
    }
}

static void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::min_out(tout.second, tself.second, tother_final);
        } else {
            torch::min_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::min_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 2 && shape_other.size() == 2 &&
        shape_self[0] == shape_other[0] && 
        shape_self[1] != shape_other[1]) {
        
        if (shape_other[1] == 8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index(
                {torch::indexing::Slice(), 
                 torch::indexing::Slice(0, cols_self)});
            torch::max_out(tout.second, tself.second, tother_final);
        } else {
            torch::max_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::max_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    auto tself = From(self);
    torch::clamp_max_out(tout.second, tself.second, From(elem));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    auto tself = From(self);
    torch::clamp_min_out(tout.second, tself.second, From(elem));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Range(LogicalTensorDataPtr out, const Element &start, const Element &end, const Element &step) {
    auto tmp = torch::arange(From(start), From(end), From(step));
    int64_t expected_numel = 1;
    for (int64_t dim : out->GetShape()) {
        expected_numel *= dim;
    }
    ASSERT(tmp.numel() == expected_numel) << "Range numel mismatch: generated " << tmp.numel() << ", expected " << expected_numel;
    auto tout = From(out);
    tout.second.copy_(tmp);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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
            tout.second.copy_(packed);
            ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
        }
    } else {
        tout.second.copy_(tmp_result);
        ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
    }
}

static void Compare(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
             CmpOperationType operation, CmpModeType mode) {
    CompareImpl(out, From(self).second, From(other).second, operation, mode);
}

static void Cmps(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem,
             CmpOperationType operation, CmpModeType mode) {
    CompareImpl(out, From(self).second, From(elem), operation, mode);
}

#define DEFINE_BINARY_PAIR_OPS(Name, bop)                                                              \
    static void Pair##Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) { \
        auto big = self, small = other;                                                                \
        if (self->GetShape() < other->GetShape()) {                                                    \
            big = other, small = self;                                                                 \
        }                                                                                              \
        auto tout = From(out);                                                                         \
        std::vector<int64_t> offset(self->GetShape().size(), 0);                                       \
        auto tbig = View(tout.second, big->GetShape(), offset);                                        \
        tbig.copy_(From(big).second);                                                                  \
        auto tsmall = View(tout.second, small->GetShape(), offset);                                    \
        torch::bop(tsmall, tsmall, From(small).second);                                                \
        ToOperand(tout.second, tout.first, out->GetData()->GetDataType());                               \
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

    auto tself_pair = From(self);
    auto tself = tself_pair.second.reshape({-1, m, n}); // [b, m1*m0, n1*n0]
    if (padm != m || padn != n) {
        tself = torch::constant_pad_nd(tself, {0, padn - n, 0, padm - m}, 0); // [b, padm, padn]
    }

    tself = tself.reshape({-1, padm, n1, n0}); // [b, padm, n1, n0]
    tself = tself.permute({0, 0x2, 1, 0x3});   // [b, n1, padm, n0]

    std::vector<int64_t> nzShape(shape.begin(), shape.end() - 2); // remove last 2 dim, keep only batch dims
    nzShape.push_back(padm);
    nzShape.push_back(padn);
    tself = tself.reshape(nzShape); // [b, padm, padn]
    auto tout = From(out);
    ToOperand(tself, tout.first, out->GetData()->GetDataType());
}

static void FormatNZ2ND(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto &shape = self->GetShape();
    ASSERT(shape.size() >= 0x2) << "Input tensor must have at least 2 dimensions";

    auto tself_pair = From(self);
    auto tself = tself_pair.second; // [b, m1*m0, n1*n0]
    int64_t ndim = shape.size();
    int64_t m = shape[ndim - 0x2];
    int64_t n0 = BLOCK_SIZE / BytesOf(self->GetDataType());
    int64_t n1 = shape[ndim - 1] / n0;

    tself = tself.reshape({-1, n1, m, n0});  // [b, n1, m1*m0, n0]
    tself = tself.permute({0, 0x2, 1, 0x3}); // [b, m1*m0, n1, n0]
    tself = tself.reshape(shape);            // [b, m1*m0, n1*n0]

    std::vector<int64_t> offset(ndim, 0);
    auto tout = From(out);
    auto view = View(tself, out->GetShape(), offset);
    tout.second.copy_(view);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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
    auto dtype = tout.second.scalar_type();
    auto calcType = dtype;
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
        calcType = torch::kFloat;
        tout.second = tout.second.to(calcType);
    }

    auto tself = From(self);
    auto tother = From(other);
    if (acc) {
        tout.second.copy_(From(acc).second);
    } else {
        tout.second.zero_();
    }
    if (param.aTrans) {
        tself.second.transpose_(-1, AXIS_TO_LAST);
    }
    if (param.bTrans) {
        tother.second.transpose_(-1, AXIS_TO_LAST);
    }
    if (tself.second.scalar_type() != calcType) {
        tself.second = tself.second.to(calcType);
    }
    if (tother.second.scalar_type() != calcType) {
        tother.second = tother.second.to(calcType);
    }
    if (!param.kStep || param.kStep == self->GetShape(-1)) {
        tout.second.add_(torch::matmul(tself.second, tother.second));
    } else {
        MatmulSplitK(tout.second, tself.second, tother.second, param.kStep);
    }
    if (calcType != dtype) {
        tout.second = tout.second.to(dtype);
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void OneHot(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int numClasses) {
    auto ret = From(out);
    auto src = From(self);
    ret.second.copy_(torch::nn::functional::one_hot(src.second.to(torch::kInt64), numClasses));
    ToOperand(ret.second, ret.first, out->GetData()->GetDataType());
}

static void ExpandS(LogicalTensorDataPtr out, const Element &elem) {
    auto tout = From(out);
    torch::full_out(tout.second, out->GetShape(), From(elem));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    if (self->GetShape(-1) != out->GetShape(-1) && self->GetShape(-1) != 1) {
        // possible block align
        tself.second = tself.second.slice(tself.second.dim() - 1, 0, 1);
    }
    auto tout = From(out);
    ToOperand(tself.second, tout.first, out->GetData()->GetDataType());
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
    auto idxFlat = tindices.second.to(torch::kLong).reshape({-1});
    auto gathered = tparams.second.index_select(/*dim=*/axis, /*index=*/idxFlat);
    std::vector<int64_t> outSize{};
    outSize.insert(outSize.end(), tparams.second.sizes().begin(), tparams.second.sizes().begin() + axis);
    outSize.insert(outSize.end(), tindices.second.sizes().begin(), tindices.second.sizes().end());
    outSize.insert(outSize.end(), tparams.second.sizes().begin() + axis + 1, tparams.second.sizes().end());
    tout.second = tout.second.view(outSize);
    tout.second.copy_(gathered.reshape(outSize));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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
    GatherINUBGolden(tout.second, tparams, tindices.second, tpageTable.second, blockSize, axis);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void GatherElements(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int axis) {
    auto ret = From(out);
    auto src = From(params);
    auto index = From(indices).second.to(torch::kInt64);
    torch::gather_out(ret.second, src.second, axis, index);
    ToOperand(ret.second, ret.first, out->GetData()->GetDataType());
}

void IndexAdd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr src, LogicalTensorDataPtr indices, int axis, const Element &alpha) {
    auto tout = From(out);
    auto inputSelf = From(self);
    auto inputSrc = From(src);
    auto inputIndices = From(indices);
    torch::index_add_out(tout.second, inputSelf.second, axis, inputIndices.second, inputSrc.second, From(alpha));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void TriU(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int diagonal) {
    auto output = From(out);
    auto input = From(in);

    torch::triu_out(output.second, input.second, diagonal);
    ToOperand(output.second, output.first, out->GetData()->GetDataType());
}

void TriL(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int diagonal) {
    auto output = From(out);
    auto input = From(in);

    torch::tril_out(output.second, input.second, diagonal);
    ToOperand(output.second, output.first, out->GetData()->GetDataType());
}

void CumSum(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int axis) {
    auto tout = From(out);
    auto input = From(in);
    torch::cumsum_out(tout.second, input.second, axis);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void IndexPut(LogicalTensorDataPtr out, LogicalTensorDataPtr self, std::vector<LogicalTensorDataPtr> indices, LogicalTensorDataPtr values, bool accumulate) {
    c10::List<c10::optional<at::Tensor>> indicesList;
    for (const auto idx : indices) {
        indicesList.push_back(From(idx).second);
    }
    auto tout = From(out);
    auto result = torch::index_put(From(self).second, indicesList, From(values).second, accumulate);
    ToOperand(result, tout.first, out->GetData()->GetDataType());
}

static void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans) {
    auto tout = From(out);
    auto tself = From(self);
    if (trans) {
        auto res = tself.second.transpose(-1, AXIS_TO_LAST);
        ToOperand(res, tout.first, out->GetData()->GetDataType());
    } else {
        ToOperand(tself.second, tout.first, out->GetData()->GetDataType());
    }
}

static void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tout = From(out);
    auto tself = From(self);
    auto res = torch::sum(tself.second, {dim}, true);
    ToOperand(res, tout.first, out->GetData()->GetDataType());
}

static void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tout = From(out);
    auto tself = From(self);
    torch::sum_out(tout.second, tself.second, {dim}, true);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto ret = torch::min(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto ret = torch::max(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto ret = torch::min(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void RowMinLine(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto tout = From(out);
    auto ret = torch::min(tself.second, dim, true);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto ret = torch::max(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void RowMaxLine(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tself = From(self);
    auto tout = From(out);
    auto ret = torch::max(tself.second, dim, true);
    ToOperand(std::get<0>(ret), tout.first, out->GetData()->GetDataType());
}

static void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    auto tself = From(self);
    auto res = torch::reshape(tself.second, out->GetShape());
    ToOperand(res, tout.first, out->GetData()->GetDataType());
}

static void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1) {
    auto tout = From(out);
    auto tself = From(self);
    torch::transpose_copy_out(tout.second, tself.second, dim0, dim1);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t> &dim) {
    auto tout = From(out);
    auto tself = From(self);
    torch::permute_copy_out(tout.second, tself.second, dim);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr> &tdatas) {
    auto tout = From(out);
    std::vector<torch::Tensor> tensors;
    for (auto &tdata : tdatas) {
        tensors.push_back(From(tdata).second);
    }
    torch::sum_out(tout.second, torch::stack(tensors, 0), 0);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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
static void BitSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, bool descending, int64_t offset) {
    auto tself = From(self);
    auto tout = From(out);
    constexpr int DIM_SIZE_TWO = 2;
    axis = axis < 0?(axis + tself.second.dim()):axis;

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    const int64_t groupSize = 32;
    auto tselfAlignShape = tself.second.sizes().vec();
    tselfAlignShape[axis] = (tself.second.size(axis) + groupSize - 1) / groupSize * groupSize;
    float padValue = descending ? (-1.0f / 0.0f) : (1.0f / 0.0f);
    auto tselfAlign = torch::full(tselfAlignShape, padValue);
    torch::Tensor tselfAlignSubview = View(tselfAlign, tself.second.sizes().vec(), viewOffset);
    tselfAlignSubview.copy_(tself.second);
    if (!descending) {
        tselfAlign.neg_();
    }

    auto indices = torch::arange(0, tselfAlign.size(axis), 1, torch::dtype(torch::kLong)) + offset;
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
    torch::Tensor expanded = torch::cat({tres, torch::zeros_like(tself.second)}, axis);
    torch::Tensor dstSubview = View(tout.second, expanded.sizes().vec(), viewOffset);
    dstSubview.copy_(expanded);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    int dim = tself.second.dim() - 1;
    if (tself.second.size(dim) == 0) {
        return;
    }
    auto indices = torch::arange(
        (mod == 1?1:0),
        tself.second.size(dim),
        INDICE_STEP,
        torch::dtype(torch::kLong)
    );
    torch::Tensor selfSubview = View(tself.second.index_select(dim, indices), tout.second.sizes().vec(), viewOffset);
    tout.second.copy_(selfSubview);

    if (!descending && mod == 0) {
        tout.second.neg_();
    }
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
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
    constexpr int DIM_SIZE_TWO = 2;
    constexpr int ACTUAL_VALID_RATIO = 2;
    (void)descending;
    axis = axis < 0 ? (axis + tself.second.dim()) : axis;
    int actShape = tself.second.size(axis) - (tself.second.size(axis) + 31) / 32 * 32 / 3 * 2;
    auto sliceIndices = torch::arange(actShape * 2, torch::dtype(torch::kLong));
    auto tselfHalf = tself.second.index_select(axis, sliceIndices);

    ASSERT(axis >= 0 && axis < tselfHalf.dim()) <<
        "axis" << axis << " is out of bounds for tensor of dimension " << tselfHalf.dim();

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);

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
    torch::Tensor dstSubview = View(tout.second, dstShape, {0, 0});
    dstSubview.copy_(topkGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void TiledMrgSort(LogicalTensorDataPtr out, LogicalTensorDataPtr src1, LogicalTensorDataPtr src2,
    LogicalTensorDataPtr src3, LogicalTensorDataPtr src4, int validBit, int kvalue) {
    auto self1 = From(src1);
    auto self2 = From(src2);
    auto self3 = From(src3);
    auto self4 = From(src4);
    auto tout = From(out);
    constexpr int SORT_NUM_TWO = 2;
    constexpr int SORT_NUM_THREE = 3;
    constexpr int SORT_NUM_FOUR = 4;
    torch::Tensor tself;
    if (validBit == SORT_NUM_TWO) {
        tself = torch::cat({self1.second, self2.second}, -1);
    } else if (validBit == SORT_NUM_THREE) {
        tself = torch::cat({self1.second, self2.second, self3.second}, -1);
    } else if (validBit == SORT_NUM_FOUR) {
        tself = torch::cat({self1.second, self2.second, self3.second, self4.second}, -1);
    }
    constexpr int ACTUAL_VALID_RATIO = 2;
    auto axis = tself.dim() - 1;

    std::vector<int64_t>newShape;
    newShape.reserve(tself.dim() + 1);
    for (int64_t i = 0; i < tself.dim(); ++i) {
        if (i == axis) {
            newShape.push_back(tself.size(axis) / ACTUAL_VALID_RATIO);
            newShape.push_back(SORT_NUM_TWO);
        } else {
            newShape.push_back(tself.size(i));
        }
    }
    auto tselfGrouped = tself.reshape(torch::IntArrayRef(newShape));
    torch::Tensor sortedIndices;
    std::tie(std::ignore, sortedIndices) = tselfGrouped.select(-1, 0).sort(axis, true);

    std::vector<int64_t>indexShape;
    for (int64_t i = 0; i < sortedIndices.dim(); ++i) {
        indexShape.push_back(sortedIndices.size(i));
    }
    indexShape.push_back(SORT_NUM_TWO);
    auto expanded_indices = sortedIndices.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    auto sortedGroups = tselfGrouped.gather(axis, expanded_indices);
    auto indicesk = torch::arange(kvalue, torch::dtype(torch::kLong));
    auto topkGroups = sortedGroups.index_select(axis, indicesk);

    std::vector<int64_t> dstShape;
    dstShape.reserve(topkGroups.dim() - 1);
    for (int64_t i = 0; i < topkGroups.dim(); ++i) {
        if (i == axis) {
            dstShape.push_back(SORT_NUM_TWO * kvalue);
        } else if (i !=axis + 1) {
            dstShape.push_back(topkGroups.size(i));
        }
    }
    torch::Tensor dstSubview = View(tout.second, dstShape, {0, 0});
    dstSubview.copy_(topkGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void TopK(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex,
                 LogicalTensorDataPtr self, int k, int axis, bool descending) {
    auto tself = From(self);
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    axis = axis < 0 ? (axis + tself.second.dim()) : axis;
    torch::Tensor tempIdxInt64 = torch::zeros(toutValue.second.sizes().vec(), torch::kInt64);
    torch::topk_out(toutValue.second, tempIdxInt64, tself.second, k, axis, descending);
    auto tempIdxInt32 = tempIdxInt64.to(torch::kInt32);
    toutIndex.second.copy_(tempIdxInt32);
    ToOperand(toutValue.second, toutValue.first, outValue->GetData()->GetDataType());
    ToOperand(toutIndex.second, toutIndex.first, outIndex->GetData()->GetDataType());
}

static void TopkSort(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outTemp,
                     LogicalTensorDataPtr self, int startIndex) {
    auto tself = From(self);
    auto toutValue = From(outValue);
    auto toutTemp = From(outTemp);

    constexpr int GROUP_SIZE = 32;
    int axis = tself.second.dim() - 1;

    // 1. Generate indices starting from startIndex*len
    int64_t len = tself.second.size(axis);
    int64_t baseIdx = startIndex * len;
    auto indices = torch::arange(baseIdx, baseIdx + len, 1, torch::dtype(torch::kFloat));
    std::vector<int64_t> indexShape(tself.second.dim(), 1);
    indexShape[axis] = len;
    indices = indices.reshape(indexShape).broadcast_to(tself.second.sizes());

    // 2. Align to GROUP_SIZE (32)
    auto tselfAlignShape = tself.second.sizes().vec();
    int64_t alignedLen = (len + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
    tselfAlignShape[axis] = alignedLen;

    float padValue = -1.0f / 0.0f;  // Negative infinity for descending sort
    auto valuesAlign = torch::full(tselfAlignShape, padValue, tself.second.dtype());
    torch::Tensor valueView = View(valuesAlign, tself.second.sizes().vec(), {0, 0});
    valueView.copy_(tself.second);

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
    torch::Tensor tempView = View(toutTemp.second, packed.sizes().vec(), {0, 0});
    tempView.copy_(packed);
    torch::Tensor valView = View(toutValue.second, packed.sizes().vec(), {0, 0});
    valView.copy_(packed);
    ToOperand(toutTemp.second, toutTemp.first, outTemp->GetData()->GetDataType());
    ToOperand(toutValue.second, toutValue.first, outValue->GetData()->GetDataType());
}

static void TopkMerge(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mergeSize) {
    (void) mergeSize;
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.second.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // mergeSize: number of already-sorted packs
    // Note: Current implementation uses global sort for simplicity (sufficient for precision verification)
    (void)mergeSize;  // Suppress unused parameter warning

    // Extract all values (even positions)
    auto evenIndices = torch::arange(0, tself.second.size(axis), 2, torch::dtype(torch::kLong));
    auto values = tself.second.index_select(axis, evenIndices);

    // Global sort to get pack order
    torch::Tensor sortIndices;
    std::tie(std::ignore, sortIndices) = values.sort(axis, true);  // Descending

    // Build actual element indices (each pack occupies 2 positions)
    auto packIdx0 = sortIndices * 2;      // value position
    auto packIdx1 = packIdx0 + 1;         // index position
    // Stack and flatten to 1D vector for index_select
    auto allIndices = torch::stack({packIdx0.flatten(), packIdx1.flatten()}, 1).flatten();

    // Rearrange packs
    auto sorted = tself.second.index_select(axis, allIndices);

    torch::Tensor outView = View(tout.second, sorted.sizes().vec(), {0, 0});
    outView.copy_(sorted);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

static void TopkExtract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int k, bool isIndex) {
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.second.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // isIndex=false: extract first k values (even positions: 0, 2, 4, ...)
    // isIndex=true:  extract first k indices (odd positions: 1, 3, 5, ...)

    int startOffset = isIndex ? 1 : 0;  // index starts from 1, value from 0
    int stride = 2;                      // Values and indices are interleaved in pack

    // Generate extraction indices: startOffset, startOffset+2, startOffset+4, ..., startOffset+2*(k-1)
    auto indices = torch::arange(startOffset, startOffset + k * stride, stride, torch::dtype(torch::kLong));

    // Extract
    auto extracted = tself.second.index_select(axis, indices);

    // If extracting indices, convert to INT32
    if (isIndex) {
        extracted = extracted.to(torch::kInt);
    }

    // Reshape to [1, k] (according to output shape in operation_impl.cpp)
    extracted = extracted.reshape({1, k});

    torch::Tensor outView = View(tout.second, extracted.sizes().vec(), {0, 0});
    outView.copy_(extracted);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void TwoTileMrgSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    auto tout = From(out);
    constexpr int SIZE_TWO = 2;
    int axis = tself.second.dim() - 1;

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    std::vector<int64_t> newShape;
    newShape.reserve(tself.second.dim() + 1);
    for (int64_t i = 0; i < tself.second.dim(); i ++ ) {
        if (i == axis) {
            newShape.push_back(tself.second.size(axis) / SIZE_TWO);
            newShape.push_back(SIZE_TWO);
        } else {
            newShape.push_back(tself.second.size(i));
        }
    }

    auto tselfGrouped = tself.second.reshape(torch::IntArrayRef(newShape));
    torch::Tensor sortedIndices;
    std::tie(std::ignore, sortedIndices) = tselfGrouped.select(-1, 0).sort(axis, true);

    std::vector<int64_t> indexShape;
    for (int64_t i = 0; i < sortedIndices.dim(); i ++ ) {
        indexShape.push_back(sortedIndices.size(i));
    }
    indexShape.push_back(SIZE_TWO);

    auto expanded_indices = sortedIndices.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    auto sortedGroups = tselfGrouped.gather(axis, expanded_indices);

    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < tself.second.dim(); i ++ ) {
        dstShape.push_back(tself.second.size(i));
    }
    torch::Tensor dstSubview = View(tout.second, dstShape, viewOffset);
    dstSubview.copy_(sortedGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}

void Sort(LogicalTensorDataPtr value, LogicalTensorDataPtr index, LogicalTensorDataPtr self, int64_t axis, bool descending) {
    auto tself = From(self);
    auto tvalue = From(value);
    auto tindex = From(index);
    auto [sortValue, sortIndex] = tself.second.sort(axis, descending);
    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < tvalue.second.dim(); i ++ ) {
        dstShape.push_back(tvalue.second.size(i));
    }
    torch::Tensor outValue = View(tvalue.second, dstShape, viewOffset);
    torch::Tensor outIndex = View(tindex.second, dstShape, viewOffset);
    outValue.copy_(sortValue);
    outIndex.copy_(sortIndex);
    ToOperand(tvalue.second, tvalue.first, value->GetData()->GetDataType());
    ToOperand(tindex.second, tindex.first, index->GetData()->GetDataType());
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
            return flag;
        }
        ret[dataIdx] = src[srcIdx];
        flag = true;
    } else if (ret.dim() == 4) { // 4 dim
        int64_t bIdx = dataIdx / blockSize;
        int64_t sIdx = dataIdx % blockSize;
        if ((bIdx < 0 || bIdx >= ret.size(0)) || (sIdx < 0 || sIdx >= ret.size(1))) {
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
    ret.second.copy_(inplace.second);
    auto src = From(self);
    auto indices = From(index);

    ASSERT(indices.second.dim() == 2);                   // indices should be 2 dim
    ASSERT((src.second.dim() == 2) || (src.second.dim() == 4)); // only 2, 4 dim support
    ASSERT((ret.second.dim() == 2) || (ret.second.dim() == 4)); // only 2, 4 dim support
    ASSERT(src.second.dim() == ret.second.dim());

    int64_t b = indices.second.size(0);
    int64_t s = indices.second.size(1);
    for (int64_t i = 0; i < b; i++) {
        for (int64_t j = 0; j < s; j++) {
            if (ScatterDateCopy({i, j}, src.second, indices.second, ret.second, blockSize) == false) {
                return;
            }
        }
    }
    ToOperand(ret.second, ret.first, out->GetData()->GetDataType());
}

static const std::vector<std::string> scatterModeString = {"add", "multiply"};

static void ScatterElement(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    const Element &src, int axis, int reduce) {
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);

    if (index->GetDataType() == DT_INT32) {
        inputIndices.second = inputIndices.second.to(torch::kInt64);
    }
    if (reduce == 0) {
        auto res = torch::scatter(inputSelf.second, axis, inputIndices.second, From(src));
        ToOperand(res, output.first, out->GetData()->GetDataType());
    } else {
        auto res = torch::scatter(inputSelf.second, axis, inputIndices.second, From(src), scatterModeString.at(reduce - 1));
        ToOperand(res, output.first, out->GetData()->GetDataType());
    }
}

static void Brcb(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    auto tout = From(out);
    
    std::vector<int64_t> input_shape = tself.second.sizes().vec();
    std::vector<int64_t> output_shape = tout.second.sizes().vec();
    
    int64_t M = input_shape[0];
    int64_t N = output_shape[1]; 
    auto first_col = tself.second.index({torch::indexing::Slice(), 0}); 
    auto expanded = first_col.unsqueeze(1).expand({M, N}); 
    tout.second.copy_(expanded);
    ToOperand(tout.second, tout.first, out->GetData()->GetDataType());
}
    
static void Scatter(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
    LogicalTensorDataPtr src, int axis, int reduce) {
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);
    auto inputSrc = From(src);

    if (index->GetDataType() == DT_INT32) {
        inputIndices.second = inputIndices.second.to(torch::kInt64);
    }
    if (reduce == 0) {
        output.second = torch::scatter(inputSelf.second, axis, inputIndices.second, inputSrc.second);
        ToOperand(output.second, output.first, out->GetData()->GetDataType());
    } else {
        output.second = torch::scatter(inputSelf.second, axis, inputIndices.second, inputSrc.second, scatterModeString.at(reduce - 1));
        ToOperand(output.second, output.first, out->GetData()->GetDataType());
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
    .Ceil = Ceil,
    .Floor = Floor,
    .Trunc = Trunc,
    .Round = Round,
    .Reciprocal = Reciprocal,
    .BitwiseNot = BitwiseNot,
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
    .FmodS = FmodS,
    .BitwiseAndS = BitwiseAndS,
    .BitwiseOrS = BitwiseOrS,
    .BitwiseXorS = BitwiseXorS,
    .Add = Add,
    .Sub = Sub,
    .Mul = Mul,
    .Div = Div,
    .Fmod = Fmod,
    .BitwiseAnd = BitwiseAnd,
    .BitwiseOr = BitwiseOr,
    .BitwiseXor = BitwiseXor,
    .CopySign = CopySign,
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
    .RowMinLine = RowMinLine,
    .RowMaxLine = RowMaxLine,
    .OneHot = OneHot,
    .ExpandS = ExpandS,
    .Expand = Expand,
    .GatherElements = GatherElements,
    .IndexAdd = IndexAdd,
    .TriU = TriU,
    .TriL = TriL,
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
    .TiledMrgSort = TiledMrgSort,
    .Extract = Extract,
    .Topk = Topk,
    .TopK = TopK,
    .TopkSort = TopkSort,
    .TopkMerge = TopkMerge,
    .TopkExtract = TopkExtract,
    .TwoTileMrgSort = TwoTileMrgSort,
    .Sort = Sort,
    .Gather = Gather,
    .GatherINUB = GatherINUB,
    .BitwiseRightShift = BitwiseRightShift,
    .BitwiseLeftShift = BitwiseLeftShift,
    .BitwiseRightShiftS = BitwiseRightShiftS,
    .BitwiseLeftShiftS = BitwiseLeftShiftS,
    .SBitwiseRightShift = SBitwiseRightShift,
    .SBitwiseLeftShift = SBitwiseLeftShift,
};

extern "C" struct CalcOps *GetCalcOps() {
    return &calcOps;
}
}

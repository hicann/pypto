/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp8_convert.cpp
 * \brief FP8 format conversion implementations (E4M3, E5M2, E8M0).
 */

#include <cmath>
#include <limits>
#include <torch/torch.h>
#include "tilefwk/data_type.h"
#include "interpreter/calculator/fp8_convert.h"

namespace npu::tile_fwk {

// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits. Exponent bias: 7.
// Bit layout: [S][EEEE][MMM]. Special: 0x7F=NaN, 0x7E=+Inf, 0xFE=-Inf, 0xFF=NaN.
// Runtime decoding below treats exponent=15 as a saturating "max finite" value 240,
// and does not distinguish INF from finite overflow; NaNs are only produced explicitly in tests.

// Round to nearest integer, ties to even, for non‑negative inputs.
static inline int RoundToNearestEvenFloatPos(float x)
{
    // Assume x >= 0 on all call sites.
    float floorVal = std::floor(x);
    float frac = x - floorVal;
    int base = static_cast<int>(floorVal);
    if (frac > 0.5f) {
        return base + 1;
    }
    if (frac < 0.5f) {
        return base;
    }
    // Exactly at .5 – choose even integer.
    return (base % 2 == 0) ? base : (base + 1);
}

static torch::Tensor Fp8E4M3ToFloat32(const torch::Tensor &self) {
    auto x = self.to(torch::kInt32);
    auto sign =
        1.0f -
        (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) * 2.0f;
    auto exp_bits = torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(3)), at::Scalar(0xF));
    auto mant_bits = torch::bitwise_and(x, at::Scalar(0x7));

    // Subnormal: exp=0, value = 2^(-6) * (mant/8)
    auto is_subnormal = (exp_bits == 0);
    auto subnormal_val = mant_bits.to(torch::kFloat32) * (1.0f / 8.0f) * (1.0f / 64.0f);

    // Normal: exp 1-14, value = 2^(exp-7) * (1 + mant/8)
    auto is_normal = (exp_bits >= 1) & (exp_bits <= 14);
    auto exp_val = (exp_bits.to(torch::kFloat32) - 7.0f);
    auto mant_val = 1.0f + mant_bits.to(torch::kFloat32) / 8.0f;
    auto normal_val = torch::pow(2.0f, exp_val) * mant_val;

    // Special: Inf (0x7E/0xFE) or NaN (0x7F/0xFF)
    auto is_max = (exp_bits == 15);
    auto max_val = 240.0f;

    auto result = torch::zeros_like(self, torch::TensorOptions().dtype(torch::kFloat32));
    result = torch::where(is_subnormal, subnormal_val * sign, result);
    result = torch::where(is_normal, normal_val * sign, result);
    result = torch::where(is_max, sign * max_val, result);
    return result;
}

// FP8 E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits. Exponent bias: 15.
// Bit layout: [S][EEEEE][MM]. Special: 0x7F=NaN, 0x7C=+Inf, 0xFC=-Inf.
static torch::Tensor Fp8E5M2ToFloat32(const torch::Tensor &self) {
    auto x = self.to(torch::kInt32);
    auto sign =
        1.0f -
        (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) * 2.0f;
    auto exp_bits = torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(2)), at::Scalar(0x1F));
    auto mant_bits = torch::bitwise_and(x, at::Scalar(0x3));

    // Subnormal: exp=0, value = 2^(-14) * (mant/4)
    auto is_subnormal = (exp_bits == 0);
    auto subnormal_val = mant_bits.to(torch::kFloat32) * (1.0f / 4.0f) * (1.0f / 16384.0f);

    // Normal: exp 1-30, value = 2^(exp-15) * (1 + mant/4)
    auto is_normal = (exp_bits >= 1) & (exp_bits <= 30);
    auto exp_val = (exp_bits.to(torch::kFloat32) - 15.0f);
    auto mant_val = 1.0f + mant_bits.to(torch::kFloat32) / 4.0f;
    auto normal_val = torch::pow(2.0f, exp_val) * mant_val;

    // Special: Inf (exp=31, mant=0) or NaN (exp=31, mant!=0)
    auto is_special = (exp_bits == 31);
    auto is_inf = is_special & (mant_bits == 0);
    auto is_nan = is_special & (mant_bits != 0);
    auto inf_val = std::numeric_limits<float>::infinity();
    auto nan_val = std::numeric_limits<float>::quiet_NaN();

    auto result = torch::zeros_like(self, torch::TensorOptions().dtype(torch::kFloat32));
    result = torch::where(is_subnormal, subnormal_val * sign, result);
    result = torch::where(is_normal, normal_val * sign, result);
    result = torch::where(is_inf, sign * inf_val, result);
    result = torch::where(is_nan, nan_val, result);
    return result;
}

// FP8 E8M0 format: 1 sign bit, 7 exponent bits, 0 mantissa. Exponent bias: 63.
// Value = (-1)^s * 2^(exp-63). All values are powers of 2.
static torch::Tensor Fp8E8M0ToFloat32(const torch::Tensor &self) {
    auto x = self.to(torch::kInt32);
    auto sign =
        1.0f -
        (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) * 2.0f;
    auto exp_bits = torch::bitwise_and(x, at::Scalar(0x7F));
    auto exp_val = exp_bits.to(torch::kFloat32) - 63.0f;
    return sign * torch::pow(2.0f, exp_val);
}

torch::Tensor Fp8ToFloat32(const torch::Tensor &self, DataType actualType) {
    if (actualType == DT_UINT8) {
        return self;
    }
    switch (actualType) {
        case DT_FP8:
        case DT_FP8E4M3: return Fp8E4M3ToFloat32(self);
        case DT_FP8E5M2: return Fp8E5M2ToFloat32(self);
        case DT_FP8E8M0: return Fp8E8M0ToFloat32(self);
        default: return self.to(torch::kFloat32);
    }
}

// Float32 to FP8 E4M3. E4M3 value range is approximately [2^-9, 240].
// Implements round-to-nearest with ties-to-even under the decode defined in Fp8E4M3ToFloat32.
static inline uint8_t EncodeFloatToFp8E4M3(float v)
{
    constexpr float kMinSubnormal = 1.0f / 512.0f; // 2^-9: smallest positive subnormal (mant=1, exp=0)
    constexpr float kMinNormal = 1.0f / 64.0f;     // 2^-6: exp=1, mant=0
    constexpr float kMaxVal = 240.0f;              // max finite value produced by Fp8E4M3ToFloat32

    if (std::isnan(v) || std::isinf(v)) {
        int sign = std::signbit(v) ? 1 : 0;
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    if (std::fpclassify(v) == FP_ZERO) {
        // Preserve signed zero in the encoding.
        return static_cast<uint8_t>(std::signbit(v) ? 0x80 : 0x00);
    }

    float absv = std::fabs(v);
    int sign = std::signbit(v) ? 1 : 0;

    // Underflow to signed zero.
    if (absv < kMinSubnormal) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Handle large magnitudes (including very large finite and infinities) by saturation.
    if (absv >= kMaxVal) {
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    // Subnormal region [kMinSubnormal, kMinNormal).
    if (absv < kMinNormal) {
        // Values are mant * 2^-9, mant = 1..7.
        float mant_scaled = absv / kMinSubnormal; // in [1, 8)
        if (mant_scaled < 0.0f) {
            mant_scaled = 0.0f;
        }
        int mant = RoundToNearestEvenFloatPos(mant_scaled);
        if (mant <= 0) {
            // Round down to zero.
            return static_cast<uint8_t>(sign << 7);
        }
        if (mant >= 8) {
            // Rounded up to the smallest normal value: exp=1, mant=0 -> kMinNormal.
            uint8_t exp_bits = 1;
            uint8_t mant_bits = 0;
            return static_cast<uint8_t>((sign << 7) | (exp_bits << 3) | mant_bits);
        }
        // Proper subnormal.
        return static_cast<uint8_t>((sign << 7) | mant);
    }

    // Normal numbers: value = 2^(exp-7) * (1 + mant/8), exp in [1,14], mant in [0,7].
    int exp_raw;
    float frac = std::frexp(absv, &exp_raw); // absv = frac * 2^exp_raw, frac in [0.5,1)
    float norm_mant = frac * 2.0f;           // in [1,2)
    int unbiased_exp = exp_raw - 1;          // because absv = norm_mant * 2^(unbiased_exp)
    int stored_exp = unbiased_exp + 7;       // add FP8 E4M3 bias

    float mant_scaled = (norm_mant - 1.0f) * 8.0f; // ideally in [0,8)
    if (mant_scaled < 0.0f) {
        mant_scaled = 0.0f;
    }
    int mant = RoundToNearestEvenFloatPos(mant_scaled); // 0..8 (8 means carry)

    if (mant >= 8) {
        // Carry into exponent.
        mant = 0;
        stored_exp += 1;
    }

    if (stored_exp >= 15) {
        // Exponent overflow: encode as saturating max value (exp=15 region).
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    if (stored_exp <= 0) {
        // Underflow from normal into subnormal: recompute in subnormal grid.
        float scaled = absv / kMinSubnormal;
        if (scaled < 0.0f) {
            scaled = 0.0f;
        }
        int sub_mant = RoundToNearestEvenFloatPos(scaled);
        if (sub_mant <= 0) {
            return static_cast<uint8_t>(sign << 7);
        }
        if (sub_mant >= 8) {
            uint8_t exp_bits = 1;
            uint8_t mant_bits = 0;
            return static_cast<uint8_t>((sign << 7) | (exp_bits << 3) | mant_bits);
        }
        return static_cast<uint8_t>((sign << 7) | sub_mant);
    }

    uint8_t exp_bits = static_cast<uint8_t>(stored_exp & 0xF);
    uint8_t mant_bits = static_cast<uint8_t>(mant & 0x7);
    return static_cast<uint8_t>((sign << 7) | (exp_bits << 3) | mant_bits);
}

static torch::Tensor Float32ToFp8E4M3(const torch::Tensor &self) {
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();

    for (int64_t i = 0; i < flat.numel(); ++i) {
        out_ptr[i] = EncodeFloatToFp8E4M3(ptr[i]);
    }

    return result.reshape(x.sizes());
}

// Float32 to FP8 E5M2. E5M2 range: [2^-16, 57344]. Round to nearest.
static inline uint8_t EncodeFloatToFp8E5M2(float v)
{
    constexpr float kMinSubnormal = 1.0f / 65536.0f; // 2^-16
    constexpr float kMinNormal = 1.0f / 16384.0f;    // 2^-14
    constexpr float kMaxVal = 57344.0f;              // 2^15 * 1.75
    if (std::isnan(v)) {
        return 0x7F;
    }
    if (std::isinf(v)) {
        return static_cast<uint8_t>((v < 0) ? 0xFC : 0x7C);
    }
    if (std::fpclassify(v) == FP_ZERO) {
        return static_cast<uint8_t>(std::signbit(v) ? 0x80 : 0x00);
    }
    float absv = std::fabs(v);
    int sign = std::signbit(v) ? 1 : 0;
    if (absv < kMinSubnormal) {
        // Underflow to signed zero.
        return static_cast<uint8_t>(sign << 7);
    }
    if (absv > kMaxVal) {
        // Saturate to signed infinity encoding region.
        return static_cast<uint8_t>((sign << 7) | 0x7C);
    }
    if (absv < kMinNormal) {
        // Subnormal region [kMinSubnormal, kMinNormal).
        int mant = static_cast<int>(std::round(absv / kMinSubnormal));
        mant = std::clamp(mant, 1, 3);
        return static_cast<uint8_t>((sign << 7) | mant);
    }
    // Normal numbers.
    float log2v = std::log2(absv);
    int exp = static_cast<int>(std::round(log2v + 15.0f));
    exp = std::clamp(exp, 1, 30);
    float scale = std::exp2(static_cast<float>(exp - 15));
    float scale_safe = (scale > 0.0f) ? scale : 1.0f;
    int mant = static_cast<int>(std::round((absv / scale_safe - 1.0f) * 4.0f));
    mant = std::clamp(mant, 0, 3);
    return static_cast<uint8_t>((sign << 7) | (exp << 2) | mant);
}
static torch::Tensor Float32ToFp8E5M2(const torch::Tensor &self) {
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();

    for (int64_t i = 0; i < flat.numel(); ++i) {
        out_ptr[i] = EncodeFloatToFp8E5M2(ptr[i]);
    }
    return result.reshape(x.sizes());
}

// Float32 to FP8 E8M0. Value = sign * 2^(exp-63). Round to nearest power of 2.
static torch::Tensor Float32ToFp8E8M0(const torch::Tensor &self) {
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    const float kMinVal = std::exp2(-63.0f);
    const float kMaxVal = std::exp2(63.0f);
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        float v = ptr[i];
        uint8_t enc = 0;
        if (std::isnan(v) || std::isinf(v) || std::fpclassify(v) == FP_ZERO) {
            enc = (std::signbit(v) && !std::isnan(v)) ? 0x80 : 0;
        } else {
            float absv = std::fabs(v);
            int sign = std::signbit(v) ? 1 : 0;
            absv = std::clamp(absv, kMinVal, kMaxVal);
            int exp = static_cast<int>(std::round(std::log2(absv) + 63.0f));
            exp = std::clamp(exp, 0, 127);
            enc = (sign << 7) | exp;
        }
        out_ptr[i] = enc;
    }
    return result.reshape(x.sizes());
}

torch::Tensor Float32ToFp8(const torch::Tensor &self, DataType actualType) {
    switch (actualType) {
        case DT_FP8:
        case DT_FP8E4M3: return Float32ToFp8E4M3(self);
        case DT_FP8E5M2: return Float32ToFp8E5M2(self);
        case DT_FP8E8M0: return Float32ToFp8E8M0(self);
        default: return self.to(torch::kUInt8);
    }
}

} // namespace npu::tile_fwk

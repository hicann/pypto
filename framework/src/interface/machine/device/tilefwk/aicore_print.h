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
 * \file aicore_print.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include "aikernel_data.h"

#ifdef __TILE_FWK_AICORE__
#include "tileop/utils/layout.h"
#endif

#define ENABLE_AICORE_PRINT 0

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef __TILE_FWK_HOST__
#include <inttypes.h>
#include <string>
#include <sstream>
#include <securec.h>
#endif

// ============================================================================
// Platform Detection and Feature Macros
// ============================================================================

#if defined(__DAV_M300__) || defined(__DAV_310R6__) || defined(__DAV_L510__) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3801)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
#define SUPPORT_FP8_HF8_PRINT 1
#define SUPPORT_L1_COPY 0
#else
#define SUPPORT_FP8_HF8_PRINT 0
#define SUPPORT_L1_COPY 1
#endif

// ============================================================================
// Constants Namespace Definitions
// ============================================================================

namespace AicorePrintConst {
    constexpr size_t INDEXED_INDEX_SIZE     = 8;
    constexpr size_t TENSOR_RANGE_SIZE      = 8;
    constexpr size_t NAMELEN_FIELD_SIZE     = 2;
    constexpr size_t MAX_SHAPE_DIMS         = 6;
    constexpr size_t REMOTE_HEADER_SIZE     = 16;
    constexpr size_t WARNING_RESERVE_SPACE  = 10;
    constexpr size_t MIN_BUFFER_TOTAL_SIZE  = WARNING_RESERVE_SPACE + REMOTE_HEADER_SIZE;
    constexpr size_t SHORT_MAX_VALUE        = 32767;
}

namespace Fp32Const {
    constexpr uint32_t EXP_BIAS     = 127;
    constexpr uint32_t EXP_INF_NAN  = 0xFFu;
    constexpr uint32_t SIGN_SHIFT   = 31;
    constexpr uint32_t EXP_SHIFT    = 23;
    constexpr uint32_t MANT_BITS    = 23;
}

namespace Fp16Const {
    constexpr uint16_t SIGN_MASK       = 0x8000u;
    constexpr uint16_t EXP_MASK        = 0x7C00u;
    constexpr uint16_t MANT_MASK       = 0x03FFu;
    constexpr uint16_t NORM_HIDDEN     = 0x0400u;
    constexpr uint16_t EXP_INF_NAN     = 0x1Fu;
    constexpr uint16_t SIGN_SHIFT      = 15;
    constexpr uint16_t EXP_SHIFT       = 10;
    constexpr uint16_t EXP_BIAS        = 15;
    constexpr uint32_t MANT_TO_FP32_SHIFT = 13;
    constexpr uint32_t SUBNORMAL_EXP_BASE = Fp32Const::EXP_BIAS - (EXP_BIAS - 1);
}

namespace Bf16Const {
    constexpr uint32_t TO_FP32_SHIFT = 16;
}

namespace Fp8Const {
    constexpr uint8_t SIGN_SHIFT = 7;
    constexpr uint8_t BIT_MASK_1 = 0x1;
}

namespace Hf8Const {
    constexpr uint8_t SIGN_SHIFT         = 7;
    constexpr uint8_t LOWER_BITS_MASK    = 0x7F;
    constexpr uint8_t MV_MASK_3BIT       = 0x7;
    constexpr uint8_t MV_MASK_2BIT       = 0x3;
    constexpr uint8_t MV_MASK_1BIT       = 0x1;
    constexpr uint8_t TOP4_SHIFT         = 3;
    constexpr uint8_t TOP3_SHIFT         = 4;
    constexpr uint8_t TOP2_SHIFT         = 5;
    constexpr uint8_t EB_SHIFT_MEDIUM    = 3;
    constexpr uint8_t EB_SHIFT_HUGE      = 2;  // 用于 DecodeHf8Huge 提取 eb
    constexpr uint8_t EB_SHIFT_MAX       = 1;  // 用于 DecodeHf8Max 提取 eb
    constexpr uint8_t EB_MASK_WIDTH_1BIT = 1;
    constexpr uint8_t MANT_BITS_3        = 3;
    constexpr uint8_t MANT_BITS_2        = 2;
    constexpr uint8_t MANT_BITS_1        = 1;
    constexpr int     MV_TO_FP32_EXP_BASE = 23;
    constexpr int     EB_MASK_WIDTH_LARGE = 2;
    constexpr int     EB_MASK_WIDTH_HUGE  = 3;
    constexpr int     EB_MASK_WIDTH_MAX   = 4;
    constexpr int     EV_BASE_LARGE       = 2;
    constexpr int     EV_BASE_HUGE        = 4;
    constexpr int     EV_BASE_MAX         = 8;
    constexpr int     RANGE_TINY          = 0;
    constexpr int     RANGE_SMALL         = 1;
    constexpr int     RANGE_MEDIUM        = 1;
    constexpr int     RANGE_LARGE         = 1;
    constexpr int     RANGE_HUGE          = 2;
    constexpr int     EB_ZERO             = 0;
    constexpr int     EV_POSITIVE         = 1;
    constexpr int     EV_NEGATIVE         = -1;
    constexpr int     MV_ZERO             = 0;
}

namespace Fp8E4M3Const {
    constexpr uint8_t EXP_BITS  = 4;
    constexpr uint8_t MANT_BITS = 3;
    constexpr uint8_t EXP_BIAS  = 7;
    constexpr uint8_t EXP_MASK  = 0xF;
    constexpr uint8_t EXP_MAX   = 15;
    constexpr float   MAX_VALUE = 240.0f;
}

namespace Fp8E5M2Const {
    constexpr uint8_t EXP_BITS  = 5;
    constexpr uint8_t MANT_BITS = 2;
    constexpr uint8_t EXP_BIAS  = 15;
}

namespace Fp8E8M0Const {
    constexpr uint8_t EXP_BIAS   = 127;
    constexpr uint8_t SIGN_SHIFT = Fp8Const::SIGN_SHIFT;
}

// ============================================================================
// Data Type Enumeration
// ============================================================================

namespace AicorePrint {
    enum class DataType : uint8_t {
        End             = 0,
        Normal          = 1,
        Fp32            = 2,
        Int64           = 3,
        Char            = 4,
        String          = 5,
        Pointer         = 6,
        Bf16            = 7,
        Fp16            = 8,
        TensorHeader    = 9,
        IndexedFp32     = 10,
        IndexedInt64    = 11,
        IndexedBf16     = 12,
        IndexedFp16     = 13,
        OverflowWarning = 14,
        Fp8E4M3         = 15,
        Fp8E5M2         = 16,
        Fp8E8M0         = 17,
        Hf8             = 18,
        IndexedFp8E4M3  = 19,
        IndexedFp8E5M2  = 20,
        IndexedFp8E8M0  = 21,
        IndexedHf8      = 22,
    };
}

// ============================================================================
// LogContext Structure
// ============================================================================

struct LogContext {
    void (*PrintInt64)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintRaw)(LogContext* ctx, __gm__ const char* fmt);
    void (*PrintFp8E4M3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E5M2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E8M0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintHf8)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
};

// ============================================================================
// IndexedTypeInfo Template
// ============================================================================

template <typename ElemT>
struct IndexedTypeInfo {
    static constexpr AicorePrint::DataType Type =
        std::is_same_v<ElemT, float>         ? AicorePrint::DataType::IndexedFp32 :
        std::is_same_v<ElemT, int64_t>       ? AicorePrint::DataType::IndexedInt64 :
#if IS_AICORE
        std::is_same_v<ElemT, bfloat16_t>    ? AicorePrint::DataType::IndexedBf16 :
        std::is_same_v<ElemT, half>          ? AicorePrint::DataType::IndexedFp16 :
#if SUPPORT_FP8_HF8_PRINT
        std::is_same_v<ElemT, float8_e4m3_t> ? AicorePrint::DataType::IndexedFp8E4M3 :
        std::is_same_v<ElemT, float8_e5m2_t> ? AicorePrint::DataType::IndexedFp8E5M2 :
        std::is_same_v<ElemT, float8_e8m0_t> ? AicorePrint::DataType::IndexedFp8E8M0 :
        std::is_same_v<ElemT, hifloat8_t>    ? AicorePrint::DataType::IndexedHf8 :
#endif
#endif
        AicorePrint::DataType::End;
};

// ============================================================================
// SafeBitCast Functions
// ============================================================================

template <typename T, typename U>
INLINE void SafeBitCast(T& dst, const U& src) {
    static_assert(sizeof(T) >= sizeof(U), "Target type too small");
    const unsigned char* srcBytes = reinterpret_cast<const unsigned char*>(&src);
    unsigned char* dstBytes = reinterpret_cast<unsigned char*>(&dst);

    for (std::size_t i = 0; i < sizeof(U); ++i) {
        dstBytes[i] = srcBytes[i];
    }

    for (std::size_t i = sizeof(U); i < sizeof(T); ++i) {
        dstBytes[i] = 0;
    }
}

template <typename T, typename U>
INLINE T SafeBitCast(const U& src) {
    T dst{};
    SafeBitCast(dst, src);
    return dst;
}

// ============================================================================
// BF16 Decode Function
// ============================================================================

INLINE float DecodeBf16(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << Bf16Const::TO_FP32_SHIFT;
    return SafeBitCast<float>(u);
}

// ============================================================================
// FP16 Decode Function
// ============================================================================

INLINE float DecodeF16(uint16_t bits) {
    const uint16_t sign = (bits & Fp16Const::SIGN_MASK) >> Fp16Const::SIGN_SHIFT;
    const uint16_t exp  = (bits & Fp16Const::EXP_MASK) >> Fp16Const::EXP_SHIFT;
    const uint16_t mant = bits & Fp16Const::MANT_MASK;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32  = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp16Const::SUBNORMAL_EXP_BASE;
            uint16_t normMant = mant;

            while ((normMant & Fp16Const::NORM_HIDDEN) == 0) {
                normMant <<= 1;
                --exp32;
            }

            normMant &= Fp16Const::MANT_MASK;
            mant32 = static_cast<uint32_t>(normMant) << Fp16Const::MANT_TO_FP32_SHIFT;
        }
    } else if (exp == Fp16Const::EXP_INF_NAN) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    } else {
        exp32 = static_cast<uint32_t>(exp) - Fp16Const::EXP_BIAS + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ============================================================================
// FP8 Common Decode Template
// ============================================================================

template<uint8_t ExpBits, uint8_t MantBits, uint8_t ExpBias, bool HasInf>
INLINE float DecodeFp8Common(uint8_t bits) {
    constexpr uint8_t SignShift  = Fp8Const::SIGN_SHIFT;
    constexpr uint8_t ExpShift   = MantBits;
    constexpr uint8_t ExpMax     = (1u << ExpBits) - 1;
    constexpr uint8_t MantMask   = (1u << MantBits) - 1;

    const uint8_t sign = (bits >> SignShift) & Fp8Const::BIT_MASK_1;
    const uint8_t exp  = (bits >> ExpShift) & ((1u << ExpBits) - 1);
    const uint8_t mant = bits & MantMask;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32  = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp32Const::EXP_BIAS - (ExpBias - 1);
            uint32_t normMant = static_cast<uint32_t>(mant);
            constexpr uint32_t hiddenBit = (1u << MantBits);

            while ((normMant & hiddenBit) == 0) {
                normMant <<= 1;
                --exp32;
            }

            normMant &= ((1u << MantBits) - 1);
            mant32 = normMant << (Fp32Const::MANT_BITS - MantBits);
        }
    } else if (HasInf && exp == ExpMax) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    } else {
        exp32 = static_cast<uint32_t>(exp) - ExpBias + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ============================================================================
// FP8 E4M3/E5M2/E8M0 Decode Functions
// ============================================================================

INLINE float DecodeFp8E4M3(uint8_t bits) {
    const uint8_t exp = (bits >> Fp8E4M3Const::MANT_BITS) & Fp8E4M3Const::EXP_MASK;
    if (exp == Fp8E4M3Const::EXP_MAX) {
        const uint8_t sign = (bits >> Fp8Const::SIGN_SHIFT) & Fp8Const::BIT_MASK_1;
        return sign ? -Fp8E4M3Const::MAX_VALUE : Fp8E4M3Const::MAX_VALUE;
    }
    return DecodeFp8Common<Fp8E4M3Const::EXP_BITS, Fp8E4M3Const::MANT_BITS, Fp8E4M3Const::EXP_BIAS, false>(bits);
}

INLINE float DecodeFp8E5M2(uint8_t bits) {
    return DecodeFp8Common<Fp8E5M2Const::EXP_BITS, Fp8E5M2Const::MANT_BITS, Fp8E5M2Const::EXP_BIAS, true>(bits);
}

INLINE float DecodeFp8E8M0(uint8_t bits) {
    // E8M0 format: 8-bit exponent (no sign bit, no mantissa), bias=127
    // Used for MX scaling (scale factor), only positive values
    if (bits == 0) {
        return 0.0f;
    }
    
    // E8M0 uses unsigned exponent semantics (no sign bit)
    // value = 2^(exp - 127)
    const uint32_t exp = static_cast<uint32_t>(bits);
    const uint32_t exp32 = exp - Fp8E8M0Const::EXP_BIAS + Fp32Const::EXP_BIAS;
    const uint32_t mant32 = 0;
    
    uint32_t result = (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ============================================================================
// HF8 Decode Functions
// ============================================================================

INLINE float DecodeHf8Tiny(int signBit, int mv) {
    if (mv == Hf8Const::MV_ZERO) {
        uint32_t result = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
        return SafeBitCast<float>(result);
    }

    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const int fp32Exp = mv - Hf8Const::MV_TO_FP32_EXP_BASE + Fp32Const::EXP_BIAS;
    const uint32_t result = sign32 | (static_cast<uint32_t>(fp32Exp) << Fp32Const::EXP_SHIFT);
    return SafeBitCast<float>(result);
}

INLINE float DecodeHf8Small(int signBit, int mv) {
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32  = Fp32Const::EXP_BIAS;
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - Hf8Const::MANT_BITS_3);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8WithEvMv(int signBit, int ev, int mv, int mantBits) {
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32  = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - mantBits);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Medium(int signBit, int lower7) {
    const int eb = (lower7 >> Hf8Const::EB_SHIFT_MEDIUM) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int ev = (eb == Hf8Const::EB_ZERO) ? Hf8Const::EV_POSITIVE : Hf8Const::EV_NEGATIVE;
    const int mv = lower7 & Hf8Const::MV_MASK_3BIT;
    return DecodeHf8WithEvMv(signBit, ev, mv, Hf8Const::MANT_BITS_3);
}

template<int EbShift, int EbMaskWidth, int EvBase, int MvMask, int MantBits>
INLINE float DecodeHf8EvMvPattern(int signBit, int lower7) {
    const int ebMask = (1 << EbMaskWidth) - 1;
    const int eb = (lower7 >> EbShift) & ebMask;
    const int evSign = (eb >> (EbMaskWidth - 1)) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int addMask = (1 << (EbMaskWidth - 1)) - 1;
    const int evAbs = EvBase + (eb & addMask);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & MvMask;
    return DecodeHf8WithEvMv(signBit, ev, mv, MantBits);
}

INLINE float DecodeHf8Large(int signBit, int lower7) {
    return DecodeHf8EvMvPattern<Hf8Const::TOP4_SHIFT, Hf8Const::EB_MASK_WIDTH_LARGE, Hf8Const::EV_BASE_LARGE,
                                 Hf8Const::MV_MASK_3BIT, Hf8Const::MANT_BITS_3>(signBit, lower7);
}

INLINE float DecodeHf8Huge(int signBit, int lower7) {
    return DecodeHf8EvMvPattern<Hf8Const::EB_SHIFT_HUGE, Hf8Const::EB_MASK_WIDTH_HUGE, Hf8Const::EV_BASE_HUGE,
                                 Hf8Const::MV_MASK_2BIT, Hf8Const::MANT_BITS_2>(signBit, lower7);
}

INLINE float DecodeHf8Max(int signBit, int lower7) {
    return DecodeHf8EvMvPattern<Hf8Const::EB_SHIFT_MAX, Hf8Const::EB_MASK_WIDTH_MAX, Hf8Const::EV_BASE_MAX,
                                 Hf8Const::MV_MASK_1BIT, Hf8Const::MANT_BITS_1>(signBit, lower7);
}

INLINE float DecodeHf8(uint8_t bits) {
    const int signBit = (bits >> Hf8Const::SIGN_SHIFT) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int lower7  = bits & Hf8Const::LOWER_BITS_MASK;
    const int top4    = lower7 >> Hf8Const::TOP4_SHIFT;

    if (top4 == Hf8Const::RANGE_TINY) {
        return DecodeHf8Tiny(signBit, lower7 & Hf8Const::MV_MASK_3BIT);
    }

    if (top4 == Hf8Const::RANGE_SMALL) {
        return DecodeHf8Small(signBit, lower7 & Hf8Const::MV_MASK_3BIT);
    }

    const int top3 = lower7 >> Hf8Const::TOP3_SHIFT;

    if (top3 == Hf8Const::RANGE_MEDIUM) {
        return DecodeHf8Medium(signBit, lower7);
    }

    const int top2 = lower7 >> Hf8Const::TOP2_SHIFT;

    if (top2 == Hf8Const::RANGE_LARGE) {
        return DecodeHf8Large(signBit, lower7);
    }

    if (top2 == Hf8Const::RANGE_HUGE) {
        return DecodeHf8Huge(signBit, lower7);
    }

    return DecodeHf8Max(signBit, lower7);
}

// ============================================================================
// Host-Side Decode Helpers
// ============================================================================

#ifdef __TILE_FWK_HOST__

struct DecodeState {
    int64_t tail_;
    int64_t head_;
    int64_t size_;
    __gm__ uint8_t* data_;
    std::string lastTensorName_;
};

inline uint8_t ReadDecodeByte(DecodeState& state, int64_t off) {
    return state.data_[off % state.size_];
}

template <typename T>
inline T ReadDecodeValue(DecodeState& state, int64_t off) {
    T val{};
    auto* bytes = reinterpret_cast<uint8_t*>(&val);

    for (size_t i = 0; i < sizeof(T); i++) {
        bytes[i] = ReadDecodeByte(state, off + i);
    }

    return val;
}

inline std::string ReadDecodeString(DecodeState& state, int64_t off) {
    std::string result;
    result.reserve(64);

    while (off < state.head_) {
        char c = ReadDecodeValue<char>(state, off++);

        if (c == '\0') {
            break;
        }

        result.push_back(c);
    }

    return result;
}

inline int DecodeTensorHeader(DecodeState& state, char* buf, size_t maxSize) {
    short nameLen = ReadDecodeValue<short>(state, state.tail_);
    state.tail_ += AicorePrintConst::NAMELEN_FIELD_SIZE;

    std::string name = ReadDecodeString(state, state.tail_);
    state.tail_ += nameLen;

    int64_t begin = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    int64_t end = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    state.lastTensorName_ = name;

    return snprintf_s(buf, maxSize, maxSize - 1, "tensor '%s', range=[%" PRId64 ", %" PRId64 ")\n", name.c_str(), begin, end);
}

template<typename BitsT, typename DecodeFunc>
inline int DecodeIndexedFloat(DecodeState& state, char* buf, size_t maxSize, DecodeFunc decodeFunc) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;

    BitsT bits = ReadDecodeValue<BitsT>(state, state.tail_);
    state.tail_ += sizeof(BitsT);

    float value = decodeFunc(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%" PRId64 "] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedFp32(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<float>(state, buf, maxSize, [](float v) { return v; });
}

inline int DecodeIndexedInt64(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;

    int64_t value = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %" PRId64 "\n",
                      state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedBf16(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint16_t>(state, buf, maxSize, DecodeBf16);
}

inline int DecodeIndexedFp16(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint16_t>(state, buf, maxSize, DecodeF16);
}

inline int DecodeIndexedFp8E4M3(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E4M3);
}

inline int DecodeIndexedFp8E5M2(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E5M2);
}

inline int DecodeIndexedFp8E8M0(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E8M0);
}

inline int DecodeIndexedHf8(DecodeState& state, char* buf, size_t maxSize) {
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeHf8);
}

inline int DecodeOverflowWarning(DecodeState& state, char* buf, size_t maxSize) {
    int64_t bufferSize = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += sizeof(int64_t);

    constexpr int64_t remoteHeaderSize = static_cast<int64_t>(AicorePrintConst::REMOTE_HEADER_SIZE);
    int64_t fullBufferSize = bufferSize + remoteHeaderSize;
    int64_t recommendedSize = fullBufferSize * 2;

    return snprintf_s(buf, maxSize, maxSize - 1,
        "[WARNING] The PRINT_BUFFER_SIZE (ring buffer) is full! "
        "Current buffer: %ld bytes (%ld KB). "
        "Recommend: set PRINT_BUFFER_SIZE >= %ld (%ld KB, double current size) "
        "in framework/src/interface/machine/device/tilefwk/aicpu_common.h, "
        "then rebuild and reinstall.\n",
        fullBufferSize, fullBufferSize / 1024, recommendedSize, recommendedSize / 1024);
}

inline int DecodeLegacyRecord(DecodeState& state, AicorePrint::DataType type,
                              char* buf, size_t maxSize) {
    auto valOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) +
                   AicorePrintConst::NAMELEN_FIELD_SIZE;

    auto fmtOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    std::string fmt = ReadDecodeString(state, fmtOff);
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) +
                   AicorePrintConst::NAMELEN_FIELD_SIZE;

    auto formatValue = [&](auto v) {
        return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), v);
    };
    auto formatString = [&](const std::string& s) {
        return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), s.c_str());
    };

    switch (type) {
        case AicorePrint::DataType::Normal:
            return formatValue(0);

        case AicorePrint::DataType::Fp32:
            return formatValue(ReadDecodeValue<float>(state, valOff));

        case AicorePrint::DataType::Int64:
            return formatValue(ReadDecodeValue<int64_t>(state, valOff));

        case AicorePrint::DataType::Char:
            return formatValue(ReadDecodeValue<char>(state, valOff));

        case AicorePrint::DataType::String:
            return formatString(ReadDecodeString(state, valOff));

        case AicorePrint::DataType::Pointer:
            return formatValue(ReadDecodeValue<int64_t>(state, valOff));

        case AicorePrint::DataType::Bf16:
            return formatValue(DecodeBf16(ReadDecodeValue<uint16_t>(state, valOff)));

        case AicorePrint::DataType::Fp16:
            return formatValue(DecodeF16(ReadDecodeValue<uint16_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E4M3:
            return formatValue(DecodeFp8E4M3(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E5M2:
            return formatValue(DecodeFp8E5M2(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E8M0:
            return formatValue(DecodeFp8E8M0(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Hf8:
            return formatValue(DecodeHf8(ReadDecodeValue<uint8_t>(state, valOff)));

        default:
            buf[0] = '?';
            return 1;
    }
}

inline int DecodeRecordImpl(DecodeState& state, AicorePrint::DataType type, char* buf, size_t maxSize) {
    switch (type) {
        case AicorePrint::DataType::TensorHeader:
            return DecodeTensorHeader(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp32:
            return DecodeIndexedFp32(state, buf, maxSize);

        case AicorePrint::DataType::IndexedInt64:
            return DecodeIndexedInt64(state, buf, maxSize);

        case AicorePrint::DataType::IndexedBf16:
            return DecodeIndexedBf16(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp16:
            return DecodeIndexedFp16(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E4M3:
            return DecodeIndexedFp8E4M3(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E5M2:
            return DecodeIndexedFp8E5M2(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E8M0:
            return DecodeIndexedFp8E8M0(state, buf, maxSize);

        case AicorePrint::DataType::IndexedHf8:
            return DecodeIndexedHf8(state, buf, maxSize);

        case AicorePrint::DataType::OverflowWarning:
            return DecodeOverflowWarning(state, buf, maxSize);

        default:
            return DecodeLegacyRecord(state, type, buf, maxSize);
    }
}

#endif

// ============================================================================
// AicoreLogger Class
// ============================================================================

class AicoreLogger {
public:
    struct RemoteHeader {
        int64_t head_;
        int64_t tail_;
    };

#define AICORE_STATIC_PRINT_WRAPPER(Name, Func, T) \
    static __aicore__ void Name(LogContext* ctx, __gm__ const char** fmt, T val) { \
        auto* self = reinterpret_cast<AicoreLogger*>(ctx); \
        if (self) { \
            self->Func(fmt, val); \
        } \
    }

    AICORE_STATIC_PRINT_WRAPPER(StaticPrintInt64,    PrintInt64,    int64_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp32,     PrintFp32,     float)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintBf16,     PrintBf16,     uint16_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp16,     PrintFp16,     uint16_t)

    static __aicore__ void StaticPrintRaw(LogContext* ctx, __gm__ const char* fmt) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintRaw(fmt);
        }
    }

    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E4M3, PrintFp8E4M3, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E5M2, PrintFp8E5M2, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E8M0, PrintFp8E8M0, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintHf8,     PrintHf8,     uint8_t)

#undef AICORE_STATIC_PRINT_WRAPPER

    __aicore__ void Init(__gm__ uint8_t* buf, size_t n) {
        if (n < AicorePrintConst::MIN_BUFFER_TOTAL_SIZE) {
            overflowed_ = true;
            size_ = 0;
            remote_ = nullptr;
            data_ = nullptr;
            head_ = 0;
            tail_ = 0;
            return;
        }

        remote_ = reinterpret_cast<volatile __gm__ RemoteHeader*>(buf);
        remote_->head_ = 0;
        remote_->tail_ = 0;
        head_ = 0;
        tail_ = 0;
        size_ = n - sizeof(RemoteHeader);
        data_ = buf + sizeof(RemoteHeader);
        overflowed_ = false;

        ctx_.PrintInt64    = StaticPrintInt64;
        ctx_.PrintFp32     = StaticPrintFp32;
        ctx_.PrintBf16     = StaticPrintBf16;
        ctx_.PrintFp16     = StaticPrintFp16;
        ctx_.PrintRaw      = StaticPrintRaw;
        ctx_.PrintFp8E4M3  = StaticPrintFp8E4M3;
        ctx_.PrintFp8E5M2  = StaticPrintFp8E5M2;
        ctx_.PrintFp8E8M0  = StaticPrintFp8E8M0;
        ctx_.PrintHf8      = StaticPrintHf8;
    }

    __aicore__ __gm__ uint8_t* GetBuffer() const {
        return data_ - sizeof(RemoteHeader);
    }

    INLINE LogContext* Context() {
        return &ctx_;
    }

    __aicore__ void PrintInt64(__gm__ const char** fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);

        if (idx == -1) {
            return;
        }

        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char*>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                EncodeTyped(AicorePrint::DataType::String,
                            reinterpret_cast<__gm__ const uint8_t*>(tmp), StringLength(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u':
                EncodeTyped(AicorePrint::DataType::Int64, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;

            case 'p':
                EncodeTyped(AicorePrint::DataType::Pointer, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;

            case 'c': {
                char c = static_cast<char>(val);
                EncodeTyped(AicorePrint::DataType::Char, reinterpret_cast<uint8_t*>(&c), 1, *fmt, idx);
                break;
            }

            default:
                EncodeTyped(AicorePrint::DataType::Normal, static_cast<uint8_t*>(nullptr), 0, *fmt, idx);
                break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFp32(__gm__ const char** fmt, float val) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp32, reinterpret_cast<uint8_t*>(&val), sizeof(val));
    }

    __aicore__ void PrintBf16(__gm__ const char** fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Bf16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp16(__gm__ const char** fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    #define AICORE_ENCODE_FP8_FUNC(Name, DataTypeVal) \
    __aicore__ void Name(__gm__ const char** fmt, uint8_t rawBits) { \
        EncodeFloatType(fmt, DataTypeVal, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits)); \
    }

    AICORE_ENCODE_FP8_FUNC(PrintFp8E4M3, AicorePrint::DataType::Fp8E4M3)
    AICORE_ENCODE_FP8_FUNC(PrintFp8E5M2, AicorePrint::DataType::Fp8E5M2)
    AICORE_ENCODE_FP8_FUNC(PrintFp8E8M0, AicorePrint::DataType::Fp8E8M0)
    AICORE_ENCODE_FP8_FUNC(PrintHf8,     AicorePrint::DataType::Hf8)

#undef AICORE_ENCODE_FP8_FUNC

    __aicore__ void PrintRaw(__gm__ const char* str) {
        auto n = StringLength(str);

        if (n) {
            EncodeTyped(AicorePrint::DataType::Normal, reinterpret_cast<const __gm__ uint8_t*>(str), n, str, n);
        }

        Sync();
    }

    __aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end) {
        size_t nameLenRaw = StringLength(name) + 1;

        if (nameLenRaw > static_cast<size_t>(AicorePrintConst::SHORT_MAX_VALUE)) {
            return;
        }

        short nameLen = static_cast<short>(nameLenRaw);
        int64_t recordSize = 1 + sizeof(short) + nameLen + sizeof(int64_t) * 2 + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::TensorHeader));
        EncodeValue<short>(nameLen);

        for (short i = 0; i < nameLen; i++) {
            EncodeByte(static_cast<uint8_t>(name[i]));
        }

        EncodeValue<int64_t>(begin);
        EncodeValue<int64_t>(end);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeIndexed(AicorePrint::DataType ty, int64_t index, const uint8_t* val, short valLen) {
        int64_t recordSize = 1 + sizeof(int64_t) + valLen + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<int64_t>(index);

        for (short i = 0; i < valLen; i++) {
            EncodeByte(val[i]);
        }

        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeOverflowWarning(int64_t bufferSize) {
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::OverflowWarning));
        EncodeValue<int64_t>(bufferSize);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template<typename PtrT>
    __aicore__ void EncodeTyped(AicorePrint::DataType ty, [[maybe_unused]]PtrT val,
                                short valLen, __gm__ const char* fmt, int fmtLen) {
        short paddedFmtLen = fmtLen + 1;
        int64_t recordSize = 1 + sizeof(short) + valLen + sizeof(short) + paddedFmtLen + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<short>(valLen);

        if constexpr (!std::is_same_v<std::remove_cv_t<PtrT>, std::nullptr_t>) {
            if (val) {
                for (short i = 0; i < valLen; i++) {
                    EncodeByte(val[i]);
                }
            }
        }

        EncodeValue<short>(paddedFmtLen);

        for (int i = 0; i < fmtLen; i++) {
            EncodeByte(static_cast<uint8_t>(fmt[i]));
        }

        EncodeByte('\0');
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template<typename PtrT>
    __aicore__ void EncodeFloatType(__gm__ const char** fmt, AicorePrint::DataType ty, PtrT val, short valLen) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);

        if (idx == -1) {
            return;
        }

        if (curFmt[idx] == 'f') {
            EncodeTyped(ty, val, valLen, *fmt, idx + 1);
        } else {
            EncodeTyped(AicorePrint::DataType::Normal, nullptr, 0, *fmt, idx + 1);
        }

        *fmt = *fmt + idx + 1;
    }

    __aicore__ void Sync() {
        if (remote_ == nullptr || data_ == nullptr) {
            return;
        }

#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE - 1);
        int64_t off = remote_->head_ - delta;

        while (off < head_) {
            dcci(&data_[off % size_], SINGLE_CACHE_LINE, CACHELINE_OUT);
            off += CACHE_LINE_SIZE;
        }

        remote_->head_ = head_;
        remote_->tail_ = tail_;
        dcci(remote_, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
        remote_->head_ = head_;
        remote_->tail_ = tail_;
#endif
    }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char* buf, size_t maxSize, uint32_t maxIterations = 1000) {
        if (remote_ == nullptr || data_ == nullptr || size_ == 0) {
            return 0;
        }

        size_t totalWritten = 0;
        head_ = remote_->head_;

        if (tail_ < remote_->tail_) {
            tail_ = remote_->tail_;
        }

        DecodeState state{tail_, head_, size_, data_, lastTensorName_};
        uint32_t iterationCount = 0;

        while (state.tail_ != state.head_ && iterationCount < maxIterations) {
            iterationCount++;
            AicorePrint::DataType type = static_cast<AicorePrint::DataType>(ReadDecodeByte(state, state.tail_++));

            if (type == AicorePrint::DataType::End) {
                if (totalWritten > 0) {
                    tail_ = state.tail_;
                    lastTensorName_ = state.lastTensorName_;
                    return static_cast<int>(totalWritten);
                }
                continue;
            }

            if (maxSize == 0) {
                continue;
            }

            int written = DecodeRecordImpl(state, type, buf, maxSize);

            if (written > 0) {
                buf += written;
                totalWritten += written;
                maxSize -= written;
            }
        }

        tail_ = state.tail_;
        lastTensorName_ = state.lastTensorName_;
        return 0;
    }
#pragma GCC diagnostic pop
#endif

private:
    __aicore__ bool CheckSpaceForRecord(int64_t recordSize) {
        if (overflowed_ || size_ == 0) {
            return false;
        }

        int64_t freeSpace = size_ - (head_ - tail_);
        int64_t requiredSpace = recordSize + AicorePrintConst::WARNING_RESERVE_SPACE;

        if (freeSpace < requiredSpace) {
            EncodeOverflowWarning(size_);
            Sync();
            overflowed_ = true;
            return false;
        }

        return true;
    }

    __aicore__ void EncodeByte(uint8_t val) {
        volatile __gm__ uint8_t* p = &data_[head_++ % size_];
        *p = val;
    }

    template<typename T>
    __aicore__ void EncodeValue(T value) {
        auto* bytes = reinterpret_cast<uint8_t*>(&value);

        for (size_t i = 0; i < sizeof(T); i++) {
            EncodeByte(bytes[i]);
        }
    }

    INLINE bool IsFormatFlagChar(char c) {
        return c == '0' || c == '+' || c == '-' || c == ' ' || c == '#';
    }

    INLINE int64_t SkipFormatFlags(__gm__ const char* fmt, int64_t idx) {
        while (fmt[idx] && IsFormatFlagChar(fmt[idx])) {
            idx++;
        }
        return idx;
    }

    INLINE int64_t SkipLengthModifier(__gm__ const char* fmt, int64_t idx) {
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l') {
                idx++;
            }
        }
        return idx;
    }

    __aicore__ int64_t ParseNextFormat(__gm__ const char* fmt) {
        int64_t idx = 0;

        while (fmt[idx]) {
            if (fmt[idx] == '%') {
                if (fmt[idx + 1] == '%') {
                    idx += 2;
                } else {
                    break;
                }
            } else {
                idx++;
            }
        }

        if (!fmt[idx]) {
            return -1;
        }

        idx++;
        idx = SkipFormatFlags(fmt, idx);

        while (IsDigit(fmt[idx])) {
            idx++;
        }

        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) {
                idx++;
            }
        }

        idx = SkipLengthModifier(fmt, idx);

        if (fmt[idx]) {
            return idx;
        } else {
            return -1;
        }
    }

    INLINE size_t StringLength(__gm__ const char* str) {
        size_t n = 0;

        while (*str++) {
            n++;
        }

        return n;
    }

    INLINE bool IsDigit(char c) {
        return c >= '0' && c <= '9';
    }

    LogContext ctx_{};
    int64_t head_ = 0;
    int64_t tail_ = 0;
    int64_t size_ = 0;
    volatile __gm__ RemoteHeader* remote_ = nullptr;
    __gm__ uint8_t* data_ = nullptr;
    bool overflowed_ = false;

#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_{};
#endif
};

// ============================================================================
// Internal Helper Functions
// ============================================================================

template <typename T>
INLINE void DispatchPrint(LogContext* ctx, __gm__ const char** fmt, T val) {
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt64(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFp32(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt64(ctx, fmt, reinterpret_cast<int64_t>(val));
#if IS_AICORE
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ctx->PrintBf16(ctx, fmt, SafeBitCast<uint16_t>(val));
    } else if constexpr (std::is_same_v<T, half>) {
        ctx->PrintFp16(ctx, fmt, SafeBitCast<uint16_t>(val));
#if SUPPORT_FP8_HF8_PRINT
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        ctx->PrintFp8E4M3(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        ctx->PrintFp8E5M2(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        ctx->PrintFp8E8M0(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, hifloat8_t>) {
        ctx->PrintHf8(ctx, fmt, SafeBitCast<uint8_t>(val));
#endif
#endif
    }
}

template <typename... Ts>
INLINE void AiCoreLogF(LogContext* ctx, __gm__ const char* fmt, Ts... args) {
    if (ctx && fmt) {
        (DispatchPrint(ctx, &fmt, args), ...);
        ctx->PrintRaw(ctx, fmt);
    }
}

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)

template <size_t I, typename ShapeTuple>
INLINE void FillShapeDims(int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS], const ShapeTuple& shape) {
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AicorePrintConst::MAX_SHAPE_DIMS) ? n : AicorePrintConst::MAX_SHAPE_DIMS;

    if constexpr (I < m) {
        dims[I] = static_cast<int64_t>(Std::get<I>(shape));
        FillShapeDims<I + 1>(dims, shape);
    }
}

template <size_t N>
INLINE void LogShapeDims(LogContext* ctx, const int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS]) {
    if constexpr (N == 1) {
        AiCoreLogF(ctx, "shape=[%ld]\n", dims[0]);
    } else if constexpr (N == 2) {
        AiCoreLogF(ctx, "shape=[%ld,%ld]\n", dims[0], dims[1]);
    } else if constexpr (N == 3) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld]\n", dims[0], dims[1], dims[2]);
    } else if constexpr (N == 4) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3]);
    } else if constexpr (N == 5) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
    } else if constexpr (N == 6) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    }
}

#endif

template <typename T, typename PtrT>
INLINE void PrintTensorImpl(LogContext* ctx, PtrT data, int64_t end, int64_t begin, __gm__ const char* name) {
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
    logger->EncodeTensorHeader(name, begin, end);
    logger->Sync();

    for (int64_t i = begin; i < end; ++i) {
        ElemT tmp = data[i];

        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<float>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<int64_t>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (sizeof(ElemT) == 2) {
            auto bits = SafeBitCast<uint16_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        } else if constexpr (sizeof(ElemT) == 1) {
            uint8_t bits = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        }

        logger->Sync();
    }
}

// ============================================================================
// Public Interface Functions
// ============================================================================

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
template <typename... Dims>
INLINE void AiCorePrintShape(LogContext* ctx, const TileOp::Shape<Dims...>& shape) {
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;

    if constexpr (N == 0 || N > AicorePrintConst::MAX_SHAPE_DIMS) {
        return;
    }

    int64_t dims[AicorePrintConst::MAX_SHAPE_DIMS]{};
    FillShapeDims<0>(dims, shape);
    LogShapeDims<N>(ctx, dims);
}
#endif

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data, int64_t end,
                                int64_t begin, __gm__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data, int64_t end,
                                int64_t begin, __ubuf__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}
#endif

#if IS_AICORE && SUPPORT_L1_COPY
template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count) {
    int64_t totalBytes = count * sizeof(T);

    if (totalBytes == 0) {
        return;
    }

    uint16_t nBurst    = 1;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    uint16_t lenBurst  = 0;

    if (totalBytes >= 32) {
        lenBurst = static_cast<uint16_t>((totalBytes + 31) / 32);
    } else if (totalBytes > 0) {
        lenBurst = static_cast<uint16_t>(totalBytes);
    } else {
        lenBurst = 1;
    }

    copy_cbuf_to_gm(dst, src, 0, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data, int64_t end,
                                int64_t begin, __gm__ T* staging, __gm__ const char* name) {
    int64_t count = end - begin;

    if (count <= 0) {
        return;
    }

    uint64_t stagingAddr = reinterpret_cast<uint64_t>(staging);

    if ((stagingAddr & 0x1F) != 0) {
        AiCoreLogF(ctx, "[WARNING] AiCorePrintL1Tensor: Parameter 5 (L1 staging address) is not aligned to 32 bytes. "
                    "Unable to print L1 data. The address must be 32-byte aligned. Please adjust and retry. "
                    "Current L1 staging address: 0x%lx", stagingAddr);
        return;
    }

    pipe_barrier(PIPE_ALL);
    L1RawCopyToGM(staging, data + begin, count);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    pipe_barrier(PIPE_ALL);

    // Force flush staging buffer cache lines (CACHELINE_OUT = clean + invalidate)
    // This ensures Scalar Processor can see the new data written by DMA
    int64_t totalBytes = count * sizeof(T);
    int64_t offset = 0;

    while (offset < totalBytes) {
        dcci((__gm__ uint8_t*)staging + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        offset += CACHE_LINE_SIZE;
    }

    AiCorePrintGmTensor<T>(ctx, staging, count, 0, name);
}
#endif

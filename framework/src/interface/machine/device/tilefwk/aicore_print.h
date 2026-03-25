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
#include <string>
#include <sstream>
#include <securec.h>
#endif

#define BF16_TO_FP32_SHIFT 16
#define F16_SIGN_MASK 0x8000u
#define F16_EXP_MASK 0x7C00u
#define F16_MANT_MASK 0x03FFu
#define F16_NORM_HIDDEN_BIT 0x0400u
#define F16_EXP_INF_NAN 0x1Fu
#define FP32_EXP_INF_NAN 0xFFu
#define F16_SIGN_SHIFT 15
#define F16_EXP_SHIFT 10
#define FP32_SIGN_SHIFT 31
#define FP32_EXP_SHIFT 23
#define F16_TO_FP32_MANT_SHIFT 13
#define F16_EXP_BIAS 15
#define FP32_EXP_BIAS 127
#define F16_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (F16_EXP_BIAS - 1))

template <typename T, typename U>
INLINE void SafeBitCast(T &dst, const U &src)
{
    const unsigned char *srcBytes = reinterpret_cast<const unsigned char *>(&src);
    unsigned char *dstBytes = reinterpret_cast<unsigned char *>(&dst);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dstBytes[i] = srcBytes[i];
    }
}

template <typename T, typename U>
INLINE T SafeBitCast(const U &src)
{
    T dst;
    SafeBitCast(dst, src);
    return dst;
}

INLINE float DecodeBf16(uint16_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits) << BF16_TO_FP32_SHIFT;
    return SafeBitCast<float>(u);
}

INLINE float DecodeF16(uint16_t bits)
{
    uint16_t sign = static_cast<uint16_t>((bits & F16_SIGN_MASK) >> F16_SIGN_SHIFT);
    uint16_t exp = static_cast<uint16_t>((bits & F16_EXP_MASK) >> F16_EXP_SHIFT);
    uint16_t mant = static_cast<uint16_t>(bits & F16_MANT_MASK);

    uint32_t sign32 = static_cast<uint32_t>(sign) << FP32_SIGN_SHIFT;
    uint32_t exp32;
    uint32_t mant32;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = F16_SUBNORMAL_FP32_EXP_BASE;
            while ((mant & F16_NORM_HIDDEN_BIT) == 0) {
                mant <<= 1;
                --exp32;
            }
            mant &= F16_MANT_MASK;
            mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
        }
    } else if (exp == F16_EXP_INF_NAN) {
        exp32 = FP32_EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
    } else {
        exp32 = static_cast<uint32_t>(exp) - F16_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
    }

    uint32_t u = sign32 | (exp32 << FP32_EXP_SHIFT) | mant32;
    return SafeBitCast<float>(u);
}

enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16 };

struct LogContext {
    void (*PrintInt)(LogContext *ctx, __gm__ const char **fmt, int64_t val);
    void (*PrintFp32)(LogContext *ctx, __gm__ const char **fmt, float val);
    void (*PrintBf16)(LogContext *ctx, __gm__ const char **fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext *ctx, __gm__ const char **fmt, uint16_t rawBits);
    void (*Print)(LogContext *ctx, __gm__ const char *fmt);
};

template <typename T>
INLINE void __AiCorePrint(LogContext *ctx, __gm__ const char **fmt, T val) {
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFp32(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt(ctx, fmt, reinterpret_cast<int64_t>(val));
#if IS_AICORE
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ctx->PrintBf16(ctx, fmt, SafeBitCast<uint16_t>(val));
    } else if constexpr (std::is_same_v<T, half>) {
        ctx->PrintFp16(ctx, fmt, SafeBitCast<uint16_t>(val));
#endif
    }
}

template <typename... Ts>
INLINE void AiCoreLogF(LogContext *ctx, __gm__ const char *fmt, Ts... Args) {
    if (ctx && fmt) {
        (__AiCorePrint(ctx, &fmt, Args), ...);
        ctx->Print(ctx, fmt);
    }
}

struct AicoreLogger {
    struct Remote {
        int64_t head_;
        int64_t tail_;
    };

    static __aicore__ void __PrintInt(LogContext *ctx, __gm__ const char **fmt, int64_t val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintInt(fmt, val);
        }
    }

    static __aicore__ void __PrintFloat(LogContext *ctx, __gm__ const char **fmt, float val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintFp32(fmt, val);
        }
    }

    static __aicore__ void __PrintBf16(LogContext *ctx, __gm__ const char **fmt, uint16_t rawBits) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintBf16(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintF16(LogContext *ctx, __gm__ const char **fmt, uint16_t rawBits) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintFp16(fmt, rawBits);
        }
    }

    static __aicore__ void __Print(LogContext *ctx, __gm__ const char *fmt) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->Print(fmt);
        }
    }

    __aicore__ void Init(__gm__ uint8_t *buf, size_t n) {
        remote_ = reinterpret_cast<volatile __gm__ Remote *>(buf);
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(Remote);
        data_ = buf + sizeof(Remote);
        ctx.PrintInt = __PrintInt;
        ctx.PrintFp32 = __PrintFloat;
        ctx.PrintBf16 = __PrintBf16;
        ctx.PrintFp16 = __PrintF16;
        ctx.Print = __Print;
    }

    __aicore__ __gm__ uint8_t *GetBuffer()  {
        return data_ - sizeof(Remote);
    }

    __aicore__ void PrintInt(__gm__ const char **fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char *>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                Encode(STRING, reinterpret_cast<__gm__ const uint8_t *>(tmp), Length(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u': {
                Encode(INT, reinterpret_cast<uint8_t *>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'p': {
                Encode(POINTER, reinterpret_cast<uint8_t *>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'c': {
                char c = static_cast<char>(val);
                Encode(CHAR, reinterpret_cast<uint8_t *>(&c), 1, *fmt, idx);
                break;
            }
            default: Encode(NORMAL, static_cast<uint8_t *>(nullptr), 0, *fmt, idx); break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFp32(__gm__ const char **fmt, float val) {
        EncodeFloatType(fmt, FP32, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    }

    __aicore__ void PrintBf16(__gm__ const char **fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, BF16, reinterpret_cast<uint8_t *>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp16(__gm__ const char **fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, FP16, reinterpret_cast<uint8_t *>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void Print(__gm__ const char *str) {
        auto n = Length(str);
        if (n) {
            Encode(NORMAL, reinterpret_cast<const __gm__ uint8_t *>(str), n, str, n);
        }
        Encode(END);
        Sync();
    }

    __aicore__ void Sync() {
#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE -1);
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

    INLINE LogContext *context() { return &ctx; }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char *buf, size_t maxSize) {
        size_t size = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) {
            // lose some data
            tail_ = remote_->tail_;
        }
        while (tail_ != head_) {
            auto type = Read<uint8_t>(tail_++);
            if (type == END) {
                if (size == 0)
                    continue;
                else
                    return size;
            } else if (maxSize == 0) {
                continue;
            }

            auto valOff = tail_ + sizeof(short);
            tail_ += Read<short>(tail_) + sizeof(short);
            auto fmtOff = tail_ + sizeof(short);
            std::string fmt = ReadString(fmtOff);
            tail_ += Read<short>(tail_) + sizeof(short);
            int n = 0;
            switch (type) {
                case NORMAL: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0); break;
                case FP32: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<float>(valOff)); break;
                case INT: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff)); break;
                case CHAR: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<char>(valOff)); break;
                case STRING: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadString(valOff).c_str()); break;
                case POINTER: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff)); break;
                case BF16: {
                    uint16_t bits = Read<uint16_t>(valOff);
                    float fv = DecodeBf16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                    break;
                }
                case FP16: {
                    uint16_t bits = Read<uint16_t>(valOff);
                    float fv = DecodeF16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                    break;
                }
                default: if (n) { buf[0] = '?'; n = 1;} break;
            }
            buf += n;
            size += n;
            maxSize -= n;
        }
        return 0;
    }
#pragma GCC diagnostic pop
#endif

private:
    __aicore__ void EncodeFloatType(__gm__ const char **fmt, NodeTy ty, uint8_t *val, short valLen) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 'f': {
                Encode(ty, val, valLen, *fmt, idx);
                break;
            }
            default: Encode(NORMAL, static_cast<uint8_t *>(nullptr), 0, *fmt, idx); break;
        }
        *fmt = *fmt + idx;
    }
    __aicore__ int64_t ParseNextFormat(__gm__ const char *fmt) {
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

        // skip fmt
        while (fmt[idx]) {
            if (fmt[idx] != '0' && fmt[idx] != '+' && fmt[idx] != '-' && fmt[idx] != ' ' && fmt[idx] != '#') {
                break;
            }
            idx++;
        }

        // width
        while (IsDigit(fmt[idx])) {
            idx++;
        }

        // precision
        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) {
                idx++;
            }
        }

        // Length
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l')
                idx++;
        }

        return fmt[idx] ? idx : -1;
    }

    template <typename T>
    INLINE T Read(int64_t off) {
        T val;
        char tmp[sizeof(T)];
        for (size_t i = 0; i < sizeof(T); i++) {
            tmp[i] = data_[(off + i) % size_];
        }
        val = *reinterpret_cast<T *>(tmp);
        return val;
    }

#ifdef __TILE_FWK_HOST__
    std::string ReadString(int64_t off) {
        std::stringstream ss;
        while (off < head_) {
            auto c = Read<char>(off++);
            if (c == '\0') break;
            ss << c;
        }
        return ss.str();
    }
#endif

    __aicore__ void Encode(uint8_t val) {
        if (head_ == tail_ + size_) {
            while (Read<uint8_t>(tail_) != END) {
                tail_++;
                tail_ += Read<short>(tail_) + sizeof(short);
                tail_ += Read<short>(tail_) + sizeof(short);
            }
            tail_++;
        }
        volatile __gm__ uint8_t *p = &data_[head_++ % size_];
        *p = val;
    }

    template<typename T>
    __aicore__ void Encode(NodeTy ty, const T *val, short valLen, __gm__ const char *fmt, int fmtLen) {
        Encode(ty);

        auto bytes = reinterpret_cast<uint8_t *>(&valLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < valLen; i++) {
            Encode(val[i]);
        }

        fmtLen += 1; // pad '\0'
        bytes = reinterpret_cast<uint8_t *>(&fmtLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < fmtLen - 1; i++) {
            Encode(fmt[i]);
        }
        Encode('\0');
    }

    INLINE size_t Length(__gm__ const char *str) {
        size_t n = 0;
        while (*str++) {
            n++;
        }
        return n;
    }

    INLINE bool IsDigit(char c) {
        return c >= '0' && c <= '9';
    }

private:
    LogContext ctx;
    int64_t head_;
    int64_t tail_;
    int64_t size_;
    volatile __gm__ Remote *remote_;
    __gm__ uint8_t *data_;
};

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
constexpr size_t AICORE_PRINT_SHAPE_MAX_DIMS = 6;
template <size_t I, typename ShapeTuple>
INLINE void __AiCoreFillShapeDims(int64_t (&d)[AICORE_PRINT_SHAPE_MAX_DIMS], const ShapeTuple &shape)
{
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AICORE_PRINT_SHAPE_MAX_DIMS) ? n : AICORE_PRINT_SHAPE_MAX_DIMS;
    if constexpr (I < m) {
        d[I] = static_cast<int64_t>(Std::get<I>(shape));
        __AiCoreFillShapeDims<I + 1>(d, shape);
    }
}

template <size_t N>
INLINE void __AiCoreLogShapeDims(LogContext *ctx, const int64_t (&d)[6])
{
    if constexpr (N == 1) {
        AiCoreLogF(ctx, "shape=[%ld]\n", d[0]);
    } else if constexpr (N == 2) {
        AiCoreLogF(ctx, "shape=[%ld,%ld]\n", d[0], d[1]);
    } else if constexpr (N == 3) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld]\n", d[0], d[1], d[2]);
    } else if constexpr (N == 4) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3]);
    } else if constexpr (N == 5) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3], d[4]);
    } else if constexpr (N == 6) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3], d[4], d[5]);
    }
}

template <typename... Dims>
INLINE void AiCorePrintShape(LogContext *ctx, const TileOp::Shape<Dims...> &shape)
{
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;
    if constexpr (N == 0 || N > AICORE_PRINT_SHAPE_MAX_DIMS) {
        return;
    }
    int64_t d[AICORE_PRINT_SHAPE_MAX_DIMS]{};
    __AiCoreFillShapeDims<0>(d, shape);
    __AiCoreLogShapeDims<Std::tuple_size<TileOp::Shape<Dims...>>::value>(ctx, d);
}
#endif

template <typename T, typename PtrT>
INLINE void __AiCorePrintTensorImpl(LogContext *ctx, PtrT data, int64_t end, int64_t begin = 0)
{
    using ElemT = std::remove_cv_t<T>;
    AiCoreLogF(ctx, "tensor data, range=[%ld, %ld)\n", begin, end);
    for (int64_t i = begin; i < end; ++i) {
        if constexpr (std::is_integral_v<ElemT>) {
            AiCoreLogF(ctx, "%lld\n", data[i]);
        } else if constexpr (std::is_floating_point_v<ElemT>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_pointer_v<ElemT>) {
            AiCoreLogF(ctx, "%p\n", data[i]);
#if IS_AICORE
        } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_same_v<ElemT, half>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
#endif
        }
    }
}

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext *ctx, __gm__ const T *data, int64_t end, int64_t begin = 0)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext *ctx, __ubuf__ const T *data, int64_t end, int64_t begin = 0)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin);
}
#endif
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
 * \file mock_types.h
 * \brief
 */

#ifndef __MOCK_TYPES_H__
#define __MOCK_TYPES_H__

#include <iostream>
#include <cstdint>


#define __ubuf__
#define __gm__
#define __aicore__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define TILEOP static __attribute__((always_inline))
#define INLINE static __attribute__((always_inline))

typedef enum {
#if (defined __DAV_N350__)
    VAReg0 = 0,
    VAReg1,
    VAReg2,
    VAReg3,
#else
    VA0 = 0,
    VA1,
    VA2,
    VA3,
    VA4,
    VA5,
    VA6,
    VA7,
#endif
} ub_addr8_t;

typedef enum {
  NoQuant= 0,
#if (defined __DAV_N350__)
  VREQ8 = 2,
  REQ8 = 3,
  VDEQF16 = 4,
  DEQF16 = 5,
  VREQ16 = 6,
  REQ16 = 7,
#else
  F322F16 = 1,
  VQF322HIF8_PRE = 2,
  QF322HIF8_PRE = 3,
  VQF322HIF8_PRE_HYBRID = 4,
  QF322HIF8_PRE_HYBRID = 5,
#if __CCE_AICORE__ < 300  // Available for V100 V200 V210 V220
  AttachF16Mul = 6,
#else
  VDEQS32_INT = 6,
  DEQS32_INT = 7,
#endif // __CCE_AICORE__ < 300
  VREQ8 = 8,
  REQ8 = 9,
  VDEQF16 = 10,
  DEQF16 = 11,
#if (defined __DAV_C310__)
  VQF322FP8_PRE = 12,
  QF322FP8_PRE = 13,
  VQF322F32_PRE = 14,
  QF322F32_PRE = 15,
#else 
  VSHIFTS322S16 = 12,
  SHIFTS322S16 = 13,
#endif // if (defined __DAV_C310__)
  F322BF16 = 16,
  VQF162B8_PRE = 17,
  QF162B8_PRE = 18,
  VQF162S4_PRE = 19,
  QF162S4_PRE = 20,
  VREQ4 = 21,
  REQ4 = 22,
  VQF322B8_PRE = 23,
  QF322B8_PRE = 24,
  VQF322S4_PRE = 25,
  QF322S4_PRE = 26,
  VDEQS16 = 27,
  DEQS16 = 28,
  VQF162S16_PRE = 29,
  QF162S16_PRE = 30,
  VQF322F16_PRE = 31,
  QF322F16_PRE = 32,
  VQF322BF16_PRE = 33,
  QF322BF16_PRE = 34,
  VQS322BF16_PRE = 35,
  QS322BF16_PRE = 36,
#endif // if (defined __DAV_N350__)
} QuantMode_t;

typedef enum {
// PAD MODE is only valid for OUT->L1 data path
  PAD_NONE  = 0,
  PAD_MODE1 = 1,
  PAD_MODE2 = 2,
  PAD_MODE3 = 3,
  PAD_MODE4 = 4,
  PAD_MODE5 = 5,
  PAD_MODE6 = 6,
  PAD_MODE7 = 7,
  PAD_MODE8 = 8,
} pad_t;


// hardware pipeline
enum PipeType {
    PIPE_S = 0, // Scalar Pipe
    PIPE_V,     // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
    PIPE_M,     // Matrix Pipe, including{}
    PIPE_MTE1,  // L1->L0{A,B}
    PIPE_MTE2,  // OUT ->{L1, L0{A,B}, UB}
    PIPE_MTE3,  // UB ->{OUT,L1}
    PIPE_ALL,
    PIPE_MTE4 = 7, // MOV_UB_TO_OUT
    PIPE_MTE5 = 8, // MOV_OUT_TO_UB
    PIPE_V2 = 9,   // Lower priority vector pipe,
    PIPE_FIX = 10, // {L0C} ->{L1,UB,L1UB}
};

typedef enum {
  inc = 0,
  dec = 1,
} addr_cal_mode_t;

enum {
    EVENT_ID0 = 0,
    EVENT_ID1 = 1,
    EVENT_ID2 = 2,
    EVENT_ID3 = 3,
    EVENT_ID4 = 4,
    EVENT_ID5 = 5,
    EVENT_ID6 = 6,
    EVENT_ID7 = 7,
    EVENT_ID8 = 8,
    EVENT_ID9 = 9,
};

typedef enum {
   // Special for dual output fix pipe instruction.
// only valid for L0C->{L1,UB,L1UB} data path
  DUAL_MODE0  = 0,     // dual output mode is disabled.
  DUAL_MODE1  = 1,
  DUAL_MODE2  = 2,     // only valid for DST={L1}
  DUAL_MODE3  = 3,     // only valid for DST={L1}
  DUAL_MODE4  = 4,     // only valid for DST={L1}
} DualMode_t;

typedef enum {
  // For VCMAX and VCMIN
    VALUE_INDEX = 0,
    INDEX_VALUE = 1,
    ONLY_VALUE = 2,
    ONLY_INDEX = 3,
} Order_t;

typedef enum {
  // the byte mode enable bit for UB->OUT
  BM_DISABLE = 0,
  BM_ENABLE = 1,
} bm_t;

class half
{
private:
    short value_{0};

public:
    half(const double& value) : value_(value) {}

    half& operator=(const half& other)
    {
        value_ = other.value_;
        return *this; 
    }

    half& operator=(const double& other)
    {
        value_ = other;
        return *this; 
    }

    friend std::ostream& operator<<(std::ostream& out, const half& c)
    {
        out << c.value_;
        return out;
    }
};

class bfloat16_t
{
private:
    short value_{0};

public:
    bfloat16_t(const double& value) : value_(value) {}

    bfloat16_t& operator=(const bfloat16_t& other)
    {
        value_ = other.value_;
        return *this; 
    }

    bfloat16_t& operator=(const double& other)
    {
        value_ = other;
        return *this; 
    }

    friend std::ostream& operator<<(std::ostream& out, const bfloat16_t& c)
    {
        out << c.value_;
        return out;
    }
};


#endif

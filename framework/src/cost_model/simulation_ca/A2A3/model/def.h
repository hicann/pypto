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
 * \file def.h
 * \brief
 */

#ifndef __COST_MODEL_DEF__
#define __COST_MODEL_DEF__

#include <vector>
#include <functional>
#include <bitset>
#include <map>
#include <memory>
#include <deque>
#include "cost_model/simulation/common/CommonType.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {

enum class PipeId {
    S,
    V,
    M,
    MTE1,
    MTE2,
    MTE3,
    ALL,
    MTE4,
    MTE5,
    V2,
    F,
    UNDEF
};

const std::vector<std::string> PIPE_ID_NAME_TAB = {
    "SCALAR", "VEC", "CUBE", "MTE1", "MTE2", "MTE3", "ALL", "MTE4", "MTE5", "VEC2","FIXP", "UNDEF"
};
constexpr uint32_t PIPE_NUM = static_cast<uint32_t>(PipeId::UNDEF);
constexpr uint32_t MAX_EVENT_ID_NUM = 8;
constexpr uint32_t BLK_SIZE = 32;

enum class SprId {
    MASK0,
    MASK1,
    FPC,
    FMATRIX,
    NDPARA,
    NUM
};

enum class InstrName {
    // mte2
    NDNZ_OUT_L1,
    MOV_OUT_UB,
    MOV_OUT_UB_PAD,
    // mte3
    MOV_UB_OUT,
    MOV_UB_OUT_PAD,
    // mte1
    LOAD_L1_L0A_2D,
    LOAD_L1_L0B_2D,
    LOAD_L1_L0A_3D,
    LOAD_L1_L0B_3D,
    // fixp
    FIX_L0C_OUT,
    // vector
    VADD,
    VSEL,
    VBITSORT,
    VMRGSORT4,
    VSUB,
    VMUL,
    VDIV,
    VMIN,
    VCOPY,
    VEXP,
    VLN,
    VSQRT,
    VRSQRT,
    VMAX,
    VCADD,
    MOVEV,
    MOVEMASK,
    VCMAX,
    VCMIN,
    VADDS,
    VMULS,
    VMINS,
    VMAXS,
    VCGADD,
    VCGMAX,
    VCGMIN,
    VREC,
    VCONV,
    MOV_UB_UB,
    MOVEVA,
    VNCHWCONV,
    VREDUCEV2,
    VBRCB,
    VABS,
    // cube
    MMAD,
    // scalar
    SET_FLAG,
    WAIT_FLAG,
    DCCI,
    BAR,
    LD,
    ST,
    ALU,

    UNDEF
};

const std::map<uint32_t, std::vector<uint32_t>> VEC_VALU_TAB = {
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VADD), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VADD), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VADD), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VADD), { 256, 1, 6 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VSEL), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VSEL), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VSEL), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VSEL), { 256, 1, 6 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VBITSORT), { 128, 1, 18 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VBITSORT), { 64, 1, 18 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMRGSORT4), { 128, 1, 18 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMRGSORT4), { 64, 1, 18 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VSUB), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VSUB), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VSUB), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VSUB), { 256, 1, 6 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMUL), { 128, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMUL), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMUL), { 256, 1, 7 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMUL), { 256, 1, 7 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VDIV), { 64, 1, 13 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VDIV), { 128, 1, 13 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VEXP), { 64, 1, 12 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VEXP), { 128, 1, 12 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VLN), { 64, 1, 14 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VLN), { 128, 1, 14 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VSQRT), { 64, 1, 13 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VSQRT), { 128, 1, 13 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VRSQRT), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VRSQRT), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VABS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VABS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMAX), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMAX), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMAX), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMAX), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMIN), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMIN), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMIN), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMIN), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCADD), { 256, 1, 23 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCADD), { 256, 1, 20 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::MOVEV), { 256, 1, 1 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::MOVEV), { 256, 1, 1 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCMAX), { 256, 1, 9 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCMAX), { 256, 1, 8 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VCMAX), { 256, 1, 10 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCMIN), { 256, 1, 9 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCMIN), { 256, 1, 8 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VCMIN), { 256, 1, 10 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VADDS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VADDS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VADDS), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VADDS), { 256, 1, 6 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMULS), { 128, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMULS), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMULS), { 256, 1, 7 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMULS), { 256, 1, 7 } },

    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VREC), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VREC), { 256, 1, 5 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMINS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMINS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMINS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMINS), { 256, 1, 4 } },

    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VMAXS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VMAXS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VMAXS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VMAXS), { 256, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VREDUCEV2), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VREDUCEV2), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VREDUCEV2), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VREDUCEV2), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCOPY), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCOPY), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VBRCB), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VBRCB), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT16) << 24 | static_cast<uint32_t>(InstrName::VBRCB), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_INT32) << 24 | static_cast<uint32_t>(InstrName::VBRCB), { 128, 1, 4 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCGADD), { 256, 1, 15 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCGADD), { 256, 1, 12 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCGMAX), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCGMAX), { 256, 1, 5 } },
    { static_cast<uint32_t>(DataType::DT_FP16) << 24 | static_cast<uint32_t>(InstrName::VCGMIN), { 256, 1, 6 } },
    { static_cast<uint32_t>(DataType::DT_FP32) << 24 | static_cast<uint32_t>(InstrName::VCGMIN), { 256, 1, 5 } },
};

const std::map<uint32_t, uint32_t> CUBE_PE_TAB = { // c << 16 | a << 8 | b
    { static_cast<uint32_t>(DataType::DT_INT32) << 16 | static_cast<uint32_t>(DataType::DT_INT8) << 8 | static_cast<uint32_t>(DataType::DT_INT8), 32 }, // s8
    { static_cast<uint32_t>(DataType::DT_FP32) << 16 | static_cast<uint32_t>(DataType::DT_FP16) << 8 | static_cast<uint32_t>(DataType::DT_FP16), 16 }, // f162f32
    { static_cast<uint32_t>(DataType::DT_FP32) << 16 | static_cast<uint32_t>(DataType::DT_BF16) << 8 | static_cast<uint32_t>(DataType::DT_BF16), 16 }, // bf162f32
    { static_cast<uint32_t>(DataType::DT_FP32) << 16 | static_cast<uint32_t>(DataType::DT_FP32) << 8 | static_cast<uint32_t>(DataType::DT_FP32), 4 }, // f322f32
    { static_cast<uint32_t>(DataType::DT_INT32) << 16 | static_cast<uint32_t>(DataType::DT_INT4) << 8 | static_cast<uint32_t>(DataType::DT_INT4), 64 }, // s4
};

const std::vector<std::string> INSTR_NAME_TAB = {
    // mte2
    "NDNZ_OUT_L1",
    "MOV_OUT_UB",
    // mte3
    "MOV_UB_OUT",
    // mte1
    "LOAD_L1_L0A_2D",
    "LOAD_L1_L0B_2D",
    "LOAD_L1_L0A_3D",
    "LOAD_L1_L0B_3D",
    // fixp
    "FIX_L0C_OUT",
    // vector
    "VADD",
    "VSUB",
    "VMUL",
    "VDIV",
    "VMIN",
    "VCOPY",
    "VEXP",
    "VLN",
    "VSQRT",
    "VRSQRT",
    "VMAX",
    "VCADD",
    "MOVEV",
    "MOVEMASK",
    "VCMAX",
    "VCMIN",
    "VADDS",
    "VMULS",
    "VMINS",
    "VMAXS",
    "VCGADD",
    "VCGMAX",
    "VCGMIN",
    "VREC",
    "VCONV",
    "MOV_UB_UB",
    "MOVEVA",
    "VNCHWCONV",
    "VREDUCEV2",
    "VBRCB",
    "VABS",
    // cube
    "MMAD",
    // scalar
    "SET_FLAG",
    "WAIT_FLAG",
    "BAR",
    "LD",
    "ST",
    "MOV_SPR",
    "UNDEF"
};

enum class NdNzParam {
    TYPE,
    ND_NUM,
    N,
    D,
    SRC_ND_MTX_STD,
    SRC_D,
    DST_NZ_C0_STD,
    DST_NZ_N_STD,
    DST_NZ_MTX_STD,
    NUM
};

enum class MovParam {
    BURST_NUM,
    BURST_LEN,
    SRC_STD,
    DST_STD,
    NUM
};

enum class MovPadParam {
    TYPE,
    BURST_NUM,
    BURST_LEN,
    LPAD,
    RPAD,
    SRC_STD,
    DST_STD,
    NUM
};

enum class FixpParam {
    TYPE,
    N,
    M,
    SRC_STD,
    DST_STD,
    ND_NUM,
    NUM
};

enum class VecOp0Param {
    TYPE,
    REPEAT,
    NUM
};

enum class VecOp1Param {
    TYPE,
    REPEAT,
    MODE,
    NUM
};

enum class ConvParam {
    SRC_TYPE,
    DST_TYPE,
    REPEAT,
    NUM
};

enum class Load3dParam {
    TYPE,
    M,
    K,
    NUM
};

enum class MmadParam {
    CTYPE,
    ATYPE,
    BTYPE,
    M,
    N,
    K,
    NUM
};

enum class SetFlagParam {
    PIPE,
    TRIGGER_PIPE,
    EVENT_ID,
    NUM
};

enum class BarParam {
    PIPE,
    NUM
};

struct InstrParam {
    uint64_t popTime = 0;
    uint64_t exeTime = 0;
    uint64_t endTime = 0;
    InstrName name = InstrName::UNDEF;
    PipeId pipe = PipeId::UNDEF;
    std::vector<uint32_t> param;
    uint32_t id = 0;
    int seqNo = 0;
};
using PInstrParam  = std::shared_ptr<InstrParam>;

inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(a) + b - 1) / b);
}

using namespace std;

// 查找指定括号对之间的内容
inline string FindContentBetween(const string& str, char open, char close, size_t& startPos) {
    size_t openPos = str.find(open, startPos);
    if (openPos == string::npos) {
        return "";
    }
    size_t closePos = str.find(close, openPos + 1);
    if (closePos == string::npos) {
        return "";
    }
    startPos = closePos + 1;
    return str.substr(openPos + 1, closePos - openPos - 1);
}

// 解析函数字符串
inline void ParseFunction(const string& funcStr, string& funcName, string& templateContent, string& paramContent) {
    size_t pos = 0;

    // 查找函数名
    size_t openAnglePos = funcStr.find('<');
    size_t openParenPos = funcStr.find('(');
    if (openAnglePos != string::npos && openAnglePos < openParenPos) {
        funcName = funcStr.substr(0, openAnglePos);
        pos = openAnglePos;
    } else if (openParenPos != string::npos) {
        funcName = funcStr.substr(0, openParenPos);
        pos = openParenPos;
    }

    // 查找 < > 中的内容
    templateContent = FindContentBetween(funcStr, '<', '>', pos);

    // 查找 ( ) 中的内容
    paramContent = FindContentBetween(funcStr, '(', ')', pos);
}
// 去除字符串首尾空格的函数
inline string Trim(const string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}
// 分割字符串的函数
inline vector<string> SplitString(const string& input, char delimiter = ',') {
    if (input.empty()) {
        return vector<string>();
    }
    vector<string> result;
    string token;
    size_t start = 0;
    size_t end = input.find(delimiter);
    while (end != string::npos) {
        token = input.substr(start, end - start);
        // 去除分割后子字符串的首尾空格
        token = Trim(token);
        result.push_back(token);
        start = end + 1;
        end = input.find(delimiter, start);
    }
    // 处理最后一个子字符串
    token = input.substr(start);
    token = Trim(token);
    result.push_back(token);
    return result;
}

inline uint32_t GetDataType(string dataType)
{
    DataType ret = DataType::DT_FP32;
    if (dataType == "int4") {
        ret = DataType::DT_INT4;
    } else if (dataType == "int8_t") {
        ret = DataType::DT_INT8;
    } else if (dataType == "int16_t") {
        ret = DataType::DT_INT16;
    } else if (dataType == "int32_t") {
        ret = DataType::DT_INT32;
    } else if (dataType == "float8_t") {
        ret = DataType::DT_FP8;
    } else if (dataType == "half") {
        ret = DataType::DT_FP16;
    } else if (dataType == "float") {
        ret = DataType::DT_FP32;
    } else if (dataType == "bfloat16_t") {
        ret = DataType::DT_BF16;
    } else if (dataType == "hifloat8_t") {
        ret = DataType::DT_HF8;
    } else if (dataType == "hfloat4") {
        ret = DataType::DT_HF4;
    }
    return static_cast<uint32_t>(ret);
}

inline uint32_t GetParam(string param)
{
    try {
        unsigned long num = std::stoul(param);
        return static_cast<uint32_t>(num);
    } catch (const std::invalid_argument& e) {
        SIMULATION_LOGE("invalid parameter: %s", e.what());
    } catch (const std::out_of_range& e) {
        SIMULATION_LOGE("out of range: %s", e.what());
    }
    return 0;
}

inline uint64_t GetLongParam(string param)
{
    try {
        uint64_t num = std::stoull(param);
        return static_cast<uint64_t>(num);
    } catch (const std::invalid_argument& e) {
        SIMULATION_LOGE("invalid parameter: %s", e.what());
    } catch (const std::out_of_range& e) {
        SIMULATION_LOGE("out of range: %s", e.what());
    }
    return 0;
}

inline PInstrParam  ScalarInstr(InstrName name, PipeId pipe)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = pipe;
    return instr;
    // pipe_barrier, set_mask_count, set_mask_norm, set_flag, wait_flag
}

inline PInstrParam  SetSpr(InstrName name, PipeId pipe, SprId spr, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = pipe;
    uint64_t sprVal = GetLongParam(params[0]);
    uint32_t sprValLo = sprVal & 0xffffffff;
    uint32_t sprValHi = (sprVal >> 32) & 0xffffffff;
    if (spr == SprId::MASK1) {
        uint32_t maskValLo = sprVal & 0xffffffff;
        uint32_t maskValHi = (sprVal >> 32) & 0xffffffff;
        instr->param = { static_cast<uint32_t>(spr), sprValLo, sprValHi, maskValLo, maskValHi };
    } else {
        instr->param = { static_cast<uint32_t>(spr), sprValLo, sprValHi };
    }

    return instr;
    // set_vector_mask, set_fmatrix set_nd_para set_fpc
}

inline PInstrParam  VecTemplate0(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::V;
    instr->param = { GetDataType(templates[0]), GetParam(params[3]) };
    return instr;
    // vadd vsub vmul vdiv vmin vmax
    // vadds vmuls vmins vmaxs
}

inline PInstrParam  VecTemplate1(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::V;
    instr->param = { GetDataType(templates[0]), GetParam(params[2]) };
    return instr;
    // vcopy vexp vln vsqrt vrsqrt vector_dup vrec vbrcb
    // vcgadd vcgmax vcgmin
}

inline PInstrParam  VecTemplate2(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::V;
    instr->param = { GetDataType(templates[0]), GetParam(params[2]), GetParam(params[6]) };
    return instr;
    // vcadd vcmax vcmin
    // vreducev2
}

inline PInstrParam  VecReduceV2(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::V;
    instr->param = { GetDataType(templates[0]), GetParam(params[3]), 1 };
    return instr;
    // vcadd vcmax vcmin
    // vreducev2
}

inline PInstrParam  VecVnchwconv(DataType type, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = InstrName::VNCHWCONV;
    instr->pipe = PipeId::V;
    instr->param = { type, GetParam(params[4]), };
    return instr;
    // vnchwconv
}

inline PInstrParam  VecVconv(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::V;
    instr->param = { GetDataType(templates[1]), GetDataType(templates[0]), GetParam(params[2]) };
    return instr;
    // vconv
}

inline PInstrParam  MteDma(InstrName name, PipeId pipe, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = pipe;
    instr->param = { GetParam(params[3]) /* nBurst */, GetParam(params[4]) /* lenBurst */,
        GetParam(params[5]) /* srcStride */, GetParam(params[6]) /* dstStride */ };
    return instr;
    // copy_ubuf_to_ubuf copy_ubuf_to_gm copy_gm_to_ubuf copy_cbuf_to_gm copy_gm_to_cbuf
}

inline PInstrParam  MteMovAlign(InstrName name, PipeId pipe, DataType type, const std::vector<std::string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = pipe;
    instr->param = { static_cast<uint32_t>(type) /* type */, GetParam(params[3]) /* nBurst */, GetParam(params[4]) /* lenBurst */,
        GetParam(params[5]) /* lpad */, GetParam(params[6]) /* rpad */,
        GetParam(params[7]) /* srcStride */, GetParam(params[8]) /* dstStride */ };
    return instr;
    // copy_gm_to_ubuf_align_b8 copy_gm_to_ubuf_align_b16 copy_gm_to_ubuf_align_b32
    // copy_ubuf_to_gm_align_b8 copy_ubuf_to_gm_align_b16 copy_ubuf_to_gm_align_b32
}

inline PInstrParam  Nd2Nz(DataType type, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = InstrName::NDNZ_OUT_L1;
    instr->pipe = PipeId::MTE2;
    instr->param = { static_cast<uint32_t>(type) /* type */, GetParam(params[3]) /* ndNum */, GetParam(params[4]) /* nValue */,
        GetParam(params[5]) /* dValue */, GetParam(params[6]) /* srcNdMatrixStride */,
        GetParam(params[7]) /* srcDValue */,
        GetParam(params[8]) /* dstNzC0Stride */, GetParam(params[9]) /* dstNzNStride */,
        GetParam(params[10]) /* dstNzMatrixStride */ };
    return instr;
    // copy_gm_to_cbuf_multi_nd2nz_b16 copy_gm_to_cbuf_multi_nd2nz_b32s
}

inline PInstrParam  Img2Col(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::MTE1;
    instr->param = { GetDataType(templates[0]) /* type */, GetParam(params[3]) /* M */, GetParam(params[2])/* K */ };
    return instr;
    // img2colv2_cbuf_to_ca img2colv2_cbuf_to_cb
}

inline PInstrParam  Load(InstrName name, const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = name;
    instr->pipe = PipeId::MTE1;
    instr->param = { GetDataType(templates[0]) /* type */, GetParam(params[3]) /* repeat */ };
    return instr;
    // load_cbuf_to_ca load_cbuf_to_cb
}

inline PInstrParam  Mmad(const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = InstrName::MMAD;
    instr->pipe = PipeId::M;
    instr->param = { GetDataType(templates[0]) /* ctype */, GetDataType(templates[1]) /* atype */,
        GetDataType(templates[2]) /* btype */, GetParam(params[3]) /* m */, GetParam(params[5]) /* n */,
        GetParam(params[4]) /* k */ };
    return instr;
    // mad
}

inline PInstrParam  FixL0cOut(const vector<string> templates, const vector<string> params)
{
    auto instr = make_shared<InstrParam>();
    instr->name = InstrName::FIX_L0C_OUT;
    instr->pipe = PipeId::F;
    instr->param = { GetDataType(templates[0]) /* type */, GetParam(params[3]) /* n */,
        GetParam(params[4]) /* m */, GetParam(params[6]) /* src_std */, GetParam(params[5]) /* dst_std */,
        1 /* nd_num */ };
    return instr;
    // copy_matrix_cc_to_gm
}

inline deque<PInstrParam> GetProgram(vector<string> program)
{
    deque<PInstrParam> ret;
    for (auto& funcStr : program) {
        if (funcStr.empty()) {
            continue;
        }
        string funcName, templateContent, paramContent;
        ParseFunction(funcStr, funcName, templateContent, paramContent);
        vector<string> templates = SplitString(templateContent);
        vector<string> params = SplitString(paramContent);
        if (funcName == "set_vector_mask") {
            ret.push_back(SetSpr(InstrName::MOVEMASK, PipeId::V, SprId::MASK1, params));
        } else if (funcName == "pipe_barrier") {
            ret.push_back(ScalarInstr(InstrName::BAR, PipeId::V));
        } else if (funcName == "set_mask_count" || funcName == "set_mask_norm" || funcName == "set_va_reg_sb" || funcName == "set_deqscale") {
            ret.push_back(ScalarInstr(InstrName::ALU, PipeId::S));
        } else if (funcName == "set_flag") {
            ret.push_back(ScalarInstr(InstrName::SET_FLAG, PipeId::S));
        } else if (funcName == "wait_flag") {
            ret.push_back(ScalarInstr(InstrName::WAIT_FLAG, PipeId::S));
        } else if (funcName == "dcci") {
            ret.push_back(ScalarInstr(InstrName::DCCI, PipeId::S));
        } else if (funcName == "set_fmatrix") {
            ret.push_back(SetSpr(InstrName::ALU, PipeId::MTE1, SprId::FMATRIX, params));
        } else if (funcName == "set_fmatrix_b") {
            ret.push_back(SetSpr(InstrName::ALU, PipeId::MTE1, SprId::FMATRIX, params));
        } else if (funcName == "set_nd_para") {
            ret.push_back(SetSpr(InstrName::ALU, PipeId::F, SprId::NDPARA, params));
        } else if (funcName == "set_fpc") {
            ret.push_back(SetSpr(InstrName::ALU, PipeId::F, SprId::FPC, params));
        } else if (funcName == "vadd") {
            ret.push_back(VecTemplate0(InstrName::VADD, templates, params));
        } else if (funcName == "vsel") {
            ret.push_back(VecTemplate0(InstrName::VSEL, templates, params));
        } else if (funcName == "vsub") {
            ret.push_back(VecTemplate0(InstrName::VSUB, templates, params));
        } else if (funcName == "vmul") {
            ret.push_back(VecTemplate0(InstrName::VMUL, templates, params));
        } else if (funcName == "vdiv") {
            ret.push_back(VecTemplate0(InstrName::VDIV, templates, params));
        } else if (funcName == "vmin") {
            ret.push_back(VecTemplate0(InstrName::VMIN, templates, params));
        } else if (funcName == "vmax") {
            ret.push_back(VecTemplate0(InstrName::VMAX, templates, params));
        } else if (funcName == "vadds") {
            ret.push_back(VecTemplate0(InstrName::VADDS, templates, params));
        } else if (funcName == "vmuls") {
            ret.push_back(VecTemplate0(InstrName::VMULS, templates, params));
        } else if (funcName == "vmins") {
            ret.push_back(VecTemplate0(InstrName::VMINS, templates, params));
        } else if (funcName == "vmaxs") {
            ret.push_back(VecTemplate0(InstrName::VMAXS, templates, params));
        } else if (funcName == "vbitsort") {
            ret.push_back(VecTemplate0(InstrName::VBITSORT, templates, params));
        } else if (funcName == "vmrgsort4") {
            ret.push_back(VecTemplate0(InstrName::VMRGSORT4, templates, params));
        } else if (funcName == "vreducev2") {
            ret.push_back(VecReduceV2(InstrName::VREDUCEV2, templates, params));
        } else if (funcName == "vcopy") {
            ret.push_back(VecTemplate1(InstrName::VCOPY, templates, params));
        } else if (funcName == "vabs") {
            ret.push_back(VecTemplate1(InstrName::VABS, templates, params));
        } else if (funcName == "vexp") {
            ret.push_back(VecTemplate1(InstrName::VEXP, templates, params));
        } else if (funcName == "vln") {
            ret.push_back(VecTemplate1(InstrName::VLN, templates, params));
        } else if (funcName == "vsqrt") {
            ret.push_back(VecTemplate1(InstrName::VSQRT, templates, params));
        } else if (funcName == "vrsqrt") {
            ret.push_back(VecTemplate1(InstrName::VRSQRT, templates, params));
        } else if (funcName == "vector_dup") {
            ret.push_back(VecTemplate1(InstrName::MOVEV, templates, params));
        } else if (funcName == "vrec") {
            ret.push_back(VecTemplate1(InstrName::VREC, templates, params));
        } else if (funcName == "vbrcb") {
            ret.push_back(VecTemplate1(InstrName::VBRCB, templates, params));
        } else if (funcName == "vcgadd") {
            ret.push_back(VecTemplate1(InstrName::VCGADD, templates, params));
        } else if (funcName == "vcgmax") {
            ret.push_back(VecTemplate1(InstrName::VCGMAX, templates, params));
        } else if (funcName == "vcgmin") {
            ret.push_back(VecTemplate1(InstrName::VCGMIN, templates, params));
        } else if (funcName == "vcadd") {
            ret.push_back(VecTemplate2(InstrName::VCADD, templates, params));
        } else if (funcName == "vcmax") {
            ret.push_back(VecTemplate2(InstrName::VCMAX, templates, params));
        } else if (funcName == "vcmin") {
            ret.push_back(VecTemplate2(InstrName::VCMIN, templates, params));
        } else if (funcName.find("vconv") != funcName.npos){
            ret.push_back(VecVconv(InstrName::VCONV, templates, params));
        } else if (funcName == "scatter_vnchwconv_b8") {
            ret.push_back(VecVnchwconv(DataType::DT_INT8, params));
        } else if (funcName == "scatter_vnchwconv_b16") {
            ret.push_back(VecVnchwconv(DataType::DT_INT16, params));
        } else if (funcName == "scatter_vnchwconv_b32") {
            ret.push_back(VecVnchwconv(DataType::DT_INT32, params));
        } else if (funcName == "copy_ubuf_to_ubuf") {
            ret.push_back(MteDma(InstrName::MOV_UB_UB, PipeId::V, params));
        } else if (funcName == "copy_gm_to_ubuf" || funcName == "copy_gm_to_cbuf") {
            ret.push_back(MteDma(InstrName::MOV_OUT_UB, PipeId::MTE2, params));
        } else if (funcName == "copy_ubuf_to_gm" || funcName == "copy_cbuf_to_gm") {
            ret.push_back(MteDma(InstrName::MOV_UB_OUT, PipeId::MTE3, params));
        } else if (funcName == "copy_gm_to_ubuf_align_b8") {
            ret.push_back(MteMovAlign(InstrName::MOV_OUT_UB_PAD, PipeId::MTE2, DataType::DT_INT8, params));
        } else if (funcName == "copy_gm_to_ubuf_align_b16") {
            ret.push_back(MteMovAlign(InstrName::MOV_OUT_UB_PAD, PipeId::MTE2, DataType::DT_INT16, params));
        } else if (funcName == "copy_gm_to_ubuf_align_b32") {
            ret.push_back(MteMovAlign(InstrName::MOV_OUT_UB_PAD, PipeId::MTE2, DataType::DT_INT32, params));
        } else if (funcName == "copy_ubuf_to_gm_align_b8") {
            ret.push_back(MteMovAlign(InstrName::MOV_UB_OUT_PAD, PipeId::MTE3, DataType::DT_INT8, params));
        } else if (funcName == "copy_ubuf_to_gm_align_b16") {
            ret.push_back(MteMovAlign(InstrName::MOV_UB_OUT_PAD, PipeId::MTE3, DataType::DT_INT16, params));
        } else if (funcName == "copy_ubuf_to_gm_align_b32") {
            ret.push_back(MteMovAlign(InstrName::MOV_UB_OUT_PAD, PipeId::MTE3, DataType::DT_INT32, params));
        } else if (funcName == "copy_gm_to_cbuf_multi_nd2nz_b8") {
            ret.push_back(Nd2Nz(DataType::DT_INT8, params));
        } else if (funcName == "copy_gm_to_cbuf_multi_nd2nz_b16") {
            ret.push_back(Nd2Nz(DataType::DT_INT16, params));
        } else if (funcName == "copy_gm_to_cbuf_multi_nd2nz_b32s") {
            ret.push_back(Nd2Nz(DataType::DT_INT32, params));
        } else if (funcName == "img2colv2_cbuf_to_ca") {
            ret.push_back(Img2Col(InstrName::LOAD_L1_L0A_3D, templates, params));
        } else if (funcName == "img2colv2_cbuf_to_cb") {
            ret.push_back(Img2Col(InstrName::LOAD_L1_L0B_3D, templates, params));
        } else if (funcName == "load_cbuf_to_ca") {
            ret.push_back(Load(InstrName::LOAD_L1_L0A_2D, templates, params));
        } else if (funcName == "load_cbuf_to_cb" || funcName == "load_cbuf_to_cb_transpose") {
            ret.push_back(Load(InstrName::LOAD_L1_L0B_2D, templates, params));
        } else if (funcName == "mad") {
            ret.push_back(Mmad(templates, params));
        } else if (funcName == "copy_matrix_cc_to_gm") {
            ret.push_back(FixL0cOut(templates, params));
        } else {
            cout << funcStr << " not support" << endl;
        }
    }
    return ret;
}

}  // namespace CostModel

#endif

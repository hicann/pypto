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
 * \file data_type.h
 * \brief
 */

#pragma once

#include <string>
#include <stdexcept>
#include <unordered_map>

#ifndef UNUSED
#define UNUSED(n)       (void)(n)
#endif

namespace npu::tile_fwk {
const std::unordered_map<size_t, size_t> BLOCK_PADDING_DIM = {
    {1, 32},
    {2, 16},
    {4,  8},
    {8,  4}
};

enum DataType {
    DT_INT4 = 0,
    DT_INT8 = 1,
    DT_INT16 = 2,
    DT_INT32 = 3,
    DT_INT64 = 4,
    DT_FP8 = 5,
    DT_FP16 = 6,
    DT_FP32 = 7,
    DT_BF16 = 8,
    DT_HF4 = 9,
    DT_HF8 = 10,
    DT_UINT8 = 11,
    DT_UINT16 = 12,
    DT_UINT32 = 13,
    DT_UINT64 = 14,
    DT_BOOL = 15,
    DT_DOUBLE = 16,
    DT_BOTTOM
};

enum class NodeType {
    LOCAL = 0,
    INCAST = 1,
    OUTCAST = 2,
};

enum class TileOpFormat {
    TILEOP_ND = 0,
    TILEOP_NZ = 1,
    TILEOP_FORMAT_NUM [[maybe_unused]],
};

enum MemoryType {
    MEM_UB = 0,
    MEM_L1 = 1,
    MEM_L0A = 2,
    MEM_L0B = 3,
    MEM_L0C = 4,
    MEM_FIX = 5,
    MEM_FIX_QUANT_PRE = 6,
    MEM_FIX_RELU_PRE = 7,
    MEM_FIX_RELU_POST = 8,
    MEM_FIX_QUANT_POST = 9,
    MEM_FIX_ELT_ANTIQ = 10,
    MEM_FIX_MTE2_ANTIQ = 11,
    MEM_BT = 12,
    MEM_L2 = 13,
    MEM_L3 = 14,
    MEM_DEVICE_DDR = 15,
    MEM_HOST1 = 16,
    MEM_FAR1 = 17,
    MEM_FAR2 = 18,
    MEM_WORKSPACE = 19,
    MEM_VECTOR_REG = 20,
    MEM_UNKNOWN
};

enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

inline std::string NodeType2String(NodeType n) {
    switch (n) {
        case NodeType::LOCAL: return "LOCAL";
        case NodeType::INCAST: return "INCAST";
        case NodeType::OUTCAST: return "OUTCAST";
        default: throw std::invalid_argument("Unknown NodeType");
    }
}

inline bool IsFloat(DataType t) {
    if ((t == DT_FP8) || (t == DT_FP16) || (t == DT_FP32) || (t == DT_BF16)) {
        return true;
    }
    return false;
}

inline std::string DataType2String(DataType t) {
    switch (t) {
        case DT_INT4: return "DT_INT4";
        case DT_INT8: return "DT_INT8";
        case DT_INT16: return "DT_INT16";
        case DT_INT32: return "DT_INT32";
        case DT_INT64: return "DT_INT64";
        case DT_FP8: return "DT_FP8";
        case DT_FP16: return "DT_FP16";
        case DT_FP32: return "DT_FP32";
        case DT_BF16: return "DT_BF16";
        case DT_HF8: return "DT_HF8";
        case DT_HF4: return "DT_HF4";
        case DT_BOOL: return "DT_BOOL";
        case DT_UINT8: return "DT_UINT8";
        case DT_UINT16: return "DT_UINT16";
        case DT_UINT32: return "DT_UINT32";
        case DT_UINT64: return "DT_UINT64";
        default: throw std::invalid_argument("Unknown DataType");
    }
}

[[maybe_unused]] inline std::string BriefDataType2String(DataType t) {
    switch (t) {
        case DT_INT4: return "INT4";
        case DT_INT8: return "INT8";
        case DT_INT16: return "INT16";
        case DT_INT32: return "INT32";
        case DT_INT64: return "INT64";
        case DT_FP8: return "FP8";
        case DT_FP16: return "FP16";
        case DT_FP32: return "FP32";
        case DT_BF16: return "BF16";
        case DT_HF4: return "HF4";
        case DT_HF8: return "HF8";
        case DT_UINT8: return "UINT8";
        case DT_UINT16: return "UINT16";
        case DT_UINT32: return "UINT32";
        case DT_UINT64: return "UINT64";
        case DT_BOOL: return "BOOL";
        case DT_DOUBLE: return "DOUBLE";
        default: throw std::invalid_argument("Unknown DataType");
    }
}

inline std::string DataType2CCEStr(DataType t) {
    switch (t) {
        case DT_INT4: return "int4";
        case DT_INT8: return "int8_t";
        case DT_INT16: return "int16_t";
        case DT_INT32: return "int32_t";
        case DT_INT64: return "int64_t";
        case DT_FP8: return "float8_t";
        case DT_FP16: return "half";
        case DT_FP32: return "float";
        case DT_BF16: return "bfloat16_t";
        case DT_HF8: return "hfloat8";
        case DT_HF4: return "hfloat4";
        case DT_BOOL: return "bool";
        case DT_UINT8: return "uint8_t";
        case DT_UINT16: return "uint16_t";
        case DT_UINT32: return "uint32_t";
        case DT_UINT64: return "uint64_t";
        default: throw std::invalid_argument("Unknown DataType");
    }
}

inline std::string DataType2VectorRegStr(DataType t) {
    switch (t) {
        case DT_FP16: return "vector_f16";
        case DT_FP32: return "vector_f32";
        default: return "Unknown DataType";
    }
}

const std::unordered_map<std::string, DataType> STR_DATA_TYPE_MAP = {
    {"int4",     DT_INT4},
    {"int8",     DT_INT8},
    {"int16",    DT_INT16},
    {"int32",    DT_INT32},
    {"int",      DT_INT32},
    {"int64",    DT_INT64},
    {"fp8",      DT_FP8},
    {"float8",   DT_FP8},
    {"fp16",     DT_FP16},
    {"float16",  DT_FP16},
    {"float",    DT_FP32},
    {"float32",  DT_FP32},
    {"bf16",     DT_BF16},
    {"bfloat16", DT_BF16},
    {"hf4",      DT_HF4},
    {"hf8",      DT_HF8},
    {"uint8",    DT_UINT8},
    {"uint16",   DT_UINT16},
    {"uint32",   DT_UINT32},
    {"uint64",   DT_UINT64},
    {"bool",     DT_BOOL},
    {"double",   DT_DOUBLE}
};

inline std::string MemoryTypeToString(MemoryType mt) {
    switch (mt) {
        case MEM_UB: return "MEM_UB";
        case MEM_L1: return "MEM_L1";
        case MEM_L0A: return "MEM_L0A";
        case MEM_L0B: return "MEM_L0B";
        case MEM_L0C: return "MEM_L0C";
        case MEM_L2: return "MEM_L2";
        case MEM_L3: return "MEM_L3";
        case MEM_DEVICE_DDR: return "MEM_DEVICE_DDR";
        case MEM_HOST1: return "MEM_HOST1";
        case MEM_FAR1: return "MEM_FAR1";
        case MEM_FAR2: return "MEM_FAR2";
        case MEM_BT: return "MEM_BT";
        case MEM_VECTOR_REG: return "MEM_VECTOR_REG";
        case MEM_FIX: return "MEM_FIX";
        case MEM_FIX_QUANT_PRE: return "MEM_FIX_QUANT_PRE";
        case MEM_FIX_RELU_PRE: return "MEM_FIX_RELU_PRE";
        case MEM_FIX_RELU_POST: return "MEM_FIX_RELU_POST";
        case MEM_FIX_QUANT_POST: return "MEM_FIX_QUANT_POST";
        case MEM_FIX_ELT_ANTIQ: return "MEM_FIX_ELT_ANTIQ";
        case MEM_FIX_MTE2_ANTIQ: return "MEM_FIX_MTE2_ANTIQ";
        default: return "MEM_UNKNOWN";
    }
#undef CASE
}

[[maybe_unused]] inline std::string BriefMemoryTypeToString(MemoryType mt) {
    switch (mt) {
        case MEM_UB: return "UB";
        case MEM_L1: return "L1";
        case MEM_L0A: return "L0A";
        case MEM_L0B: return "L0B";
        case MEM_L0C: return "L0C";
        case MEM_L2: return "L2";
        case MEM_L3: return "L3";
        case MEM_DEVICE_DDR: return "DDR";
        case MEM_HOST1: return "HOST1";
        case MEM_FAR1: return "FAR1";
        case MEM_FAR2: return "FAR2";
        case MEM_BT: return "BT";
        default: return "MEM_UNKNOWN";
    }
}

inline size_t BytesOf(DataType t) {
    switch (t) {
        case DT_INT4:
        case DT_HF4: return 1; // 4bits still need 1 byte
        case DT_INT8:
        case DT_UINT8:
        case DT_BOOL:
        case DT_FP8:
        case DT_HF8: return 1;
        case DT_INT16:
        case DT_UINT16:
        case DT_FP16:
        case DT_BF16: return 0x2;
        case DT_INT32:
        case DT_UINT32:
        case DT_FP32: return 0x4;
        case DT_DOUBLE:
        case DT_UINT64:
        case DT_INT64: return 0x8;
        default: throw std::invalid_argument("Unknown DataType");
    }
}

[[maybe_unused]] inline std::string CastMode2String(CastMode mode) {
    switch (mode) {
        case CastMode::CAST_NONE: return "CAST_NONE";
        case CastMode::CAST_RINT: return "CAST_RINT";
        case CastMode::CAST_ROUND: return "CAST_ROUND";
        case CastMode::CAST_FLOOR: return "CAST_FLOOR";
        case CastMode::CAST_CEIL: return "CAST_CEIL";
        case CastMode::CAST_TRUNC: return "CAST_TRUNC";
        case CastMode::CAST_ODD: return "CAST_ODD";
        default: throw std::invalid_argument("Unknown CastMode");
    }
}

enum class CachePolicy {
    PREFETCH,
    NONE_CACHEABLE,
    MAX_NUM
};
} // namespace npu::tile_fwk

namespace std {
inline std::string to_string(npu::tile_fwk::TileOpFormat format) {
    switch (format) {
        case npu::tile_fwk::TileOpFormat::TILEOP_ND: return "TILEOP_ND";
        case npu::tile_fwk::TileOpFormat::TILEOP_NZ: return "TILEOP_NZ";
        default: return "UNKNOWN";
    }
}
} // namespace std

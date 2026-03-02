/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dlpack_dtype.h
 * \brief DLPack ABI structs and dtype code to DataType conversion.
 * \see https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
 */

#pragma once

#include "tilefwk/data_type.h"

#include <cstdint>

namespace npu::tile_fwk {

/*!
 * \brief DLPack ABI structs, layout-compatible with dlpack.h, no external dependency.
 */
struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct DLManagedTensor {
    struct DLTensor {
        void *data;
        int32_t device_type;
        int32_t device_id;
        int32_t ndim;
        DLDataType dtype;
        int64_t *shape;
        int64_t *strides;
        uint64_t byte_offset;
    } dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *);
};

/*! \brief The type code options DLDataType. */
typedef enum {
    /*! \brief signed integer */
    kDLInt = 0U,
    /*! \brief unsigned integer */
    kDLUInt = 1U,
    /*! \brief IEEE floating point */
    kDLFloat = 2U,
    /*! \brief Opaque handle type, reserved for testing purposes. */
    kDLOpaqueHandle = 3U,
    /*! \brief bfloat16 */
    kDLBfloat = 4U,
    /*! \brief complex number */
    kDLComplex = 5U,
    /*! \brief boolean */
    kDLBool = 6U,
    /*! \brief FP8 data types */
    kDLFloat8_e3m4 = 7U,
    kDLFloat8_e4m3 = 8U,
    kDLFloat8_e4m3b11fnuz = 9U,
    kDLFloat8_e4m3fn = 10U,
    kDLFloat8_e4m3fnuz = 11U,
    kDLFloat8_e5m2 = 12U,
    kDLFloat8_e5m2fnuz = 13U,
    kDLFloat8_e8m0fnu = 14U,
    kDLFloat6_e2m3fn = 15U,
    kDLFloat6_e3m2fn = 16U,
    kDLFloat4_e2m1fn = 17U,
} DLDataTypeCode;

/*!
 * \brief Convert DLPack dtype (code, bits, lanes) to DataType.
 * \param code DLPack type code (DLDataTypeCode)
 * \param bits Number of bits
 * \param lanes Number of lanes (must be 1 for scalar types)
 * \param out Output DataType on success
 * \return true if conversion succeeded, false otherwise
 */
inline bool DlpackDtypeToDataType(uint8_t code, uint8_t bits, uint16_t lanes, DataType *out) {
    if (lanes != 1) return false;
    switch (code) {
        case kDLInt:
            switch (bits) {
                case 8: *out = DT_INT8; return true;
                case 16: *out = DT_INT16; return true;
                case 32: *out = DT_INT32; return true;
                case 64: *out = DT_INT64; return true;
                default: return false;
            }
        case kDLUInt:
            switch (bits) {
                case 8: *out = DT_UINT8; return true;
                case 16: *out = DT_UINT16; return true;
                case 32: *out = DT_UINT32; return true;
                case 64: *out = DT_UINT64; return true;
                default: return false;
            }
        case kDLFloat:
            switch (bits) {
                case 16: *out = DT_FP16; return true;
                case 32: *out = DT_FP32; return true;
                case 64: *out = DT_DOUBLE; return true;
                default: return false;
            }
        case kDLBfloat:
            if (bits == 16) { *out = DT_BF16; return true; }
            return false;
        case kDLBool:
            if (bits == 8) { *out = DT_BOOL; return true; }
            return false;
        case kDLFloat8_e5m2:
            if (bits == 8) { *out = DT_FP8E5M2; return true; }
            return false;
        case kDLFloat8_e4m3:
            if (bits == 8) { *out = DT_FP8E4M3; return true; }
            return false;
        default:
            return false;
    }
}

}  // namespace npu::tile_fwk

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
 * \file enum.cpp
 * \brief
 */

#include "pybind_common.h"
#include "pybind11/native_enum.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindEnum(py::module& m)
{
    // clang-format off
    py::native_enum<DataType>(m, "DataType", "enum.IntEnum")
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_value) \
        .value(#name, DataType::name)
        DATA_TYPE_ALL
#undef DTYPE_DESC
        .value("DT_BOTTOM", DataType::DT_BOTTOM)
        .export_values()
        .finalize();
    // clang-format on

    py::native_enum<NodeType>(m, "NodeType", "enum.IntEnum")
        .value("LOCAL", NodeType::LOCAL)
        .value("INCAST", NodeType::INCAST)
        .value("OUTCAST", NodeType::OUTCAST)
        .finalize();

    py::native_enum<TileOpFormat>(m, "TileOpFormat", "enum.IntEnum")
        .value("TILEOP_ND", TileOpFormat::TILEOP_ND)
        .value("TILEOP_NZ", TileOpFormat::TILEOP_NZ)
        .finalize();

    py::native_enum<CachePolicy>(m, "CachePolicy", "enum.IntEnum")
        .value("NONE_CACHEABLE", CachePolicy::NONE_CACHEABLE)
        .finalize();

    py::native_enum<ReduceMode>(m, "ReduceMode", "enum.IntEnum").value("ATOMIC_ADD", ReduceMode::ATOMIC_ADD).finalize();

    py::native_enum<ScatterMode>(m, "ScatterMode", "enum.IntEnum")
        .value("NONE", ScatterMode::NONE)
        .value("ADD", ScatterMode::ADD)
        .value("MULTIPLY", ScatterMode::MULTIPLY)
        .finalize();

    py::native_enum<FunctionType>(m, "FunctionType", "enum.IntEnum")
        .value("STATIC", FunctionType::STATIC)
        .value("DYNAMIC", FunctionType::DYNAMIC)
        .value("DYNAMIC_LOOP", FunctionType::DYNAMIC_LOOP)
        .finalize();

    py::native_enum<GraphType>(m, "GraphType", "enum.IntEnum")
        .value("TENSOR_GRAPH", GraphType::TENSOR_GRAPH)
        .finalize();

    py::native_enum<CastMode>(m, "CastMode", "enum.IntEnum")
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .finalize();

    py::native_enum<SaturationMode>(m, "SaturationMode", "enum.IntEnum")
        .value("OFF", SaturationMode::OFF)
        .value("ON", SaturationMode::ON)
        .finalize();

    py::native_enum<PrecisionType>(m, "PrecisionType", "enum.IntEnum")
        .value("INTRINSIC", PrecisionType::INTRINSIC)
        .value("HIGH_PRECISION", PrecisionType::HIGH_PRECISION)
        .finalize();

    py::enum_<DequantScaleRoundingMode>(m, "DequantScaleRoundingMode")
        .value("ROUND_UP", DequantScaleRoundingMode::ROUND_UP)
        .value("ROUND_DOWN", DequantScaleRoundingMode::ROUND_DOWN)
        .export_values();

    py::native_enum<TileType>(m, "TileType", "enum.IntEnum")
        .value("VEC", TileType::VEC)
        .value("CUBE", TileType::CUBE)
        .value("DIST", TileType::DIST)
        .value("MAX", TileType::MAX)
        .finalize();

    py::native_enum<OpType>(m, "OpType", "enum.IntEnum")
        .value("EQ", OpType::EQ)
        .value("NE", OpType::NE)
        .value("LT", OpType::LT)
        .value("LE", OpType::LE)
        .value("GT", OpType::GT)
        .value("GE", OpType::GE)
        .finalize();

    py::native_enum<OutType>(m, "OutType", "enum.IntEnum")
        .value("BOOL", OutType::BOOL)
        .value("BIT", OutType::BIT)
        .finalize();

    py::native_enum<TopKAlgo>(m, "TopKAlgo", "enum.IntEnum")
        .value("MERGE_SORT", TopKAlgo::MERGE_SORT)
        .value("RADIX_SELECT", TopKAlgo::RADIX_SELECT)
        .finalize();

    py::native_enum<Matrix::ReLuType>(m, "ReLuType", "enum.IntEnum")
        .value("NO_RELU", Matrix::ReLuType::NoReLu)
        .value("RELU", Matrix::ReLuType::ReLu)
        .finalize();

    py::native_enum<Conv::ReLuType>(m, "ConvReLuType", "enum.IntEnum")
        .value("NO_RELU", Conv::ReLuType::NoReLu)
        .value("RELU", Conv::ReLuType::ReLu)
        .finalize();

    py::native_enum<Matrix::TransMode>(m, "TransMode", "enum.IntEnum")
        .value("CAST_NONE", Matrix::TransMode::CAST_NONE)
        .value("CAST_RINT", Matrix::TransMode::CAST_RINT)
        .value("CAST_ROUND", Matrix::TransMode::CAST_ROUND)
        .finalize();

    py::native_enum<LogBaseType>(m, "LogBaseType", "enum.IntEnum")
        .value("LOG_E", LogBaseType::LOG_E)
        .value("LOG_2", LogBaseType::LOG_2)
        .value("LOG_10", LogBaseType::LOG_10)
        .finalize();

    py::native_enum<Distributed::AtomicType>(m, "AtomicType", "enum.IntEnum")
        .value("SET", Distributed::AtomicType::SET)
        .value("ADD", Distributed::AtomicType::ADD)
        .finalize();
}
} // namespace pypto

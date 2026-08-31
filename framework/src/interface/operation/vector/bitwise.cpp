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
 * \file bitwise.cpp
 * \brief
 */
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"

namespace npu::tile_fwk {

Tensor BitwiseAnd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEAND");
    static const std::unordered_set<DataType> BITWISE_A2A3_TYPES = {DT_INT16, DT_UINT16, DT_INT8,
                                                                    DT_UINT8, DT_INT32,  DT_UINT32};
    static const std::unordered_set<DataType> BITWISE_A5_TYPES = {DT_INT16, DT_UINT16, DT_INT8,  DT_UINT8,
                                                                  DT_INT32, DT_UINT32, DT_INT64, DT_UINT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISE_A2A3_TYPES, BITWISE_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEAND");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseOr(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEOR");
    static const std::unordered_set<DataType> BITWISE_A2A3_TYPES = {DT_INT16, DT_UINT16, DT_INT8,
                                                                    DT_UINT8, DT_INT32,  DT_UINT32};
    static const std::unordered_set<DataType> BITWISE_A5_TYPES = {DT_INT16, DT_UINT16, DT_INT8,  DT_UINT8,
                                                                  DT_INT32, DT_UINT32, DT_INT64, DT_UINT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISE_A2A3_TYPES, BITWISE_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEOR");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseXor(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "BITWISEXOR");
    static const std::unordered_set<DataType> BITWISE_A2A3_TYPES = {DT_INT16, DT_UINT16, DT_INT8,
                                                                    DT_UINT8, DT_INT32,  DT_UINT32};
    static const std::unordered_set<DataType> BITWISE_A5_TYPES = {DT_INT16, DT_UINT16, DT_INT8,  DT_UINT8,
                                                                  DT_INT32, DT_UINT32, DT_INT64, DT_UINT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISE_A2A3_TYPES, BITWISE_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEXOR");
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, BinaryOperationTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEOR, Opcode::OP_BITWISEOR, BinaryOperationTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, BinaryOperationTileFunc<BinaryOpType::BITWISEXOR>);

} // namespace npu::tile_fwk

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
 * \file codegen_cube.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "securec.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::PrintMatmulTileTensor(bool isAcc) const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));

    std::ostringstream oss;
    if (opAttrs.count(OP_ATTR_PREFIX + "has_bias")) {
        bool hasBias = npu::tile_fwk::AnyCast<bool>(opAttrs.at(OP_ATTR_PREFIX + "has_bias"));
        if (hasBias) {
            std::string biasTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC2_IDX));
            oss << tileOpName << "(" << dstTensor << ", " << src0Tensor << ", " << src1Tensor << ", " << biasTensor
                << ");\n";
            return oss.str();
        }
    }
    oss << tileOpName << "<" << std::to_string(isAcc) << ">" << "(" << dstTensor << ", " << src0Tensor << ", "
        << src1Tensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenCubeOp(bool zeroC) const {
    if (isSupportLayout) {
        return PrintMatmulTileTensor(!zeroC);
    }
    // shape: dst, src0, src1
    bool isOffsetValid =
        (offset[ID1][ID0] == 0) && (offset[ID1][ID1] == 0) && (offset[ID2][ID0] == 0) && (offset[ID2][ID1] == 0);
    ASSERT(isOffsetValid) << "CUBE: haven't consider l0A/B offset yet.";
    bool isShapeValid = (shape[ID0][ID0] == shape[ID1][ID0]) &&
                        (shape[ID0][ID1] == shape[ID2][ID1] || shape[ID0][ID1] == shape[ID2][ID0]) &&
                        (shape[ID1][ID1] == shape[ID2][ID1] || shape[ID1][ID1] == shape[ID2][ID0]);
    ASSERT(isShapeValid) << "CUBE: m k n is invalid.";
    int64_t m = shape[ID0][ID0];
    int64_t k = shape[ID1][ID1]; // NEXTNEXT assume A is not transposed for now
    int64_t n = shape[ID0][ID1];
    unsigned uf = 0;

    std::string aVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string bVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string cVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::string aDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string bDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string cDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::ostringstream oss;

    if (isDynamicFunction) {
        auto l0cShapeDyn = dynamicValidShape[ID0];
        auto l0aShapeDyn = dynamicValidShape[ID1];
        auto l0bShapeDyn = dynamicValidShape[ID2];
        auto mSymbol = l0cShapeDyn[ID0];
        auto kSymbol = l0aShapeDyn[ID1];
        auto nSymbol = l0cShapeDyn[ID1];
        bool hasBias = 0;
        if (opAttrs.count(OP_ATTR_PREFIX + "has_bias")) {
            hasBias = npu::tile_fwk::AnyCast<bool>(opAttrs.at(OP_ATTR_PREFIX + "has_bias"));
        }
        std::string biasStr = ", " + std::to_string(hasBias);

        oss << tileOpName << "<" << cDtypeStr << ", " << aDtypeStr << ", " << bDtypeStr << ", " << offset[ID0][ID0]
            << ", " << offset[ID0][ID1] << biasStr << ">"
            << "((" << GetAddrTypeByOperandType(operandType[ID0]) << " " << cDtypeStr << "*)" << cVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID1]) << " " << aDtypeStr << "*)" << aVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID2]) << " " << bDtypeStr << "*)" << bVar << ", "
            << SymbolicExpressionTable::BuildExpression(mSymbol) << ", "
            << SymbolicExpressionTable::BuildExpression(kSymbol) << ", "
            << SymbolicExpressionTable::BuildExpression(nSymbol) << ", " << (zeroC ? "true" : "false") << ", " << uf
            << ", " << SymbolicExpressionTable::BuildExpression(l0cShapeDyn[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(l0cShapeDyn[ID1]) << ");\n";
    } else { // static function
        oss << tileOpName << "<" << cDtypeStr << ", " << aDtypeStr << ", " << bDtypeStr << ", " << offset[ID0][ID0]
            << ", " << offset[ID0][ID1] << ", " << shape[ID0][ID0] << ", " << shape[ID0][ID1] << ">"
            << "((" << GetAddrTypeByOperandType(operandType[ID0]) << " " << cDtypeStr << "*)" << cVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID1]) << " " << aDtypeStr << "*)" << aVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID2]) << " " << bDtypeStr << "*)" << bVar << ", " << m
            << ", " << k << ", " << n << ", " << (zeroC ? "true" : "false") << ", " << uf << ");\n";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::GenCubeOpMatmul() const {
    return GenCubeOp(true);
}

std::string CodeGenOpCloudNPU::GenCubeOpMatmulAcc() const {
    return GenCubeOp(false);
}

std::string CodeGenOpCloudNPU::GenParamsStr(const std::unordered_set<int32_t> &skipOperands) const {
    std::vector<std::string> params;
    for (int i = 0; i < MAX_OPERANDS; i++) {
        if (operand[i] == NULL_OPERAND) {
            continue;
        }

        std::string dtypeStr = DataType2CCEStr(operandDtype[i]);
        std::string prefix = GetAddrTypeByOperandType(operandType[i]);

        if (skipOperands.find(i) != skipOperands.end()) {
            continue;
        }

        if (operandType[i] == BUF_DDR) {
            std::string var = GenGmParamVar(i);
            std::ostringstream oss;
            oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            params.emplace_back(oss.str());
        } else {
            std::string var = sm->QueryVarNameByTensorMagic(operandWithMagic[i]);

            if (opCode != Opcode::OP_L1_TO_L0A && opCode != Opcode::OP_L1_TO_L0B && opCode != Opcode::OP_L1_TO_L0_BT &&
                opCode != Opcode::OP_L1_TO_L0_AT) {
                // 大包搬运场景下，L1搬运至L0不需要计算L1地址偏移
                // 非大包搬运场景下，L1与L0数据大小一致，也不需要地址偏移
                // 偏移计算仅用于L1_Copy_In 和 L1_Copy_Out
                AppendLocalBufferVarOffset({
                    {static_cast<unsigned>(i), std::ref(var)}
                });
            }

            std::ostringstream oss;
            ALOG_DEBUG_F("GenParamsStr var: %s", var.c_str());
            oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            params.emplace_back(oss.str());
        }
    }
    return JoinString(params, ", ");
}

} // namespace npu::tile_fwk

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
 * \file codegen_op.h
 * \brief
 */

#ifndef CODEGEN_OP_H
#define CODEGEN_OP_H

#include <map>
#include <tuple>
#include <cstdint>
#include <string>
#include <utility>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
const std::unordered_set<Opcode> SKIP_OPCODE = {
    Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_RESHAPE,
    Opcode::OP_UB_ALLOC,
    Opcode::OP_L1_ALLOC,
    Opcode::OP_L0A_ALLOC,
    Opcode::OP_L0B_ALLOC,
    Opcode::OP_L0C_ALLOC,
    Opcode::OP_FIX_ALLOC,
    Opcode::OP_BT_ALLOC,
    Opcode::OP_BIND_TENSOR,
    Opcode::OP_NOP,
    Opcode::OP_HUB,
};

const int MAX_OPERANDS = 11;
const int NULL_OPERAND = 0;

class CodeGenOp {
public:
    CodeGenOp(const std::shared_ptr<SymbolManager> &symbolManager, FunctionType funcType,
        const std::map<int, int> &locToOffset = {}, bool isUnderDynamicFunc = false, bool isMainBlk = false)
        : functionType(funcType),
          paramLocToParamListOffset(locToOffset),
          isUnderDynamicFunction(isUnderDynamicFunc),
          isMainBlock(isMainBlk) {
        for (size_t i = 0; i < MAX_OPERANDS; i++) {
            operand[i] = NULL_OPERAND;
            operandType[i] = BUF_UNKNOWN;
        }
        sm = symbolManager;
    }
    virtual ~CodeGenOp() = default;

    virtual void Init(const Operation &ops);

    virtual std::string GenBarrier() const;
    virtual std::string GenSyncSetOp() const;
    virtual std::string GenSyncWaitOp() const;

    virtual std::string GenOpCode() const = 0;

    bool hasNan{false};
    bool hasInf{false};

protected:
    std::string GenOpAttr(bool hasExistingParam = true) const;

    std::string opCodeStr;
    Opcode opCode{Opcode::OP_UNKNOWN};
    std::string aliasOp; // alias op name

    int operand[MAX_OPERANDS] = {}; // buffer id
    int operandWithMagic[MAX_OPERANDS] = {};
    OperandType operandType[MAX_OPERANDS] = {BUF_UNKNOWN, BUF_UNKNOWN, BUF_UNKNOWN, BUF_UNKNOWN};
    DataType operandDtype[MAX_OPERANDS] = {
        DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM};
    Element extOperandVal;
    SymbolicScalar extSymbolicScalar;
    std::vector<Element> extScalarVec;
    std::vector<int64_t> offset[MAX_OPERANDS] = {};
    std::vector<int64_t> shape[MAX_OPERANDS] = {};
    std::vector<int64_t> rawShape[MAX_OPERANDS] = {};
    // need adapt unaligned scene
    // Used for unaligned scene. In AST 1.0 it was padded in LogicalTensor constructor
    std::vector<int64_t> originShape[MAX_OPERANDS] = {};
    std::vector<SymbolicScalar> dynamicOffset[MAX_OPERANDS] = {};
    std::vector<SymbolicScalar> dynamicValidShape[MAX_OPERANDS] = {}; // valid shape
    std::vector<SymbolicScalar> offsetGmSymbolic[MAX_OPERANDS] = {};  // for spilling into GM scene
    bool isPartialMem[MAX_OPERANDS] = {};
    // if operand is an variable, record its related argument location
    // In COA(Call Operation Attribute), 0-index is the callee's cce info. So the tensor list starts from 1.
    int paramLocation[MAX_OPERANDS] = {1, 1, 1, 1, 1, 1};
    int GmTensorParamIdxInCallFunc{0};
    OpSyncQueue syncQueue;

    // add for ooo sched
    int addrOffset[MAX_OPERANDS] = {};
    std::vector<long> convParams;
    std::vector<int> poolParams;

    std::map<std::string, npu::tile_fwk::Any> opAttrs;

    std::shared_ptr<SymbolManager> sm{nullptr};

    const FunctionType functionType;
    std::string tileOpName;
    bool isInputForceCombineAxis{false};
    bool isSupportDynamicAligned{false}; // NEXTNEXT delete after all TileOp is changed to TileTensor Mode
    bool isDynamicFunction{false};
    bool isSupportLayout{false};
    const std::map<int, int> &paramLocToParamListOffset{};
    bool isUnderDynamicFunction{false};
    int operandCnt{0};
    bool isMainBlock{false};

private:
    void UpdateCodegenOpInfoByTensor(const Operation &ops, bool isInput, const std::shared_ptr<LogicalTensor> &tensor,
        int &operandIdx, size_t ioIdx);

    void UpdateTileOpInfo(const Operation &ops);

    void GetGmParamIdx(const Operation &oper);

    void ConvertPoolAttribute(const Operation &operation);
    void ConvertAttribute(const Operation &operation);
    void UpdateShape(
        const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx, bool isInput, size_t ioIdx);
    void UpdateOffsetForInput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx);
    void UpdateOffsetForOutput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx);
    void UpdateOffsetValueForGM(const std::vector<OpImmediate> &offsets, int operandIdx);
    void UpdateScalarValue(const npu::tile_fwk::Operation &ops);
    void UpdateOpAttribute(const npu::tile_fwk::Operation &ops);
    void CombineAxis(const Operation &oper, int operandIdx, bool isInput, size_t ioIdx);
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_H

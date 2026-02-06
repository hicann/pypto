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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
std::unordered_map<Opcode, std::set<int>> SKIP_PROC_PRARAM_IDX_IN_LOOP = {
    // scene: reduce for last axis
    // Parameter at 1st index (after numbering by CodeGenOp::Init) is used as temp buffer which is reused in loop body.
    {Opcode::OP_ROWSUM_SINGLE, {ID1}},
    {Opcode::OP_ROWMAX_SINGLE, {ID1}},
    {Opcode::OP_ROWMIN_SINGLE, {ID1}},
};

CodeGenOpCloudNPU::CodeGenOpCloudNPU(const std::shared_ptr<SymbolManager> &symbolManager, FunctionType funcType,
    const std::map<int, int> &locToOffset, bool isUnderDynamicFunc, bool isMainBlk)
    : CodeGenOp(symbolManager, funcType, locToOffset, isUnderDynamicFunc, isMainBlk),
      mteFixPipeOps_({
          // UB <-> GM
          {         Opcode::OP_UB_COPY_IN,              [this]() { return GenUBCopyIn(); }},
          {        Opcode::OP_UB_COPY_OUT,             [this]() { return GenUBCopyOut(); }},
          {    Opcode::OP_RESHAPE_COPY_IN,         [this]() { return GenReshapeCopyIn(); }},
          {   Opcode::OP_RESHAPE_COPY_OUT,        [this]() { return GenReshapeCopyOut(); }},
          {Opcode::OP_L1_TO_FIX_QUANT_PRE,             [this]() { return GenMemL1ToFB(); }},
          {       Opcode::OP_GATHER_IN_UB,            [this]() { return GenGatherInUB(); }},
          {             Opcode::OP_GATHER,              [this]() { return GenGatherOp(); }},
          // L1 <-> GM/BT/L1
          {         Opcode::OP_L1_COPY_IN,           [this]() { return GenMemL1CopyIn(); }},
          { Opcode::OP_L1_COPY_IN_A_SCALE,           [this]() { return GenMemL1CopyIn(); }},
          { Opcode::OP_L1_COPY_IN_B_SCALE,           [this]() { return GenMemL1CopyIn(); }},
          {        Opcode::OP_L1_COPY_OUT,          [this]() { return GenMemL1CopyOut(); }},
          {       Opcode::OP_GATHER_IN_L1,            [this]() { return GenGatherInL1(); }},

          // L0C <-> GM
          {       Opcode::OP_L0C_COPY_OUT,         [this]() { return GenMemL0CCopyOut(); }},

          {          Opcode::OP_L0C_TO_L1,            [this]() { return GenMemL0CToL1(); }},

          // L1 <-> L0
          {          Opcode::OP_L1_TO_L0A,             [this]() { return GenMemL1ToL0(); }},
          {          Opcode::OP_L1_TO_L0B,             [this]() { return GenMemL1ToL0(); }},
          {        Opcode::OP_L1_TO_L0_BT,             [this]() { return GenMemL1ToL0(); }},
          {        Opcode::OP_L1_TO_L0_AT,             [this]() { return GenMemL1ToL0(); }},
          {    Opcode::OP_L1_TO_L0A_SCALE,             [this]() { return GenMemL1ToL0(); }},
          {    Opcode::OP_L1_TO_L0B_SCALE,             [this]() { return GenMemL1ToL0(); }},
          {           Opcode::OP_L1_TO_BT,             [this]() { return GenMemL1ToBt(); }},

          // load op
          {               Opcode::OP_LOAD,                [this]() { return GenLoadOp(); }},

          // transpose with gm
          {  Opcode::OP_TRANSPOSE_MOVEOUT,     [this]() { return GenTransposeDataMove(); }},
          {   Opcode::OP_TRANSPOSE_MOVEIN,     [this]() { return GenTransposeDataMove(); }},

          // index outcast
          {      Opcode::OP_INDEX_OUTCAST,        [this]() { return GenIndexOutCastOp(); }},
          // lOC -> UB
          {        Opcode::OP_L0C_COPY_UB,     [this]() { return GenL0CToUBTileTensor(); }},

          {         Opcode::OP_UB_COPY_L1,      [this]() { return GenUBToL1TileTensor(); }},
          {      Opcode::OP_UB_COPY_ND2NZ, [this]() { return GenUBToUBND2NZTileTensor(); }},
}),
      unaryOps_({
          // cast op
          {Opcode::OP_CAST, [this]() { return GenCastOp(); }},

          // unary op
          {Opcode::OP_EXP, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_NEG, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_RSQRT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_BITWISENOT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_SQRT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_CEIL, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_FLOOR, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_TRUNC, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_EXPAND, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ONEHOT, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_RECIPROCAL, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWSUM, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMAX, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWEXPSUM, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWEXPMAX, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_COPY_UB_TO_UB, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMAXLINE, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ROWMINLINE, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_ABS, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_LN, [this]() { return GenUnaryOp(); }},
          {Opcode::OP_BRCB, [this]() { return GenUnaryOp(); }},

          // unary with temp buffer
          {Opcode::OP_COMPACT, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROUND, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWSUMLINE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWSUM_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWMAX_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWMIN_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_TRANSPOSE_VNCHWCONV, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
          {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
      }),
      binaryOps_({
          // binary op: vector operations
          {Opcode::OP_ADD, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_SUB, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MUL, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_DIV, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MAXIMUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_MINIMUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRSUM, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRMAX, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_PAIRMIN, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_POW, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_BITWISEAND, [this]() { return GenBinaryOp(); }},
          {Opcode::OP_BITWISEOR, [this]() { return GenBinaryOp(); }},

          // binary op: vector operations with tmp
          {Opcode::OP_MOD, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISERIGHTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISELEFTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_BITWISEXOR, [this]() { return GenBinaryOpWithTmp(); }},
          {Opcode::OP_COPYSIGN, [this]() { return GenBinaryOpWithTmp(); }},

          // binary op: broadcast associated vector
          {Opcode::OP_ADD_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_SUB_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MUL_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_DIV_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MAX_BRC, [this]() { return GenBinaryWithBrc(); }},
          {Opcode::OP_MIN_BRC, [this]() { return GenBinaryWithBrc(); }},

          // binary op: vector scalar
          {Opcode::OP_ADDS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_SUBS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MULS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_DIVS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MAXS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_MINS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISEANDS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISEORS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISERIGHTSHIFTS, [this]() { return GenVectorScalarOp(); }},
          {Opcode::OP_BITWISELEFTSHIFTS, [this]() { return GenVectorScalarOp(); }},

          // binary op: vector scalar with tmp
          {Opcode::OP_MODS, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_SBITWISERIGHTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_SBITWISELEFTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
          {Opcode::OP_BITWISEXORS, [this]() { return GenVectorScalarOpWithTmp(); }},

          // binary op: vector scalar, scalar mode
          {Opcode::OP_S_ADDS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_SUBS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MULS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_DIVS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MAXS, [this]() { return GenVectorScalarOpScalarMode(); }},
          {Opcode::OP_S_MINS, [this]() { return GenVectorScalarOpScalarMode(); }},
      }),
      compositeOps_({
          // range op
          {Opcode::OP_RANGE, [this]() { return GenRangeOp(); }},

          // logicalnot
          {Opcode::OP_LOGICALNOT, [this]() { return GenLogicalNotOp(); }},
          // logicaland
          {Opcode::OP_LOGICALAND, [this]() { return GenLogicalAndOp(); }},

          // indexadd
          {Opcode::OP_INDEX_ADD, [this]() { return GenIndexAddOp(); }},

          // indexput
          {Opcode::OP_INDEX_PUT, [this]() { return GenIndexPutOp(); }},

          // cumsum
          {Opcode::OP_CUM_SUM, [this]() { return GenCumSumOp(); }},

          // triUL
          {Opcode::OP_TRIUL, [this]() { return GenTriULOp(); }},

          // vector where
          {Opcode::OP_WHERE_SS, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_TS, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_ST, [this]() { return GenWhereOp(); }},
          {Opcode::OP_WHERE_TT, [this]() { return GenWhereOp(); }},

          // cmp op
          {Opcode::OP_CMP, [this]() { return GenCmpOp(); }},
          {Opcode::OP_CMPS, [this]() { return GenCmpOp(); }},
      }),
      sortOps_({
          // sort
          {Opcode::OP_BITSORT, [this]() { return GenBitSortOp(); }},
          {Opcode::OP_MRGSORT, [this]() { return GenMrgSortOp(); }},
          {Opcode::OP_EXTRACT, [this]() { return GenExtractOp(); }},
          {Opcode::OP_TILEDMRGSORT, [this]() { return GenTiledMrgSortOp(); }},

          {Opcode::OP_TOPK_SORT, [this]() { return GenTopKSortOp(); }},
          {Opcode::OP_TOPK_MERGE, [this]() { return GenTopKMergeOp(); }},
          {Opcode::OP_TOPK_EXTRACT, [this]() { return GenTopKExtractOp(); }},

          // parallel sort
          {Opcode::OP_SORT, [this]() { return GenSortOp(); }},
          {Opcode::OP_COMPARE_SWAP, [this]() { return GenCompareAndSwapOp(); }},
          {Opcode::OP_MERGE, [this]() { return GenMergeOp(); }},
      }),
      cubeOps_({
          // matmul
          {Opcode::OP_A_MUL_B, [this]() { return GenCubeOpMatmul(); }},
          {Opcode::OP_A_MUL_BT, [this]() { return GenCubeOpMatmul(); }},
          {Opcode::OP_A_MULACC_B, [this]() { return GenCubeOpMatmulAcc(); }},
          {Opcode::OP_A_MULACC_BT, [this]() { return GenCubeOpMatmulAcc(); }},
      }),
      syncOps_({
          // sync
          {Opcode::OP_SYNC_SRC, [this]() { return GenSyncSetOp(); }},
          {Opcode::OP_SYNC_DST, [this]() { return GenSyncWaitOp(); }},
          {Opcode::OP_BAR_V, [this]() { return GenBarrier(); }},
          {Opcode::OP_BAR_M, [this]() { return GenBarrier(); }},
          {Opcode::OP_BAR_ALL, [this]() { return GenBarrier(); }},
          {Opcode::OP_CV_SYNC_SRC, [this]() { return GenCVSyncSetOp(); }},
          {Opcode::OP_CV_SYNC_DST, [this]() { return GenCVSyncWaitOp(); }},
      }),
      distributeOps_({
          // distribute op
          {Opcode::OP_FFN_SCHED, [this]() { return GenDistOp(); }},
          {Opcode::OP_FFN_BATCHING, [this]() { return GenDistOp(); }},
          {Opcode::OP_FFN_COMBINEINFO, [this]() { return GenDistOp(); }},
          {Opcode::OP_FFN_VALIDCNT, [this]() { return GenDistOp(); }},
          {Opcode::OP_SEND_TO_ROUTING_EXPERT, [this]() { return GenDistOp(); }},
          {Opcode::OP_SEND_TO_SHARED_EXPERT, [this]() { return GenDistOp(); }},
          {Opcode::OP_DISPATCH_SET_FLAG, [this]() { return GenDistOp(); }},
          {Opcode::OP_COPY_TO_LOCAL_EXPERT, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_SET, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_PUT, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_PUT_UB2GM, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_SIGNAL, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_GET, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_GET_GM2UB, [this]() { return GenDistOp(); }},
          {Opcode::OP_SHMEM_REDUCE, [this]() { return GenDistOp(); }},
          {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, [this]() { return GenDistOp(); }},
          {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, [this]() { return GenDistOp(); }},
      }),
      gatherScatterOps_({
          // gather/scatter op
          {Opcode::OP_GATHER_FROM_UB, [this]() { return GenGatherFromUBOp(); }},
          {Opcode::OP_GATHER_ELEMENT, [this]() { return GenGatherElementOp(); }},
          {Opcode::OP_SCATTER_ELEMENT, [this]() { return GenScatterElementSOp(); }},
          {Opcode::OP_SCATTER, [this]() { return GenScatterOp(); }},
      }),
      normalVecOps_({
          // vector dup
          {Opcode::OP_VEC_DUP, [this]() { return GenDupOp(); }},
      }),
      perfOps_({
          // for performace optimization
          {Opcode::OP_PHASE1, []() { return "SUBKERNEL_PHASE1\n"; }},
          {Opcode::OP_PHASE2, []() { return "SUBKERNEL_PHASE2\n"; }},
      }),
      aicpuOps_({
          // for aicpu call
          {Opcode::OP_AICPU_CALL_AIC, [this]() { return GenAicpuCallOp(); }},
          {Opcode::OP_AICPU_CALL_AIV, [this]() { return GenAicpuCallOp(); }},
      }) {
    InitOpsGenMap();
}

CodeGenOpCloudNPU::CodeGenOpCloudNPU(const CodeGenOpCloudNPUCtx &ctx)
    : CodeGenOpCloudNPU(ctx.symbolManager, ctx.topFunc.GetFunctionType(), ctx.locToOffset,
          ctx.topFunc.IsUnderDynamicFunction(), ctx.isMainBlock) {
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}
void CodeGenOpCloudNPU::InitOpsGenMap() {
    InitScalaOpsMap();
    InitMTEOpsMap();
    InitVecOpsMap();
    InitCubeOpsMap();
    InitDistOpsMap();
    InitPerfOpsMap();
    InitAICPUOpsMap();
}

void CodeGenOpCloudNPU::InitScalaOpsMap() {
    opsGenMap_.insert(syncOps_.cbegin(), syncOps_.cend());
}

void CodeGenOpCloudNPU::InitMTEOpsMap() {
    opsGenMap_.insert(mteFixPipeOps_.cbegin(), mteFixPipeOps_.cend());
}

void CodeGenOpCloudNPU::InitVecOpsMap() {
    opsGenMap_.insert(unaryOps_.cbegin(), unaryOps_.cend());
    opsGenMap_.insert(binaryOps_.cbegin(), binaryOps_.cend());
    opsGenMap_.insert(compositeOps_.cbegin(), compositeOps_.cend());
    opsGenMap_.insert(sortOps_.cbegin(), sortOps_.cend());
    opsGenMap_.insert(gatherScatterOps_.cbegin(), gatherScatterOps_.cend());
    opsGenMap_.insert(normalVecOps_.cbegin(), normalVecOps_.cend());
}

void CodeGenOpCloudNPU::InitCubeOpsMap() {
    opsGenMap_.insert(cubeOps_.cbegin(), cubeOps_.cend());
}

void CodeGenOpCloudNPU::InitDistOpsMap() {
    opsGenMap_.insert(distributeOps_.cbegin(), distributeOps_.cend());
}

void CodeGenOpCloudNPU::InitPerfOpsMap() {
    opsGenMap_.insert(perfOps_.cbegin(), perfOps_.cend());
}

void CodeGenOpCloudNPU::InitAICPUOpsMap() {
    opsGenMap_.insert(aicpuOps_.cbegin(), aicpuOps_.cend());
}

void CodeGenOpCloudNPU::AppendLocalBufferVarOffset(
    const std::map<unsigned, std::reference_wrapper<std::string>> &vars) const {
    for (auto &kv : vars) {
        auto operandIdx = kv.first;
        int64_t resOffset{0};

        std::vector<int64_t> varOffset = offset[operandIdx];
        if (varOffset.empty()) {
            continue;
        }

        std::vector<int64_t> varRawShape = rawShape[operandIdx];
        ASSERT(!varRawShape.empty()) << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        resOffset = CalcLinearOffset(varRawShape, varOffset);
        if (resOffset == 0) {
            continue;
        }

        std::string &var = kv.second.get();

        ASSERT(!var.empty()) << "operandIdx: " << operandIdx << ", var is empty !!";
        ALOG_INFO_F("var: %s, varRawShape: %s, varOffset: %s, resOffset: %lld", var.c_str(),
            IntVecToStr(varRawShape).c_str(), IntVecToStr(varOffset).c_str(), resOffset);

        var.append(" + ").append(std::to_string(resOffset));
    }
}

SymbolicScalar CodeGenOpCloudNPU::GetOperandStartOffset(int operandIdx) const {
    std::vector varOffset = offset[operandIdx];
    if (varOffset.empty()) {
        return 0;
    }

    const auto &dynOffset = dynamicOffset[operandIdx];
    if (!dynOffset.empty()) {
        std::vector varRawShape = rawShape[operandIdx]; // 内部应该不能出现dynRawShape，所以这里用立即数即可
        ASSERT(!varRawShape.empty()) << "varRawShape is empty!!";
        ASSERT(dynOffset.size() == varRawShape.size())
            << "dynOffset " << SymbolicVecToStr(dynOffset) << ", size " << dynOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!!";

        SymbolicScalar resOffset = 0;
        for (size_t i = 0; i < dynOffset.size(); i++) {
            resOffset = resOffset * varRawShape[i];
            resOffset = resOffset + dynOffset[i];
        }

        ASSERT(operandIdx < operandCnt) << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
        ALOG_DEBUG_F(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
        ALOG_DEBUG_F(" varOffset: %s", SymbolicVecToStr(dynOffset).c_str());
        ALOG_DEBUG_F(" resOffset: %s", resOffset.Dump().c_str());
        if (resOffset.ConcreteValid()) {
            return resOffset.Concrete();
        }
        return SymbolicExpressionTable::BuildExpression(resOffset);
    }

    std::vector varRawShape = rawShape[operandIdx];
    ASSERT(!varRawShape.empty()) << "varRawShape is empty!!";
    ASSERT(varOffset.size() == varRawShape.size())
        << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
        << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!!";

    int64_t resOffset = CalcLinearOffset(varRawShape, varOffset);
    if (resOffset == 0) {
        return 0;
    }

    ASSERT(operandIdx < operandCnt) << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
    ALOG_DEBUG_F(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
    ALOG_DEBUG_F(" varOffset: %s", IntVecToStr(varOffset).c_str());
    ALOG_DEBUG_F(" resOffset: %d", resOffset);
    return resOffset;
}

std::string CodeGenOpCloudNPU::GenGmParamVar(unsigned gmParamIdx) const {
    if (isUnderDynamicFunction) {
        std::ostringstream os;
        os << "GET_PARAM_ADDR(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ")";
        return os.str();
    }

    auto paramLoc = paramLocation[gmParamIdx];
    auto iter = paramLocToParamListOffset.find(paramLoc);
    ASSERT(iter != paramLocToParamListOffset.end())
        << "paramLoc " << paramLoc << " can not be found in paramLocToParamListOffset";
    std::string gmVar = "((" + GM_PARAM_TYPE_FOR_STATIC + "*)(param) + " + std::to_string(iter->second) + ")->Addr";
    return gmVar;
}

// Used for parameter of GM shape and offset, e.g.
// GET_PARAM_RAWSHAPE_2(param, 19, 9), GET_PARAM_OFFSET_2(param, 19, 9)
// If dim is 2, the macro would be expanded into "shape0, shape1" which is implemented in aicore_runtime.h
std::vector<std::string> CodeGenOpCloudNPU::GenGetParamMacroPacked(
    unsigned gmParamIdx, int dim, const std::string &prefix) const {
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    os << "GET_PARAM_" << prefix << "_" << dim << "(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc
       << ", " << paramLocation[gmParamIdx] << ")";
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpCloudNPU::GenParamIdxExprByIndex(
    unsigned gmParamIdx, int dim, const std::string &prefix) const {
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    for (int index = 0; index < dim; ++index) {
        os << "GET_PARAM_" << prefix << "_BY_IDX(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ", " << dim << ", " << index << ")";
        paramExpr.emplace_back(os.str());
        os.str("");
    }
    return paramExpr;
}

std::vector<std::string> CodeGenOpCloudNPU::GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const {
    std::vector<std::string> argList;
    for (auto &expr : exprList) {
        std::string exprStr = SymbolicExpressionTable::BuildExpression(expr);
        argList.push_back(exprStr);
    }
    return argList;
}

std::vector<std::string> CodeGenOpCloudNPU::BuildStride(const std::vector<int64_t> &input) {
    if (input.empty()) {
        return {};
    }

    std::vector<std::string> res(input.size(), "1");
    int64_t base = 1;
    for (int i = input.size() - 2; i >= 0; --i) {
        base *= input[i + 1];
        res[i] = std::to_string(base);
    }

    return res;
}

void CodeGenOpCloudNPU::UpdateTileTensorShapeAndStride(
    int paramIdx, TileTensor &tileTensor, bool isSpillToGm, const ShapeInLoop &shapeInLoop) {
    auto newOriginShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.originShape : originShape[paramIdx];
    auto newRawShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.rawShape : rawShape[paramIdx];
    auto newDynValidShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.dynamicValidShape : dynamicValidShape[paramIdx];
    ALOG_INFO_F("newOriginShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newOriginShape).c_str(),
        IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

    tileTensor.rawShape = newRawShape;

    // ---- static ----
    if (functionType == FunctionType::STATIC) {
        for (auto s : newOriginShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
        tileTensor.stride = BuildStride(newRawShape);
        return;
    }

    // ---- dynamic ----
    // gm tensor
    if (tileTensor.bufType == OperandType::BUF_DDR) {
        if (isSpillToGm) {
            for (auto s : shape[paramIdx]) {
                tileTensor.shape.emplace_back(std::to_string(s));
            }
            tileTensor.stride = BuildStride(shape[paramIdx]);
        } else {
            tileTensor.shape = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_RAW_SHAPE);
            tileTensor.stride = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_STRIDE);
        }
        return;
    }

    // local tensor

    for (const auto &s : newDynValidShape) {
        tileTensor.shape.emplace_back(SymbolicExpressionTable::BuildExpression(s));
    }
    tileTensor.stride = BuildStride(newRawShape);
}

TileTensor CodeGenOpCloudNPU::BuildTileTensor(
    int paramIdx, const std::string &usingType, const ShapeInLoop &shapeInLoop) {
    bool isSpillToGm = operand[paramIdx] == SYMBOL_STACK_BASE;

    TileTensor tileTensor;
    tileTensor.isStatic = functionType == FunctionType::STATIC;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.shapeInLoop = shapeInLoop;

    if (tileTensor.isStatic) {
        tileTensor.dim = shapeInLoop.loopDepth > 0 ? shapeInLoop.originShape.size() : originShape[paramIdx].size();
    } else {
        tileTensor.dim =
            shapeInLoop.loopDepth > 0 ? shapeInLoop.dynamicValidShape.size() : dynamicValidShape[paramIdx].size();
    }

    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = isSpillToGm ? GenGMAddrExprWithOffset(GM_STACK_BASE, paramIdx) : GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic, true);
    }

    tileTensor.usingType = usingType;

    tileTensor.tensorName = BUFFER_TYPE_TO_PREFIX_LC.at(tileTensor.bufType) + "Tensor_" +
                            std::to_string(IdGen<IdType::CG_VAR_NAME>::Inst().NewId());
    if (shapeInLoop.loopDepth != 0) {
        std::string tensorName = tensorNames_[paramIdx];
        if (!tensorName.empty()) {
            tensorName.append("_low").append(std::to_string(tileTensor.dim)).append("DimInLoop");
            tileTensor.tensorName = tensorName;
        }
    }
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, shapeInLoop);
    
    if (!offset[paramIdx].empty()) {
        tileTensor.localBufOffset = offset[paramIdx];
    }

    return tileTensor;
}

void CodeGenOpCloudNPU::UpdateSaturateStatus(FloatSaturateStatus &fs) {
    auto checkValue = [&](float value) {
        fs.hasNan |= std::isnan(value);
        fs.hasInf |= std::isinf(value);
    };

    if (extOperandVal.IsFloat()) {
        float value = extOperandVal.Cast<float>();
        checkValue(value);
    }
    for (const auto &scalar : extScalarVec) {
        if (scalar.IsFloat()) {
            float value = scalar.Cast<float>();
            checkValue(value);
        }
    }
}

void CodeGenOpCloudNPU::UpdateTileTensorInfo() {
    if (!isSupportLayout) {
        return;
    }

    auto iter = SUPPORT_TILETENSOR_OPS.find(opCode);
    if (iter == SUPPORT_TILETENSOR_OPS.end()) {
        ASSERT(iter != SUPPORT_TILETENSOR_OPS.end()) << "opCode: " << opCodeStr << " not support tile tensor!";
        return;
    }

    tileOpName = iter->second; // update tileOpName from SUPPORT_TILETENSOR_OPS

    for (int i = 0; i < operandCnt; ++i) {
        TileTensorUsing tileTensorUsing{operandDtype[i], operandType[i], static_cast<int>(rawShape[i].size()),
            originShape[i], rawShape[i], functionType == FunctionType::STATIC};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType);
        std::string tensorName = sm->AddTileTensor(tileTensor);
        tensorNames_[i] = tensorName;
    }
}

bool CodeGenOpCloudNPU::ShouldSkipProcInLoop(int paramIdx) {
    auto iter = SKIP_PROC_PRARAM_IDX_IN_LOOP.find(opCode);
    if (iter == SKIP_PROC_PRARAM_IDX_IN_LOOP.end()) {
        return false;
    }
    return iter->second.find(paramIdx) != iter->second.end();
}

std::vector<SymbolicScalar> CodeGenOpCloudNPU::GetLoopAxes() {
    std::vector<SymbolicScalar> loopAxes;
    GetAttr(OpAttributeKey::loopAxes, loopAxes);

    if (!isMainBlock) {
        return loopAxes;
    }
    // use dst shape as loop axes in main block
    std::vector<SymbolicScalar> newLoopAxes;
    for (size_t i = 0; i < loopAxes.size(); ++i) {
        SymbolicScalar axis = isDynamicFunction ? dynamicValidShape[0][i] : SymbolicScalar(originShape[0][i]);
        newLoopAxes.emplace_back(axis);
    }

    return newLoopAxes;
}

void CodeGenOpCloudNPU::UpdateLoopInfo() {
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        return;
    }

    std::vector<SymbolicScalar> loopAxes = GetLoopAxes();
    if (loopAxes.empty()) {
        return;
    }

    bool isLoopStart{false};
    if (GetAttr(OpAttributeKey::loopGroupStart, isLoopStart) && isLoopStart) {
        forBlkMgr_->LoopStart();
        forBlkMgr_->UpdateAxesList(loopAxes);
    }

    // Add TileTensor info in loop
    ALOG_INFO_F("opCode %s has loopAxes: %s", opCodeStr.c_str(), IntVecToStr(loopAxes).c_str());
    size_t loopDepth = loopAxes.size();
    for (int i = 0; i < operandCnt; ++i) {
        if (ShouldSkipProcInLoop(i)) {
            continue;
        }
        ShapeInLoop shapeInLoop = BuildShapeInLoop(i, loopDepth);
        ALOG_INFO_F("shapeInLoop: loopDepth is %d newOriginShape is %s, newRawShape is %s, newDynValidShape is %s",
            loopDepth, IntVecToStr(shapeInLoop.originShape).c_str(), IntVecToStr(shapeInLoop.rawShape).c_str(),
            IntVecToStr(shapeInLoop.dynamicValidShape).c_str());
        TileTensorUsing tileTensorUsing{operandDtype[i], operandType[i], static_cast<int>(shapeInLoop.rawShape.size()),
            shapeInLoop.originShape, shapeInLoop.rawShape, functionType == FunctionType::STATIC};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType, shapeInLoop);
        forBlkMgr_->AddTensorInLoopBody(tensorNames_[i], tileTensor);
    }
}

ShapeInLoop CodeGenOpCloudNPU::BuildShapeInLoop(int paramIdx, size_t loopDepth) {
    auto newOriginShape = GetShapeInLoop(originShape[paramIdx], loopDepth);
    auto newRawShape = GetShapeInLoop(rawShape[paramIdx], loopDepth);
    auto newDynValidShape = GetShapeInLoop<SymbolicScalar>(dynamicValidShape[paramIdx], loopDepth);
    return {loopDepth, newOriginShape, newRawShape, newDynValidShape};
}

std::string CodeGenOpCloudNPU::PrintCoord(size_t dim, const std::string &coord) const {
    std::string ret = COORD;
    ret.append(std::to_string(dim)).append(DIM).append(coord);
    return ret;
}

void CodeGenOpCloudNPU::FillParamWithFullShape(
    std::vector<std::string> &paramList, const std::vector<int64_t> &input) const {
    FillParamWithInput(paramList, input, 0, input.size());
}

void CodeGenOpCloudNPU::FillParamWithShapeExceptFirst(
    std::vector<std::string> &paramList, const std::vector<int64_t> &input) const {
    FillParamWithInput(paramList, input, 1, input.size());
}

std::string CodeGenOpCloudNPU::QueryTileTensorNameByIdx(int paramIdx) const {
    std::vector<TileTensor> res;
    if (forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop()) {
        res = sm->QueryTileTensorInLoopByMagic(operandWithMagic[paramIdx]);
        // some tensor in loop is reused same tensor out of loop
        if (res.empty()) {
            res = sm->QueryTileTensorByMagic(operandWithMagic[paramIdx]);
        }
    } else {
        res = sm->QueryTileTensorByMagic(operandWithMagic[paramIdx]);
    }

    if (res.size() == 1) {
        return res[0].tensorName;
    }
    ALOG_INFO_F("paramIdx is %d, tensor magic is %d, res size is %d", paramIdx, operandWithMagic[paramIdx], res.size());

    for (const auto &tileTensor : res) {
        auto targetRawShape =
            forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop() ? tileTensor.shapeInLoop.rawShape : rawShape[paramIdx];
        // Currently only support additional comparison of rawShape
        if (tileTensor.rawShape == targetRawShape) {
            return tileTensor.tensorName;
        }
    }

    ASSERT(false) << "paramIdx " << paramIdx << ", tensor magic " << operandWithMagic[paramIdx]
                  << " is not found !!! res size is " << res.size();
    return "";
}
std::string CodeGenOpCloudNPU::GenOpCode() const {
    std::string ret;
    auto iter = opsGenMap_.find(opCode);
    if (iter != opsGenMap_.end()) {
        ret = iter->second();
    } else {
        // To aid in testing, do not use ASSERT.
        return std::string{"CAN NOT HANDLE OP: " + opCodeStr};
    }

    if (forBlkMgr_ == nullptr || !forBlkMgr_->IsInLoop()) {
        return ret;
    }

    forBlkMgr_->AddOpInLoopBody(ret);

    bool isLoopEnd{false};
    GetAttr(OpAttributeKey::loopGroupEnd, isLoopEnd);
    if (!isLoopEnd) {
        return "";
    }

    ret = forBlkMgr_->Print();
    forBlkMgr_->OutLoop();
    return ret;
}

std::string CodeGenOpCloudNPU::GetLastUse() const {
    if (!opAttrs.count(OpAttributeKey::lastUse)) {
        return "";
    }
    std::vector<int64_t> val = GetVectorIntAttribute(OpAttributeKey::lastUse);
    int valSize = val.size();
    ASSERT(valSize != 0) << "GetLastUse error!!!";
    std::ostringstream oss;
    oss << "LastUse" << valSize << "Dim";
    oss << WrapParamByAngleBrackets(val);
    return oss.str();
}

} // namespace npu::tile_fwk

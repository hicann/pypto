/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#ifndef CODEGEN_OP_CLOUDNPU_H
#define CODEGEN_OP_CLOUDNPU_H

#include <utility>
#include <unordered_set>

#include "op_print_param_def.h"
#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/stmt_mgr/codegen_for_block.h"
#include "codegen/codegen_op.h"

namespace npu::tile_fwk {
struct CodeGenOpCloudNPUCtx : public CodeGenOpCtx {
    const Operation &operation;
    std::shared_ptr<ForBlockManager> forBlockManager{nullptr};

    CodeGenOpCloudNPUCtx(std::shared_ptr<SymbolManager> sm, Function &tf, Function &sf, const Operation &op,
        const std::map<int, int> &lto = {}, bool isMainBlk = false)
        : CodeGenOpCtx(std::move(sm), tf, sf, lto, isMainBlk), operation(op) {}

    CodeGenOpCloudNPUCtx(std::shared_ptr<SymbolManager> sm, std::shared_ptr<ForBlockManager> fbm, Function &tf,
        Function &sf, const Operation &op, const std::map<int, int> &lto = {}, bool isMainBlk = false)
        : CodeGenOpCtx(std::move(sm), tf, sf, lto, isMainBlk), operation(op), forBlockManager(fbm) {}
};

class CodeGenOpCloudNPU : public CodeGenOp {
public:
    explicit CodeGenOpCloudNPU(const CodeGenOpCloudNPUCtx &ctx);
    CodeGenOpCloudNPU(const std::shared_ptr<SymbolManager> &symbolManager, FunctionType funcType,
        const std::map<int, int> &locToOffset = {}, bool isUnderDynamicFunc = false, bool isMainBlk = false);

    ~CodeGenOpCloudNPU() override = default;

    std::string GenBarrier() const;
    std::string GenSyncSetOp() const;
    std::string GenSyncWaitOp() const;
    std::string GenCVSyncSetOp() const;
    std::string GenCVSyncWaitOp() const;
    std::string GenMemL1ToBt() const;
    std::string GenMemL1CopyIn() const;
    std::string GenMemL1CopyOut() const;
    std::string GenMemL1ToFB() const;
    std::string GenMemL0CCopyOut() const;
    std::string GenMemL0CToL1() const;

    std::string GenMemL1ToL0() const;

    std::string GenUBCopyIn() const;
    std::string GenUBCopyOut() const;
    std::string GenUBToL1TileTensor() const;
    std::string GenUBToUBND2NZTileTensor() const;
    std::string GenReshapeCopyIn() const;
    std::string GenReshapeCopyOut() const;

    std::string GenLoadOp() const;
    std::string PrintGatherInL1TileTensor() const;
    std::string GenGatherInL1() const;
    std::string GenGatherInUB() const;

    std::string GenUnaryOp() const;
    std::string GenUnaryOpWithTmpBuff() const;

    std::string GenLogicalNotOp() const;
    std::string GenLogicalAndOp() const;

    std::string GenBinaryOp() const;
    std::string GenVectorScalarOp() const;
    std::string GenBinaryOpWithTmp() const;
    std::string GenVectorScalarOpWithTmp() const;

    std::string GenCubeOpMatmul() const;
    std::string GenCubeOpMatmulAcc() const;

    std::string GenBitwiseShiftOp() const;
    std::string GenBitwiseShiftScalarOp() const;

    std::string GenCastOp() const;

    std::string GenDupOp() const;

    std::string GenTransposeDataMove() const;

    std::string GenGatherElementOp() const;

    std::string GenRangeOp() const;
    std::string PrintRangeTileTensor(
        const std::string &startVal, const std::string &stepVal, const std::string &tileIdxExpr) const;
    std::string GenL0CToUBTileTensor() const;

    std::string GenScatterElementSOp() const;
    std::string GenScatterOp() const;

    std::string GenIndexAddOp() const;

    std::string GenIndexPutOp() const;

    std::string GenIndexOutCastOp() const;
    std::string PrintIndexOutCastTileTensor() const;

    std::string GenCumSumOp() const;
    std::string PrintGatherDynamicUnaligned() const;
    std::string PrintGatherLayout() const;
    std::string GenGatherOp() const;
    std::string GenGatherFromUBOp() const;

    std::string GenMemCopyCube(bool isLocalToGM, unsigned uf = 0) const;
    std::string GenMemL1SpillIntoGM(bool isLocalToGM, unsigned uf) const;

    std::string GenBinaryWithBrc() const;

    std::string GenBitSortOp() const;
    std::string GenMrgSortOp() const;
    std::string GenExtractOp() const;
    std::string GenTiledMrgSortOp() const;
    std::string GenSortOp() const;
    std::string GenCompareAndSwapOp() const;
    std::string GenMergeOp() const;

    std::string GenTopKSortOp() const;
    std::string GenTopKMergeOp() const;
    std::string GenTopKExtractOp() const;

    std::string GenParamsStr(const std::unordered_set<int32_t> &skipOperands = {}) const;

    std::string GenDistOp() const;
    std::string GetTemplateDType() const;
    std::string GenTemplateParams() const;
    std::string GenExtraTemplateParamsForMoeDistributedCombine(int32_t operandIndex) const;
    std::string GenOffsets(int32_t operandIndex, int32_t dim) const;
    std::string GenShapes(int32_t operandIndex, int32_t dim) const;
    std::string GenRawShapes(int32_t operandIndex, int32_t dim) const;
    std::string GenExtraParamsStr() const;
    std::string GenOffsetsAndRawShapes(int32_t operandIndex, int32_t dim) const;

    std::string GenAicpuCallOp() const;

    std::string GenWhereOp() const;

    std::string GenOpCode() const override;

    void UpdateSaturateStatus(FloatSaturateStatus &fs);

private:
    std::string QueryTileTensorNameByIdx(int paramIdx) const;

    std::string GenTemplateParamsForPutAndGet() const;
    std::string GenTemplateParamsForSignal() const;
    std::string GenTemplateParamsForMoeDistributedCombineSend() const;
    std::string GenTemplateParamsForMoeDistributedCombineReceive() const;
    std::string GenTemplateParamsForSet() const;
    std::string GenTemplateParamsDefault() const;

    std::string GenOffsetsAndRawShapesForShmemPutAndGet() const;
    std::string GenOffsetsAndRawShapesForShmemPutAndGetUB() const;
    std::string GenOffsetsAndRawShapesForShmemSignal() const;
    std::string GenOffsetsAndRawShapesForMoeDistributedCombineSend() const;
    std::string GenOffsetsAndRawShapesForMoeDistributedCombineReceive() const;
    std::string GenOffsetsAndRawShapesForSendToRoutingExpert() const;
    std::string GenOffsetsAndRawShapesForSendToSharedExpert() const;
    std::string GenOffsetsAndRawShapesForCopyToLocalExpert() const;
    std::string GenOffsetsAndRawShapesForDispatchSetFlag() const;
    std::string GenOffsetsAndRawShapesForFfnOperations() const;
    std::string GenOffsetsAndRawShapesForFfnCombineInfo() const;
    std::string GenOffsetsAndRawShapesForShmemSet() const;
    std::string GenOffsetsAndRawShapesDefault() const;

    void UpdateTileTensorInfo();
    bool NeedUpdateLoopInfo();
    void UpdateLoopInfo();
    std::vector<SymbolicScalar> GetLoopAxes();
    ShapeInLoop BuildShapeInLoop(int paramIdx, size_t loopDepth);
    bool ShouldSkipProcInLoop(int paramIdx);

    template <typename T = int64_t>
    std::vector<T> GetShapeInLoop(const std::vector<T> &input, size_t loopDepth) {
        ASSERT(loopDepth < input.size()) << "loopDepth " << loopDepth << " must be small than dim size" << input.size();
        std::vector<T> reservedShapeExceptLoopAxes;
        for (size_t i = loopDepth; i < input.size(); ++i) {
            reservedShapeExceptLoopAxes.emplace_back(input[i]);
        }
        return reservedShapeExceptLoopAxes;
    }

    int GetCacheModeFlag(const std::string &cacheMode) const;

    template <typename T>
    bool GetAttr(const std::string &key, T &value) const {
        auto it = opAttrs.find(key);
        if (it == opAttrs.end()) {
            ALOG_INFO_F("can not find key: %s in opAttrs", key.c_str());
            return false;
        }
        if (it->second.Type() == typeid(T)) {
            value = npu::tile_fwk::AnyCast<T>(it->second);
            return true;
        }
        ALOG_ERROR_F("Type mismatch: %s != %s", it->second.Type().name(), typeid(T).name());
        return false;
    }

    TileTensor BuildTileTensor(int paramIdx, const std::string &usingType, const ShapeInLoop &shapeInLoop = {});
    void UpdateTileTensorShapeAndStride(
        int paramIdx, TileTensor &tileTensor, bool isSpillToGm, const ShapeInLoop &shapeInLoop = {});
    std::vector<std::string> BuildStride(const std::vector<int64_t> &input);

    std::vector<int64_t> GetTileShapeForMemTransfer(
        OperandType localType, std::vector<int64_t> gmShape, unsigned localIdx) const;
    std::string GenMemCopyVar(bool isCopyLocalToGM, unsigned uf = 0) const;

    std::string GenGMAddrExprWithOffset(const std::string &addrExpr, unsigned gmIdx) const;
    std::string GenAddrExpr(const std::string &addrExpr, unsigned offsetParam) const;

    // Add offset of local buffer variable when the variable is generated by spliting from "view" operation.
    template <typename T = std::string, typename... Args>
    void AppendLocalBufVarOffsetInOrder(Args &...args) const {
        tempVarsMap.clear();
        tempKey = 0;
        AppendLocalBufVarOffsetInOrderImpl<T>(args...);
    }
    void AppendLocalBufferVarOffset(const std::map<unsigned, std::reference_wrapper<std::string>> &vars) const;

    // get start offset in total block
    SymbolicScalar GetOperandStartOffset(int operandIdx) const;

    std::string GenGmParamVar(unsigned gmParamIdx) const;

    std::vector<std::string> GenGetParamMacroPacked(unsigned gmParamIdx, int dim, const std::string &prefix) const;

    std::vector<std::string> GenParamIdxExprByIndex(unsigned gmParamIdx, int dim, const std::string &prefix) const;

    std::vector<std::string> GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const;

    std::string GenMemUBTransfer(bool isCopyUBToGM) const;
    std::string GenMemUBSpillIntoGM(bool isCopyUBToGM) const;
    std::string GenVectorScalarOpByMode(VecScalMode mode) const;
    std::string GenVectorScalarOpScalarMode() const;
    std::string GenCubeOp(bool zeroC) const;
    std::string GenCmpOp() const;

    std::string PrintDupOp(const PrintDupOpParam &param) const;
    std::string PrintDupOpDynUnaligned(const PrintDupOpParam &param) const;
    std::string PrintDupOpStatic(const PrintDupOpParam &param) const;
    std::string PrintDupTileTensor(const PrintDupOpParam &param) const;

    std::string PrintRowMaxline(const PrintUnaryParam &param) const;
    std::string PrintRowMaxlineTileTensor() const;
    std::string PrintRowMaxlineDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintRowMaxlineStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceEx(const PrintUnaryParam &param) const;
    std::string PrintReduceExStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceSum(const PrintUnaryParam &param) const;
    std::string PrintReduceSumStatic(const PrintUnaryParam &param) const;

    std::string PrintVcopy(const PrintUnaryParam &param) const;
    std::string PrintVcopyStatic(const PrintUnaryParam &param) const;

    std::string PrintVnchwconv(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvTileTensor() const;

    std::string PrintCompact(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam &param,
        std::vector<std::string> &gmShapeExpr, std::vector<std::string> &gmOffsetExpr) const;
    std::string PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam &param) const;

    std::string PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const;
    std::string PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param &param) const;
    std::string PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param &param) const;
    std::string PrintL1CopyInTileTensor(const PrintMemCopyWithL1Param &param) const;

    std::string PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam &param) const;
    std::vector<std::string> GetGmOffsetForTileTensor(unsigned gmIdx, bool isSpillingToGM = false) const;

    std::string PrintGather(const PrintGatherParam &param) const;
    std::string PrintGatherDynamicUnaligned(const PrintGatherParam &param) const;
    std::string PrintGatherStatic(const PrintGatherParam &param) const;

    std::string PrintBinaryScalar(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const;

    std::string PrintUnary(const PrintUnaryParam &param) const;
    std::string PrintUnaryTileTensor() const;
    std::string PrintUnaryDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintUnaryStatic(const PrintUnaryParam &param) const;

    std::string PrintBitwiseNot() const;

    SortParam PrepareSortParam() const;
    TiledSortParam PrepareTiledSortParam() const;
    std::string PrintTileSortTileTensor() const;
    std::string PrintTiledSortDynamicUnaligned(const TiledSortParam &param) const;
    std::string PrintTiledMrgSortDynamicUnaligned(const TiledSortParam &param) const;
    std::string PrintSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintSortStatic(const SortParam &param) const;
    std::string PrintSortTileTensor() const;
    std::string PrintBitSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintBitSortStatic(const SortParam &param) const;
    std::string PrintMrgSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintMrgSortStatic(const SortParam &param) const;

    std::string PrintBinaryStatic(const PrintBinaryParam &param) const;
    std::string PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const;
    std::string PrintBinaryTileTensor() const;
    std::string PrintBinary(const PrintBinaryParam &param) const;

    std::string PrintBinaryTmpTileTensor() const;
    std::string PrintBinaryTmp(const PrintBinaryTmpParam &param) const;

    std::string PrintBinaryBrcStatic(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrc(const PrintBinaryBrcParam &param) const;

    std::string PrintTransposeDataMove(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveLayout(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam &param) const;

    std::string PrintGatherElementDynamicUnaligned(const PrintGatherEleParam &param) const;
    std::string PrintGatherElementStatic(const PrintGatherEleParam &param) const;
    std::string PrintGatherElementTileTensor(const PrintGatherEleParam &param) const;

    std::string PrintIndexOutCast(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastStatic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam &param) const;

    std::string PrintExpandDynamicUnaligned(const PrintUnaryParam &param, int expandAxis) const;
    std::string PrintExpandLayout(int expandAxis) const;
    std::string PrintExpand(const std::string &s0Var, const std::string &dVar, const std::string &srcDtypeStr,
        const std::string &dstDtypeStr) const;
    std::string PrintOneHot(const PrintUnaryParam &param) const;
    std::string PrintOneHotLayout() const;

    DynamicParamPackMTE PrepareDynamicShapeInfoForMTE(
        int dynShapeIdx, int ShapeDim = SHAPE_DIM4, bool isGmSpill = false) const;

    std::string PrintReduceLastAxis(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintReduceLastAxisTileTensor() const;

    std::string PrintRowSumline(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintRowSumlineTileTensor() const;
    std::string PrintRowSumlineDynamicUnaligned(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintRowSumlineStatic(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintExtractStatic() const;
    std::string PrintExtractDynamicUnaligned() const;
    std::string PrintExtractTileTensor() const;

    std::string PrintCastDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintCastTileTensor() const;
    std::string PrintReduceCombine(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVectorScalarTileTensor(const PrintUnaryParam &param) const;
    std::string PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam &param) const;
    std::string PrintMemL1ToL0TileTensor() const;
    std::string PrintMatmulTileTensor(bool isAcc) const;
    std::string PrintTmove() const;
    std::string PrintL0CToL1TileTensor() const;

    std::string PrintScatterElementSOpStatic(const PrintScatterElemParam &param) const;
    std::string PrintScatterElementSOpDynamicUnaligned(const PrintScatterElemParam &param) const;
    std::string PrintScatterElementSTileTensor(const PrintScatterElemParam &param) const;
    std::string PrintScatterOpDynamicUnaligned(const PrintScatterParam &param) const;
    std::string PrintScatterTileTensor(const PrintScatterParam &param) const;

    std::string PrintIndexAddDynamicUnaligned(const PrintIndexAddParam &param) const;
    std::string PrintIndexAddTileTensor(const PrintIndexAddParam &param) const;

    std::string PrintIndexPut(const PrintIndexPutParam &param) const;
    std::string PrintIndexPutLayout(size_t indicesSize, bool accumulate) const;
    std::string PrintIndexPutDynamicUnaligned(const PrintIndexPutParam &param) const;

    std::string PrintCumSumDynamicUnaligned(const PrintCumSumParam &param) const;
    std::string PrintCumSumTileTensor(int axis) const;

    WhereParam PrepareWhereParam() const;
    void GetVarAndTypeParam(std::vector<std::string> &varExpr, std::vector<std::string> &dataTypeExpr) const;
    std::string PrintWhereOp(const WhereParam &param) const;
    std::string PrintWhereOpTileTensor(const WhereParam &param) const;

    std::string PrintCmpTileTensor() const;
    std::string PrintLogicalAndTileTensor() const;
    std::string PrintLogicalNotTileTensor() const;
    std::string PrintBitwiseShiftTensor() const;
    std::string PrintBitwiseShiftScalar() const;

    void InitOpsGenMap();
    void InitScalaOpsMap();
    void InitMTEOpsMap();
    void InitVecOpsMap();
    void InitCubeOpsMap();
    void InitDistOpsMap();
    void InitPerfOpsMap();
    void InitAICPUOpsMap();

    std::string PrintCoord(size_t dim, const std::string &coord) const;
    std::string PrintTensorForCopyBetweenGM(unsigned operandIdx, unsigned gmIdx, const std::string &gmVarName) const;
    void FillParamWithFullShape(std::vector<std::string> &paramList, const std::vector<int64_t> &input) const;
    void FillParamWithShapeExceptFirst(std::vector<std::string> &paramList, const std::vector<int64_t> &input) const;

    const std::unordered_map<Opcode, std::function<std::string()>> mteFixPipeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> unaryOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> binaryOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> compositeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> sortOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> cubeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> syncOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> distributeOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> gatherScatterOps_;

    const std::unordered_map<Opcode, std::function<std::string()>> normalVecOps_;

    std::unordered_map<Opcode, std::function<std::string()>> perfOps_;

    std::unordered_map<Opcode, std::function<std::string()>> aicpuOps_;

    std::unordered_map<Opcode, std::function<std::string()>> opsGenMap_;

    std::shared_ptr<ForBlockManager> forBlkMgr_;

    // <parameter index, tensor name>
    std::unordered_map<int, std::string> tensorNames_;

    mutable std::map<unsigned, std::reference_wrapper<std::string>> tempVarsMap;
    mutable unsigned tempKey = 0;
    template <typename T>
    void AppendLocalBufVarOffsetInOrderImpl() const {
        if (!tempVarsMap.empty()) {
            AppendLocalBufferVarOffset(tempVarsMap);
            tempVarsMap.clear();
            tempKey = 0;
        }
    }

    template <typename T, typename FirstArg, typename... RestArgs>
    void AppendLocalBufVarOffsetInOrderImpl(FirstArg &first_arg, RestArgs &...rest_args) const {
        static_assert(
            std::is_same_v<std::remove_reference_t<FirstArg>, T>, "All arguments must be T (default: std::string)!");
        tempVarsMap.emplace(tempKey++, std::ref(first_arg));
        AppendLocalBufVarOffsetInOrderImpl<T>(rest_args...);
    }
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_CLOUDNPU_H

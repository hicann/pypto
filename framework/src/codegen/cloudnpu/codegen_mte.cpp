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
 * \file codegen_mte.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"
#include <string>
#include <iterator>
namespace npu::tile_fwk {
DynamicParamPackMTE CodeGenOpCloudNPU::PrepareDynamicShapeInfoForMTE(
    int dynShapeIdx, int shapeDim, bool isGmSpill) const {
    DynamicParamPackMTE pack;
    int dim = static_cast<int>(rawShape[dynShapeIdx].size());
    if (isGmSpill) {
        for (auto s : shape[dynShapeIdx]) {
            pack.gmShapeExpr.emplace_back(std::to_string(s));
        }
    } else {
        pack.gmShapeExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_RAW_SHAPE);
    }
    FillIntVecWithDummyInHead<std::string>(pack.gmShapeExpr, shapeDim - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(pack.gmShapeExpr).c_str());

    if (offsetFromAttr[dynShapeIdx][ID0].IsValid()) {
        pack.gmOffsetExpr = GenSymbolicArgument(offsetFromAttr[dynShapeIdx]);
    } else {
        pack.gmOffsetExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_OFFSET);
    }
    FillIntVecWithDummyInHead<std::string>(pack.gmOffsetExpr, shapeDim - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(pack.gmOffsetExpr).c_str());

    for (const auto &gs : pack.gmShapeExpr) {
        pack.paramList.emplace_back(gs);
    }
    if (!isGmSpill) {
        for (const auto &go : pack.gmOffsetExpr) {
            pack.paramList.emplace_back(go);
        }
    }

    return pack;
}

std::string CodeGenOpCloudNPU::GenMemL1CopyIn() const {
    return GenMemCopyCube(false, 0);
}

std::string CodeGenOpCloudNPU::GenMemL1CopyOut() const {
    return GenMemCopyCube(true, 0);
}

std::string CodeGenOpCloudNPU::GenMemL0CCopyOut() const {
    return GenMemCopyCube(true, 0);
}

std::string CodeGenOpCloudNPU::GenMemCopyCube(bool isLocalToGM, unsigned uf) const {
    unsigned gmIdx = isLocalToGM ? 0 : 1;
    if (operand[gmIdx] != SYMBOL_STACK_BASE) {
        return GenMemCopyVar(isLocalToGM, uf);
    }

    return GenMemL1SpillIntoGM(isLocalToGM, uf);
}

std::string CodeGenOpCloudNPU::GenMemL1SpillIntoGM(bool isLocalToGM, unsigned int uf) const {
    unsigned gmIdx = isLocalToGM ? 0 : 1;
    unsigned l1Idx = isLocalToGM ? 1 : 0;
    DataType gmDtype = operandDtype[gmIdx];
    DataType l1Dtype = operandDtype[l1Idx];

    std::vector<std::string> addrTypeHead(ID2);
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[l1Idx] = GetAddrTypeByOperandType(BUF_L1);

    // Query ub variable name
    std::vector<int64_t> gmOffset = offset[gmIdx];
    std::vector<int64_t> l1TileOffset = offset[l1Idx];
    unsigned l1Offset = l1TileOffset[ID0] * l1TileOffset[ID1];
    std::vector<std::string> addrExpr(ID2);
    addrExpr[gmIdx] = GenGMAddrExprWithOffset(GM_STACK_BASE, gmIdx);
    addrExpr[l1Idx] = GenAddrExpr(sm->QueryVarNameByTensorMagic(operandWithMagic[l1Idx]), l1Offset);

    std::vector<int64_t> gmShape = rawShape[gmIdx];
    ALOG_INFO_F("GenMemOpL1 op: %s, gmShape: %s", tileOpName.c_str(), IntVecToStr(gmShape).c_str());

    std::vector<int64_t> l1Shape = rawShape[l1Idx];
    ALOG_INFO_F("GenMemOpL1 op: %s, l1Shape: %s", tileOpName.c_str(), IntVecToStr(l1Shape).c_str());

    // Spilling out scene only support 2-dim shape
    ASSERT(l1Shape.size() == SHAPE_DIM2) << "L1 shape must be 2-dim!";
    int tileShape0 = l1Shape[ID0];
    int tileShape1 = l1Shape[ID1];

    std::string typeExpr[ID2];
    typeExpr[gmIdx] = DataType2CCEStr(gmDtype);
    typeExpr[l1Idx] = DataType2CCEStr(l1Dtype);

    int ret{0};
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (isDynamicFunction) {
        // The layout of gm stack data is continuous, so just use dynValidShape[ID1] as gm stride to match the
        // implementation of "Copy TileOp" in order to make gm gap value be zero in copy intrinsic.
        auto dynValidShape = dynamicValidShape[l1Idx];
        std::ostringstream oss;
        oss << tileOpName << "<" << typeExpr[gmIdx] << ", " << typeExpr[l1Idx] << ">"
            << "((" << addrTypeHead[ID0] << " " << typeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
            << "(" << addrTypeHead[ID1] << " " << typeExpr[ID1] << "*)" << addrExpr[ID1] << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << uf << ");\n";

        return oss.str();
    } else {
        ret = sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
            tileOpName.c_str(), typeExpr[gmIdx].c_str(), typeExpr[l1Idx].c_str(), tileShape0, tileShape1, gmShape[ID0],
            gmShape[ID1], addrTypeHead[ID0].c_str(), typeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
            addrTypeHead[ID1].c_str(), typeExpr[ID1].c_str(), addrExpr[ID1].c_str(), uf);
    }
    ASSERT(ret >= 0) << "sprintf_s failed in genMemOp_L1, return value:" << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::GenL0CToUBTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(offset[ToUnderlying(MISOIdx::SRC0_IDX)]);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), coordCp);
    int64_t copyInMode = 1;
    if (opAttrs.count(OP_ATTR_PREFIX + "is_nz")) {
        copyInMode = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "is_nz"));
    }
    std::string nzVar = copyInMode ? "CopyOutMode::NZ2ND" : "CopyOutMode::NZ2NZ";
    std::ostringstream oss;
    int64_t aivId = 0;
    GetAttr(OpAttributeKey::subBlockIdx, aivId);
    oss << tileOpName << "<" << nzVar << ">" << "(" << dstTensor << ", " << src0Tensor << ", " << coord << ", " << aivId
        << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintMemL1ToL0TileTensor() const {
    bool isTrans = false;
    if ((opCode == Opcode::OP_L1_TO_L0_BT) || (opCode == Opcode::OP_L1_TO_L0_AT)) {
        isTrans = true;
    }
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    unsigned srcOffset0 = 0;
    unsigned srcOffset1 = 0;
    auto dynoffset = offsetFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];
    if (!dynoffset.empty()) {
        ASSERT(dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToL0 only support 2-dim!";
        srcOffset0 = dynoffset[ID0];
        srcOffset1 = dynoffset[ID1];
    }
    std::vector<std::string> l0Offset = {
        SymbolicExpressionTable::BuildExpression(srcOffset0), SymbolicExpressionTable::BuildExpression(srcOffset1)};
    // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(l0Offset);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), coordCp);
    std::ostringstream oss;
    oss << tileOpName << "<" << std::to_string(isTrans) << ">" << "(" << dstTensor << ", " << src0Tensor << ", "
        << coord << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL1ToL0() const {
    if (isSupportLayout) {
        return PrintMemL1ToL0TileTensor();
    }

    std::string paramStr = GenParamsStr();

    std::vector<int64_t> l1Shape = this->rawShape[ID1];
    ALOG_INFO_F("GenMemL1ToL0 %s, l1Shape is %s", tileOpName.c_str(), IntVecToStr(l1Shape).c_str());

    std::vector<int64_t> l0Shape = this->rawShape[ID0];
    ALOG_INFO_F("GenMemL1ToL0 %s, l0Shape is %s", tileOpName.c_str(), IntVecToStr(l0Shape).c_str());

    unsigned srcOffset0 = 0;
    unsigned srcOffset1 = 0;
    auto dynoffset = offsetFromAttr[ID1];
    if (!dynoffset.empty()) {
        ASSERT(dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToL0 only support 2-dim!";
        srcOffset0 = dynoffset[ID0];
        srcOffset1 = dynoffset[ID1];
    }

    unsigned srcShape0 = l1Shape[ID0];
    unsigned srcShape1 = l1Shape[ID1];
    unsigned tileShape0 = l0Shape[ID0];
    unsigned tileShape1 = l0Shape[ID1];

    std::string dtypeStr = DataType2CCEStr(operandDtype[ID0]);
    auto l1ShapeDyn = dynamicValidShape[ID1];
    auto l0ShapeDyn = dynamicValidShape[ID0];

    std::ostringstream oss;
    if (isDynamicFunction) {
        oss << tileOpName << "<" << dtypeStr << ", " << srcOffset0 << ", " << srcOffset1 << ">"
            << "(" << paramStr << ", " << SymbolicExpressionTable::BuildExpression(l0ShapeDyn[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(l0ShapeDyn[ID1]) << ", "
            << SymbolicExpressionTable::BuildExpression(l1ShapeDyn[ID0]) << ", "
            << SymbolicExpressionTable::BuildExpression(l1ShapeDyn[ID1]) << ");\n";
    } else {
        oss << tileOpName << "<" << dtypeStr << ", " << tileShape0 << ", " << tileShape1 << ", " << srcOffset0 << ", "
            << srcOffset1 << ", " << srcShape0 << ", " << srcShape1 << ">"
            << "(" << paramStr << ");\n";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTmove() const {
    std::ostringstream oss;
    std::vector<int64_t> tmpoffset(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), 0);
    std::string coordCp = WrapParamByParentheses(tmpoffset);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), coordCp);
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    oss << tileOpName << "<" << 0 << ">" << "(" << dstTensor << ", " << src0Tensor << ", " << coord << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL1ToBt() const {
    if (isSupportLayout) {
        return PrintTmove();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    auto dynValidShape = dynamicValidShape[ID0];
    auto dynoffset = offsetFromAttr[ID1];
    // only support 2-dim shape
    ASSERT(dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToBt only support 2-dim!";

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.emplace_back(dstDtypeStr);
    // only need the valid offset of tail axis
    ASSERT(dynoffset[ID1].IsValid()) << "GenMemL1ToBt offset is invalid";
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynoffset[ID1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(uint64_t)" + dstVar;
    std::string src = "(";
    src.append(GetAddrTypeByOperandType(BUF_L1)).append(" ").append(srcDtypeStr).append("*)").append(srcVar);
    paramList.insert(paramList.end(), {dst, src});
    // only need the valid shape of tail axis
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">" << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenMemUBSpillIntoGM(bool isCopyUBToGM) const {
    unsigned gmIdx = isCopyUBToGM ? 0 : 1;
    unsigned ubIdx = isCopyUBToGM ? 1 : 0;

    std::vector<std::string> addrTypeHead(ID2);
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[ubIdx] = GetAddrTypeByOperandType(BUF_UB);

    // Query ub variable name
    std::vector<std::string> addrExpr(ID2);
    addrExpr[ubIdx] = sm->QueryVarNameByTensorMagic(operandWithMagic[ubIdx]);
    // In spilling out scene,  GM offset is added after "GMStackBase" var.
    // "GMStackBase" is a base address of a gm workspace which is using for spilled tensors.
    addrExpr[gmIdx] = GenGMAddrExprWithOffset(GM_STACK_BASE, gmIdx);

    std::vector<std::string> dataTypeExpr(ID2);
    dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
    dataTypeExpr[ubIdx] = DataType2CCEStr(operandDtype[ubIdx]);

    PrintMemCopyWithUBParam param = {gmIdx, ubIdx, addrTypeHead, addrExpr, dataTypeExpr, true};
    return PrintMemCopyWithUB(param);
}

std::string CodeGenOpCloudNPU::GenMemUBTransfer(bool isCopyUBToGM) const {
    unsigned gmIdx = isCopyUBToGM ? 0 : 1;
    if (operand[gmIdx] != SYMBOL_STACK_BASE) {
        return GenMemCopyVar(isCopyUBToGM, 0);
    }

    return GenMemUBSpillIntoGM(isCopyUBToGM);
}

std::string CodeGenOpCloudNPU::GenUBCopyIn() const {
    return GenMemUBTransfer(false);
}

std::string CodeGenOpCloudNPU::GenUBCopyOut() const {
    return GenMemUBTransfer(true);
}

std::string CodeGenOpCloudNPU::GenReshapeCopyIn() const {
    return GenMemCopyVar(false, 0);
}

std::string CodeGenOpCloudNPU::GenReshapeCopyOut() const {
    return GenMemCopyVar(true, 0);
}

std::string CodeGenOpCloudNPU::PrintIndexOutCastTileTensor() const {
    auto cacheMode = npu::tile_fwk::AnyCast<std::string>(opAttrs.at(OpAttributeKey::cacheMode));
    auto blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::panzBlockSize));
    int cacheModeFlag = GetCacheModeFlag(cacheMode);
    std::string dstTensor = sm->QueryTileTensorByBufVarName(GenGmParamVar(ID0));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));

    int dim = rawShape[ID0].size();
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(ID0);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(dim, coordCp);
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, src1Tensor, coord};

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({std::to_string(cacheModeFlag), std::to_string(blockSize)});
    oss << WrapParamByParentheses(tileOpParamList);
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenIndexOutCastOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::cacheMode)) << "cannot get cacheMode attr";
    ASSERT(opAttrs.count(OpAttributeKey::panzBlockSize)) << "cannot get panzBlockSize attr";
    auto cacheMode = npu::tile_fwk::AnyCast<std::string>(opAttrs.at(OpAttributeKey::cacheMode));
    auto blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::panzBlockSize));
    unsigned gmIdx = 0;
    unsigned localIdx = 1;

    std::vector<std::string> addrExpr(ID2);
    addrExpr[gmIdx] = GenGmParamVar(gmIdx);

    std::vector<int64_t> src0OriginShape = this->originShape[ID1];
    std::vector<int64_t> src1OriginShape = this->originShape[ID2];
    std::vector<int64_t> src0RawShape = this->rawShape[ID1];
    std::vector<int64_t> src1RawShape = this->rawShape[ID2];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};

    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    AppendLocalBufferVarOffset({
        {localIdx, std::ref(s0Var)}
    });

    std::vector gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("genIndexOutCastOp gm shape: %s", IntVecToStr(gmShape).c_str());

    std::vector<int64_t> s0os = NormalizeShape(src0OriginShape, SHAPE_DIM4);
    std::vector<int64_t> gms = NormalizeShape(gmShape, SHAPE_DIM4);
    std::vector<int64_t> s0rs = NormalizeShape(src0RawShape, SHAPE_DIM4);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM4);
    std::string blockSizeStr = std::to_string(blockSize);

    return PrintIndexOutCast(
        {s0Var, s1Var, addrExpr, gms, s0os, s0rs, src1OriginShape, s1rs, dataTypeExpr, cacheMode, blockSizeStr});
}

std::string CodeGenOpCloudNPU::PrintL0CToL1TileTensor() const {
    auto l1Offset = offsetFromAttr[ID0];
    std::vector<std::string> dstOffset;
    for (auto tmpOffset : l1Offset) {
        dstOffset.emplace_back(SymbolicExpressionTable::BuildExpression(tmpOffset));
    }
    auto l0cOffset = offsetFromAttr[ID1];
    std::vector<std::string> srcOffset;
    for (auto tmpOffset : l0cOffset) {
        srcOffset.emplace_back(SymbolicExpressionTable::BuildExpression(tmpOffset));
    }
    std::string coordCpDst = WrapParamByParentheses(dstOffset);
    std::string coordDst = PrintCoord(rawShape[ID0].size(), coordCpDst);
    std::string coordCpSrc = WrapParamByParentheses(srcOffset);
    std::string coordSrc = PrintCoord(rawShape[ID1].size(), coordCpSrc);
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = srcTensor;
    int64_t isAcc = 0;
    int64_t reluMode = 0;

    GetAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    std::string nzVar = "CopyOutMode::NZ2NZ";
    std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
    std::string storeConfig = WrapParamByAngleBrackets(storeConfigList);
    npu::tile_fwk::Element scaleValue = npu::tile_fwk::Element(DataType::DT_UINT64, 0);

    GetAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    if ((!scaleValue.GetUnsignedData()) && ((operandDtype[ID1] == DT_INT32) && (operandDtype[ID0] == DT_FP16))) {
        src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    }
    std::vector<std::string> tileOpParamList = {
        dstTensor, srcTensor, src1Tensor, coordDst, coordSrc, std::to_string(scaleValue.GetUnsignedData())};
    std::ostringstream oss;
    oss << tileOpName << "<" << "TileOp::TStoreConfig" << storeConfig << ">";
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL0CToL1() const {
    if (isSupportLayout) {
        return PrintL0CToL1TileTensor();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    auto dstValidShape = dynamicValidShape[ID0];
    auto srcValidShape = dynamicValidShape[ID1];
    auto dynValidShapeFromAttr = dynValidShapeFromOpAttr[ID0];

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back(srcDtypeStr);
    int64_t reluMode = 0;
    GetAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    paramList.emplace_back(std::to_string(reluMode));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_L1) + " " + dstDtypeStr + "*)" + dstVar;
    std::string src = "(" + GetAddrTypeByOperandType(BUF_L0C) + " " + srcDtypeStr + "*)" + srcVar;
    paramList.insert(paramList.end(), {dst, src});
    for (const auto &realShape : dynValidShapeFromAttr) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(realShape));
    }
    for (const auto &dstShape : dstValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dstShape));
    }
    auto l1Offset = offsetFromAttr[ID0];
    for (auto dstOffset : l1Offset) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dstOffset));
    }
    for (auto srcShape : srcValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(srcShape));
    }
    auto l0COffset = offsetFromAttr[ID1];
    for (auto srcOffset : l0COffset) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(srcOffset));
    }
    npu::tile_fwk::Element scaleValue = npu::tile_fwk::Element(DataType::DT_UINT64, 0);
    GetAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    paramList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">" << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenUBToL1TileTensor() const {
    if (!isSupportLayout) {
        return "";
    }
    std::vector<int64_t> dstOffset = this->offset[ID0];
    std::string coordCp = WrapParamByParentheses(dstOffset);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[ID0].size(), coordCp);
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, coord};

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenUBToUBND2NZTileTensor() const {
    if (!isSupportLayout) {
        return "";
    }

    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor};

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintIndexOutCast(const PrintIndexOutCastParam &param) const {
    if (isSupportLayout) {
        return PrintIndexOutCastTileTensor();
    }
    if (isSupportDynamicAligned) {
        return PrintIndexOutCastDynamic(param);
    } else if (isDynamicFunction) {
        return PrintIndexOutCastDynamicUnaligned(param);
    }
    return PrintIndexOutCastStatic(param);
}

int CodeGenOpCloudNPU::GetCacheModeFlag(const std::string &cacheMode) const {
    const int PA_BNSD = 0;
    const int PA_NZ = 1;
    const int PA_BSND = 2;
    int cacheModeFlag = PA_BNSD;
    if (cacheMode == "PA_NZ") {
        cacheModeFlag = PA_NZ;
    } else if (cacheMode == "PA_BSND") {
        cacheModeFlag = PA_BSND;
    }
    return cacheModeFlag;
}
// template <typename T,unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned src1Shape0,
// unsigned src1Shape1, unsigned GmShape0, unsigned GmShape1, unsigned GmShape2, unsigned GmShape3>
// TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ int32_t* index, unsigned Offset0, unsigned
// Offset1) {
std::string CodeGenOpCloudNPU::PrintIndexOutCastStatic(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    std::vector<int64_t> &src0OriginShape = param.src0OriginShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1OriginShape = param.src1OriginShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                          std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.insert(paramList.end(), {std::to_string(gmShape[ID2]), std::to_string(gmShape[ID3])});
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintIndexOutCastDynamic(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    std::vector<int64_t> &src0OriginShape = param.src0OriginShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1OriginShape = param.src1OriginShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                          std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);

    auto src0ValidShape = dynamicValidShape[ID1];
    FillIntVecWithDummyInHead<SymbolicScalar>(src0ValidShape, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    auto src1ValidShape = dynamicValidShape[ID2];

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(paramList.end(), {SymbolicExpressionTable::BuildExpression(src0ValidShape[ID0]),
                                          SymbolicExpressionTable::BuildExpression(src0ValidShape[ID1]),
                                          SymbolicExpressionTable::BuildExpression(src0ValidShape[ID3])});
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(src1ValidShape[ID0]));
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(src1ValidShape[ID1]));
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

// In static shape scene, GM Offset is already calculated and added to GM Addr in host side, so TileOp do not need
// GM offset
std::string CodeGenOpCloudNPU::GenMemCopyVar(bool isCopyLocalToGM, unsigned uf) const {
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;
    OperandType localType = operandType[localIdx];
    std::vector<std::string> addrTypeHead(ID2);
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[localIdx] = GetAddrTypeByOperandType(localType);

    std::vector<int64_t> gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("gmShape is %s", IntVecToStr(gmShape).c_str());
    std::vector<int64_t> localRawShape = this->rawShape[localIdx];
    ALOG_INFO_F("localRawShape is %s", IntVecToStr(localRawShape).c_str());

    std::vector<std::string> addrExpr(ID2);
    addrExpr[localIdx] = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
    addrExpr[gmIdx] = GenGmParamVar(gmIdx);

    std::vector<std::string> dataTypeExpr(ID2);
    dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
    dataTypeExpr[localIdx] = DataType2CCEStr(operandDtype[localIdx]);

    if (localType == BUF_L0C) {
        return PrintMemCopyWithL0C({uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, localRawShape, dataTypeExpr});
    } else if (localType == BUF_L1) {
        return PrintMemCopyWithL1({uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, localRawShape, dataTypeExpr});
    } else if (localType == BUF_UB) {
        PrintMemCopyWithUBParam param = {gmIdx, localIdx, addrTypeHead, addrExpr, dataTypeExpr, false};
        return PrintMemCopyWithUB(param);
    }

    ASSERT(0) << "GenMemCopyVar: cannot support current localType!!!" << localType;
    return {};
}

std::string CodeGenOpCloudNPU::PrintTensorForCopyBetweenGM(
    unsigned operandIdx, unsigned gmIdx, const std::string &gmVarName) const {
    std::string tensor =
        operandIdx == gmIdx ? sm->QueryTileTensorByBufVarName(gmVarName) : QueryTileTensorNameByIdx(operandIdx);
    return tensor;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam &param) const {
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);
    std::string gmVarName = GenGmParamVar(param.gmIdx);
    std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
    std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
    int64_t reluMode = 0;
    GetAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    std::string src1Tensor = srcTensor;
    int64_t nzValue = 0;
    int64_t isAcc = 0;
    auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape);
    GetAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
    GetAttr("op_attr_is_nz", nzValue);
    std::string nzVar = nzValue ? "CopyOutMode::NZ2NZ" : "CopyOutMode::NZ2ND";
    std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
    std::string storeConfig = PrintParams({"<", ">"}, storeConfigList, ", ");

    npu::tile_fwk::Element scaleValue = npu::tile_fwk::Element(DataType::DT_UINT64, 0);
    if (!isAcc) {
        GetAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    }

    if ((!scaleValue.GetUnsignedData()) &&
        ((operandDtype[param.localIdx] == DT_INT32) && (operandDtype[param.gmIdx] == DT_FP16))) {
        src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    }

    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, src1Tensor, coord, outerValueStr, innerValueStr,
        std::to_string(scaleValue.GetUnsignedData())};
    std::ostringstream oss;
    oss << tileOpName << "<" << "TileOp::TStoreConfig" << storeConfig << ">";
    oss << PrintParams({"(", ")"}, tileOpParamList, ", ");
    oss << STMT_END;
    return oss.str();
}
std::string CodeGenOpCloudNPU::PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const {
    if (isSupportLayout) {
        return PrintMemCopyWithL0CTileTensor(param);
    }
    if (isDynamicFunction) {
        return PrintMemCopyWithL0CDynamic(param);
    }
    return PrintMemCopyWithL0CStatic(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    const std::vector<int64_t> &localRawShape = param.localRawShape;
    const std::vector<SymbolicScalar> &outputOffset = offsetFromAttr[gmIdx];
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(originShape[localIdx][ID0], localRawShape[ID0]);
    int oriTileShape1 = std::min(originShape[localIdx][ID1], localRawShape[ID1]);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
        "%s<%s, %s, %u, %u, %d, %d, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n", tileOpName.c_str(),
        dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0], localRawShape[ID1],
        gmShape[ID0], gmShape[ID1], SymbolicExpressionTable::BuildExpression(outputOffset[ID0]).c_str(),
        SymbolicExpressionTable::BuildExpression(outputOffset[ID1]).c_str(), oriTileShape0, oriTileShape1,
        addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(),
        dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(), uf);
    ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_L0C), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam &param,
    std::vector<std::string> &gmShapeExpr, std::vector<std::string> &gmOffsetExpr) const {
    std::ostringstream os;
    std::vector<std::string> paramList;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    paramList.emplace_back(dataTypeExpr[gmIdx]);
    paramList.emplace_back(dataTypeExpr[localIdx]);
    int64_t nzValue = 0;
    int64_t isAcc = 0;
    auto ret = GetAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
    if (ret) {
        paramList.emplace_back(std::to_string(isAcc));
    }
    ret = GetAttr("op_attr_is_nz", nzValue);
    if (ret && nzValue == 1) {
        paramList.emplace_back("false");
    } else {
        paramList.emplace_back("true");
    }
    int64_t reluMode = 0;
    ret = GetAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
    paramList.emplace_back(std::to_string(reluMode));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[0] + " " + dataTypeExpr[0] + "*)" + addrExpr[0];
    std::string src = "(" + addrTypeHead[1] + " " + dataTypeExpr[1] + "*)" + addrExpr[1];
    paramList.insert(paramList.end(), {dst, src});
    auto dynValidShape = dynamicValidShape[localIdx];
    for (int i = 0; i < SHAPE_DIM2; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynValidShape[i]));
    }
    paramList.emplace_back(gmShapeExpr[0]);
    paramList.emplace_back(gmOffsetExpr[0]);
    int64_t outerValue = 0, innerValue = 0;
    ret = GetAttr("op_attr_curH", outerValue);
    ret = GetAttr("op_attr_curW", innerValue);
    auto gmShapeExprByIndex = GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    std::string outerValueStr = outerValue == 0 ? gmShapeExprByIndex[0] : std::to_string(outerValue);
    std::string innerValueStr = innerValue == 0 ? gmShapeExprByIndex[1] : std::to_string(innerValue);
    paramList.insert(paramList.end(), {outerValueStr, innerValueStr, std::to_string(param.uf)});
    npu::tile_fwk::Element scaleValue = npu::tile_fwk::Element(DataType::DT_UINT64, 0);
    ret = GetAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
    if (!isAcc) {
        paramList.emplace_back(std::to_string(scaleValue.GetUnsignedData()));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<int64_t> &localRawShape = param.localRawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(originShape[localIdx][ID0], localRawShape[ID0]);
    int oriTileShape1 = std::min(originShape[localIdx][ID1], localRawShape[ID1]);

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    int printRet{0};

    if (isSupportDynamicAligned) {
        printRet =
            sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %s, %s, %u);\n",
                tileOpName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
                localRawShape[ID1], oriTileShape0, oriTileShape1, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(),
                addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(),
                gmShapeExpr[ID0].c_str(), gmOffsetExpr[ID0].c_str(), uf);
        ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_L0C), return value:" << printRet;
        return buffer;
    }

    return PrintL0CCopyOutDynamicUnalign(param, gmShapeExpr, gmOffsetExpr);
}

std::pair<std::string, std::string> CodeGenOpCloudNPU::GetOuterInnerValueStr(
    unsigned gmIdx, const std::vector<int64_t> &gmShape) const {
    int64_t outerValue = 0;
    int64_t innerValue = 0;
    GetAttr("op_attr_outer_value", outerValue);
    GetAttr("op_attr_inner_value", innerValue);
    auto gmShapeExprByIndex = GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    std::string outerFromGmShape = isDynamicFunction ? gmShapeExprByIndex[0] : std::to_string(gmShape[0]);
    std::string innerFromGmShape = isDynamicFunction ? gmShapeExprByIndex[1] : std::to_string(gmShape[1]);
    std::string outerValueStr = outerValue == 0 ? outerFromGmShape : std::to_string(outerValue);
    std::string innerValueStr = innerValue == 0 ? innerFromGmShape : std::to_string(innerValue);
    return {outerValueStr, innerValueStr};
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1TileTensor(const PrintMemCopyWithL1Param &param) const {
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
    // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);
    std::string gmVarName = GenGmParamVar(param.gmIdx);
    std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
    std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
    auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape);

    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, coord, outerValueStr, innerValueStr};
    int64_t copyInMode = -1;
    std::string cpModeStr = "";
    const int64_t ND2ND = 0;
    const int64_t ND2NZ = 1;
    if (opAttrs.count(OP_ATTR_PREFIX + "copy_in_mode")) {
        copyInMode = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "copy_in_mode"));
    }
    int64_t nzValue = 0;
    auto ret = GetAttr(OP_ATTR_PREFIX + "is_nz", nzValue);
    if (copyInMode == ND2ND) {
        cpModeStr = "CopyInMode::ND2ND";
    } else if (copyInMode == ND2NZ) {
        cpModeStr = "CopyInMode::ND2NZ";
    } else if (ret && nzValue) {
        cpModeStr = "CopyInMode::NZ2NZ";
    } else {
        cpModeStr = "CopyInMode::ND2NZ";
    }
    std::ostringstream oss;
    oss << tileOpName << "<" << cpModeStr << ">";
    oss << PrintParams({"(", ")"}, tileOpParamList, ", ");
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const {
    if (isSupportLayout) {
        return PrintMemCopyWithL1TileTensor(param);
    }
    if (isDynamicFunction) {
        return PrintMemCopyWithL1Dynamic(param);
    }
    return PrintMemCopyWithL1Static(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    const std::vector<int64_t> &localRawShape = param.localRawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    std::string opName = tileOpName;
    char addrBuffer[BUFFER_SIZE_1024] = "";
    char oriAddrBuffer[BUFFER_SIZE_1024] = "";

    int printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
    ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    int64_t nzValue = 0;
    auto ret = GetAttr("op_attr_is_nz", nzValue);
    if (ret && nzValue == 1) {
        opName = "TileOp::L1CopyInNZ2NZ";
        std::string curAddrBuffer =
            "((__gm__ GMTensorInfo*)(oriAddrParam) + " + std::to_string(paramLocation[gmIdx]) + ")->Addr";
        printRet = sprintf_s(
            oriAddrBuffer, BUFFER_SIZE_1024, "(__gm__ %s*)%s", dataTypeExpr[ID1].c_str(), curAddrBuffer.c_str());
        ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(gmIdx, gmShape);
        printRet =
            sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %u, %u, %d, %d, %s, %s>((%s %s*)%s, (%s %s*)%s, %s, %u);\n",
                opName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
                localRawShape[ID1], gmShape[ID0], gmShape[ID1], outerValueStr.c_str(), innerValueStr.c_str(),
                addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(),
                dataTypeExpr[ID1].c_str(), addrBuffer, oriAddrBuffer, uf);
        ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar, return value:" << printRet;
    } else {
        std::vector<SymbolicScalar> gmOffset = this->offsetFromAttr[gmIdx];
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet =
            sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %u, %u, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
                opName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), localRawShape[ID0],
                localRawShape[ID1], SymbolicExpressionTable::BuildExpression(gmOffset[ID0]).c_str(),
                SymbolicExpressionTable::BuildExpression(gmOffset[ID1]).c_str(), gmShape[ID0], gmShape[ID1],
                addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(),
                dataTypeExpr[ID1].c_str(), addrBuffer, uf);
    }
    ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param &param) const {
    std::ostringstream oss;

    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<int64_t> &localRawShape = param.localRawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::string opName = tileOpName;
    std::string addrBuffer = addrExpr[ID1];

    int64_t nzValue = 0;
    auto ret = GetAttr(OP_ATTR_PREFIX + "is_nz", nzValue);
    if (ret && nzValue == 1) {
        opName = tileOpName + "NZ2NZ";
        auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(gmIdx, param.gmShape);
        if (isSupportDynamicAligned) {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << localRawShape[ID0]
                << ", " << localRawShape[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr
                << ", " << uf << ");\n";
        } else {
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr << ", " << uf << ");\n";
        }
    } else {
        if (isSupportDynamicAligned) {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << localRawShape[ID0]
                << ", " << localRawShape[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        } else {
            int64_t copyInMode = 1;
            std::string cpModeStr = "";
            const int64_t ND2ND = 0;
            if (opAttrs.count(OP_ATTR_PREFIX + "copy_in_mode")) {
                copyInMode = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "copy_in_mode"));
            }
            if (copyInMode == ND2ND) {
                cpModeStr = ", CopyInMode::ND2ND";
            }
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << cpModeStr << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID0]) << ", "
                << SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]) << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        }
    }

    return oss.str();
}

// When ub tensor spilling to GM occurred, the spilling unit is entire raw shape of ub tensor.
// So ub offset is always zero under this scene, do not need to calculate anymore.
std::string CodeGenOpCloudNPU::PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const {
    unsigned localIdx = param.localIdx;
    std::vector<std::string> &addrExpr = param.addrExpr;
    if (isSupportLayout) {
        return PrintMemCopyWithUBTileTensor(param);
    }
    if (isSupportDynamicAligned) {
        AppendLocalBufferVarOffset({
            {localIdx, addrExpr[localIdx]}
        });
        return PrintMemCopyWithUBDynamic(param);
    }
    if (isDynamicFunction) {
        return PrintMemCopyWithUBDynamicSupportUnaligned(param);
    }
    AppendLocalBufferVarOffset({
        {localIdx, addrExpr[localIdx]}
    });
    return PrintMemCopyWithUBStatic(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam &param) const {
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> os = NormalizeShape(originShape[localIdx], SHAPE_DIM5);
    std::vector<int64_t> dstStride = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::vector<int64_t> srcStride = NormalizeShape(rawShape[ID1], SHAPE_DIM5);

    // Support int64
    if (dataTypeExpr[localIdx] == "int64_t") {
        constexpr int numAlign = 2;
        dataTypeExpr[localIdx] = "int32_t";
        os[SHAPE_DIM5 - 1] *= numAlign;
        srcStride[SHAPE_DIM5 - 1] *= numAlign;
        dstStride[SHAPE_DIM5 - 1] *= numAlign;
    }
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
        "%s<%s, %u, %u, %u, %u, %u, /*dst stride*/ %u, %u, %u, %u,"
        "/*src stride*/ %u, %u, %u, %u %s>((%s %s*)%s, (%s %s*)%s);\n",
        tileOpName.c_str(), dataTypeExpr[localIdx].c_str(), os[ID0], os[ID1], os[ID2], os[ID3], os[4], dstStride[ID1],
        dstStride[ID2], dstStride[ID3], dstStride[4], srcStride[ID1], srcStride[ID2], srcStride[ID3], srcStride[4],
        GenOpAttr().c_str(), addrTypeHead[ID0].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID0].c_str(),
        addrTypeHead[ID1].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID1].c_str());
    ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_UB), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam &param) const {
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> newOriginShape = originShape[localIdx];
    FillIntVecWithDummyInHead<int64_t>(newOriginShape, MAX_DIM - originShape[localIdx].size(), 1);
    const std::vector<int64_t> &localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, param.isSpillIntoGM);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (auto ts : newOriginShape) {
        paramList.emplace_back(std::to_string(ts));
    }
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam &param) const {
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::vector<std::string> &addrTypeHead = param.addrTypeHead;
    const std::vector<std::string> &addrExpr = param.addrExpr;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    std::vector<SymbolicScalar> newDynamicShape = dynamicValidShape[localIdx];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynamicShape, MAX_DIM - dynamicValidShape[localIdx].size(), 1);
    const std::vector<int64_t> &localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, param.isSpillIntoGM);
    std::vector<std::string> gmShapeExpr = paramPack.gmOffsetExpr;
    std::vector<std::string> gmOffsetExpr = paramPack.gmOffsetExpr;

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    if (localIdx == 0) { // means op is COPY_IN
        if (isPartialMem[localIdx]) {
            paramList.emplace_back("true");
        }
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    for (auto ts : newDynamicShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(ts));
    }
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());
    auto startOffset = GetOperandStartOffset(localIdx);
    if (!startOffset.ConcreteValid() || startOffset.Concrete() != 0) {
        paramList.emplace_back(startOffset.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::GenLoadOp() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType srcDtype = operandDtype[ID1];
    const DataType offsetsDtype = operandDtype[ID2];
    ASSERT(dstDtype == srcDtype) << "src and dst dtype must be same!";

    std::string srcVar = GenGmParamVar(0);
    std::string offsetsVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    auto dstRawShapes = rawShape[ID0];
    auto offsetsRawShapes = rawShape[ID2];
    auto dstOriShapes = dynamicValidShape[ID0];
    auto offsetsOriShapes = dynamicValidShape[ID2];
    ASSERT(dstRawShapes == offsetsRawShapes) << "raw shape must be same!";
    ASSERT(dstOriShapes.size() == offsetsOriShapes.size()) << "ori shape must be same!";

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string srcDtypeStr = DataType2CCEStr(srcDtype);
    std::string offsetsDtypeStr = DataType2CCEStr(offsetsDtype);

    ASSERT(offsetsDtypeStr == "int64_t" || offsetsDtypeStr == "int32_t") << "offsets dtype must be int32_t or int64_t";

    int ret = -1;
    if (dstRawShapes.size() == SHAPE_DIM2) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %s, %lld>((__ubuf__ %s *)%s, (__gm__ %s *)%s, (__ubuf__ %s *)%s, %s, %s);\n", tileOpName.c_str(),
            dstDtypeStr.c_str(), offsetsDtypeStr.c_str(), dstRawShapes[ID1], dstDtypeStr.c_str(), dstVar.c_str(),
            srcDtypeStr.c_str(), srcVar.c_str(), offsetsDtypeStr.c_str(), offsetsVar.c_str(),
            dstOriShapes[ID0].Dump().c_str(), dstOriShapes[1].Dump().c_str());
    } else if (dstRawShapes.size() == SHAPE_DIM3) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %s, %lld, %lld>((__ubuf__ %s *)%s, (__gm__ %s *)%s, (__ubuf__ %s *)%s, %s, %s, %s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), offsetsDtypeStr.c_str(), dstRawShapes[ID1], dstRawShapes[ID2],
            dstDtypeStr.c_str(), dstVar.c_str(), srcDtypeStr.c_str(), srcVar.c_str(), offsetsDtypeStr.c_str(),
            offsetsVar.c_str(), dstOriShapes[ID0].Dump().c_str(), dstOriShapes[ID1].Dump().c_str(),
            dstOriShapes[ID2].Dump().c_str());
    } else {
        ASSERT(false) << "unsupport dim " << dstRawShapes.size() << " , only support 2 or 3 now.";
    }

    ASSERT(ret >= 0) << "GenLoadOp sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}

std::vector<std::string> CodeGenOpCloudNPU::GetGmOffsetForTileTensor(unsigned gmIdx, bool isSpillingToGM) const {
    int dim = static_cast<int>(rawShape[gmIdx].size());
    std::vector<std::string> gmOffsetExpr;
    if (isSpillingToGM || functionType == FunctionType::STATIC) {
        return std::vector<std::string>(dim, "0");
    }

    if (offsetFromAttr[gmIdx][ID0].IsValid()) {
        return GenSymbolicArgument(offsetFromAttr[gmIdx]);
    }

    return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam &param) const {
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx, param.isSpillIntoGM);
    // e.g. ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
    std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);

    std::string gmVarName =
        param.isSpillIntoGM ? GenGMAddrExprWithOffset(GM_STACK_BASE, param.gmIdx) : GenGmParamVar(param.gmIdx);
    std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
    std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, coord};

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL1ToFB() const {
    if (isSupportLayout) {
        return PrintTmove();
    }
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    auto dynValidShape = dynamicValidShape[ID0];
    auto dynoffset = offsetFromAttr[ID1];
    ASSERT(dynoffset.size() == SHAPE_DIM2) << "GenMemL1ToFB only support 2-dim!";

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    // only need the valid offset of tail axis
    ASSERT(dynoffset[ID1].IsValid()) << "GenMemL1TFB offset is invalid";
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynoffset[ID1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_FIX) + " " + srcDtypeStr + "*)" + dstVar;
    std::string src = "(" + GetAddrTypeByOperandType(BUF_L1) + " " + srcDtypeStr + "*)" + srcVar;
    paramList.insert(paramList.end(), {dst, src});
    // only need the valid shape of tail axis
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynValidShape[ID1]));
    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName << "<" << templateParam << ">" << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenGMAddrExprWithOffset(const std::string &addrExpr, unsigned gmIdx) const {
    // gm offset of spilling workspace is calculated by pass, the value is saved in dim 0.
    SymbolicScalar gmOffset = this->offsetFromAttr[gmIdx][ID0];
    bool isZero = gmOffset.IsValid() && gmOffset.ConcreteValid() && gmOffset.Concrete() == 0;

    std::ostringstream oss;
    if (isZero) {
        oss << addrExpr;
    } else {
        oss << "((__gm__ uint8_t*)" << addrExpr << " + " << SymbolicExpressionTable::BuildExpression(gmOffset) << ")";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::GenAddrExpr(const std::string &addrExpr, unsigned offsetParam) const {
    std::ostringstream oss;
    if (offsetParam != 0) {
        oss << addrExpr << " + 0x" << std::hex << offsetParam;
    } else {
        oss << addrExpr;
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintGatherInL1TileTensor() const {
    std::string srcVar = sm->QueryTileTensorByBufVarName(GenGmParamVar(ID1));
    std::string offsetsVar = sm->QueryTileTensorByBufVarName(GenGmParamVar(ID2));
    std::string blockTableVar = sm->QueryTileTensorByBufVarName(GenGmParamVar(ID3));
    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    int64_t blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("op_attr_blocksize"));

    auto startOffset = opAttrs.at(OpAttributeKey::startOffset);
    ASSERT(startOffset.HasValue() && (startOffset.Type() == typeid(int64_t)))
        << "GenGatherInL1 startOffset must be int64_t!";
    auto srcColumnStartOffset = npu::tile_fwk::AnyCast<int64_t>(startOffset);
    std::string srcCoordCp = WrapParamByParentheses({std::to_string(srcColumnStartOffset)});
    std::string srcCoord = PrintCoord(SHAPE_DIM1, srcCoordCp);

    auto offsetsStartOffsets = GenParamIdxExprByIndex(ID2, SHAPE_DIM2, PREFIX_STR_OFFSET);
    std::string offsetCoordCp = WrapParamByParentheses(offsetsStartOffsets);
    std::string offsetCoord = PrintCoord(SHAPE_DIM2, offsetCoordCp);

    auto blockTableStartOffsets = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_OFFSET);
    std::string blockTableCoordCp = WrapParamByParentheses(blockTableStartOffsets);
    std::string blockTableCoord = PrintCoord(SHAPE_DIM2, blockTableCoordCp);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({std::to_string(blockSize)});
    oss << WrapParamByParentheses({dstVar, srcVar, blockTableVar, offsetsVar, srcCoord, offsetCoord, blockTableCoord});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGatherInL1() const {
    if (isSupportLayout) {
        return PrintGatherInL1TileTensor();
    }
    const DataType dstDtype = operandDtype[ID0];
    const DataType srcDtype = operandDtype[ID1];
    const DataType offsetsDtype = operandDtype[ID2];
    ASSERT(dstDtype == srcDtype) << "dstDtype and srcDtype must be same!";

    std::string srcVar = GenGmParamVar(ID1);
    std::string offsetsVar = GenGmParamVar(ID2);
    std::string blockTableVar = GenGmParamVar(ID3);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    auto dstRawShapes = rawShape[ID0];
    auto srcRawShapes = rawShape[ID1];
    auto offsetsRawShapes = rawShape[ID2];
    auto dstOriShapes = dynamicValidShape[ID0];
    ASSERT(srcRawShapes.size() == SHAPE_DIM2) << "GenGatherInL1 only support 2-dim!";
    ASSERT(dstRawShapes.size() == SHAPE_DIM2) << "GenGatherInL1 only support 2-dim!";
    ASSERT(offsetsRawShapes.size() == SHAPE_DIM2) << "GenGatherInL1 only support 2-dim!";
    ASSERT(dstOriShapes.size() == SHAPE_DIM2) << "GenGatherInL1 only support 2-dim!";

    auto offsetsStartOffsets = GenParamIdxExprByIndex(ID2, SHAPE_DIM2, PREFIX_STR_OFFSET);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string srcDtypeStr = DataType2CCEStr(srcDtype);
    std::string offsetsDtypeStr = DataType2CCEStr(offsetsDtype);
    std::string blockTableDtypeStr = DataType2CCEStr(operandDtype[ID3]);

    ASSERT(dstDtypeStr == srcDtypeStr);
    ASSERT(offsetsDtypeStr == "int64_t" || offsetsDtypeStr == "int32_t");
    ASSERT(opAttrs.find("op_attr_blocksize") != opAttrs.end()) << "GenGatherOp: There is nop axis attribute here";
    const int64_t blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("op_attr_blocksize"));

    auto startOffset = opAttrs.at(OpAttributeKey::startOffset);
    ASSERT(startOffset.HasValue() && (startOffset.Type() == typeid(int64_t)))
        << "GenGatherInL1 startOffset must be int64_t!";
    auto srcColumnStartOffset = npu::tile_fwk::AnyCast<int64_t>(startOffset);
    auto blockTableGMStride = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    auto blockTableStartOffsets = GenParamIdxExprByIndex(ID3, SHAPE_DIM2, PREFIX_STR_OFFSET);

    auto ret = sprintf_s(buffer, sizeof(buffer),
        "%s<%s, %s, %s, %lld, %lld, %lld, %lld>((__cbuf__ %s *)%s, %s, %s, (__gm__ %s *)%s, %lld, (__gm__ %s *)%s, "
        "(__gm__ %s *)%s, %s, %s, %s, %s, %s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), offsetsDtypeStr.c_str(), blockTableDtypeStr.c_str(), dstRawShapes[ID0],
        offsetsRawShapes[ID1], srcColumnStartOffset, blockSize, dstDtypeStr.c_str(), dstVar.c_str(),
        dstOriShapes[ID0].Dump().c_str(), dstOriShapes[ID1].Dump().c_str(), srcDtypeStr.c_str(), srcVar.c_str(),
        srcRawShapes[1], offsetsDtypeStr.c_str(), offsetsVar.c_str(), blockTableDtypeStr.c_str(), blockTableVar.c_str(),
        offsetsStartOffsets[ID0].c_str(), offsetsStartOffsets[ID1].c_str(), blockTableGMStride[ID1].c_str(),
        blockTableStartOffsets[ID0].c_str(), blockTableStartOffsets[ID1].c_str());

    ASSERT(ret >= 0) << "GenGatherInL1 sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}
/**
 * 辅助函数，对 axis 参数进行归一化
 * example:
 * parma [a,b]
 * axis 0
 * 归一化后
 * parma [1,1,a,b]
 * axis 2
 *
 */
inline int NormalizeAxis(int axis, int paramDim) {
    return axis + (SHAPE_DIM4 - paramDim);
}
/**
 * 归一化的 gather 参数维度
 * example:
 * param [a,b]
 * indices [c]
 * axis 1
 * result [a,c]
 * 归一化后:
 * [1,1,a,b]
 * [1,c]
 * axis 3
 * result  [1,1,a,1,c]
 * 处理逻辑:
 * 1. 根据result的形状，还原出来的 param 和 indices 的维度
 * 2. param 归一化四维，indices 归一化到 两维
 * 3. 重新拼装处 result 形状
 */
template <typename T>
void NormalizeGatherShape(std::vector<T> &rawShape, const int paramDim, const int indicesDim, const int axis) {
    static_assert((std::is_same_v<T, int64_t> || std::is_same_v<T, SymbolicScalar>), "类型错误");
    std::vector<T> paramShape{};
    std::vector<T> indicesShape{};
    indicesShape.assign(rawShape.begin() + axis, rawShape.begin() + axis + indicesDim);
    paramShape.assign(rawShape.begin(), rawShape.begin() + axis);
    paramShape.push_back(-1);
    paramShape.insert(paramShape.end(), rawShape.begin() + axis + indicesDim, rawShape.end());
    if constexpr (std::is_same_v<T, int64_t>) {
        paramShape = NormalizeShape(paramShape, SHAPE_DIM4);
        indicesShape = NormalizeShape(indicesShape, SHAPE_DIM2);
    } else if constexpr (std::is_same_v<T, SymbolicScalar>) {
        FillIntVecWithDummyInHead<SymbolicScalar>(paramShape, SHAPE_DIM4 - paramDim, 1);
        FillIntVecWithDummyInHead<SymbolicScalar>(indicesShape, SHAPE_DIM2 - indicesDim, 1);
    }
    rawShape = paramShape;
    int normalizedAxis = NormalizeAxis(axis, paramDim);
    rawShape.erase(rawShape.begin() + normalizedAxis);
    rawShape.insert(rawShape.begin() + normalizedAxis, indicesShape.begin(), indicesShape.end());
}
void HelpNormalize(std::vector<size_t> &index, int axis, int paramDim) {
    size_t delNum = NUM4 - paramDim;
    index.erase(index.begin() + delNum);
    axis = NormalizeAxis(axis, paramDim);
    index.insert(index.begin() + axis, delNum);
}
std::string CodeGenOpCloudNPU::PrintGatherDynamicUnaligned() const {
    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::string resultDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string paramDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    ASSERT(resultDtypeStr == paramDtypeStr);
    const int64_t axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    const int paramDim = paramRawShapes.size();
    const int indicesDim = indicesRawShapes.size();
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    auto normalizedOutputRawShapes = outputRawShapes;
    NormalizeGatherShape<int64_t>(normalizedOutputRawShapes, paramDim, indicesDim, axis);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(paramDtypeStr);
    paramList.emplace_back(indicesDtypeStr);
    paramList.emplace_back(std::to_string(NormalizeAxis(axis, paramDim)));
    std::transform(normalizedOutputRawShapes.begin() + 1, normalizedOutputRawShapes.end(), back_inserter(paramList),
        [](int x) { return std::to_string(x); });

    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string paramVar = GenGmParamVar(paramIndex);
    std::string indicesVar = GenGmParamVar(indicesIndex);
    std::string outputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string outputParamStr = "(__ubuf__ " + resultDtypeStr + "*)" + outputVar;
    std::string paramParamStr = "(__gm__ " + paramDtypeStr + "*)" + paramVar;
    std::string indicesParamStr = "(__gm__ " + indicesDtypeStr + "*)" + indicesVar;
    paramList.emplace_back(outputParamStr);
    paramList.emplace_back(paramParamStr);
    paramList.emplace_back(indicesParamStr);
    NormalizeGatherShape<SymbolicScalar>(outputValidShapes, paramDim, indicesDim, axis);

    std::transform(outputValidShapes.begin(), outputValidShapes.end(), back_inserter(paramList),
        [](SymbolicScalar x) { return SymbolicExpressionTable::BuildExpression(x); });

    auto paramGMStride = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_RAW_SHAPE);
    auto paramStartOffsets = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(paramGMStride, SHAPE_DIM4 - paramDim, std::string("1"));
    FillIntVecWithDummyInHead<std::string>(paramStartOffsets, SHAPE_DIM4 - paramDim, std::string("0"));
    paramList.insert(paramList.end(), paramGMStride.begin() + 1, paramGMStride.end());
    paramList.insert(paramList.end(), paramStartOffsets.begin(), paramStartOffsets.end());

    auto indicesGMStride = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_RAW_SHAPE);
    auto indicesStartOffsets = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(indicesGMStride, SHAPE_DIM2 - indicesDim, std::string("1"));
    FillIntVecWithDummyInHead<std::string>(indicesStartOffsets, SHAPE_DIM2 - indicesDim, std::string("0"));
    paramList.insert(paramList.end(), indicesGMStride.begin() + 1, indicesGMStride.end());
    paramList.insert(paramList.end(), indicesStartOffsets.begin(), indicesStartOffsets.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    paramList.clear();
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}
std::string CodeGenOpCloudNPU::PrintGatherLayout() const {
    // constexpr int paramIndex = 0;
    // constexpr int indicesIndex = 1;
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    size_t paramDim = paramValidShapes.size();
    size_t indicesDim = indicesValidShapes.size();
    const int64_t axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    std::vector<size_t> helpIndex = {ID0, ID1, ID2, ID3, ID4};
    if (indicesDim == 1 && axis != 0) {
        HelpNormalize(helpIndex, axis, paramDim);
    }
    auto paramOffsetSymbol = GenGetParamMacroPacked(ID1, paramDim, PREFIX_STR_OFFSET);
    auto indicesOffsetSymbol = GenGetParamMacroPacked(ID2, indicesDim, PREFIX_STR_OFFSET);
    std::string coordCpparamOffset = WrapParamByParentheses(paramOffsetSymbol);
    std::string coordCpindicesOffset = WrapParamByParentheses(indicesOffsetSymbol);
    std::string coord4Param = PrintCoord(paramDim, coordCpparamOffset);
    std::string coord4Indices = PrintCoord(indicesDim, coordCpindicesOffset);

    std::string outputTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string paramTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string indicesTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::vector<std::string> paramList;
    paramList.emplace_back(std::to_string(NormalizeAxis(axis, paramDim)));
    std::transform(
        helpIndex.begin(), helpIndex.end(), back_inserter(paramList), [](size_t x) { return std::to_string(x); });
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    std::vector<std::string> tileOpParamList = {outputTensor, paramTensor, indicesTensor, coord4Param, coord4Indices};
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGatherOp() const {
    if (isSupportLayout) {
        return PrintGatherLayout();
    }
    if (isDynamicFunction) {
        return PrintGatherDynamicUnaligned();
    }
    ASSERT(false) << "Gather operator does not support static graph";
    return "";
}

std::string CodeGenOpCloudNPU::PrintGatherInUBLayout() const {
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    constexpr int blockTableIndex = 3;
    constexpr int paramDim = 2;
    constexpr int indicesDim = 2;
    constexpr int blockTableDim = 2;

    auto paramOffsetSymbol = GenGetParamMacroPacked(paramIndex, paramDim, PREFIX_STR_OFFSET);
    auto indicesOffsetSymbol = GenGetParamMacroPacked(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    auto blockTableOffsetSymbol = GenGetParamMacroPacked(blockTableIndex, blockTableDim, PREFIX_STR_OFFSET);

    std::string coordCpparamOffset = WrapParamByParentheses(paramOffsetSymbol);
    std::string coordCpindicesOffset = WrapParamByParentheses(indicesOffsetSymbol);
    std::string coordCpblockTableOffset = WrapParamByParentheses(blockTableOffsetSymbol);
    std::string coord4Param = PrintCoord(paramDim, coordCpparamOffset);
    std::string coord4Indices = PrintCoord(indicesDim, coordCpindicesOffset);
    std::string coord4BlockTable = PrintCoord(blockTableDim, coordCpblockTableOffset);
    std::string outputTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string paramTensor = sm->QueryTileTensorByBufVarName(GenGmParamVar(ToUnderlying(MISOIdx::SRC0_IDX)));
    std::string indicesTensor = sm->QueryTileTensorByBufVarName(GenGmParamVar(ToUnderlying(MISOIdx::SRC1_IDX)));
    std::string pageTableTensor = sm->QueryTileTensorByBufVarName(GenGmParamVar(ToUnderlying(MISOIdx::SRC2_IDX)));
    std::vector<std::string> paramList;
    ASSERT(opAttrs.find(OpAttributeKey::blockSize) != opAttrs.end()) << "GenGatherOp: There is nop axis attribute here";
    const int64_t blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::blockSize));
    paramList.emplace_back(std::to_string(blockSize));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    std::vector<std::string> tileOpParamList = {
        outputTensor, paramTensor, indicesTensor, pageTableTensor, coord4Param, coord4Indices, coord4BlockTable};
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}
std::string CodeGenOpCloudNPU::PrintGatherInUBDynamicUnaligned() const {
    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::string resultDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string paramDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string blockTableDtypeStr = DataType2CCEStr(operandDtype[ID3]);
    ASSERT(resultDtypeStr == paramDtypeStr);
    ASSERT(opAttrs.find(OpAttributeKey::blockSize) != opAttrs.end()) << "GenGatherOp: There is nop axis attribute here";
    const int64_t blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::blockSize));
    auto outputRawShapes = rawShape[ID0];
    auto paramRawShapes = rawShape[ID1];
    auto indicesRawShapes = rawShape[ID2];
    auto outputValidShapes = dynamicValidShape[ID0];
    auto paramValidShapes = dynamicValidShape[ID1];
    auto indicesValidShapes = dynamicValidShape[ID2];
    constexpr int paramDim = 2;
    constexpr int indicesDim = 2;
    constexpr int blockTableDim = 2;
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(paramDtypeStr);
    paramList.emplace_back(indicesDtypeStr);
    paramList.emplace_back(blockTableDtypeStr);
    paramList.emplace_back(std::to_string(outputRawShapes[0]));
    paramList.emplace_back(std::to_string(outputRawShapes[1]));
    paramList.emplace_back(std::to_string(blockSize));
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    constexpr int paramIndex = 1;
    constexpr int indicesIndex = 2;
    constexpr int blockTableIndex = 3;
    std::string paramVar = GenGmParamVar(paramIndex);
    std::string indicesVar = GenGmParamVar(indicesIndex);
    std::string blockTableVar = GenGmParamVar(blockTableIndex);
    std::string outputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string outputParamStr = "(__ubuf__ " + resultDtypeStr + "*)" + outputVar;
    std::string paramParamStr = "(__gm__ " + paramDtypeStr + "*)" + paramVar;
    std::string indicesParamStr = "(__gm__ " + indicesDtypeStr + "*)" + indicesVar;
    std::string blockTableParamStr = "(__gm__ " + blockTableDtypeStr + "*)" + blockTableVar;
    paramList.emplace_back(outputParamStr);
    paramList.emplace_back(paramParamStr);
    paramList.emplace_back(indicesParamStr);
    paramList.emplace_back(blockTableParamStr);
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(outputValidShapes[1]));

    auto paramGMStride = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_RAW_SHAPE);
    auto paramStartOffsets = GenParamIdxExprByIndex(paramIndex, paramDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(paramGMStride[1]);
    paramList.emplace_back(paramStartOffsets[0]);
    paramList.emplace_back(paramStartOffsets[1]);

    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(outputValidShapes[0]));
    auto indicesGMStride = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_RAW_SHAPE);
    auto indicesStartOffsets = GenParamIdxExprByIndex(indicesIndex, indicesDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(indicesGMStride[1]);
    paramList.emplace_back(indicesStartOffsets[0]);
    paramList.emplace_back(indicesStartOffsets[1]);
    auto blockTableGMStride = GenParamIdxExprByIndex(blockTableIndex, blockTableDim, PREFIX_STR_RAW_SHAPE);
    auto blockTableStartOffsets = GenParamIdxExprByIndex(blockTableIndex, blockTableDim, PREFIX_STR_OFFSET);
    paramList.emplace_back(blockTableGMStride[1]);
    paramList.emplace_back(blockTableStartOffsets[0]);
    paramList.emplace_back(blockTableStartOffsets[1]);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    paramList.clear();
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::GenGatherInUB() const {
    if (isSupportLayout) {
        return PrintGatherInUBLayout();
    }
    if (isDynamicFunction) {
        return PrintGatherInUBDynamicUnaligned();
    }
    ASSERT(false) << "Gather operator does not support static graph";
    return "";
}
} // namespace npu::tile_fwk

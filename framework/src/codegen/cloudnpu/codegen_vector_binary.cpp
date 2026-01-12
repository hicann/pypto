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
 * \file codegen_vector.cpp
 * \brief
 */

#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen_op_cloudnpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
std::string GetBrcOprandIdxStr(int64_t brcbOperandIdx) {
    ALOG_INFO_F("input brcbOperandIdx is %d", brcbOperandIdx);
    std::string ret = "TileOp::";
    switch (brcbOperandIdx) {
        case ToUnderlying(BroadcastOperand::NONE): ret.append("BroadcastOperand::NONE"); break;
        case ToUnderlying(BroadcastOperand::LEFT_OPERAND): ret.append("BroadcastOperand::LEFT_OPERAND"); break;
        case ToUnderlying(BroadcastOperand::RIGHT_OPERAND): ret.append("BroadcastOperand::RIGHT_OPERAND"); break;
        default: ret.append("BroadcastOperand::NONE");
    }
    return ret;
}

std::string CodeGenOpCloudNPU::PrintBinaryStatic(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> os1 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*OS1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os1[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    int64_t brcOperandIdx = 0;
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        paramList.emplace_back(GetBrcOprandIdxStr(brcOperandIdx));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::vector<SymbolicScalar> dynSrcShape0 = dynamicValidShape[ID1];
    std::vector<SymbolicScalar> dynSrcShape1 = dynamicValidShape[ID2];

    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape0, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape1, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    int64_t brcOperandIdx = 0;
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        paramList.emplace_back(GetBrcOprandIdxStr(brcOperandIdx));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    for (auto dynShape : dynSrcShape0) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    for (auto dynShape : dynSrcShape1) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryTileTensor() const {
    std::string dstTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(MISOIdx::DST_IDX)]);
    std::string src0Tensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC0_IDX)]);
    std::string src1Tensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC1_IDX)]);
    std::vector<std::string> tileOpCallParamList = {dstTensor, src0Tensor, src1Tensor};

    std::vector<std::string> templateParamList;
    int64_t brcOperandIdx = 0;
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        templateParamList.emplace_back(GetBrcOprandIdxStr(brcOperandIdx));
    }

    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBinary(const PrintBinaryParam &param) const {
    if (isSupportLayout) {
        return PrintBinaryTileTensor();
    }
    if (isDynamicFunction) {
        return PrintBinaryDynamicUnaligned(param);
    }
    return PrintBinaryStatic(param);
}

std::string CodeGenOpCloudNPU::GenBinaryOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector src0RawShape = this->rawShape[ID1];
    ALOG_INFO_F("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);

    auto offset0 = GetOperandStartOffset(ID0);
    auto offset1 = GetOperandStartOffset(ID1);
    auto offset2 = GetOperandStartOffset(ID2);
    if (!offset0.ConcreteValid() || offset0.Concrete() != 0) {
        dVar += "+" + GetOperandStartOffset(ID0).Dump();
    }
    if (!offset1.ConcreteValid() || offset1.Concrete() != 0) {
        s0Var += "+" + GetOperandStartOffset(ID1).Dump();
    }
    if (!offset2.ConcreteValid() || offset2.Concrete() != 0) {
        s1Var += "+" + GetOperandStartOffset(ID2).Dump();
    }
    return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}

std::string CodeGenOpCloudNPU::PrintBinaryBrcStatic(const PrintBinaryBrcParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> brcParamList;
    brcParamList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s1[i]));
    }
    brcParamList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(brcParamList, ", ");

    brcParamList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    brcParamList.emplace_back(dst);
    brcParamList.emplace_back(src0);
    brcParamList.emplace_back(src1);
    brcParamList.emplace_back(tmp);

    std::string tiloOpCallParam = JoinString(brcParamList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    paramList.emplace_back("/*isCombineAxis*/");
    paramList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src0, src1, tmp});
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryBrc(const PrintBinaryBrcParam &param) const {
    if (isDynamicFunction) {
        return PrintBinaryBrcDynamicUnaligned(param);
    }
    return PrintBinaryBrcStatic(param);
}

std::string CodeGenOpCloudNPU::GenBinaryWithBrc() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector src0RawShape = this->rawShape[ID2];
    std::vector src1RawShape = this->rawShape[ID3];
    ALOG_INFO_F("GenBinaryWithBrc %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    char buffer[256] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);

    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var, tmpVar);
    int ret = 0;
    if (opCode == Opcode::OP_ADD_BRC || opCode == Opcode::OP_SUB_BRC || opCode == Opcode::OP_MUL_BRC ||
        opCode == Opcode::OP_DIV_BRC || opCode == Opcode::OP_MAX_BRC) {
        return PrintBinaryBrc({s0Var, s1Var, dVar, tmpVar, src0DtypeStr, src1DtypeStr, dstDtypeStr, tmpDtypeStr});
    }
    ASSERT(ret >= 0) << "GenBinaryWithBrc sprintf_s failed ";
    return buffer;
}

std::string CodeGenOpCloudNPU::GenVectorScalarOp() const {
    return GenVectorScalarOpByMode(VecScalMode::VEC_MODE);
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpScalarMode() const {
    return GenVectorScalarOpByMode(VecScalMode::SCALAR_MODE);
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM3);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    std::ostringstream os;
    std::vector<std::string> binScalParmList;
    binScalParmList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(binScalParmList, ", ");
    templateParam += GenOpAttr();
    binScalParmList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());
    binScalParmList.emplace_back(dst);
    binScalParmList.emplace_back(src0);
    binScalParmList.emplace_back(scalarTmpBuffer);
    std::string tiloOpCallParam = JoinString(binScalParmList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM3 - dynamicValidShape[1].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    paramList.emplace_back("/*DstRawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*Src0RawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(scalarTmpBuffer);
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynSrcShape[i]));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVectorScalarTileTensor(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(MISOIdx::DST_IDX)]);
    std::string srcTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC0_IDX)]);
    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());

    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor};

    std::ostringstream oss;
    oss << tileOpName << "<" << dstDtypeStr << ">"
        << "(" << dstTensor << ", " << srcTensor << ", " << scalarTmpBuffer << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalar(const PrintBinaryScalarParam &param) const {
    if (isDynamicFunction) {
        return PrintBinaryScalarDynamicUnaligned(param);
    }
    return PrintBinaryScalarStatic(param);
}

std::string CodeGenOpCloudNPU::PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    auto newDynSrcValidShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    char scalarTmp[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmp, sizeof(scalarTmp), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode sprintf_s failed ";

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0S*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(" + dstDtypeStr + ")" + scalarTmp;
    paramList.insert(paramList.end(), {dst, src, tmp});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpByMode(VecScalMode mode) const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    char buffer[BUFFER_SIZE_512] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var);

    std::vector src0RawShape = this->rawShape[1];
    std::vector dstRawShape = this->rawShape[0];
    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    if (mode == VecScalMode::SCALAR_MODE) {
        // Scalar op
        return PrintBinaryScalar({s0Var, dVar, dstDtypeStr, dstDtypeStr, rawShape[0].size()});
    }

    if (opAttrs.count(npu::tile_fwk::OP_EMUOP_PREFIX + "opc")) {
        // Hack: should be optimized to memory copy in pass
        int emuopc = AnyCast<int64_t>(opAttrs.find(npu::tile_fwk::OP_EMUOP_PREFIX + "opc")->second);
        if (emuopc == npu::tile_fwk::EMUOP_TENSOR_EXTRACT) {
            int ret = sprintf_s(buffer, sizeof(buffer),
                "RUNTIME_TensorExtract(/*type=*/%s, /*mem=*/__ubuf__, /*dst*/%s, /*src*/%s);\n", dstDtypeStr.c_str(),
                dVar.c_str(), s0Var.c_str());
            ASSERT(ret >= 0) << "GenVectorScalarOpByMode " << OpcodeManager::Inst().GetOpcodeStr(opCode) << " failed "
                             << ret;
            return buffer;
        }
    }

    if (isSupportLayout) {
        return PrintVectorScalarTileTensor({s0Var, dVar, dstDtypeStr, dstDtypeStr});
    }

    if (isDynamicFunction) {
        return PrintVectorScalarOpDynamicUnalign({s0Var, dVar, dstDtypeStr, dstDtypeStr});
    }

    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());
    int ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %d, %d, %d, %d, /*DS*/ %d, %d, %d, /*S0S*/ %d, %d, %d>"
        "((__ubuf__ %s*)%s, (__ubuf__ %s*)%s, (%s)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), os0[ID0], os0[ID1], os0[ID2], os0[ID3], ds[ID1], ds[ID2], ds[ID3],
        s0[ID1], s0[ID2], s0[ID3], dstDtypeStr.c_str(), dVar.c_str(), dstDtypeStr.c_str(), s0Var.c_str(),
        dstDtypeStr.c_str(), scalarTmpBuffer.c_str());
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed "
                     << ret;
    return buffer;
}

} // namespace npu::tile_fwk

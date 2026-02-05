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

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    // func actual param
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM5 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconvTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << srcTensor << "," << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconv(const PrintUnaryTmpBuffParam &param) const {
    if (isSupportLayout) {
        return PrintVnchwconvTileTensor();
    }
    if (isDynamicFunction) {
        return PrintVnchwconvDynUnaligned(param);
    }
    return PrintVnchwconvStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxis(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;

    std::vector<int64_t> dstOriginShape = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    ALOG_INFO_F("rawShape[2] is %s", IntVecToStr(rawShape[ID2]).c_str());

    if (isSupportLayout) {
        return PrintReduceLastAxisTileTensor();
    }

    if (isDynamicFunction) {
        return PrintReduceLastAxisDynamicUnalign({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ "
        "%s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcOriginShape[ID0], srcOriginShape[ID1], srcOriginShape[ID2],
        srcOriginShape[ID3], dstRawShape[ID1], dstRawShape[ID2], dstRawShape[ID3], srcRawShape[ID1], srcRawShape[ID2],
        srcRawShape[ID3], tmpRawShape[ID3], dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(),
        tmpDtypeStr.c_str(), tmpVar.c_str());
    ASSERT(ret >= 0) << "PrintReduceLastAxis" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed "
                     << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxisTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        oss << WrapParamByAngleBrackets({lastUse});
    }
    oss << WrapParamByParentheses({dstTensor, src0Tensor, tmpTensor});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;

    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    // tmp only need the last axis shape
    paramList.emplace_back(std::to_string(tmpRawShape[ID3]));

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dstName = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcName = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmpName = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dstName, srcName, tmpName});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceCombine(const PrintUnaryTmpBuffParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> parmList;
    parmList.emplace_back(dstDtypeStr);
    // src origin shape
    for (int i = ID0; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(srcOriginShape[i]));
    }
    // dst raw shape
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(dstRawShape[i]));
    }
    // src raw shape
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(srcRawShape[i]));
    }
    parmList.emplace_back(std::to_string(tmpRawShape[ID3]));
    std::string templateParam = JoinString(parmList, ", ");
    parmList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    parmList.insert(parmList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(parmList, ", ");
    os << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintCompact(const PrintUnaryTmpBuffParam &param) const {
    return PrintCompactStatic(param);
}

std::string CodeGenOpCloudNPU::PrintRoundLayout() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::string scalarTmpBuffer = FormatFloat(extOperandVal.Cast<float>());

    std::ostringstream oss;
    oss << tileOpName.c_str() << "<float>" << "(" << dstTensor << ", " << tmpTensor << ", " << srcTensor << ", "
        << scalarTmpBuffer << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRound() const {
    ASSERT(isSupportLayout) << "Round only support tile tensor";
    return PrintRoundLayout();
}

std::string CodeGenOpCloudNPU::PrintRowSumlineStatic(const PrintUnaryTmpBuffParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1))))
        << "unsupported reduce axis" << reduceAxis;

    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(originShape[ID2], SHAPE_DIM4);

    std::vector<std::string> paramList;
    paramList.emplace_back(param.dstDtypeStr);
    FillParamWithFullShape(paramList, os);
    FillParamWithShapeExceptFirst(paramList, srcShape);
    FillParamWithShapeExceptFirst(paramList, dstShape);
    FillParamWithShapeExceptFirst(paramList, tmpShape);
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + param.dstDtypeStr + "*)" + param.dVar;
    std::string src = "(__ubuf__ " + param.srcDtypeStr + "*)" + param.s0Var;
    std::string tmp = "(__ubuf__ " + param.tmpDtypeStr + "*)" + param.tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowSumlineDynamicUnaligned(const PrintUnaryTmpBuffParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1))))
        << "unsupported reduce axis" << reduceAxis;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;

    auto dynSrcShape = dynamicValidShape[ID2];
    // adjust reduceAxis for dim4
    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - rawShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(tmpShape[i]));
    }
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    paramList.emplace_back(tmp);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintRowSumlineTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string src0Tensor = QueryTileTensorNameByIdx(ID2);
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[ID2].size()) - 1)))) << "unsupported reduce axis";
    reduceAxis += SHAPE_DIM5 - rawShape[0].size();
    std::ostringstream oss;
    oss << tileOpName << "<" << reduceAxis << ">"
        << "(" << dstTensor << ", " << src0Tensor << ", " << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowSumline(const PrintUnaryTmpBuffParam &param) const {
    if (isSupportLayout) {
        return PrintRowSumlineTileTensor();
    }
    if (isDynamicFunction) {
        return PrintRowSumlineDynamicUnaligned(param);
    }
    return PrintRowSumlineStatic(param);
}

std::string CodeGenOpCloudNPU::GenUnaryOpWithTmpBuff() const {
    // In this scenario, frontend set tmp buffer in output to optimize ooo schedule result.
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = this->rawShape[2];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s src raw shape: %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s dst raw shape: %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, tmpVar, s0Var);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (opCode == Opcode::OP_TRANSPOSE_VNCHWCONV) {
        return PrintVnchwconv({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROUND) {
        return PrintRound();
    }

    if (opCode == Opcode::OP_ROWSUMLINE) {
        return PrintRowSumline({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }
    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE ||
        opCode == Opcode::OP_ROWMIN_SINGLE) {
        return PrintReduceLastAxis({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE || opCode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE) {
        return PrintReduceCombine({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_COMPACT) {
        return PrintCompact({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    std::string ostring(buffer);
    return ostring;
}

} // namespace npu::tile_fwk

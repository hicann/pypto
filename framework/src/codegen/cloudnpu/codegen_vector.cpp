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
#include <string>

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::GenCastOp() const {
    if (isSupportLayout) {
        return PrintCastTileTensor();
    }
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = this->rawShape[ID1];
    ALOG_INFO_F("genCastOp %s, srcShape is %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[ID0];
    ALOG_INFO_F("genCastOp %s, dstShape is %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var);
    std::vector<int64_t> os = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (isDynamicFunction) {
        return PrintCastDynamicUnaligned({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u %s>((__ubuf__ %s *)%s,  (__ubuf__ %s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcDtypeStr.c_str(), os[0], os[1], os[2], os[3], ds[1], ds[2], ds[3],
        ss[1], ss[2], ss[3], GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(),
        s0Var.c_str());
    ASSERT(ret >= 0) << "GenCastOp sprintf_s failed " << ret;
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintDupOpDynUnaligned(const PrintDupOpParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &dupV = param.dupV;
    // dst origin shape
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    auto dynDstShape = dynamicValidShape[0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    for (auto dstOriShape : dynDstShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dstOriShape));
    }

    auto startOffset = GetOperandStartOffset(0);
    if (!startOffset.ConcreteValid() || startOffset.Concrete() != 0) {
        paramList.emplace_back(startOffset.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintDupOpStatic(const PrintDupOpParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dVar = param.dVar;
    const std::string &dupV = param.dupV;
    AppendLocalBufVarOffsetInOrder(dVar);
    // dst origin shape
    std::vector<int64_t> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (auto oriShape : dos) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintDupTileTensor(const PrintDupOpParam &param) const {
    const std::string &dupV = param.dupV;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));

    std::ostringstream oss;
    oss << tileOpName << "<" << dstDtypeStr << ">"
        << "(" << dstTensor << ", " << dupV << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintDupOp(const PrintDupOpParam &param) const {
    if (isSupportLayout) {
        return PrintDupTileTensor(param);
    }

    if (isDynamicFunction) {
        return PrintDupOpDynUnaligned(param);
    }
    return PrintDupOpStatic(param);
}

std::string CodeGenOpCloudNPU::GenDupOp() const {
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string dupV;
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalar = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(SymbolicScalar)))
            << npu::tile_fwk::AnyCast<SymbolicScalar>(scalar).IsValid()
            << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = npu::tile_fwk::AnyCast<SymbolicScalar>(scalar);
        dupV = SymbolicExpressionTable::BuildExpression(scalarExpr);
    } else if (dstDtypeStr == "float" || dstDtypeStr == "half" || dstDtypeStr == "bfloat16_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsFloat() << "SCALAR attribute has to have float value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<float>());
    } else if (dstDtypeStr == "int32_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsSigned() << "SCALAR attribute has to have int value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<int>());
    } else {
        ASSERT(false) << "unsupported type";
    }
    return PrintDupOp({dVar, dstDtypeStr, dupV});
}

std::string CodeGenOpCloudNPU::GenTransposeDataMove() const {
    bool isCopyLocalToGM = opCode == Opcode::OP_TRANSPOSE_MOVEOUT;
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;

    std::string localVar = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
    std::string gmVar = GenGmParamVar(gmIdx);

    std::vector<int64_t> srcShape = this->rawShape[localIdx];
    ALOG_INFO_F("GenTransposeDataMove: srcShape is %s", IntVecToStr(srcShape).c_str());
    std::vector<int64_t> gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("GenTransposeDataMove: gmShape is %s", IntVecToStr(gmShape).c_str());

    AppendLocalBufferVarOffset({
        {   gmIdx,    std::ref(gmVar)},
        {localIdx, std::ref(localVar)}
    });

    std::string localDtypeStr = DataType2CCEStr(operandDtype[localIdx]);
    std::string gmDtypeStr = DataType2CCEStr(operandDtype[gmIdx]);
    return PrintTransposeDataMove({gmIdx, localIdx, localVar, gmShape, localDtypeStr, gmDtypeStr});
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMove(const PrintTransposeDataMoveParam &param) const {
    if (isSupportLayout) {
        return PrintTransposeDataMoveLayout(param);
    }
    if (isSupportDynamicAligned) {
        return PrintTransposeDataMoveDynamic(param);
    } else if (isDynamicFunction) {
        return PrintTransposeDataMoveDynamicUnaligned(param);
    }
    return PrintTransposeDataMoveStatic(param);
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveLayout(const PrintTransposeDataMoveParam &param) const {
    std::string gmVarName = GenGmParamVar(param.gmIdx);
    std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
    std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM5 - originShape[param.localIdx].size();
    std::vector<std::string> uselessVector0;
    std::vector<std::string> uselessVector1;
    std::vector<std::string> uselessVector2;
    std::vector<std::string> gmOffsetExpr =
        GetGmOffsetForTileTensor({param.gmIdx, param.localIdx, uselessVector0, uselessVector1, uselessVector2, false});
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    int dim = static_cast<int>(rawShape[param.gmIdx].size());
    std::string coord = "Coord" + std::to_string(dim) + DIM + coordCp;

    std::ostringstream oss;
    oss << tileOpName << "<" << (transposeAxis[0] + correctionAxis) << ", " << (transposeAxis[1] + correctionAxis)
        << ">" << WrapParamByParentheses({dstTensor, srcTensor, coord}) << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam &param) const {
    const std::string &localVar = param.localVar;
    const std::string &localDtypeStr = param.localDtypeStr;
    const std::string &gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> gmShape = NormalizeShape(param.gmShape, SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;

    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(gmShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[0].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    oss << tileOpName.c_str() << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const {
    const std::string &localVar = param.localVar;
    const std::string &localDtypeStr = param.localDtypeStr;
    const std::string &gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);

    int dim = static_cast<int>(rawShape[ID0].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM4 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM4 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[1].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam &param) const {
    const int gmIdx = param.gmIdx;
    const int localIdx = param.localIdx;
    std::string gmVar = GenGmParamVar(gmIdx);

    int dim = static_cast<int>(rawShape[gmIdx].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM5 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM5 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());
    auto newDynLocalValidShape = dynamicValidShape[localIdx];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynLocalValidShape, SHAPE_DIM5 - dim, 1);

    std::vector<int64_t> localShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(param.gmDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; i++) {
        paramList.emplace_back(std::to_string(localShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM5 - originShape[localIdx].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string gm = "(__gm__ " + param.gmDtypeStr + "*)" + gmVar;
    std::string ub = "(__ubuf__ " + param.localDtypeStr + "*)" + param.localVar;

    if (gmIdx == 0) {
        paramList.insert(paramList.end(), {gm, ub});
    } else {
        paramList.insert(paramList.end(), {ub, gm});
    }

    for (auto localDynShape : newDynLocalValidShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(localDynShape));
    }
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintGatherStatic(const PrintGatherParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID1];

    std::vector<int64_t> dos = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*DOS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dos[i]));
    }
    paramList.emplace_back("/*SS*/");
    paramList.emplace_back(std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/");
    paramList.emplace_back(std::to_string(ds[ID3]));
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

std::string CodeGenOpCloudNPU::PrintGatherDynamicUnaligned(const PrintGatherParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const int64_t axis = param.axis;
    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID1];
    std::vector src1Shape = this->rawShape[ID2];

    auto mul = [](uint32_t data, const int64_t in) { return data * in; };
    std::vector<int64_t> indexShape = NormalizeShape(src1Shape, SHAPE_DIM4);

    size_t inputRank = src0Shape.size();
    size_t outputRank = dstShape.size();
    int afterAxis = inputRank - axis - 1;
    int outputUBStride = dstShape[outputRank - afterAxis - 1];
    uint32_t before = std::accumulate(src0Shape.begin(), src0Shape.begin() + axis, 1, mul);
    uint32_t after = axis == (static_cast<int64_t>(src0Shape.size() - 1)) ?
                         1 :
                         std::accumulate(src0Shape.begin() + axis + 1, src0Shape.end(), 1, mul);
    auto dynIndexShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynIndexShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*before*/");
    paramList.emplace_back(std::to_string(before));

    paramList.emplace_back("/*after*/");
    paramList.emplace_back(std::to_string(after));
    paramList.emplace_back("/*axis_shape*/");
    paramList.emplace_back(std::to_string(src0Shape[axis]));

    for (int i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(indexShape[i]));
    }

    paramList.emplace_back(std::to_string(outputUBStride));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + param.dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + param.s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + param.s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    for (int i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynIndexShape[i]));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintGather(const PrintGatherParam &param) const {
    if (isDynamicFunction) {
        return PrintGatherDynamicUnaligned(param);
    }
    return PrintGatherStatic(param);
}

std::string CodeGenOpCloudNPU::GenGatherFromUBOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    ASSERT(opAttrs.find("op_attr_axis") != opAttrs.end()) << "GenGatherOp: There is nop axis attribute here";
    const int64_t axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("op_attr_axis"));
    // shape: dst, src0, src1
    int src0Rank = shape[ID1].size();
    ASSERT(src0Rank <= RANK4) << "GenGatherOp: src0 shape rank is not supported!";

    std::vector dstShape = this->rawShape[0];

    std::vector src0Shape = this->rawShape[1];
    ALOG_INFO_F("GenGatherOp, src0 Shape is [%d,%d]", src0Shape[0], src0Shape[1]);

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var);
    return PrintGather({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr, axis});
}

std::string CodeGenOpCloudNPU::PrintGatherElementStatic(const PrintGatherEleParam &param) const {
    // Static only support 2Dim
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    std::vector<int64_t> &dstOriginShape = param.dstOriginShape;
    std::vector<int64_t> &dstRawShape = param.dstRawShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(dstOriginShape[ID0]), std::to_string(dstOriginShape[ID1])});
    paramList.emplace_back(std::to_string(src0RawShape[ID1]));
    paramList.emplace_back(std::to_string(dstRawShape[ID1]));
    paramList.emplace_back(std::to_string(param.axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintGatherElementDynamicUnaligned(const PrintGatherEleParam &param) const {
    // support 1-4 dims
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src0RawShape = NormalizeShape(param.src0RawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    for (size_t i = 1; i < src0RawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(src0RawShape[i]));
    }
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    auto dstValidShape = dynamicValidShape[ID0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dstValidShape, SHAPE_DIM4 - dstValidShape.size(), 1);
    for (int i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dstValidShape[i]));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintGatherElementTileTensor(const PrintGatherEleParam &param) const {
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string src0Tensor = QueryTileTensorNameByIdx(ID2);
    std::string src1Tensor = QueryTileTensorNameByIdx(ID3);
    std::vector<std::string> paramList;
    int axis = param.axis + SHAPE_DIM5 - param.src1RawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << dstTensor << ", " << src0Tensor << ", " << src1Tensor << ", " << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGatherElementOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID2];
    std::vector src1Shape = this->rawShape[ID3];
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);
    AppendLocalBufVarOffsetInOrder(dVar, s0Var, s1Var);

    // [case1] src0: [S2,D], src1: [B,S], axis: 0, dst: [B,S]
    std::vector<int64_t> dos = originShape[ID0];
    std::vector<int64_t> s0s = src0Shape;
    std::vector<int64_t> s1s = src1Shape;
    std::vector<int64_t> ds = dstShape;
    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    int gatherEleAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "axis");
    if (axis.HasValue()) {
        gatherEleAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    if (isSupportLayout) {
        return PrintGatherElementTileTensor({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
    }
    if (isDynamicFunction) {
        return PrintGatherElementDynamicUnaligned({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
    }
    return PrintGatherElementStatic({gatherEleAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
}
std::string CodeGenOpCloudNPU::PrintIndexPutDynamicUnaligned(const PrintIndexPutParam &param) const {
    const std::string &dstVar = param.dVar;
    const std::string &src1Var = param.s1Var;
    std::vector<std::string> src2Var = param.s2Var;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    size_t dstRank = param.gmShape.size();
    std::vector<int64_t> s1rs = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    int dim = static_cast<int>(rawShape[ID0].size());
    auto paramPack = GenParamIdxExprByIndex(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(paramPack, ID4 - dim, "1");
    bool accumulate = param.accumulate;
    
    // template param
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID3]});
    paramList.emplace_back(std::to_string(dstRank));
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(s1rs[i]));
    }
    paramList.emplace_back(std::to_string(accumulate));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // function actual params
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + src1Var;
    paramList.insert(paramList.end(), {dst, src1});
    for (size_t i = 0; i < src2Var.size(); i++) {
        std::string src2Temp = "(__ubuf__ " + dataTypeExpr[ID3] + "*)" + src2Var[i];
        paramList.emplace_back(src2Temp);
    }
    auto validShape = dynamicValidShape[ID2]; // src1
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(validShape[0]));
    paramList.insert(paramList.end(), paramPack.begin(), paramPack.end());

    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintIndexPut(const PrintIndexPutParam &param) const {
    ASSERT(isDynamicFunction) << "Only Support the DynamicUnaligned tileOp";
    return PrintIndexPutDynamicUnaligned(param);
}

std::string CodeGenOpCloudNPU::PrintIndexPutLayout(size_t indicesSize, bool accumulate) const {
    std::string gmVarName = GenGmParamVar(ID0);
    std::string dstTensor = sm->QueryTileTensorByBufVarName(gmVarName);
    std::string valuesTensor = QueryTileTensorNameByIdx(ID2);
    std::vector<std::string> paramList = {dstTensor, valuesTensor};
    for (size_t i = 0; i < SHAPE_DIM4; ++i) {
        if (i < indicesSize) {
            std::string indices = QueryTileTensorNameByIdx(ID3 + i);
            paramList.push_back(indices);
        } else {
            paramList.push_back(paramList.back());
        }
    }
    std::ostringstream oss;
    oss << tileOpName << "<" << accumulate << ", " << indicesSize << ">" << WrapParamByParentheses(paramList) << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenIndexPutOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::accumulate)) << "cannot get accumulate attr";
    ASSERT(opAttrs.count(OpAttributeKey::indicesSize)) << "cannot get indicesSize attr";
    bool accumulate = npu::tile_fwk::AnyCast<bool>(opAttrs.at(OpAttributeKey::accumulate));
    int64_t indicesSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::indicesSize));
    if (isSupportLayout) {
        return PrintIndexPutLayout(indicesSize, accumulate);
    }
    // dst:gm, s0/self:gm, s1/values:ub, s2/indices:ub
    std::string dstVar = GenGmParamVar(ID0);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::vector<std::string> s2Var;
    for (int i = 0; i < indicesSize; i++) {
        std::string s2VarTemp = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3 + i]);
        s2Var.emplace_back(s2VarTemp);
    }
    std::vector gmShape = this->rawShape[ID0];
    std::vector src1RawShape = this->rawShape[ID2];

    std::vector<std::string> dataTypeExpr;
    for (int i = 0; i < NUM4; i++) {
        dataTypeExpr.emplace_back(DataType2CCEStr(operandDtype[i]));
    }

    std::map<unsigned, std::reference_wrapper<std::string>> vars;
    vars.insert({ID1, s1Var});
    for (int i = 0; i < indicesSize; i++) {
        vars.insert({i + ID2, s2Var[i]});
    }
    AppendLocalBufferVarOffset(vars);

    return PrintIndexPut({dstVar, s1Var, s2Var, gmShape, src1RawShape, dataTypeExpr, accumulate});
}

std::string CodeGenOpCloudNPU::PrintRangeTileTensor(const std::string& startVal, const std::string& stepVal, const std::string& tileIdxExpr) const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    auto dstValidShape = dynamicValidShape[ToUnderlying(MISOIdx::DST_IDX)];
    std::vector<std::string> paramList = {
        dstTensor, SymbolicExpressionTable::BuildExpression(dstValidShape[ID0]), startVal, stepVal, tileIdxExpr};
    std::ostringstream oss;
    oss << tileOpName;
    oss << PrintParams({"(", ")"}, paramList, ", ");
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenRangeOp() const {
    auto start = opAttrs.at(OP_ATTR_PREFIX + "START");
    auto step = opAttrs.at(OP_ATTR_PREFIX + "STEP");
    std::string startVal, stepVal, tileIdxExpr;
    ASSERT(start.HasValue() && step.HasValue()) << "GenRangeOp failed ";

    switch (operandDtype[ID0]) {
        case DataType::DT_FP32:
            startVal = std::to_string(npu::tile_fwk::AnyCast<Element>(start).Cast<float>());
            stepVal = std::to_string(npu::tile_fwk::AnyCast<Element>(step).Cast<float>());
            break;
        case DataType::DT_INT32:
            startVal = std::to_string(npu::tile_fwk::AnyCast<Element>(start).Cast<int>());
            stepVal = std::to_string(npu::tile_fwk::AnyCast<Element>(step).Cast<int>());
            break;
        case DataType::DT_INT64:
            startVal = std::to_string(npu::tile_fwk::AnyCast<Element>(start).Cast<int64_t>());
            stepVal = std::to_string(npu::tile_fwk::AnyCast<Element>(step).Cast<int64_t>());
            break;
        default: ALOG_ERROR_F("GenRangeOp: Unsupport type: DataType=%d", operandDtype[ID0]); return "CG_ERROR";
    }
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalarAny = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT((scalarAny.HasValue()) && (scalarAny.Type() == typeid(SymbolicScalar)))
            << npu::tile_fwk::AnyCast<SymbolicScalar>(scalarAny).IsValid()
            << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = npu::tile_fwk::AnyCast<SymbolicScalar>(scalarAny);
        tileIdxExpr = "((int64_t)(" + SymbolicExpressionTable::BuildExpression(scalarExpr) + "))";
    }
    if (isSupportLayout) {
        return PrintRangeTileTensor(startVal, stepVal, tileIdxExpr);
    }
    // only support 1 dim
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back(std::to_string(rawShape[0][0]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + dstDtypeStr + "*)" + dVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(dynamicValidShape[ID0][ID0].Dump());
    paramList.insert(paramList.end(), {startVal, stepVal});
    paramList.emplace_back(tileIdxExpr);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintIndexAddDynamicUnaligned(const PrintIndexAddParam &param) const {
    // support 2-4 dims
    const std::string &dstVar = param.dstVar;
    const std::string &srcVar = param.srcVar;
    const std::string &indicesVar = param.indicesVar;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(param.srcRawShape, SHAPE_DIM4);
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    const Element &alpha = extOperandVal;

    // template params
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2], DataType2CCEStr(alpha.GetDataType())});
    for (size_t i = 1; i < srcRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.srcRawShape.size(); // 调用4维tileop需要切换axis
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // function actual params
    paramList.clear();
    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    std::string dst = "(" + addrType + " " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string src = "(" + addrType + " " + dataTypeExpr[ID1] + "*)" + srcVar;
    std::string indices = "(" + addrType + " " + dataTypeExpr[ID2] + "*)" + indicesVar;
    paramList.insert(paramList.end(), {dst, src, indices});
    std::string scalarTmpBuffer = FormatFloat(alpha.Cast<float>());
    paramList.emplace_back("(" + DataType2CCEStr(alpha.GetDataType()) + ")" + scalarTmpBuffer);
    auto validShape = dynamicValidShape[ID2]; // srcvalidshape
    FillIntVecWithDummyInHead<SymbolicScalar>(validShape, SHAPE_DIM4 - validShape.size(), 1);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(validShape[i]));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintIndexAddTileTensor(const PrintIndexAddParam &param) const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string idxTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC2_IDX));
    std::vector<std::string> paramList;
    int axis = param.axis + SHAPE_DIM5 - param.srcRawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    paramList.insert(paramList.end(), {dstTensor, src0Tensor, src1Tensor, idxTensor});
    const Element &alpha = extOperandVal;
    std::string scalarTmpBuffer = FormatFloat(alpha.Cast<float>());
    paramList.emplace_back("(" + DataType2CCEStr(alpha.GetDataType()) + ")" + scalarTmpBuffer);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenIndexAddOp() const {
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string selfVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string indicesVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);

    std::vector dstRawShape = this->rawShape[ID0];
    std::vector srcRawShape = this->rawShape[ID2];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID3]);
    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, srcDtypeStr, indicesDtypeStr};

    AppendLocalBufVarOffsetInOrder(dstVar, selfVar, srcVar, indicesVar);

    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    if (isSupportLayout) {
        return PrintIndexAddTileTensor({axis, dstVar, srcVar, indicesVar, dstRawShape, srcRawShape, dataTypeExpr});
    }
    return PrintIndexAddDynamicUnaligned({axis, dstVar, srcVar, indicesVar, dstRawShape, srcRawShape, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintCumSumDynamicUnaligned(const PrintCumSumParam &param) const {
    // support 2-4 dims
    const std::string &dstVar = param.dVar;
    const std::string &inputVar = param.inputVar;

    std::vector<int64_t> inputRawShape = NormalizeShape(param.inputRawShape, SHAPE_DIM4);
    const std::string *dataTypeExpr = param.dataTypeExpr;

    // template params
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0]});
    for (size_t i = 0; i < inputRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(inputRawShape[i]));
    }

    bool flag = param.flag;
    paramList.emplace_back(std::to_string(param.axis));
    paramList.emplace_back(std::to_string(flag));
    std::string templateParam = JoinString(paramList, ", ");

    // function actual params
    paramList.clear();
    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    std::string dst = "(" + addrType + " " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string input = "(" + addrType + " " + dataTypeExpr[ID1] + "*)" + inputVar;

    paramList.insert(paramList.end(), {dst, input});

    auto validShape = dynamicValidShape[ID1];
    FillIntVecWithDummyInHead<SymbolicScalar>(validShape, SHAPE_DIM4 - validShape.size(), 1);
    for (int i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(validShape[i]));
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenCumSumOp() const {
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string inputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::vector inputRawShape = this->rawShape[ID1];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string inputDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    constexpr int NumOperands = 2;
    std::string dataTypeExpr[NumOperands] = {dstDtypeStr, inputDtypeStr};
    AppendLocalBufVarOffsetInOrder(dstVar, inputVar);

    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    axis = axis + SHAPE_DIM4 - inputRawShape.size(); // 调用4维tileop需要切换axis

    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "flag")) << "cannot get flag attr";
    bool flag = npu::tile_fwk::AnyCast<bool>(opAttrs.at(OP_ATTR_PREFIX + "flag"));

    return PrintCumSumDynamicUnaligned({axis, flag, dstVar, inputVar, inputRawShape, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintScatterElementSOpStatic(const PrintScatterElemParam &param) const {
    // Static only support 2Dim
    int dstRank = shape[ToUnderlying(MISOIdx::DST_IDX)].size();
    int src1Rank = shape[ToUnderlying(MISOIdx::SRC1_IDX)].size();
    ASSERT(src1Rank == RANK2) << "GenScatterElementSOp: src1 shape rank is not supported!";
    ASSERT(dstRank == RANK2) << "GenScatterElementSOp: dst shape rank is not supported!";

    const std::string &dstVar = param.dVar;
    const std::string &src0Var = param.s0Var;
    const std::string &src1Var = param.s1Var;
    std::vector<int64_t> &dstShape = param.dstRawShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    const Element &scala = extOperandVal;

    std::vector src1Shape = this->originShape[ToUnderlying(MISOIdx::SRC1_IDX)];
    std::vector<int64_t> s1os = NormalizeShape(src1Shape, SHAPE_DIM2);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM2);
    std::vector<int64_t> drs = NormalizeShape(dstShape, SHAPE_DIM2);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)]);
    templateParams.emplace_back(std::to_string(s1rs[ToUnderlying(MISOIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(drs[ToUnderlying(MISOIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(MISOIdx::DST_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(MISOIdx::SRC0_IDX)]));
    std::string templateParamStr = JoinString(templateParams, ", ");
    templateParamStr += ", " + std::to_string(param.axis);

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)] + "*)" + src1Var);
    std::string scalarTmpBuffer = FormatFloat(scala.Cast<float>());
    callParams.emplace_back("(" + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + ")" + scalarTmpBuffer);

    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintScatterElementSOpDynamicUnaligned(const PrintScatterElemParam &param) const {
    const std::string &dstVar = param.dVar;
    const std::string &src0Var = param.s0Var;
    const std::string &src1Var = param.s1Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;
    const Element &scala = extOperandVal;
    std::string scalarDtypeBuffer = DataType2CCEStr(scala.GetDataType());
    auto dynSrc1Shape = dynamicValidShape[ToUnderlying(MISOIdx::SRC1_IDX)];
    FillIntVecWithDummyInHead<SymbolicScalar>(
        dynSrc1Shape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(MISOIdx::SRC1_IDX)].size(), 1);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)]);
    templateParams.emplace_back(scalarDtypeBuffer);
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    templateParams.emplace_back(std::to_string(axis));
    templateParams.emplace_back(std::to_string(param.scatterMode));
    std::string templateParamStr = JoinString(templateParams, ", ");

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(MISOIdx::SRC1_IDX)] + "*)" + src1Var);
    std::string scalarTmpBuffer = FormatFloat(scala.Cast<float>());
    callParams.emplace_back("(" + scalarDtypeBuffer + ")" + scalarTmpBuffer);
    for (size_t i = 0; i < SHAPE_DIM4; ++i) {
        callParams.emplace_back(SymbolicExpressionTable::BuildExpression(dynSrc1Shape[i]));
    }
    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenScatterElementSOp() const {
    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "scatter_mode")) << "cannot get scatter mode attr";
    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    int scatterMode = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "scatter_mode"));
    const DataType dstDtype = operandDtype[ToUnderlying(MISOIdx::DST_IDX)];
    const DataType src0Dtype = operandDtype[ToUnderlying(MISOIdx::SRC0_IDX)];
    const DataType src1Dtype = operandDtype[ToUnderlying(MISOIdx::SRC1_IDX)];

    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC0_IDX)]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::SRC1_IDX)]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(MISOIdx::DST_IDX)]);

    std::vector dstRawShape = this->rawShape[ToUnderlying(MISOIdx::DST_IDX)];
    std::vector src1RawShape = this->rawShape[ToUnderlying(MISOIdx::SRC1_IDX)];

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    ALOG_INFO_F("GenScatterElementSOp, dstDtypeStr%s", dstDtypeStr.c_str());
    ALOG_INFO_F("GenScatterElementSOp, src1DtypeStr%s", src1DtypeStr.c_str());

    AppendLocalBufVarOffsetInOrder(dstVar, src0Var, src1Var);

    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    if (isDynamicFunction) {
        return PrintScatterElementSOpDynamicUnaligned(
            {axis, scatterMode, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
    }
    return PrintScatterElementSOpStatic(
        {axis, scatterMode, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintScatterOpDynamicUnaligned(const PrintScatterParam &param) const {
    const std::string &dstVar = param.dVar;
    const std::string &src1Var = param.s1Var;
    const std::string &src2Var = param.s2Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> src1RawShape = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    std::vector<int64_t> src2RawShape = NormalizeShape(param.src2RawShape, SHAPE_DIM4);
    const std::vector<std::string> &dataTypeExpr = param.dataTypeExpr;

    auto dynSrc1Shape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrc1Shape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ID0]);
    templateParams.emplace_back(dataTypeExpr[ID1]);
    for (size_t i = 1; i < src1RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src1RawShape[i]));
    }
    for (size_t i = 1; i < src2RawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(src2RawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        templateParams.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.src1RawShape.size();
    templateParams.emplace_back(std::to_string(axis));
    templateParams.emplace_back(std::to_string(param.scatterMode));
    std::string templateParamStr = JoinString(templateParams, ", ");

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID1] + "*)" + src1Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ID2] + "*)" + src2Var);
    for (size_t i = 0; i < SHAPE_DIM4; ++i) {
        callParams.emplace_back(SymbolicExpressionTable::BuildExpression(dynSrc1Shape[i]));
    }
    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenScatterOp() const {
    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "scatter_mode")) << "cannot get scatter mode attr";
    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    int scatterMode = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "scatter_mode"));
    const DataType dstDtype = operandDtype[ID0];
    const DataType src1Dtype = operandDtype[ID2];
    const DataType src2Dtype = operandDtype[ID3];

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string src2Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);

    std::vector dstRawShape = this->rawShape[ID0];
    std::vector src1RawShape = this->rawShape[ID2];
    std::vector src2RawShape = this->rawShape[ID3];

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    std::string src2DtypeStr = DataType2CCEStr(src2Dtype);

    AppendLocalBufVarOffsetInOrder(dstVar, src1Var, src2Var);

    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, src1DtypeStr, src2DtypeStr};
    return PrintScatterOpDynamicUnaligned(
        {axis, scatterMode, dstVar, src1Var, src2Var, dstRawShape, src1RawShape, src2RawShape, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintExtractStatic() const {
    SymbolManager::AllocRecord src0, dst;
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::vector src0RawShape = this->rawShape[1];
    unsigned tShape0 = 0;
    unsigned tShape1 = 0;

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::vector src0Shape = this->rawShape[0];
    AppendLocalBufVarOffsetInOrder(dVar, s0Var);
    constexpr unsigned defaultDim = 1u;
    if (this->rawShape[1].size() == 1) {
        tShape1 = std::min(src0RawShape[0], shape[0][0]);
        tShape0 = defaultDim;
    } else {
        tShape0 = std::min(src0RawShape[0], shape[0][0]);
        tShape1 = std::min(src0RawShape[1], shape[0][1]);
    }
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    paramList.insert(paramList.end(), {std::to_string(tShape0), std::to_string(tShape1)});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintExtractDynamicUnaligned() const {
    SymbolManager::AllocRecord src0, dst;
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::vector src0RawShape = this->rawShape[1];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    AppendLocalBufVarOffsetInOrder(dVar, s0Var);

    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    std::vector dstShape = this->rawShape[0];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    paramList.insert(paramList.end(), {std::to_string(ds[ID1]), std::to_string(ds[ID2]), std::to_string(ds[ID3])});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();

    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    auto dynDstShape = dynamicValidShape[0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynamicValidShape[0].size(), 1);
    for (int i = 0; i < SHAPE_DIM3; ++i) {
        auto tShape = dynSrcShape[i].Min(dynDstShape[i]);
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(tShape));
    }
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintExtractTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    std::ostringstream oss;
    oss << tileOpName << "<" << GenOpAttr(false) << ">"
        << "(" << dstTensor << ", " << src0Tensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenExtractOp() const {
    if (isSupportLayout) {
        return PrintExtractTileTensor();
    }
    if (isDynamicFunction) {
        return PrintExtractDynamicUnaligned();
    }
    return PrintExtractStatic();
}

static const std::unordered_map<int, std::string> aicpuCallNumDict = {
    {AICPU_CALL_NUM_COPYOUT_RESOLVE, "AICPU_CALL_NUM_COPYOUT_RESOLVE"},
};

std::string CodeGenOpCloudNPU::GenAicpuCallOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::aicpuCall)) << "OpAttributeKey::aicpuCall not found";
    uint32_t call =
        static_cast<uint32_t>(npu::tile_fwk::AnyCast<int64_t>(opAttrs.find(OpAttributeKey::aicpuCall)->second));
    uint16_t callNum = call >> AICPU_CALL_ARG_BIT;
    uint16_t callArg = call & ((1 << AICPU_CALL_ARG_BIT) - 1);

    std::ostringstream oss;
    std::string callNumName = std::to_string(callNum);
    if (aicpuCallNumDict.count(callNum)) {
        callNumName = aicpuCallNumDict.find(callNum)->second;
    }
    oss << tileOpName << "<" << callNumName << "," << callArg << ">(GET_CURRENT_TASKID());\n";
    return oss.str();
}

void CodeGenOpCloudNPU::GetVarAndTypeParam(
    std::vector<std::string> &varExpr, std::vector<std::string> &dataTypeExpr) const {
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::resIdx)]);
    std::string tempVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::tempIdx)]);
    std::string condVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::condIdx)]);

    varExpr = {dstVar, tempVar, condVar};
    std::map<unsigned, std::reference_wrapper<std::string>> varMap;
    const unsigned idxs[] = {
        ToUnderlying(WhereOpIdx::resIdx), ToUnderlying(WhereOpIdx::tempIdx), ToUnderlying(WhereOpIdx::condIdx)};
    for (unsigned idx : idxs) {
        varMap.emplace(idx, std::ref(varExpr[idx]));
    }
    if (opCode == Opcode::OP_WHERE_ST || opCode == Opcode::OP_WHERE_TS || opCode == Opcode::OP_WHERE_TT) {
        std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::src0Idx)]);
        varExpr.emplace_back(src0Var);
        varMap.emplace(ToUnderlying(WhereOpIdx::src0Idx), std::ref(src0Var));
    }
    if (opCode == Opcode::OP_WHERE_TT) {
        std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::src1Idx)]);
        varExpr.emplace_back(src1Var);
        varMap.emplace(ToUnderlying(WhereOpIdx::src1Idx), std::ref(src1Var));
    }

    AppendLocalBufferVarOffset(varMap);

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::resIdx)]);
    std::string tempDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::tempIdx)]);
    std::string condDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::condIdx)]);
    dataTypeExpr = {dstDtypeStr, tempDtypeStr, condDtypeStr};
}

WhereParam CodeGenOpCloudNPU::PrepareWhereParam() const {
    std::vector<std::string> varExpr;
    std::vector<std::string> dataTypeExpr;
    GetVarAndTypeParam(varExpr, dataTypeExpr);
    std::vector<int64_t> ds = NormalizeShape(this->rawShape[ToUnderlying(WhereOpIdx::resIdx)], SHAPE_DIM4);
    std::vector<int64_t> c0s = NormalizeShape(this->rawShape[ToUnderlying(WhereOpIdx::condIdx)], SHAPE_DIM4);
    std::vector<int64_t> s0s = NormalizeShape(this->rawShape[ToUnderlying(WhereOpIdx::src0Idx)], SHAPE_DIM4);
    std::vector<std::string> templateList;
    templateList.emplace_back(dataTypeExpr[ToUnderlying(WhereOpIdx::resIdx)]);
    templateList.emplace_back(dataTypeExpr[ToUnderlying(WhereOpIdx::condIdx)]);
    templateList.emplace_back("/*DstRawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(ds[i]));
    }
    templateList.emplace_back("/*ConditionRawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(c0s[i]));
    }
    templateList.emplace_back("/*Src0RawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(s0s[i]));
    }

    std::vector<std::string> paramList;
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::resIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::resIdx)]);
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::tempIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::tempIdx)]);
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::condIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::condIdx)]);
    std::vector<std::string> dynParamList;
    auto dynSrcShape = dynamicValidShape[ToUnderlying(WhereOpIdx::resIdx)];
    FillIntVecWithDummyInHead<SymbolicScalar>(
        dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(WhereOpIdx::resIdx)].size(), 1);
    for (int i = 0; i < SHAPE_DIM4; i++) {
        dynParamList.emplace_back(dynSrcShape[i].Dump());
    }
    WhereParam param{templateList, paramList, dynParamList, varExpr, dataTypeExpr};
    return param;
}

std::string CodeGenOpCloudNPU::PrintWhereOp(const WhereParam &param) const {
    std::vector<std::string> templateList = param.templateList;
    std::vector<std::string> paramList = param.paramList;
    std::vector<std::string> dynParamList = param.dynParamList;
    std::vector<std::string> varExpr = param.varExpr;
    std::vector<std::string> dataTypeExpr = param.dataTypeExpr;
    std::string templateParam = JoinString(templateList, CONN_COMMA);
    std::string funcParam = JoinString(paramList, CONN_COMMA);
    std::string dynFuncParam = JoinString(dynParamList, CONN_COMMA);
    std::vector<std::string> extList;

    std::ostringstream os;
    if (opCode == Opcode::OP_WHERE_SS) {
        std::string src0Var = FormatFloat(extScalarVec[0].GetVariantData());
        std::string src1Var = FormatFloat(extScalarVec[1].GetVariantData());
        extList.emplace_back(dataTypeExpr[0] + "(" + src0Var + ")");
        extList.emplace_back(dataTypeExpr[0] + "(" + src1Var + ")");
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else if (opCode == Opcode::OP_WHERE_ST) {
        std::string scalarVar = FormatFloat(extOperandVal.GetVariantData());
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src1DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        extList.emplace_back(dataTypeExpr[0] + "(" + scalarVar + ")");
        extList.emplace_back("(__ubuf__ " + src1DtypeStr + "*)" + src0Var);
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else if (opCode == Opcode::OP_WHERE_TS) {
        std::string scalarVar = FormatFloat(extOperandVal.GetVariantData());
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src0DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        extList.emplace_back("(__ubuf__ " + src0DtypeStr + "*)" + src0Var);
        extList.emplace_back(dataTypeExpr[0] + "(" + scalarVar + ")");
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else { // opCode == Opcode::OP_WHERE_TT
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src0DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        std::string src1Var = varExpr[ToUnderlying(WhereOpIdx::src1Idx)];
        std::string src1DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src1Idx)]);
        extList.emplace_back("(__ubuf__ " + src0DtypeStr + "*)" + src0Var);
        extList.emplace_back("(__ubuf__ " + src1DtypeStr + "*)" + src1Var);
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    }
}

std::string CodeGenOpCloudNPU::PrintWhereOpTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::resIdx));
    std::string tempTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::tempIdx));
    std::string condTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::condIdx));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src0Idx));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src1Idx));

    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << tempTensor << ", " << condTensor << ", " << src0Tensor << ", "
        << src1Tensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenWhereOp() const {
    WhereParam param = PrepareWhereParam();
    if (isSupportLayout) {
        return PrintWhereOpTileTensor();
    }
    return PrintWhereOp(param);
}

std::string CodeGenOpCloudNPU::GenLogicalNotOp() const {
    if (isSupportLayout) {
        return PrintLogicalNotTileTensor();
    }
    // Support 2 dim
    enum class OpIdx : int { resIdx = 0, tmpIdx, srcIdx };

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx)]);

    std::vector dstShape = this->rawShape[ToUnderlying(OpIdx::resIdx)];
    std::vector srcShape = this->rawShape[ToUnderlying(OpIdx::srcIdx)];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx)]);

    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, srcVar);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    int dim = dstShape.size();
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    paramList.emplace_back("(__ubuf__ " + dstDtypeStr + "*)" + dstVar);
    paramList.emplace_back("(__ubuf__ " + srcDtypeStr + "*)" + srcVar);
    paramList.emplace_back("(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar);

    auto dynSrcShape = dynamicValidShape[ToUnderlying(OpIdx::srcIdx)];
    for (auto dyn : dynSrcShape) {
        paramList.emplace_back(dyn.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintCmpTileTensor() const {
    enum class TensorIdx : int { dstIdx = 0, tmpIdx, src0Idx, src1Idx };
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(TensorIdx::dstIdx));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(TensorIdx::tmpIdx));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(TensorIdx::src0Idx));
    std::string src1Tensor = "";
    if (opCode == Opcode::OP_CMP) {
        src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(TensorIdx::src1Idx));
    }

    auto cmpOp = opAttrs.at(OP_ATTR_PREFIX + "cmp_operation");
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "cmp_mode");
    std::string cmpOpVal = std::to_string(npu::tile_fwk::AnyCast<int64_t>(cmpOp));
    std::string modeVal = std::to_string(npu::tile_fwk::AnyCast<int64_t>(mode));

    std::vector<std::string> tileOpParamList = {dstTensor, src0Tensor, src1Tensor, tmpTensor};
    std::vector<std::string> templateParamList = {cmpOpVal, modeVal};
    if (opCode == Opcode::OP_CMPS) {
        auto scalarAttr = opAttrs.at(OpAttributeKey::scalar);
        auto scalarElement = npu::tile_fwk::AnyCast<npu::tile_fwk::Element>(scalarAttr);
        float scalarValue = static_cast<float>(scalarElement.GetFloatData());
        auto scalarType = scalarElement.GetDataType();
        if (scalarType == DataType::DT_FP16) {
            templateParamList.emplace_back("half");
        } else {
            templateParamList.emplace_back("float");
        }
        tileOpParamList.erase(tileOpParamList.begin() + ID2);
        tileOpParamList.emplace_back(std::to_string(scalarValue));
    }
    std::ostringstream oss;
    oss << tileOpName;
    oss << PrintParams({"<", ">"}, templateParamList, ", ");
    oss << PrintParams({"(", ")"}, tileOpParamList, ", ");
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenCmpOp() const {
    if (isSupportLayout) {
        return PrintCmpTileTensor();
    }
    enum class TensorIdx : int { dstIdx = 0, tmpIdx, src0Idx, src1Idx };

    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::dstIdx)]);
    std::string tVar1 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::tmpIdx)]);

    bool isScalarMode = (opCode == Opcode::OP_CMPS);
    std::string s0Var, s1Var;
    std::vector<int64_t> src0RawShape, src1RawShape;
    auto newDynSrcValidShape = dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)];

    if (isScalarMode) {
        s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src0Idx)]);
        src0RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src0Idx)], SHAPE_DIM4);
        FillIntVecWithDummyInHead<SymbolicScalar>(
            newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)].size(), 1);
    } else {
        s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src0Idx)]);
        s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src1Idx)]);
        src0RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src0Idx)], SHAPE_DIM4);
        src1RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src1Idx)], SHAPE_DIM4);
        FillIntVecWithDummyInHead<SymbolicScalar>(
            newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)].size(), 1);
    }

    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::dstIdx)], SHAPE_DIM4);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(TensorIdx::src0Idx)]);

    if (isScalarMode) {
        AppendLocalBufVarOffsetInOrder(dVar, tVar1, s0Var);
    } else {
        AppendLocalBufVarOffsetInOrder(dVar, tVar1, s0Var, s1Var);
    }

    auto cmpOp = opAttrs.at(OP_ATTR_PREFIX + "cmp_operation");
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "cmp_mode");
    std::string cmpOpVal = std::to_string(npu::tile_fwk::AnyCast<int64_t>(cmpOp));
    std::string modeVal = std::to_string(npu::tile_fwk::AnyCast<int64_t>(mode));

    float scalarValue = 0.0f;
    if (isScalarMode) {
        auto scalarAttr = opAttrs.at(OpAttributeKey::scalar);
        auto scalarElement = npu::tile_fwk::AnyCast<npu::tile_fwk::Element>(scalarAttr);
        scalarValue = static_cast<float>(scalarElement.GetFloatData());
    }

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(src0RawShape[i]));
    }
    if (!isScalarMode) {
        for (int i = ID1; i < SHAPE_DIM4; ++i) {
            paramList.emplace_back(std::to_string(src1RawShape[i]));
        }
    }
    paramList.emplace_back(cmpOpVal);
    paramList.emplace_back(modeVal);
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_UB) + " uint8_t*)" + dVar;
    std::string src0 = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + srcDtypeStr + "*)" + s0Var;

    std::string tmp1 = "(" + GetAddrTypeByOperandType(BUF_UB) + " uint8_t*)" + tVar1;

    paramList.insert(paramList.end(), {dst, src0});
    if (!isScalarMode) {
        std::string src1 = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + srcDtypeStr + "*)" + s1Var;
        paramList.emplace_back(src1);
    }

    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    paramList.emplace_back(tmp1);

    if (isScalarMode) {
        paramList.emplace_back(std::to_string(scalarValue));
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintLogicalAndTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC1_IDX));
    std::vector<std::string> paramList = {dstTensor, srcTensor, src1Tensor, tmpTensor};
    std::ostringstream oss;
    oss << tileOpName;
    oss << PrintParams({"(", ")"}, paramList, ", ");
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintLogicalNotTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::vector<std::string> paramList = {dstTensor, srcTensor, tmpTensor};
    std::ostringstream oss;
    oss << tileOpName;
    oss << PrintParams({"(", ")"}, paramList, ", ");
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenLogicalAndOp() const {
    if (isSupportLayout) {
        return PrintLogicalAndTileTensor();
    }
    // Support 2 dim
    enum class OpIdx : int { resIdx = 0, tmpIdx, srcIdx0, srcIdx1 };

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcVar0 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx0)]);
    std::string srcVar1 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx1)]);

    std::vector dstShape = this->rawShape[ToUnderlying(OpIdx::resIdx)];
    std::vector srcShape0 = this->rawShape[ToUnderlying(OpIdx::srcIdx0)];
    std::vector srcShape1 = this->rawShape[ToUnderlying(OpIdx::srcIdx1)];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcDtypeStr0 = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx0)]);
    std::string srcDtypeStr1 = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx1)]);

    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, srcVar0, srcVar1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr0);
    paramList.emplace_back(srcDtypeStr1);

    int dim = dstShape.size(); // 输入输出Tensor维度相同
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape0[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape1[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    paramList.emplace_back("(" + addrType + " " + dstDtypeStr + "*)" + dstVar);
    paramList.emplace_back("(" + addrType + " " + srcDtypeStr0 + "*)" + srcVar0);
    paramList.emplace_back("(" + addrType + " " + srcDtypeStr1 + "*)" + srcVar1);
    paramList.emplace_back("(" + addrType + " " + tmpDtypeStr + "*)" + tmpVar);

    auto dynSrcShape = dynamicValidShape[ToUnderlying(OpIdx::srcIdx0)];
    for (auto dyn : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dyn));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::GenTopKSortOp() const {
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    AppendLocalBufVarOffsetInOrder(yVar, tmpVar, xVar);

    std::string startIdx;
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalar = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(SymbolicScalar)))
            << npu::tile_fwk::AnyCast<SymbolicScalar>(scalar).IsValid()
            << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = npu::tile_fwk::AnyCast<SymbolicScalar>(scalar);
        startIdx = SymbolicExpressionTable::BuildExpression(scalarExpr);
    }

    auto xShape = this->rawShape[ID2];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    if (!isDynamicFunction) {
        templateParam += GenOpAttr();
    }

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string tmp = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + tmpVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID2]) + " " + xDtypeStr + "*)" + xVar;
    if (isDynamicFunction) {
        if (startIdx.empty()) {
            startIdx = GenOpAttr().substr(NUM2);
        }
        paramList.insert(paramList.end(), {y, tmp, x, startIdx});
    } else {
        paramList.insert(paramList.end(), {y, tmp, x});
    }
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenTopKMergeOp() const {
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    AppendLocalBufVarOffsetInOrder(yVar, xVar);

    auto xShape = this->rawShape[ID0];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1] / NUM2));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + xVar;
    paramList.insert(paramList.end(), {y, x});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenTopKExtractOp() const {
    std::string yDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    AppendLocalBufVarOffsetInOrder(yVar, xVar);

    auto yShape = this->rawShape[ID0];
    auto xShape = this->rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(yDtypeStr);
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(std::to_string(yShape[0]));
    paramList.emplace_back(std::to_string(yShape[1]));
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1] / NUM2));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + yDtypeStr + "*)" + yVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + xDtypeStr + "*)" + xVar;
    paramList.insert(paramList.end(), {y, x});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}
} // namespace npu::tile_fwk

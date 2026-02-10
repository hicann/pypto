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
 * \file codegen_vector.cpp
 * \brief
 */

#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen_op_cloudnpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::PrintCastDynamicUnaligned(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    auto dynDstShape = dynamicValidShape[0];
    std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    paramList.insert(paramList.end(), {dstDtypeStr, srcDtypeStr});
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    for (auto dynShape : newDynDstShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintCastTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "mode");
    int64_t modeEnum{0};
    if (mode.HasValue()) {
        modeEnum = AnyCast<int64_t>(mode);
    }
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    templateParamList.emplace_back(std::to_string(modeEnum));
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses({dstTensor, srcTensor});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowMaxlineStatic(const PrintUnaryParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[1].size()) - 1))))
        << "unsupported reduce axis: " << reduceAxis;

    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);

    std::vector<std::string> paramList;
    paramList.emplace_back(param.dstDtypeStr);
    FillParamWithFullShape(paramList, os);
    FillParamWithShapeExceptFirst(paramList, srcShape);
    FillParamWithShapeExceptFirst(paramList, dstShape);
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + param.dstDtypeStr + "*)" + param.dVar;
    std::string src = "(__ubuf__ " + param.srcDtypeStr + "*)" + param.s0Var;
    paramList.insert(paramList.end(), {dst, src});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowMaxlineDynamicUnaligned(const PrintUnaryParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[1].size()) - 1))))
        << "unsupported reduce axis" << reduceAxis;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    auto dynSrcShape = dynamicValidShape[1];
    // adjust reduceAxis for dim4
    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - rawShape[0].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintRowMaxlineTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(rawShape[1].size()) - 1)))) << "unsupported reduce axis";
    reduceAxis += SHAPE_DIM5 - rawShape[0].size();
    std::ostringstream oss;
    oss << tileOpName << "<" << reduceAxis << ">"
        << "(" << dstTensor << ", " << src0Tensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowMaxline(const PrintUnaryParam &param) const {
    if (isSupportLayout) {
        return PrintRowMaxlineTileTensor();
    }
    if (isDynamicFunction) {
        return PrintRowMaxlineDynamicUnaligned(param);
    }
    return PrintRowMaxlineStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceExStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> oriShape = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(oriShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src0});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceEx(const PrintUnaryParam &param) const {
    return PrintReduceExStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceSumStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst0 = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst0, src0});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceSum(const PrintUnaryParam &param) const {
    return PrintReduceSumStatic(param);
}

std::string CodeGenOpCloudNPU::PrintVcopyStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src0});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVcopy(const PrintUnaryParam &param) const {
    return PrintVcopyStatic(param);
}

std::string CodeGenOpCloudNPU::PrintExpandDynamicUnaligned(const PrintUnaryParam &param, int expandAxis) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    auto dynDstShape = dynamicValidShape[0];
    std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    auto dynSrcShape = dynamicValidShape[1];
    std::vector<SymbolicScalar> newDynSrcShape = dynSrcShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcShape, SHAPE_DIM4 - dynSrcShape.size(), 1);
    std::ostringstream os;
    std::vector<std::string> paramList;
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = ID1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = ID1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    paramList.emplace_back(std::to_string(expandAxis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    for (int i = ID0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(newDynDstShape[i]));
    }
    for (int i = ID0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(newDynSrcShape[i]));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintExpandLayout(int expandAxis) const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    templateParamList.emplace_back(std::to_string(expandAxis));
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses({dstTensor, srcTensor});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintExpand(const std::string &s0Var, const std::string &dVar,
    const std::string &srcDtypeStr, const std::string &dstDtypeStr) const {
    char buffer[256] = "CG_ERROR";
    int ret = 0;
    int expandAxis{-1};
    std::vector<int64_t> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "EXPANDDIM");
    if (axis.HasValue()) {
        expandAxis = AnyCast<int64_t>(axis);
    }
    ASSERT((expandAxis >= 0) && (expandAxis <= (static_cast<int>(rawShape[1].size() - 1))))
        << "unsupported expand axis";
    // modify expandAxis for SHAPE_DIM4
    expandAxis += SHAPE_DIM4 - rawShape[1].size();

    if (isSupportLayout) {
        return PrintExpandLayout(expandAxis);
    }
    if (isDynamicFunction) {
        return PrintExpandDynamicUnaligned({s0Var, dVar, srcDtypeStr, dstDtypeStr}, expandAxis);
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %d, %d, %d, %d, %d, %d, %d, %d, /*DS*/ %d, %d, %d, /*SS*/ %d, %d, %d, %d>"
        "((__ubuf__ %s*)%s, (__ubuf__ %s*)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), dos[ID0], dos[ID1], dos[ID2], dos[ID3], os[ID0], os[ID1], os[ID2],
        os[ID3], ds[ID1], ds[ID2], ds[ID3], ss[ID1], ss[ID2], ss[ID3], expandAxis, dstDtypeStr.c_str(), dVar.c_str(),
        srcDtypeStr.c_str(), s0Var.c_str());
    ASSERT(ret >= 0) << "GenUnaryOp" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed " << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintOneHotLayout() const {
    std::string dstTensor =QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor =QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor <<","<< srcTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintOneHot(const PrintUnaryParam &param) const {
    if (isSupportLayout) {
        return PrintOneHotLayout();
    }

    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.emplace_back("/*DS*/");
    paramList.emplace_back(std::to_string(ds[SHAPE_DIM1]));
    paramList.emplace_back(std::to_string(ds[SHAPE_DIM2]));
    int numClasses{-1};
    auto num = opAttrs.at(OP_ATTR_PREFIX + "numClasses");
    if (num.HasValue()) {
        numClasses = AnyCast<int64_t>(num);
    }
    constexpr int align = BLOCK_SIZE / sizeof(int64_t);
    paramList.emplace_back(std::to_string((numClasses + align - 1) / align * align));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__ubuf__ int64_t*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM3 - dynSrcShape.size(), 1);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintUnaryDynamicUnaligned(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintUnaryStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBitwiseNot() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << srcTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintUnaryTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        oss << WrapParamByAngleBrackets({lastUse});
    }
    oss << WrapParamByParentheses({dstTensor, srcTensor});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintUnary(const PrintUnaryParam &param) const {
    if (isSupportLayout) {
        return PrintUnaryTileTensor();
    }
    if (isDynamicFunction) {
        return PrintUnaryDynamicUnaligned(param);
    }
    return PrintUnaryStatic(param);
}

std::string CodeGenOpCloudNPU::GenUnaryOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var);

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        srcDtypeStr = GetTypeForB16B32(operandDtype[ID1]);
        dstDtypeStr = GetTypeForB16B32(operandDtype[ID0]);
    }

    if (opCode == Opcode::OP_EXPAND) {
        return PrintExpand(s0Var, dVar, srcDtypeStr, dstDtypeStr);
    } else if (opCode == Opcode::OP_ONEHOT) {
        return PrintOneHot({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_ROWMAX || opCode == Opcode::OP_ROWEXPMAX || opCode == Opcode::OP_ROWEXPSUM) {
        return PrintReduceEx({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_ROWMAXLINE || opCode == Opcode::OP_ROWMINLINE) {
        return PrintRowMaxline({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_EXP || opCode == Opcode::OP_SQRT || opCode == Opcode::OP_ABS ||
               opCode == Opcode::OP_RECIPROCAL || opCode == Opcode::OP_NEG || opCode == Opcode::OP_RSQRT ||
               opCode == Opcode::OP_LN || opCode == Opcode::OP_LOGICALNOT || opCode == Opcode::OP_BRCB ||
               opCode == Opcode::OP_CEIL|| opCode == Opcode::OP_FLOOR|| opCode == Opcode::OP_TRUNC) {
        return PrintUnary({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        return PrintVcopy({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_ROWSUM) {
        return PrintReduceSum({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_BITWISENOT) {
        return PrintBitwiseNot();
    }
    ALOG_INFO_F("unsupported tileop: %s", opCodeStr.c_str());
    return "CG_ERROR";
}

} // namespace npu::tile_fwk

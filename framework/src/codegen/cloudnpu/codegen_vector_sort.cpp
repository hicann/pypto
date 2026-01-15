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
std::string CodeGenOpCloudNPU::PrintSortDynamicUnaligned(const SortParam &param) const {
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);

    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(src0Shape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynSrcShape[i]));
    }
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintSortStatic(const SortParam &param) const {
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    unsigned orisrcShape0 = 0;
    unsigned orisrcShape1 = 0;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::vector<int64_t> &oriSrc0Shape = originShape[1];
    constexpr unsigned defaultDim = 1u;
    if (oriSrc0Shape.size() == 1) {
        orisrcShape0 = defaultDim;
        orisrcShape1 = oriSrc0Shape[0];
    } else {
        orisrcShape0 = oriSrc0Shape[0];
        orisrcShape1 = oriSrc0Shape[1];
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    // static only support 1~2 dim
    paramList.insert(paramList.end(), {std::to_string(dstShape[2]), std::to_string(dstShape[3])});
    paramList.insert(paramList.end(), {std::to_string(src0Shape[2]), std::to_string(src0Shape[3])});
    paramList.insert(paramList.end(), {std::to_string(orisrcShape0), std::to_string(orisrcShape1)});

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTiledSortDynamicUnaligned(const TiledSortParam &param) const {
    auto dstShape = param.dstShape;
    auto srcShape = param.srcShape;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string &s2Var = param.s2Var;
    const std::string &s3Var = param.s3Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    auto dynSrc0Shape = dynamicValidShape[ID2];
    auto dynSrc1Shape = dynamicValidShape[ID3];
    auto dynSrc2Shape = dynamicValidShape[ID4];
    auto dynSrc3Shape = dynamicValidShape[ID5];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrc0Shape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrc1Shape, SHAPE_DIM4 - dynamicValidShape[ID3].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrc2Shape, SHAPE_DIM4 - dynamicValidShape[ID4].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrc3Shape, SHAPE_DIM4 - dynamicValidShape[ID5].size(), 1);

    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0Param = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string src1Param = "(__ubuf__ " + srcDtypeStr + "*)" + s1Var;
    std::string src2Param = "(__ubuf__ " + srcDtypeStr + "*)" + s2Var;
    std::string src3Param = "(__ubuf__ " + srcDtypeStr + "*)" + s3Var;
    std::string tmpParam = "(__ubuf__ " + srcDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dstParam, src0Param, src1Param, src2Param, src3Param, tmpParam});
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(dynSrc0Shape[i].Dump());
    }
    paramList.emplace_back(dynSrc1Shape[ID3].Dump());
    paramList.emplace_back(dynSrc2Shape[ID3].Dump());
    paramList.emplace_back(dynSrc3Shape[ID3].Dump());
    std::string tileCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBitSortDynamicUnaligned(const SortParam &param) const {
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpCloudNPU::PrintBitSortStatic(const SortParam &param) const {
    return PrintSortStatic(param);
}

std::string CodeGenOpCloudNPU::PrintSortTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    oss << tileOpName << "<" << GenOpAttr(false) << ">" << "(" << dstTensor << ", " << srcTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenBitSortOp() const {
    SortParam sortParm = PrepareSortParam();
    if (isSupportLayout) {
        return PrintSortTileTensor();
    }
    if (isDynamicFunction) {
        return PrintBitSortDynamicUnaligned(sortParm);
    }
    return PrintBitSortStatic(sortParm);
}

std::string CodeGenOpCloudNPU::PrintMrgSortDynamicUnaligned(const SortParam &param) const {
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpCloudNPU::PrintMrgSortStatic(const SortParam &param) const {
    return PrintSortStatic(param);
}

std::string CodeGenOpCloudNPU::PrintTiledMrgSortDynamicUnaligned(const TiledSortParam &param) const {
    return PrintTiledSortDynamicUnaligned(param);
}

SortParam CodeGenOpCloudNPU::PrepareSortParam() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType src0Dtype = operandDtype[ID1];

    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    AppendLocalBufVarOffsetInOrder(dstVar, src0Var);
    return {
        {ds[ID0], ds[ID1], ds[ID2], ds[ID3]},
        {ss[ID0], ss[ID1], ss[ID2], ss[ID3]},
        src0Var, dstVar, src0DtypeStr,
        dstDtypeStr
    };
}

std::string CodeGenOpCloudNPU::GenMrgSortOp() const {
    SortParam sortParm = PrepareSortParam();
    if (isSupportLayout) {
        return PrintSortTileTensor();
    }
    if (isDynamicFunction) {
        return PrintMrgSortDynamicUnaligned(sortParm);
    }
    return PrintMrgSortStatic(sortParm);
}

TiledSortParam CodeGenOpCloudNPU::PrepareTiledSortParam() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType srcDtype = operandDtype[ID2];

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string src2Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);
    std::string src3Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID5]);

    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID2];
    std::vector src3Shape = this->rawShape[ID5];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> s3 = NormalizeShape(src3Shape, SHAPE_DIM4);

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string srcDtypeStr = DataType2CCEStr(srcDtype);
    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, src0Var, src1Var, src2Var, src3Var);
    return {
        {ds[ID0], ds[ID1], ds[ID2], ds[ID3]},
        {s0[ID0], s0[ID1], s0[ID2], s0[ID3], s3[ID3]},
        src0Var, src1Var, src2Var,
        src3Var, tmpVar, dstVar, srcDtypeStr, dstDtypeStr
    };
}

std::string CodeGenOpCloudNPU::PrintTileSortTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string src1Tensor = QueryTileTensorNameByIdx(ID2);
    std::string src2Tensor = QueryTileTensorNameByIdx(ID3);
    std::string src3Tensor = QueryTileTensorNameByIdx(ID4);
    std::string src4Tensor = QueryTileTensorNameByIdx(ID5);
    std::ostringstream oss;
    oss << tileOpName << "<" << GenOpAttr(false) << ">"
        << "(" << dstTensor << ", " << src1Tensor << ", " << src2Tensor << ", " << src3Tensor << ", " << src4Tensor
        << ", " << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenTiledMrgSortOp() const {
    TiledSortParam tiledSortParm = PrepareTiledSortParam();
    if (isSupportLayout) {
        return PrintTileSortTileTensor();
    }
    return PrintTiledMrgSortDynamicUnaligned(tiledSortParm);
}

std::string CodeGenOpCloudNPU::GenSortOp() const {
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string idxDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string yIdxVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    AppendLocalBufVarOffsetInOrder(yVar, yIdxVar, tmpVar, xVar);

    auto xShape = this->rawShape[ID0];
    auto idxShape = this->rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(idxDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1]));
    paramList.emplace_back(std::to_string(idxShape[0]));
    paramList.emplace_back(std::to_string(idxShape[1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string yIdx = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + idxDtypeStr + "*)" + yIdxVar;
    std::string tmp = "(" + GetAddrTypeByOperandType(operandType[ID2]) + " " + xDtypeStr + "*)" + tmpVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID3]) + " " + xDtypeStr + "*)" + xVar;
    paramList.insert(paramList.end(), {y, yIdx, tmp, x});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenMergeOp() const {
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string idxDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::string yVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string yIdxVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string xVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string idxVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);
    AppendLocalBufVarOffsetInOrder(yVar, yIdxVar, tmpVar, xVar, idxVar);

    auto xShape = this->rawShape[ID0];
    auto idxShape = this->rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(idxDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1]));
    paramList.emplace_back(std::to_string(idxShape[0]));
    paramList.emplace_back(std::to_string(idxShape[1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + yVar;
    std::string yIdx = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + idxDtypeStr + "*)" + yIdxVar;
    std::string tmp = "(" + GetAddrTypeByOperandType(operandType[ID2]) + " " + xDtypeStr + "*)" + tmpVar;
    std::string x = "(" + GetAddrTypeByOperandType(operandType[ID3]) + " " + xDtypeStr + "*)" + xVar;
    std::string idx = "(" + GetAddrTypeByOperandType(operandType[ID4]) + " " + idxDtypeStr + "*)" + idxVar;
    paramList.insert(paramList.end(), {y, yIdx, tmp, x, idx});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenCompareAndSwapOp() const {
    std::string xDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string idxDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    std::string y0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string yIdx0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string y1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string yIdx1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string x0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);
    std::string idx0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID5]);
    std::string x1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID6]);
    std::string idx1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID7]);
    AppendLocalBufVarOffsetInOrder(y0Var, yIdx0Var, y1Var, yIdx1Var, x0Var, idx0Var, x1Var, idx1Var);

    auto xShape = this->rawShape[ID0];
    auto idxShape = this->rawShape[ID1];

    std::vector<std::string> paramList;
    paramList.emplace_back(xDtypeStr);
    paramList.emplace_back(idxDtypeStr);
    paramList.emplace_back(std::to_string(xShape[0]));
    paramList.emplace_back(std::to_string(xShape[1]));
    paramList.emplace_back(std::to_string(idxShape[0]));
    paramList.emplace_back(std::to_string(idxShape[1]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    templateParam += GenOpAttr();

    paramList.clear();
    std::string y0 = "(" + GetAddrTypeByOperandType(operandType[ID0]) + " " + xDtypeStr + "*)" + y0Var;
    std::string yIdx0 = "(" + GetAddrTypeByOperandType(operandType[ID1]) + " " + idxDtypeStr + "*)" + yIdx0Var;
    std::string y1 = "(" + GetAddrTypeByOperandType(operandType[ID2]) + " " + xDtypeStr + "*)" + y1Var;
    std::string yIdx1 = "(" + GetAddrTypeByOperandType(operandType[ID3]) + " " + idxDtypeStr + "*)" + yIdx1Var;
    std::string x0 = "(" + GetAddrTypeByOperandType(operandType[ID4]) + " " + xDtypeStr + "*)" + x0Var;
    std::string idx0 = "(" + GetAddrTypeByOperandType(operandType[ID5]) + " " + idxDtypeStr + "*)" + idx0Var;
    std::string x1 = "(" + GetAddrTypeByOperandType(operandType[ID6]) + " " + xDtypeStr + "*)" + x1Var;
    std::string idx1 = "(" + GetAddrTypeByOperandType(operandType[ID7]) + " " + idxDtypeStr + "*)" + idx1Var;
    paramList.insert(paramList.end(), {y0, yIdx0, y1, yIdx1, x0, idx0, x1, idx1});
    std::string tileOpParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tileOpParam << ");\n";
    return os.str();
}

} // namespace npu::tile_fwk

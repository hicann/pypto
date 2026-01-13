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
 * \file cube_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Matrix {
const float EPSILON = 1e-6f;
namespace Deprecate {

#define OP_CHECK(cond, exec_expr) \
    do { \
        if (cond) { \
            exec_expr; \
        } \
    } while (0)

const int32_t GMACC = 3;
const int32_t BIASINDEX = 2;
const int32_t SCALEINDEX = 3;

using AggregationMap = std::map<std::vector<int64_t>,
    std::vector<std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr>>>;

struct TensorAttributes {
    int64_t tileSize;
    int64_t offset;
    std::string name;
    MemoryType memType {MemoryType::MEM_UNKNOWN };
};

struct MatmulInputs {
    LogicalTensorPtr aTensorPtr = nullptr;
    LogicalTensorPtr bTensorPtr = nullptr;
    LogicalTensorPtr cTensorPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr scaleTensorPtr = nullptr;
};

void SetMatmulAttr(Operation &op, const std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr> &tensorPtrs,
    const std::vector<int64_t> &matrixSize) {
    int64_t nzAttr = (static_cast<int64_t>(std::get<0>(tensorPtrs)->Format())) |
                     (static_cast<int64_t>(std::get<1>(tensorPtrs)->Format()) << 1) |
                     // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                     (static_cast<int64_t>(std::get<2>(tensorPtrs)->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    if (matrixSize.size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[M_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[K_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[N_INDEX]);
}

void SetBiasAndScaleAttr(
    const MatmulInputs &matmulInputs, const MatmulAttrParam &matmulAttrParam, bool isFirstTile, Operation &op) {
    if (matmulInputs.biasTensorPtr != nullptr && isFirstTile) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, true);
    }
    if (isFirstTile) {
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(matmulAttrParam.reluType));
    }
    if (matmulAttrParam.scaleValue != 0 && isFirstTile) {
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, matmulAttrParam.scaleValue));
    }
};

struct L1DataLoadParam {
    const LogicalTensorPtr &cTilePtr;
    const int64_t mL1Idx;
    const int64_t nL1Idx;
    const int64_t stepK;
    const int64_t mL1Size;
    const int64_t nL1Size;
    const int64_t orgK;
};

struct CollectSubAMulBPara {
    const TileShape &tileShape;
    const std::array<int64_t, 3> &posK;
    const LogicalTensorPtr &aTensorPtr;
    const LogicalTensorPtr &bTensorPtr;
    const LogicalTensorPtr &cTensorPtr;
    const LogicalTensorPtr &biasTensorPtr = nullptr;
    const LogicalTensorPtr &scaleTensorPtr = nullptr;
};

struct DoAMulBParam {
    const TileShape &tileShape;
    const LogicalTensorPtr &cTensorPtr;
};

template <bool isTrans>
std::vector<SymbolicScalar> GetValidShapeFromTranspose(LogicalTensorPtr &l0Tensor) {
    auto l0ValidShape = l0Tensor->GetDynValidShape();
    if (l0ValidShape.empty()) {
        return l0ValidShape;
    }
    if constexpr (isTrans) {
        std::swap(l0ValidShape.at(0), l0ValidShape.at(1));
    }
    return l0ValidShape;
}

void AddOpView(
    Function &function, const LogicalTensorPtr &operand, LogicalTensorPtr &viewTensor, const TensorAttributes &attrs) {
    DataType dtype = (attrs.name == "bias_BT" ? DataType::DT_FP32 : operand->Datatype());
    viewTensor = std::make_shared<LogicalTensor>(function, dtype, std::vector<int64_t>{1, attrs.tileSize},
        SymbolicScalar::FromConcrete({1, attrs.tileSize}), operand->Format(), attrs.name, operand->nodetype);
    viewTensor->UpdateDynValidShape(
        GetViewValidShape(operand->GetDynValidShape(), {0, attrs.offset}, {}, {1, attrs.tileSize}));
    auto &viewOperand = function.AddOperation(Opcode::OP_VIEW, {operand}, {viewTensor});
    std::vector<int64_t> newoffset{0, attrs.offset};

    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        newoffset, SymbolicScalar::FromConcrete(newoffset), viewTensor->GetDynValidShape());
    viewAttribute->SetToType(attrs.memType);
    viewOperand.SetOpAttribute(viewAttribute);
    if (attrs.name.find("l1") != std::string::npos) {
        viewOperand.SetAttribute(A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(0));
    }
}

template <bool isTransA = false, bool isTransB = false>
void CollectSubAMulB(Function &function, const CollectSubAMulBPara &args, AggregationMap &aggregations,
    const std::vector<int64_t> &l1Offset) {
    const CubeTile &cubeTile = args.tileShape.GetCubeTile();
    const LogicalTensorPtr &aTensorPtr = args.aTensorPtr;
    const LogicalTensorPtr &bTensorPtr = args.bTensorPtr;
    const LogicalTensorPtr &cTensorPtr = args.cTensorPtr;
    const LogicalTensorPtr &biasTensorPtr = args.biasTensorPtr;
    const LogicalTensorPtr &scaleTensorPtr = args.scaleTensorPtr;

    const std::array<int64_t, 3> &posK = args.posK;
    const int32_t kL1SizeIndex = 2;
    const int64_t mL1 = isTransA ? aTensorPtr->shape[1] : aTensorPtr->shape[0];
    const int64_t nL1 = isTransB ? bTensorPtr->shape[0] : bTensorPtr->shape[1];
    const auto opCodeA = isTransA ? Opcode::OP_L1_TO_L0_AT : Opcode::OP_L1_TO_L0A;
    const auto opCodeB = isTransB ? Opcode::OP_L1_TO_L0_BT : Opcode::OP_L1_TO_L0B;

    for (int64_t mL0Idx = 0; mL0Idx < mL1; mL0Idx += cubeTile.m[0]) {
        for (int64_t nL0Idx = 0; nL0Idx < nL1; nL0Idx += cubeTile.n[0]) {
            int64_t mL0size = std::min(mL1 - mL0Idx, cubeTile.m[0]);
            int64_t nL0size = std::min(nL1 - nL0Idx, cubeTile.n[0]);
            LogicalTensorPtr biasToBTLogicalTensor = nullptr;
            LogicalTensorPtr scaleToFBLogicalTensor = nullptr;
            if (biasTensorPtr != nullptr) {
                TensorAttributes biasAttrs = {nL0size, nL0Idx, "bias_BT", MemoryType::MEM_BT};
                AddOpView(function, biasTensorPtr, biasToBTLogicalTensor, biasAttrs);
            }
            if (scaleTensorPtr != nullptr) {
                TensorAttributes scaleAttrs = {nL0size, nL0Idx, "scale_FB", MemoryType::MEM_FIX_QUANT_PRE};
                AddOpView(function, scaleTensorPtr, scaleToFBLogicalTensor, scaleAttrs);
            }
            auto cL0Tensor = cTensorPtr->View(function, {mL0size, nL0size}, {mL0Idx, nL0Idx});
            for (int64_t kL0Idx = 0; kL0Idx < posK[kL1SizeIndex]; kL0Idx += cubeTile.k[0]) {
                int64_t kL0size = std::min(posK[kL1SizeIndex] - kL0Idx, cubeTile.k[0]);
                const std::vector<int64_t> sizeVecA =
                    isTransA ? std::vector<int64_t>{kL0size, mL0size} : std::vector<int64_t>{mL0size, kL0size};
                const std::vector<int64_t> sizeVecB =
                    isTransB ? std::vector<int64_t>{nL0size, kL0size} : std::vector<int64_t>{kL0size, nL0size};
                auto aL0Tensor = isTransA ? aTensorPtr->View(function, sizeVecA, {posK[0] + kL0Idx, mL0Idx}) :
                                            aTensorPtr->View(function, sizeVecA, {mL0Idx, posK[0] + kL0Idx});
                auto bL0Tensor = isTransB ? bTensorPtr->View(function, sizeVecB, {nL0Idx, posK[1] + kL0Idx}) :
                                            bTensorPtr->View(function, sizeVecB, {posK[1] + kL0Idx, nL0Idx});
                auto aL0ValidShape = GetValidShapeFromTranspose<isTransA>(aL0Tensor);
                auto aL0LogicalTensor = std::make_shared<LogicalTensor>(function, aTensorPtr->Datatype(),
                    std::vector<int64_t>{mL0size, kL0size}, aL0ValidShape, aTensorPtr->Format(),
                    "a_l0", aTensorPtr->nodetype);
                auto bL0ValidShape = GetValidShapeFromTranspose<isTransB>(bL0Tensor);
                auto bL0LogicalTensor = std::make_shared<LogicalTensor>(function, bTensorPtr->Datatype(),
                    std::vector<int64_t>{kL0size, nL0size}, bL0ValidShape, bTensorPtr->Format(),
                    "b_l0", bTensorPtr->nodetype);
                function.AddOperation(opCodeA, {aL0Tensor}, {aL0LogicalTensor});
                function.AddOperation(opCodeB, {bL0Tensor}, {bL0LogicalTensor});
                auto l0Offset = l1Offset;
                l0Offset[0] += mL0Idx;
                l0Offset[1] += nL0Idx;

                std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr> logicalTensor;
                std::get<0>(logicalTensor) = aL0LogicalTensor;
                std::get<1>(logicalTensor) = bL0LogicalTensor;
                if (biasTensorPtr != nullptr) {
                    std::get<BIASINDEX>(logicalTensor) = biasToBTLogicalTensor;
                }
                if (scaleTensorPtr != nullptr) {
                    std::get<SCALEINDEX>(logicalTensor) = scaleToFBLogicalTensor;
                }
                aggregations[l0Offset].emplace_back(logicalTensor);
            }
        }
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1NormalLoad(Function &function, const MatmulInputs &matmulInputs, const L1DataLoadParam &L1DataLoadParam,
    const TileShape &tileShape, AggregationMap &aggregations) {
    const int64_t orgK = L1DataLoadParam.orgK;
    const LogicalTensorPtr &cTilePtr = L1DataLoadParam.cTilePtr;
    const int64_t mL1Idx = L1DataLoadParam.mL1Idx;
    const int64_t nL1Idx = L1DataLoadParam.nL1Idx;
    const int64_t mL1Size = L1DataLoadParam.mL1Size;
    const int64_t nL1Size = L1DataLoadParam.nL1Size;
    const auto operand1 = matmulInputs.aTensorPtr;
    const auto operand2 = matmulInputs.bTensorPtr;
    auto &cubeTile = tileShape.GetCubeTile();
    for (int64_t kL1Idx = 0; kL1Idx < orgK; kL1Idx += cubeTile.k[1]) {
        int64_t kL1Size = std::min(orgK - kL1Idx, cubeTile.k[1]);
        auto aL1Tensor = isTransA ? operand1->View(function, {kL1Size, mL1Size}, {kL1Idx, mL1Idx}) :
                                    operand1->View(function, {mL1Size, kL1Size}, {mL1Idx, kL1Idx});
        auto bL1Tensor = isTransB ? operand2->View(function, {nL1Size, kL1Size}, {nL1Idx, kL1Idx}) :
                                    operand2->View(function, {kL1Size, nL1Size}, {kL1Idx, nL1Idx});

        LogicalTensorPtr biasL1Tensor = nullptr;
        LogicalTensorPtr scaleL1Tensor = nullptr;
        if (matmulInputs.biasTensorPtr != nullptr) {
            TensorAttributes biasAttrs = {nL1Size, nL1Idx, "bias_l1", MemoryType::MEM_L1};
            AddOpView(function, matmulInputs.biasTensorPtr, biasL1Tensor, biasAttrs);
        }
        if (matmulInputs.scaleTensorPtr != nullptr) {
            TensorAttributes scaleAttrs = {nL1Size, nL1Idx, "scale_l1", MemoryType::MEM_L1};
            AddOpView(function, matmulInputs.scaleTensorPtr, scaleL1Tensor, scaleAttrs);
        }
        CollectSubAMulB<isTransA, isTransB>(function,
            {
                tileShape, {0, 0, kL1Size},
                 aL1Tensor, bL1Tensor, cTilePtr, biasL1Tensor, scaleL1Tensor
        },
            aggregations, {mL1Idx, nL1Idx});
    }
}

void ExtendTileOprandInputs(
    const std::vector<LogicalTensorPtr> &operandVec, MatmulInputs &matmulInputs, const MatmulAttrParam &params) {
    matmulInputs.aTensorPtr = operandVec[0];
    matmulInputs.bTensorPtr = operandVec[1];
    if (params.hasBias) {
        matmulInputs.biasTensorPtr = operandVec[SHAPE_DIM2];
        if (params.hasScale) {
            matmulInputs.scaleTensorPtr = operandVec[SHAPE_DIM3];
        }
    } else if (params.hasScale) {
        matmulInputs.scaleTensorPtr = operandVec[SHAPE_DIM2];
    } else if (operandVec.size() == GMACC) {
        matmulInputs.cTensorPtr = operandVec[SHAPE_DIM2];
    }
}

template <bool hasThirdInput = false>
void DoAMulB(Function &function, const AggregationMap &aggregations, const MatmulInputs &matmulInputs,
    const DoAMulBParam &DoAMulBPara, const MatmulAttrParam &matmulAttrParam) {
    const auto &cubeTile = DoAMulBPara.tileShape.GetCubeTile();
    const LogicalTensorPtr &cTensorPtr = DoAMulBPara.cTensorPtr;
    LogicalTensorPtr inputOperand = hasThirdInput ? matmulInputs.cTensorPtr :
                                                    std::make_shared<LogicalTensor>(function, cTensorPtr->Datatype(),
                                                        cTensorPtr->shape, cTensorPtr->GetDynValidShape());
    auto dataType = cTensorPtr->Datatype();
    std::vector<int64_t> shape = {cubeTile.m[0], cubeTile.n[0]};
    const std::vector<int64_t> matrixSize = {matmulAttrParam.mValue, matmulAttrParam.kValue, matmulAttrParam.nValue};
    for (const auto &[offset, aggregation] : aggregations) {
        OP_CHECK(true, {ASSERT(!aggregation.empty()) << "AggregationMap is empty." << std::endl;});
        auto cL0PartialTensor = std::make_shared<LogicalTensor>(function, dataType, shape);
        shape[0] = std::min(cubeTile.m[0], cTensorPtr->shape[0] - offset[0]);
        shape[1] = std::min(cubeTile.n[0], cTensorPtr->shape[1] - offset[1]);
        auto cTilePtr = cTensorPtr->View(function, shape, offset);
        auto partialGM = inputOperand->View(function, shape, offset);
        for (size_t i = 0; i < aggregation.size(); i++) {
            bool isFirstTile = i == 0;
            std::vector<LogicalTensorPtr> inputWithThird = {std::get<0>(aggregation[i]), std::get<1>(aggregation[i])};
            if (matmulInputs.biasTensorPtr != nullptr && isFirstTile) {
                inputWithThird.push_back(std::get<BIASINDEX>(aggregation[i]));
            }
            if (matmulInputs.scaleTensorPtr != nullptr && isFirstTile) {
                inputWithThird.push_back(std::get<SCALEINDEX>(aggregation[i]));
            }
            std::vector<LogicalTensorPtr> inputWithOutThird = inputWithThird;
            if (isFirstTile) {
                inputWithThird.push_back(partialGM);
            } else {
                inputWithOutThird.push_back(cL0PartialTensor);
                inputWithThird.push_back(cL0PartialTensor);
                inputWithThird.push_back(partialGM);
            }
            const std::vector<LogicalTensorPtr> inputVec =
                (isFirstTile && hasThirdInput) ? inputWithThird : inputWithOutThird;
            const std::string matmulOpStr = isFirstTile ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
            if (i == aggregation.size() - 1) {
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cTilePtr});
                SetBiasAndScaleAttr(matmulInputs, matmulAttrParam, isFirstTile, op);
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(std::get<0>(aggregation[i]), std::get<1>(aggregation[i]), cTensorPtr),
                    matrixSize);
            } else {
                auto cL0PartialSum = std::make_shared<LogicalTensor>(function, dataType, shape);
                if (!inputVec[0]->GetDynValidShape().empty() && !inputVec[1]->GetDynValidShape().empty()) {
                    cL0PartialSum->UpdateDynValidShape(
                        {inputVec[0]->GetDynValidShape()[0], inputVec[1]->GetDynValidShape()[1]});
                }
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cL0PartialSum});
                SetBiasAndScaleAttr(matmulInputs, matmulAttrParam, isFirstTile, op);
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(std::get<0>(aggregation[i]), std::get<1>(aggregation[i]), cTensorPtr),
                    matrixSize);
                cL0PartialTensor = cL0PartialSum;
            }
        }
    }
}

template <bool isTransA, bool isTransB>
void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &cTensorPtr, const MatmulAttrParam &params) {
    MatmulInputs matmulInputs;
    ExtendTileOprandInputs(operandVec, matmulInputs, params);
    const auto operand1 = matmulInputs.aTensorPtr;
    const auto operand2 = matmulInputs.bTensorPtr;

    // 2为shape的维度，当前只支持2维
    OP_CHECK(operand1->shape.size() != 2, { ASSERT(false && "only supported two dimension"); });
    const int64_t orgM = isTransA ? operand1->shape[1] : operand1->shape[0];
    const int64_t orgKa = isTransA ? operand1->shape[0] : operand1->shape[1];
    const int64_t orgKb = isTransB ? operand2->shape[1] : operand2->shape[0];
    const int64_t orgN = isTransB ? operand2->shape[0] : operand2->shape[1];
    OP_CHECK(true, {
        ASSERT(orgKa == orgKb) << "K-axis mismatch: "
                                 << "orgKa: " << orgKa << ", orgKb: " << orgKb << std::endl;
    });

    const int32_t kBL1Idx = 2;
    auto &cubeTile = tileShape.GetCubeTile();
    const int64_t stepK = std::gcd(cubeTile.k[1], cubeTile.k[kBL1Idx]);

    AggregationMap aggregations;
    // 增加计算尾块的逻辑
    for (int64_t mL1Idx = 0; mL1Idx < orgM; mL1Idx += cubeTile.m[1]) {
        for (int64_t nL0Idx = 0; nL0Idx < orgN; nL0Idx += cubeTile.n[0]) {
            auto mL1Size = std::min(orgM - mL1Idx, cubeTile.m[1]);
            auto nL1Size = std::min(orgN - nL0Idx, cubeTile.n[0]);

            auto cTilePtr = cTensorPtr->View(function, {mL1Size, nL1Size}, {mL1Idx, nL0Idx});
            L1NormalLoad<isTransA, isTransB>(function, matmulInputs,
                {cTilePtr, mL1Idx, nL0Idx, stepK, mL1Size, nL1Size, orgKa}, tileShape, aggregations);
        }
    }

    if (matmulInputs.cTensorPtr != nullptr) {
        DoAMulB<true>(function, aggregations, matmulInputs, {tileShape, cTensorPtr}, params);
    } else {
        DoAMulB<false>(function, aggregations, matmulInputs, {tileShape, cTensorPtr}, params);
    }
}

void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &cTensorPtr, const MatmulAttrParam &params)
{
    if (params.transA && params.transB) {
        TiledInnerAMulB<true, true>(function, tileShape, operandVec, cTensorPtr, params);
    } else if (params.transA && !params.transB) {
        TiledInnerAMulB<true, false>(function, tileShape, operandVec, cTensorPtr, params);
    } else if (!params.transA && params.transB) {
        TiledInnerAMulB<false, true>(function, tileShape, operandVec, cTensorPtr, params);
    } else {
        TiledInnerAMulB<false, false>(function, tileShape, operandVec, cTensorPtr, params);
    }
}
} // namespace Deprecate

const int32_t MATRIX_SHAPE_DIM = 2;

template <typename T>
auto CeilAlign(T num_1, T num_2) -> T
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

inline bool CheckValidShape(const LogicalTensorPtr &tensorPtr)
{
    if (tensorPtr == nullptr) {
        return false;
    }
    return tensorPtr->GetDynValidShape().size() == SHAPE_DIM2;
}

template <typename T1, typename T2 = T1>
LogicalTensorPtr AddOpView(Function &function, const LogicalTensorPtr &srcTensorPtr,
                           const MatmulTensorInfo &dstTensorInfo, const std::map<std::string, T1> opAttr = {},
                           const std::map<std::string, T2> extraOpAttr = {})
{
    OP_CHECK(true, {
        ASSERT(srcTensorPtr != nullptr) << "Original tensor for OpView operation is nullptr." << std::endl; });
    auto dstShape = dstTensorInfo.shape;
    if (dstTensorInfo.transFlag) {
        OP_CHECK(true, {
             ASSERT(dstShape.size() == SHAPE_DIM2) << "destination shape dimension is invalid: "
                                     << "Expected dimensions: "
                                     << SHAPE_DIM2 << ", actual dimensions: " << dstShape.size() << std::endl;
        });
        std::swap(dstShape[0], dstShape[1]);
    }
    LogicalTensorPtr dstTensorPtr =
        std::make_shared<LogicalTensor>(function, dstTensorInfo.dtype, dstShape, SymbolicScalar::FromConcrete(dstShape),
                                        dstTensorInfo.format, dstTensorInfo.name, dstTensorInfo.nodeType);
    dstTensorPtr->UpdateDynValidShape(
        GetViewValidShape(srcTensorPtr->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
    if (dstTensorInfo.transFlag) {
        auto &dstValidShape = dstTensorPtr->GetDynValidShape();
        OP_CHECK(true, {
            ASSERT(dstValidShape.size() == SHAPE_DIM2) << "dstValidShape dimension is invalid: "
                                     << "Expected dimensions: " <<SHAPE_DIM2
                                     << ", actual dimensions: " << dstValidShape.size() << std::endl;
        });
        std::swap(dstValidShape[0], dstValidShape[1]);
    }
    auto &viewOp = function.AddOperation(Opcode::OP_VIEW, {srcTensorPtr}, {dstTensorPtr});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        dstTensorInfo.offset, SymbolicScalar::FromConcrete(dstTensorInfo.offset), dstTensorPtr->GetDynValidShape());
    viewAttribute->SetToType(dstTensorInfo.memType);
    viewOp.SetOpAttribute(viewAttribute);
    for (const auto &attrPair : opAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    for (const auto &attrPair : extraOpAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    return dstTensorPtr;
}

LogicalTensorPtr AddOpView(Function &function, const LogicalTensorPtr &srcTensorPtr,
                           const MatmulTensorInfo &dstTensorInfo)
{
    return AddOpView<int64_t>(function, srcTensorPtr, dstTensorInfo);
}

void SetAMulBAttr(const MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam, Operation &op)
{
    OP_CHECK(true,
        {
                ASSERT(tensorGraphNodes.aTensorPtr != nullptr && tensorGraphNodes.bTensorPtr != nullptr &&
           tensorGraphNodes.outTensorPtr != nullptr)
        << "Expected aTensorPtr, bTensorPtr, and outTensorPtr to be non-nullptr." << std::endl;
        });

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.aTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.bTensorPtr->Format()) << 1) |
                     // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                     (static_cast<int64_t>(tensorGraphNodes.outTensorPtr->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, attrParam.mValue);
    op.SetAttribute(A_MUL_B_ACT_K, attrParam.kValue);
    op.SetAttribute(A_MUL_B_ACT_N, attrParam.nValue);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(attrParam.reluType));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, attrParam.scaleValue));
    }
}

void SetTensorGraphAttr(Operation &op, const MatmulExtendParam &param, bool gmAccumulationFlag, const MatmulAttrParam &attrParam)
{
    op.SetAttribute(A_MUL_B_GM_ACC, gmAccumulationFlag);
    op.SetAttribute(A_MUL_B_TRANS_A, attrParam.transA);
    op.SetAttribute(A_MUL_B_TRANS_B, attrParam.transB);
    op.SetAttribute(A_MUL_B_BIAS_ATTR, (param.biasTensor.GetStorage() != nullptr));
    op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(param.reluType));
    // means perchannel
    if (param.scaleTensor.GetStorage() != nullptr) {
        op.SetAttribute(A_MUL_B_VECTOR_QUANT_FLAG, true);
    }
    // means pertensor
    if (fabs(param.scaleValue - 0) > EPSILON) {
        uint32_t scaleValueTmp = 0;
        memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &param.scaleValue, sizeof(param.scaleValue));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, static_cast<uint64_t>(scaleValueTmp)));
    }

    auto matrixSize = TileShape::Current().GetMatrixSize();
    if (matrixSize.size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[M_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[N_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[K_INDEX]);
}

void SetMatmulAttrParam(const Operation &op, MatmulAttrParam &param)
{
    param.mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    param.kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    param.nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;
    param.reluType = (op.HasAttr(A_MUL_B_RELU_ATTR)) ? op.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    param.scaleValue = (op.HasAttr(A_MUL_B_SCALE_ATTR)) ? op.GetElementAttribute(A_MUL_B_SCALE_ATTR).GetUnsignedData()
                                                        : Element(DataType::DT_UINT64, 0).GetUnsignedData();
    param.hasBias = (op.HasAttr(A_MUL_B_BIAS_ATTR)) ? op.GetBoolAttribute(A_MUL_B_BIAS_ATTR) : false;
    param.hasScale = (op.HasAttr(A_MUL_B_VECTOR_QUANT_FLAG)) ? op.GetBoolAttribute(A_MUL_B_VECTOR_QUANT_FLAG) : false;
    param.transA = (op.HasAttr(A_MUL_B_TRANS_A)) ? op.GetBoolAttribute(A_MUL_B_TRANS_A) : false;
    param.transB = (op.HasAttr(A_MUL_B_TRANS_B)) ? op.GetBoolAttribute(A_MUL_B_TRANS_B) : false;
    param.gmAccumulationFlag = (op.HasAttr(A_MUL_B_GM_ACC)) ? op.GetBoolAttribute(A_MUL_B_GM_ACC) : false;
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
                         const MatmulAttrParam &param, MatmulGraphNodes &tensorGraphNodes)
{
    size_t operandVecSize = SHAPE_DIM2 + static_cast<size_t>(param.hasScale + param.hasBias + param.gmAccumulationFlag);
    OP_CHECK(true, {
            ASSERT(operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: "
        << "Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", SHAPE_DIM2: " << SHAPE_DIM2 << ", hasScale: " << param.hasScale << ", hasBias: " << param.hasBias
        << ", gmAccumulationFlag: " << param.gmAccumulationFlag << std::endl;
    });

    tensorGraphNodes.aTensorPtr = operandVec[0];
    tensorGraphNodes.bTensorPtr = operandVec[1];
    OP_CHECK(true,
    {     ASSERT(tensorGraphNodes.aTensorPtr != nullptr && tensorGraphNodes.bTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr." << std::endl; });

    OP_CHECK(true, { ASSERT(cTensorPtr != nullptr) << "cTensorPtr is nullptr." << std::endl;});
    tensorGraphNodes.outTensorPtr = cTensorPtr;

    size_t extraDim = static_cast<size_t>(param.hasScale) | (static_cast<size_t>(param.hasBias) << 1) |
                      (static_cast<size_t>(param.gmAccumulationFlag) << 2);  // 2含义：编码偏移
    switch (extraDim) {
        case 0:  // 无bias，无scale, 无gmTensor
            break;
        case 1:  // 有scale
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 2:  // 2含义：有bias
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 3:  // 3含义：有bias, 有scale
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM3];
            break;
        case 4:  // 4含义：有gmTensor
            tensorGraphNodes.gmAccumulationTensorPtr = operandVec[SHAPE_DIM2];
            break;
        default:
            OP_CHECK(true, { ASSERT(false) << "Invalid tensor graph\n";});
    }
}

void CheckOperandShape(const Tensor &operand1, const Tensor &operand2)
{
    OP_CHECK(true, {
            ASSERT(operand1.GetShape().size() == operand2.GetShape().size())
        << "Shape dimension mismatch between operand1 and operand2. "
        << "operand1 shape size: " << operand1.GetShape().size()
        << ", operand2 shape size: " << operand2.GetShape().size() << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(operand1.GetShape().size() == operand1.GetStorage()->offset.size())
        << "Shape dimension mismatch with offset size for operand1. "
        << "shape size: " << operand1.GetShape().size() << ", offset size: " << operand1.GetStorage()->offset.size()
        << std::endl;
    });

    OP_CHECK(true, {
            ASSERT(operand2.GetShape().size() == operand2.GetStorage()->offset.size())
        << "Shape dimension mismatch with offset size for operand2. "
        << "shape size: " << operand2.GetShape().size() << ", offset size: " << operand2.GetStorage()->offset.size()
        << std::endl;
    });

    OP_CHECK(true, {
    ASSERT(operand1.GetShape().size() >= SHAPE_DIM2)
        << "The dimension of operand1 must be larger than 2! The dimensin of operand1:" << operand1.GetShape().size()
        << std::endl;
    });

    OP_CHECK(true, {
            ASSERT(operand2.GetShape().size() >= SHAPE_DIM2)
        << "The dimension of operand2 must be larger than 2! The dimensin of operand2:" << operand2.GetShape().size()
        << std::endl;
    });

    for (size_t i = 0; i < operand1.GetShape().size(); ++i) {
        OP_CHECK(true, {
            ASSERT(operand1.GetShape()[i] > 0)
            << "The value of the " << i << "-th dimension of operand1 must be larger than 0" << std::endl;
        });
    }

    for (size_t i = 0; i < operand2.GetShape().size(); ++i) {
        OP_CHECK(true, {
             ASSERT(operand2.GetShape()[i] > 0)
            << "The value of the " << i << "-th dimension of operand2 must be larger than 0" << std::endl;
        });
    }
}

void CheckCubeTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int32_t kBL1Idx = 2;
    const int64_t kL0 = cubeTile.k[0];
    const int64_t kL1a = cubeTile.k[1];
    const int64_t kL1b = cubeTile.k[kBL1Idx];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t mL1 = cubeTile.m[1];
    const int64_t nL0 = cubeTile.n[0];
    const int64_t nL1 = cubeTile.n[1];
    OP_CHECK(true, {
        ASSERT(kL0 > 0 && kL1a > 0 && kL1b > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0)
            << "Current kL0: " << kL0 << ", kL1a: " << kL1a << ", kL1b: " << kL1b << ", mL0: " << mL0
            << ", mL1: " << mL1 << ", nL0: " << nL0 << ", nL1: " << nL1
            << " Requirement: kL0 > 0 && kL1a > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(kL0 <= kL1a && kL1a % kL0 == 0) << "Current kL0: " << kL0 << ", kL1a: " << kL1a
                                               << ", Requirement: kL0 <= kL1a && kL1a % kL0 == 0" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(kL0 <= kL1b && kL1b % kL0 == 0) << "Current kL0: " << kL0 << ", kL1b: " << kL1b
                                               << ", Requirement: kL0 <= kL1b && kL1b % kL0 == 0" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(nL0 <= nL1 && nL1 % nL0 == 0)
            << "Current nL0: " << nL0 << ", nL1: " << nL1 << ", Requirement: nL0 <= nL1 && nL1 % nL0 == 0" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(mL0 <= mL1 && mL1 % mL0 == 0)
            << "Current mL0: " << mL0 << ", mL1: " << mL1 << ", Requirement: mL0 <= mL1 && mL1 % mL0 == 0" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
            << "Current length of kL0: " << (kL0 * BytesOf(operand1.GetDataType()))
            << " bytes, the length must be aligned to 32 bytes" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(nL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0)
            << "Current length of nL0: " << (kL0 * BytesOf(operand1.GetDataType()))
            << " bytes, the length must be aligned to 32 bytes" << std::endl;
    });
    if (operand1.Format() == TileOpFormat::TILEOP_ND) {
        if (attrParam.transA) { // For ND A transpose, mL0 must be 32B aligned
            OP_CHECK(true, { ASSERT(mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                                << "Current length of mL0: " << (mL0 * BytesOf(operand1.GetDataType()))
                                << " bytes, the length must be aligned to 32 bytes when A is transposed" << std::endl;
            });
        }
    }
}

void CheckOperandShapeBound(const Tensor &operand) {
    auto opFormat = operand.Format();
    if (opFormat == TileOpFormat::TILEOP_ND) {
        OP_CHECK(true, {
            ASSERT(operand.GetShape().back() <= SHAPE_INNER_AXIS_MAX_SIZE)
                << "Current inner axis: " << operand.GetShape().back()
                << ", when input is ND format, inner axis must be less than 65535" << std::endl;
        });
        OP_CHECK(true, {
            ASSERT(operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] <= std::numeric_limits<int32_t>::max())
                << "Current outer axis: " << (operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2])
                << ", when input is ND format, outer axis must be less than 2^31 - 1" << std::endl;
        });
    } else {
        OP_CHECK(true, {
            ASSERT(operand.GetShape().back() * BytesOf(operand.GetDataType()) % ALIGN_SIZE_32 == 0)
                << "Current inner axis: " << operand.GetShape().back() << ", when input "
                << "is NZ format, inner axis shape must be 32-byte aligned" << std::endl;
        });
        OP_CHECK(true, {
            ASSERT(operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] % ALIGN_SIZE_16 == 0)
                << "Current outer axis: " << operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2]
                << ", when input "
                << "is NZ format, outer axis shape must be 16-element aligned" << std::endl;
        });
    }
}

void CheckNZFormatAligned(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int64_t kL0 = cubeTile.k[0];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t nL0 = cubeTile.n[0];
    auto opFormatA = operand1.Format();
    auto opFormatB = operand2.Format();
    if (opFormatA == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transA) {
            OP_CHECK(true, {ASSERT(mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                                << "Current length of mL0: " << (mL0 * BytesOf(operand1.GetDataType()))
                                << " bytes, the length must be aligned to 32 bytes" << std::endl;
            });
            OP_CHECK(true, {
                ASSERT(kL0 % ALIGN_SIZE_16 == 0) << "Current length of kL0: " << kL0
                                                 << " elements, the length must be aligned to 16 elements" << std::endl;
            });
        } else {
            OP_CHECK(true, { ASSERT(kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                                << "Current length of kL0: " << (kL0 * BytesOf(operand1.GetDataType()))
                                << " bytes, the length must be aligned to 32 bytes" << std::endl;
            });
            OP_CHECK(true, {
                ASSERT(mL0 % ALIGN_SIZE_16 == 0) << "Current length of mL0: " << mL0
                                                 << " elements, the length must be aligned to 16 elements" << std::endl;
            });
        }
    }
    if (opFormatB == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transB) {
            OP_CHECK(true, {ASSERT(kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                                << "Current length of kL0: " << (kL0 * BytesOf(operand1.GetDataType()))
                                << " bytes, the length must be aligned to 32 bytes" << std::endl;
            });
            OP_CHECK(true, {
                ASSERT(nL0 % ALIGN_SIZE_16 == 0) << "Current length of nL0: " << nL0
                                                 << " elements, the length must be aligned to 16 elements" << std::endl;
            });
        } else {
            OP_CHECK(true, {ASSERT(nL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                                << "Current length of nL0: " << (nL0 * BytesOf(operand1.GetDataType()))
                                << " bytes, the length must be aligned to 32 bytes" << std::endl;
            });
            OP_CHECK(true, {
                ASSERT(kL0 % ALIGN_SIZE_16 == 0) << "Current length of kL0: " << kL0
                                                 << " elements, the length must be aligned to 16 elements" << std::endl;
            });
        }
    }
}

void CheckCMatrixNZFormatAligned(const DataType &outType, const Tensor &operand, const MatmulAttrParam &attrParam) {
    auto &cubeType = TileShape::Current().GetCubeTile();
    const int64_t nL0 = cubeType.n[0];
    if (attrParam.isCMatrixNZ) {
        int64_t nView = attrParam.transB ? operand.GetShape()[0] : operand.GetShape()[1];
        if (outType == DataType::DT_INT32) {
            OP_CHECK(true, {
                ASSERT(nView % ALIGN_SIZE_16 == 0)
                    << "Current nView: " << nView
                    << " elements, nView must be aligned to 16 elements when CMatrix is NZ and outType is int32"
                    << std::endl;
            });

            OP_CHECK(true, {
                ASSERT(nL0 % ALIGN_SIZE_16 == 0)
                    << "Current nL0: " << nL0
                    << " elements, nL0 must be aligned to 16 elements when CMatrix is NZ and outType is int32"
                    << std::endl;
            });
        } else {
            OP_CHECK(true, {
                ASSERT(nView * BytesOf(outType) % ALIGN_SIZE_32 == 0)
                    << "Current nView: " << (nView * BytesOf(outType))
                    << " bytes, nView must be aligned to 32 bytes when CMatrix is NZ" << std::endl;
            });

            OP_CHECK(true, {
                ASSERT(nL0 * BytesOf(outType) % ALIGN_SIZE_32 == 0)
                    << "Current nL0: " << (nL0 * BytesOf(outType))
                    << " bytes, nL0 must be aligned to 32 bytes when CMatrix is NZ" << std::endl;
            });
        }
    }
}

void CheckBiasParam(DataType inDtype, const MatmulExtendParam &param = {}) {
    if (param.biasTensor.GetStorage() == nullptr) {
        return;
    }
    OP_CHECK(true,
        { ASSERT(param.biasTensor.Format() == TileOpFormat::TILEOP_ND) << "Only support TILEOP_ND." << std::endl; });

    if (inDtype == DataType::DT_BF16 || inDtype == DataType::DT_FP32) {
        OP_CHECK(true, {
            ASSERT(param.biasTensor.GetDataType() == DataType::DT_FP32) << "When input tensor is DT_BF16 or DT_FP32, "
                                                                           "bias must be DT_FP32."
                                                                        << std::endl;
        });
    } else if (inDtype == DataType::DT_FP16) {
        OP_CHECK(true, {
            ASSERT(param.biasTensor.GetDataType() == DataType::DT_FP32 ||
                   param.biasTensor.GetDataType() == DataType::DT_FP16)
                << "When input tensor is DT_FP16, bias must be DT_FP32 or DT_FP16." << std::endl;
        });
    } else if (inDtype == DataType::DT_INT8) {
        OP_CHECK(true, {
            ASSERT(param.biasTensor.GetDataType() == DataType::DT_INT32)
                << "When input tensor is DT_INT8, bias must be DT_INT32." << std::endl;
        });
    }
    OP_CHECK(true, {
        ASSERT(param.biasTensor.GetShape().size() == SHAPE_DIM2)
            << "Bias tensor shape dimension mismatch: "
            << "Expected " << SHAPE_DIM2 << " dimensions, got " << param.biasTensor.GetShape().size() << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(param.biasTensor.GetShape()[0] == 1)
            << "Bias tensor first dimension mismatch: "
            << "Expected first dimension to be 1, got " << param.biasTensor.GetShape()[0] << std::endl;
    });
}

void CheckFixpipeParam(DataType inDtype, DataType outDtype, const MatmulExtendParam &param = {}) {
    if (param.scaleTensor.GetStorage() != nullptr) {
        OP_CHECK(true, {
            ASSERT(param.scaleTensor.Format() == TileOpFormat::TILEOP_ND) << "Only support TILEOP_ND." << std::endl;
        });

        OP_CHECK(true, {
            ASSERT(param.scaleTensor.GetDataType() == DataType::DT_INT64 ||
                   param.scaleTensor.GetDataType() == DataType::DT_UINT64)
                << "scaleTensor dataType: " << DataType2String(param.scaleTensor.GetDataType())
                << ". scaleTensor only support int64 and uint64 dtype currently." << std::endl;
        });

        OP_CHECK(true, {
            ASSERT(outDtype == DataType::DT_FP16 && inDtype == DataType::DT_INT8)
                << "Data type mismatch in fixpipe scenario. "
                << "Expected inDtype to be DT_INT8 and outDtype to be DT_FP16." << std::endl;
        });

        OP_CHECK(true, {
            ASSERT(param.scaleTensor.GetShape()[0] == 1)
                << "Scale tensor first dimension mismatch. "
                << "Expected first dimension to be 1, got " << param.scaleTensor.GetShape()[0] << std::endl;
        });
    }
    if (fabs(param.scaleValue - 0) > EPSILON) {
        OP_CHECK(true, {
            ASSERT(outDtype == DataType::DT_FP16 && inDtype == DataType::DT_INT8)
                << "Data type mismatch in pertensor scenario. "
                << "Expected inDtype to be DT_INT8 and outDtype to be DT_FP16." << std::endl;
        });
    }
    if (inDtype == DataType::DT_INT8 && outDtype == DataType::DT_FP16) {
        ASSERT(fabs(param.scaleValue - 0) > EPSILON || param.scaleTensor.GetStorage() != nullptr);

        OP_CHECK(true, {
            ASSERT(fabs(param.scaleValue - 0) > EPSILON || param.scaleTensor.GetStorage() != nullptr)
                << "Quantization error in INT8→FP16 path: scaleValue must not be 0.0f, OR scaleTensor must not be null."
                << std::endl;
        });
    }
}

void CheckGmAccumulationParam(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &param = {}) {
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (!cubeTile.enableSplitK) {
        return;
    }
    OP_CHECK(attrParam.isCMatrixNZ, ASSERT(false)
                                        << "Gm accumulation with output NZ format is not supported." << std::endl;);
    OP_CHECK(true, {
        ASSERT(param.scaleTensor.GetStorage() == nullptr && param.biasTensor.GetStorage() == nullptr &&
               fabs(param.scaleValue - 0) < EPSILON)
            << "Fixpipe and bias cannot be used simultaneously with GM ACC" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(outType != DT_FP16 && outType != DT_BF16) << "Output data type only support FP32 and INT32 when using GM accumulated" << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
            << "Both aMatrix and bMatrix cannot get storage" << std::endl;
    });
    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    OP_CHECK(true, {
        ASSERT(aMatrixValidShape.size() == SHAPE_DIM2 && bMatrixValidShape.size() == SHAPE_DIM2 &&
               cubeTile.k.size() == MAX_K_DIM_SIZE)
            << "The validShapes of aMatrix and bMatrix must be 2 Dim. Additionally, the K TileShape must be 3 Dim"
            << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(aMatrix.GetShape().size() == SHAPE_DIM2 && bMatrix.GetShape().size() == SHAPE_DIM2)
            << "The shapes of aMatrix and bMatrix must be 2 Dim" << std::endl;
    });
    int64_t kSizeA = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t kSizeB = attrParam.transB ? bMatrix.GetShape()[1] : bMatrix.GetShape()[0];
    OP_CHECK(true, {
        ASSERT(kSizeA == kSizeB) << "Matrix K dimemsion mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB
                                 << std::endl;
    });
}

void CheckMatmulOperands(DataType outType, const Tensor &operand1, const Tensor &operand2,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &param = {}) {
    OP_CHECK(true, {
        ASSERT(outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16 ||
               outType == DataType::DT_INT32)
            << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16, DT_INT32 are supported.";
    });
    OP_CHECK(true, {
        ASSERT(operand1.GetDataType() == operand2.GetDataType())
            << "input dataType must be consistent. "
            << "operand1 dataType: " << DataType2String(operand1.GetDataType())
            << ", operand2 dataType: " << DataType2String(operand2.GetDataType()) << std::endl;
    });
    // GM Acc valid check
    CheckGmAccumulationParam(outType, operand1, operand2, attrParam, param);
    // shape valid check
    CheckOperandShape(operand1, operand2);
    // tile valid check
    CheckCubeTiling(operand1, operand2, attrParam);
    // shape bound valid check
    CheckOperandShapeBound(operand1);
    CheckOperandShapeBound(operand2);
    // input NZ format valid check
    CheckNZFormatAligned(operand1, operand2, attrParam);
    // output NZ format valid check
    CheckCMatrixNZFormatAligned(outType, operand2, attrParam);
    // bias and scale valid check
    CheckBiasParam(operand1.GetDataType(), param);
    CheckFixpipeParam(operand1.GetDataType(), outType, param);
}

void SetMatmulTileInfo(const TileShape &tileShape, const MatmulAttrParam &attrParam,
    const MatmulGraphNodes &tensorGraphNodes, MatmulTileInfo &tileInfo) {
    OP_CHECK(true, {
        ASSERT(tensorGraphNodes.aTensorPtr != nullptr && tensorGraphNodes.bTensorPtr != nullptr)
            << "Both inputs must be non-nullptr." << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(tensorGraphNodes.aTensorPtr->GetShape().size() == SHAPE_DIM2 &&
               tensorGraphNodes.bTensorPtr->GetShape().size() == SHAPE_DIM2)
            << "Invalid tensor shape dimension, expected both tensors to have exactly. " << SHAPE_DIM2
            << " dimensions. "
            << "aTensorPtr shape dim: " << tensorGraphNodes.aTensorPtr->GetShape().size()
            << ", bTensorPtr shape dim: " << tensorGraphNodes.bTensorPtr->GetShape().size() << std::endl;
    });

    tileInfo.mView = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[1] : tensorGraphNodes.aTensorPtr->shape[0];
    tileInfo.nView = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[0] : tensorGraphNodes.bTensorPtr->shape[1];
    int64_t kViewA = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[0] : tensorGraphNodes.aTensorPtr->shape[1];
    int64_t kViewB = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[1] : tensorGraphNodes.bTensorPtr->shape[0];

    OP_CHECK(true, {
        ASSERT(kViewA == kViewB) << "Matrix K dimemsion mismatch, kViewA: " << kViewA << ", kViewB: " << kViewB
                                 << std::endl;
    });
    tileInfo.kView = kViewA;

    auto &cubeTile = tileShape.GetCubeTile();
    tileInfo.tileML0 = cubeTile.m[0];
    tileInfo.tileML1 = cubeTile.m[1];
    tileInfo.tileNL0 = cubeTile.n[0];
    tileInfo.tileNL1 = cubeTile.n[1];
    tileInfo.tileKL0 = cubeTile.k[0];
    tileInfo.tileKAL1 = cubeTile.k[1];
    tileInfo.tileKBL1 = cubeTile.k[2]; // 2含义：kBL1 tile的偏移
    int64_t tileKL1Min = std::min(tileInfo.tileKAL1, tileInfo.tileKBL1);
    int64_t tileKL1Max = std::max(tileInfo.tileKAL1, tileInfo.tileKBL1);

    OP_CHECK(true, {
        ASSERT(tileKL1Max >= kViewA || (tileKL1Max > 0 && tileKL1Min > 0 && tileKL1Max % tileKL1Min == 0))
            << "Invalid tileKL1 configuration: tileKL1Max: " << tileKL1Max << ", kViewA: " << kViewA
            << ", tileKL1Min: " << tileKL1Min
            << ". Must satisfy: tileKL1Max >= kViewA OR (all values > 0 and tileKL1Max is divisible by tileKL1Min)."
            << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(tileInfo.tileKL0 > 0 && tileKL1Min % tileInfo.tileKL0 == 0)
            << "tileKL0: " << tileInfo.tileKL0 << ", tileKL1Min: " << tileKL1Min
            << ". Must have: tileKL0 > 0 AND tileKL1Min is divisible by tileKL0." << std::endl;
    });
}

LogicalTensorPtr LinkBias(Function &function, const MatmulGraphNodes &tensorGraphNodes, const TileInfo &tileInfoL1,
                          const TileInfo &tileInfoBT)
{
    if (tensorGraphNodes.biasTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo biasL1TensorInfo{
        "biasL1Tensor",  tensorGraphNodes.biasTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.biasTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr biasL1TensorPtr =
        AddOpView<int64_t>(function, tensorGraphNodes.biasTensorPtr, biasL1TensorInfo,
                           {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    DataType biasBtType =
        (tensorGraphNodes.aTensorPtr->Datatype() == DataType::DT_INT8) ? DataType::DT_INT32 : DataType::DT_FP32;
    MatmulTensorInfo biasBtTensorInfo{"biasBtTensor",    biasBtType,      tileInfoBT.shape,
                                      tileInfoBT.offset, NodeType::LOCAL, biasL1TensorPtr->Format(),
                                      MemoryType::MEM_BT};
    LogicalTensorPtr biasBtTensorPtr = AddOpView(function, biasL1TensorPtr, biasBtTensorInfo);
    return biasBtTensorPtr;
}

LogicalTensorPtr LinkScale(Function &function, const MatmulGraphNodes &tensorGraphNodes, const TileInfo &tileInfoL1,
                           const TileInfo &tileInfoFB)
{
    if (tensorGraphNodes.scaleTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo scaleL1TensorInfo{
        "scaleL1Tensor", tensorGraphNodes.scaleTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.scaleTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr scaleL1TensorPtr =
        AddOpView<int64_t>(function, tensorGraphNodes.scaleTensorPtr, scaleL1TensorInfo,
                           {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    MatmulTensorInfo scaleFbTensorInfo{"scaleFbTensor",
                                       scaleL1TensorPtr->Datatype(),
                                       tileInfoFB.shape,
                                       tileInfoFB.offset,
                                       NodeType::LOCAL,
                                       scaleL1TensorPtr->Format(),
                                       MemoryType::MEM_FIX_QUANT_PRE};
    LogicalTensorPtr scaleFbTensorPtr = AddOpView(function, scaleL1TensorPtr, scaleFbTensorInfo);
    return scaleFbTensorPtr;
}

LogicalTensorPtr LinkTensorA(Function &function, const MatmulGraphNodes &tensorGraphNodes,
                             const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo,
                             const MatmulIterInfo &iterInfo, LogicalTensorPtr &aL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKAL1 == 0) {
        std::vector<int64_t> aL1Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kAL1Size, iterInfo.mL0Size}
                                                           : std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kAL1Size};
        std::vector<int64_t> aL1Offset = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kOffset, iterInfo.mOffset}
                                                            : std::vector<int64_t>{iterInfo.mOffset, iterInfo.kOffset};
        MatmulTensorInfo aL1TensorInfo{
            "aL1Tensor",     tensorGraphNodes.aTensorPtr->Datatype(), aL1Shape,          aL1Offset,
            NodeType::LOCAL, tensorGraphNodes.aTensorPtr->Format(),   MemoryType::MEM_L1};
        aL1TensorPtr = AddOpView(function, tensorGraphNodes.aTensorPtr, aL1TensorInfo);
    }
    std::vector<int64_t> aL0Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kL0Size, iterInfo.mL0Size}
                                                       : std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kL0Size};
    std::vector<int64_t> aL0Offset = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKAL1, 0}
                                                        : std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKAL1};
    MatmulTensorInfo aL0TensorInfo{"aL0Tensor",
                                   tensorGraphNodes.aTensorPtr->Datatype(),
                                   aL0Shape,
                                   aL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.aTensorPtr->Format(),
                                   MemoryType::MEM_L0A,
                                   attrParam.transA};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(aL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(aL0Shape);
    LogicalTensorPtr aL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(function, aL1TensorPtr, aL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transA}}, {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return aL0TensorPtr;
}

LogicalTensorPtr LinkTensorB(Function &function, const MatmulGraphNodes &tensorGraphNodes,
                             const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo,
                             const MatmulIterInfo &iterInfo, LogicalTensorPtr &bL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKBL1 == 0) {
        std::vector<int64_t> bL1Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kBL1Size}
                                                           : std::vector<int64_t>{iterInfo.kBL1Size, iterInfo.nL0Size};
        std::vector<int64_t> bL1Offset = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nOffset, iterInfo.kOffset}
                                                            : std::vector<int64_t>{iterInfo.kOffset, iterInfo.nOffset};
        MatmulTensorInfo bL1TensorInfo{
            "bL1Tensor",     tensorGraphNodes.bTensorPtr->Datatype(), bL1Shape,          bL1Offset,
            NodeType::LOCAL, tensorGraphNodes.bTensorPtr->Format(),   MemoryType::MEM_L1};
        bL1TensorPtr = AddOpView(function, tensorGraphNodes.bTensorPtr, bL1TensorInfo);
    }
    std::vector<int64_t> bL0Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kL0Size}
                                                       : std::vector<int64_t>{iterInfo.kL0Size, iterInfo.nL0Size};
    std::vector<int64_t> bL0Offset = (attrParam.transB) ? std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKBL1}
                                                        : std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKBL1, 0};
    MatmulTensorInfo bL0TensorInfo{"bL0Tensor",
                                   tensorGraphNodes.bTensorPtr->Datatype(),
                                   bL0Shape,
                                   bL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.bTensorPtr->Format(),
                                   MemoryType::MEM_L0B,
                                   attrParam.transB};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(bL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(bL0Shape);
    LogicalTensorPtr bL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(function, bL1TensorPtr, bL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transB}}, {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return bL0TensorPtr;
}

void LinkAMulB(Function &function, const MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam,
    const MatmulIterInfo &iterInfo, MatmulGraphNodes &tileGraphNodes) {
    OP_CHECK(true, {
        ASSERT(tileGraphNodes.aTensorPtr != nullptr && tileGraphNodes.bTensorPtr != nullptr &&
               tileGraphNodes.outTensorPtr != nullptr)
            << "Inputs must be non-nullptr." << std::endl;
    });
    std::vector<LogicalTensorPtr> aMulBInputs;
    std::vector<LogicalTensorPtr> aMulBOutputs;
    const std::string matmulOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (attrParam.gmAccumulationFlag) {
        // GM 累加场景
        OP_CHECK(true, {
            ASSERT(tensorGraphNodes.gmAccumulationTensorPtr != nullptr && attrParam.hasBias == false &&
                   attrParam.hasScale == false)
                << "In GM accumulation mode, neither bias nor scale is allowed." << std::endl;
        });
        tileGraphNodes.gmAccumulationTensorPtr = tensorGraphNodes.gmAccumulationTensorPtr->View(
            function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
        if (iterInfo.isFirstK) {
            aMulBInputs = {
                tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr, tileGraphNodes.gmAccumulationTensorPtr};
        } else {
            aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr, tileGraphNodes.cL0PartialSumPtr,
                tileGraphNodes.gmAccumulationTensorPtr};
        }
    } else {
        // 普通场景
        if (iterInfo.isFirstK) {
            aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr};
            if (tileGraphNodes.biasTensorPtr != nullptr) {
                aMulBInputs.push_back(tileGraphNodes.biasTensorPtr);
            }
            if (tileGraphNodes.scaleTensorPtr != nullptr) {
                aMulBInputs.push_back(tileGraphNodes.scaleTensorPtr);
            }
        } else {
            aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr, tileGraphNodes.cL0PartialSumPtr};
        }
    }
    if (iterInfo.isLastK) {
        aMulBOutputs = {tileGraphNodes.outTensorPtr};
    } else {
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, tileGraphNodes.outTensorPtr->Datatype(), tileGraphNodes.outTensorPtr->GetShape());
        if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
            tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
                {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
        }
        aMulBOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto &aMulBOp = function.AddOperation(matmulOpStr, aMulBInputs, aMulBOutputs);
    SetAMulBAttr(tensorGraphNodes, attrParam, aMulBOp);
}

void UpdateIterInfo(const MatmulTileInfo &tileInfo, MatmulIterInfo &iterInfo) {
    iterInfo.kAL1Size = std::min(tileInfo.tileKAL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kBL1Size = std::min(tileInfo.tileKBL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kL0Size = std::min(tileInfo.tileKL0, tileInfo.kView - iterInfo.kOffset);
    iterInfo.isFirstK = (iterInfo.kOffset == 0);
    iterInfo.isLastK = (iterInfo.kOffset + tileInfo.tileKL0 >= tileInfo.kView);

    OP_CHECK(true, {
        ASSERT(tileInfo.tileKAL1 > 0 && tileInfo.tileKBL1 > 0)
            << "Both tileKAL1 and tileKBL1 must be positive: tileKAL1: " << tileInfo.tileKAL1
            << "tileKBL1: " << tileInfo.tileKBL1 << std::endl;
    });
}

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
{
    MatmulAttrParam attrParam;
    SetMatmulAttrParam(op, attrParam);
    // tensor graph中的数据节点
    MatmulGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, attrParam, tensorGraphNodes);
    MatmulTileInfo tileInfo;
    SetMatmulTileInfo(tileShape, attrParam, tensorGraphNodes, tileInfo);
    MatmulIterInfo iterInfo;
    // tile graph中的数据节点
    MatmulGraphNodes tileGraphNodes;

    auto &cubeTile = tileShape.GetCubeTile();
    if (!cubeTile.enableMultiDataLoad) {
        Deprecate::TiledInnerAMulB(function, tileShape, operandVec, cTensorPtr, attrParam);
        return;
    }

    for (iterInfo.nOffset = 0; iterInfo.nOffset < tileInfo.nView; iterInfo.nOffset += tileInfo.tileNL0) {
        iterInfo.nL0Size = std::min(tileInfo.nView - iterInfo.nOffset, tileInfo.tileNL0);
        tileGraphNodes.biasTensorPtr =
            LinkBias(function, tensorGraphNodes, TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                     TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
        tileGraphNodes.scaleTensorPtr =
            LinkScale(function, tensorGraphNodes, TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                      TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
        for (iterInfo.mOffset = 0; iterInfo.mOffset < tileInfo.mView; iterInfo.mOffset += tileInfo.tileML0) {
            iterInfo.mL0Size = std::min(tileInfo.mView - iterInfo.mOffset, tileInfo.tileML0);
            tileGraphNodes.outTensorPtr =
                cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
            LogicalTensorPtr aL1TensorPtr = nullptr;
            LogicalTensorPtr bL1TensorPtr = nullptr;
            for (iterInfo.kOffset = 0; iterInfo.kOffset < tileInfo.kView; iterInfo.kOffset += tileInfo.tileKL0) {
                UpdateIterInfo(tileInfo, iterInfo);
                tileGraphNodes.aTensorPtr =
                    LinkTensorA(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, aL1TensorPtr);
                tileGraphNodes.bTensorPtr =
                    LinkTensorB(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, bL1TensorPtr);
                LinkAMulB(function, tensorGraphNodes, attrParam, iterInfo, tileGraphNodes);
            }
        }
    }
}

void AddAMulBNode(const LogicalTensorPtr &aTensorPtr, const LogicalTensorPtr &bTensorPtr,
    const LogicalTensorPtr &cTensorPtr, const LogicalTensorPtr &gmTensorPtr, const MatmulAttrParam &attrParam,
    const MatmulExtendParam &param = {}) {
    if (CheckValidShape(aTensorPtr) && CheckValidShape(bTensorPtr)) {
        SymbolicScalar mSizeDyn =
            attrParam.transA ? aTensorPtr->GetDynValidShape()[1] : aTensorPtr->GetDynValidShape()[0];
        SymbolicScalar kSizeDyn =
            attrParam.transA ? aTensorPtr->GetDynValidShape()[0] : aTensorPtr->GetDynValidShape()[1];
        SymbolicScalar nSizeDyn =
            attrParam.transB ? bTensorPtr->GetDynValidShape()[0] : bTensorPtr->GetDynValidShape()[1];
        OP_CHECK(true, { ASSERT(cTensorPtr != nullptr) << "cTensorPtr is nullptr." << std::endl; });
        cTensorPtr->UpdateDynValidShape({mSizeDyn, nSizeDyn});
    }

    std::vector<LogicalTensorPtr> operandVec = {aTensorPtr, bTensorPtr};
    bool gmAccumulationFlag = false;
    if (gmTensorPtr != nullptr) {
        operandVec.push_back(gmTensorPtr);
        gmAccumulationFlag = true;
    }
    if (param.biasTensor.GetStorage() != nullptr) {
        operandVec.push_back(param.biasTensor.GetStorage());
    }
    if (param.scaleTensor.GetStorage() != nullptr) {
        operandVec.push_back(param.scaleTensor.GetStorage());
    }
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();

    OP_CHECK(true, { ASSERT(functionPtr != nullptr) << "functionPtr is nullptr." << std::endl; });
    auto &op = functionPtr->AddOperation(Opcode::OP_A_MUL_B, operandVec, {cTensorPtr});
    SetTensorGraphAttr(op, param, gmAccumulationFlag, attrParam);
}

Tensor ConstructTensorGraph(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix, const Tensor &gmMatrix,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &param = {}) {
    int64_t mSize = attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[0];
    int64_t kSizeA = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t kSizeB = attrParam.transB ? bMatrix.GetShape()[1] : bMatrix.GetShape()[0];
    int64_t nSize = attrParam.transB ? bMatrix.GetShape()[0] : bMatrix.GetShape()[1];

    OP_CHECK(true, {
        ASSERT(kSizeA == kSizeB) << "Matrix K dimemsion mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB
                                 << std::endl;
    });
    Tensor cMatrix(dataType, {mSize, nSize}, "TensorC");
    if (attrParam.isCMatrixNZ) {
        OP_CHECK(true, {
            ASSERT(BytesOf(dataType) > 0)
                << "BytesOf(dataType): " << BytesOf(dataType) << ". Must be positive." << std::endl;
        });
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        cMatrix = Tensor(dataType, {mSize, CeilAlign(nSize, c0Size)}, "TensorC", TileOpFormat::TILEOP_NZ);
    }
    AddAMulBNode(
        aMatrix.GetStorage(), bMatrix.GetStorage(), cMatrix.GetStorage(), gmMatrix.GetStorage(), attrParam, param);
    return cMatrix;
}

static Tensor AssembleGmAccumulationTensor(DataType outType, const Tensor gmAccumulationTensor,
    std::vector<int64_t> outSize, std::vector<SymbolicScalar> validShape, bool isCMatrixNZ) {
    OP_CHECK(true, {
        ASSERT(outSize.size() == SHAPE_DIM2 && validShape.size() == SHAPE_DIM2)
            << "Both outSize and validShape must be 2-element vectors" << std::endl;
    });
    OP_CHECK(true, { ASSERT(outSize[0] != 0 && outSize[1] != 0) << "Matrix size cannot be 0 " << std::endl; });
    Tensor assembleTensor(
        outType, {outSize[0], outSize[1]}, "", isCMatrixNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND);
    OP_CHECK(true,
        { ASSERT(assembleTensor.GetStorage() != nullptr) << "Can not get assembleTensor's storage" << std::endl; });
    assembleTensor.GetStorage()->UpdateDynValidShape({validShape[0], validShape[1]});
    Assemble(gmAccumulationTensor, {0, 0}, assembleTensor);
    return assembleTensor;
}

static Tensor ConstructGmAccumulationTensorGraph(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const MatmulAttrParam &attrParam) {
    auto &cubeTile = TileShape::Current().GetCubeTile();
    OP_CHECK(true, {
        ASSERT(aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
            << "Both aMatrix and bMatrix cannot get storage" << std::endl;
    });
    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aMatrixValidShape[1] : aMatrixValidShape[0];
    SymbolicScalar nValidShape = attrParam.transB ? bMatrixValidShape[0] : bMatrixValidShape[1];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    int64_t mSize = attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[0];
    int64_t kSize = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t nSize = attrParam.transB ? bMatrix.GetShape()[0] : bMatrix.GetShape()[1];
    TileShape::Current().SetVecTile({cubeTile.m[0], cubeTile.n[0]});
    Tensor gmAccumulationTensor =
        Full(Element(outType, static_cast<int64_t>(0)), outType, {mSize, nSize}, {mValidShape, nValidShape});
    std::vector<Tensor> gmPartialSums;
    OP_CHECK(true, { ASSERT(kL1TileShape != 0) << "kL1TileShape can not be 0" << std::endl; });
    const int64_t kLoop = (kSize + kL1TileShape - 1) / kL1TileShape;
    const int64_t kL1Size = std::min(kSize, kL1TileShape);
    for (int64_t kIdx = 0; kIdx < kLoop; ++kIdx) {
        int64_t kValidshape = std::min(kSize - kL1Size * kIdx, kL1Size);
        Tensor tensorA;
        if (attrParam.transA) {
            tensorA = View(aMatrix, {kL1Size, mSize}, {kValidshape, mValidShape}, {kL1Size * kIdx, 0});
        } else {
            tensorA = View(aMatrix, {mSize, kL1Size}, {mValidShape, kValidshape}, {0, kL1Size * kIdx});
        }
        Tensor tensorB;
        if (attrParam.transB) {
            tensorB = View(bMatrix, {nSize, kL1Size}, {nValidShape, kValidshape}, {0, kL1Size * kIdx});
        } else {
            tensorB = View(bMatrix, {kL1Size, nSize}, {kValidshape, nValidShape}, {kL1Size * kIdx, 0});
        }
        Tensor gmPartialSum = ConstructTensorGraph(outType, tensorA, tensorB, gmAccumulationTensor, attrParam);
        gmPartialSums.emplace_back(gmPartialSum);
    }
    gmAccumulationTensor = npu::tile_fwk::Reduce(gmPartialSums, ReduceMode::ATOMIC_ADD);
    return AssembleGmAccumulationTensor(
        outType, gmAccumulationTensor, {mSize, nSize}, {mValidShape, nValidShape}, attrParam.isCMatrixNZ);
}

Tensor Matmul(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    }
    return ConstructTensorGraph(outType, aMatrix, bMatrix, Tensor(), attrParam);
}

Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const MatmulExtendParam &param,
    bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam, param);
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    }
    return ConstructTensorGraph(outType, aMatrix, bMatrix, Tensor(), attrParam, param);
}

Tensor ABatchMulB3D(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    OP_CHECK(true, {
        ASSERT(operand1.GetShape().size() == operand2.GetShape().size() && operand1.GetShape().size() == SHAPE_DIM3)
            << "Shape dimension miamatch, expected exactly " << SHAPE_DIM3 << "dimension for both operands. "
            << "operand1: " << operand1.GetShape().size() << " , operand2: " << operand2.GetShape().size() << std::endl;
    });
    const int64_t batchSizeA = operand1.GetShape()[0];
    const int64_t batchSizeB = operand2.GetShape()[0];
    OP_CHECK(true, {
        ASSERT(batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1)
            << "batchSize invalid, only allowed batchSizeA = batchSizeB, batchSizeA = 1, or batchSizeB = 1: batchSizeA "
               "= "
            << batchSizeA << ". batchSizeB: " << batchSizeB << std::endl;
    });
    const int64_t orgM = attrParam.transA ? operand1.GetShape()[SHAPE_DIM2] : operand1.GetShape()[1];
    const int64_t orgKa = attrParam.transA ? operand1.GetShape()[1] : operand1.GetShape()[SHAPE_DIM2];
    const int64_t orgKb = attrParam.transB ? operand2.GetShape()[2] : operand2.GetShape()[1];
    const int64_t orgN = attrParam.transB ? operand2.GetShape()[1] : operand2.GetShape()[SHAPE_DIM2];
    OP_CHECK(true, { ASSERT(orgKa == orgKb) << "orgK mismatch: Ka: " << orgKa << ", Kb: " << orgKb << std::endl; });
    int64_t firstDimA = attrParam.transA ? orgKa : orgM;
    int64_t secondDimA = attrParam.transA ? orgM : orgKa;
    int64_t firstDimB = attrParam.transB ? orgN : orgKb;
    int64_t secondDimB = attrParam.transB ? orgKb : orgN;
    int64_t batchSize = std::max(batchSizeA, batchSizeB);
    auto operand2D1 = Reshape(operand1, {batchSizeA * firstDimA, secondDimA});
    auto operand2D2 = Reshape(operand2, {batchSizeB * firstDimB, secondDimB});
    Tensor result(dataType, {batchSize * orgM, orgN});
    if (attrParam.isCMatrixNZ) {
        result = Tensor(dataType, {batchSize * orgM, orgN}, "BatchMatmulOutputNz", TileOpFormat::TILEOP_NZ);
    }
    CheckMatmulOperands(dataType, operand2D1, operand2D2, attrParam);
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int64_t i = 0; i < batchSize; i++) {
        int64_t offsetA = batchSizeA == 1 ? 0 : i * firstDimA;
        int64_t offsetB = batchSizeB == 1 ? 0 : i * firstDimB;
        int64_t offsetC = i * orgM;
        auto tensorA = operand2D1.GetStorage()->View(curFunc, {firstDimA, secondDimA}, {offsetA, 0});
        auto tensorB = operand2D2.GetStorage()->View(curFunc, {firstDimB, secondDimB}, {offsetB, 0});
        auto tensorC = result.GetStorage()->View(curFunc, {orgM, orgN}, {offsetC, 0});
        AddAMulBNode(tensorA, tensorB, tensorC, nullptr, attrParam);
    }
    return Reshape(result, {batchSize, orgM, orgN});
};

void CheckABatchMulB4D(const Tensor &operand1, const Tensor &operand2) {
    OP_CHECK(true, {
        ASSERT(operand1.GetShape().size() == SHAPE_DIM4 && operand2.GetShape().size() == SHAPE_DIM4)
            << "Expected 4D tensor, but got: "
            << "op1Size: " << operand1.GetShape().size() << ", op2Size: " << operand2.GetShape().size() << std::endl;
    });
    const int64_t batchSizeA1 = operand1.GetShape()[0];
    const int64_t batchSizeA2 = operand1.GetShape()[1];
    const int64_t batchSizeB1 = operand2.GetShape()[0];
    const int64_t batchSizeB2 = operand2.GetShape()[1];

    OP_CHECK(true, {
        ASSERT(batchSizeA1 == batchSizeB1 || batchSizeB1 == 1 || batchSizeA1 == 1)
            << "batchSize invalid: A1=B1 or 1 allowed. A1: " << batchSizeA1 << ", B1: " << batchSizeB1 << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(batchSizeA2 == batchSizeB2 || batchSizeB2 == 1 || batchSizeA2 == 1)
            << "batchSize invalid: A2=B2 or 1 allowed. A2: " << batchSizeA2 << ", B2: " << batchSizeB2 << std::endl;
    });
}

Tensor ABatchMulB4D(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    CheckABatchMulB4D(operand1, operand2);
    const int64_t batchSizeA1 = operand1.GetShape()[0];
    const int64_t batchSizeA2 = operand1.GetShape()[1];
    const int64_t batchSizeB1 = operand2.GetShape()[0];
    const int64_t batchSizeB2 = operand2.GetShape()[1];

    const int64_t orgM = attrParam.transA ? operand1.GetShape()[SHAPE_DIM3] : operand1.GetShape()[SHAPE_DIM2];
    const int64_t orgKa = attrParam.transA ? operand1.GetShape()[SHAPE_DIM2] : operand1.GetShape()[SHAPE_DIM3];
    const int64_t orgKb = attrParam.transB ? operand2.GetShape()[SHAPE_DIM3] : operand2.GetShape()[SHAPE_DIM2];
    const int64_t orgN = attrParam.transB ? operand2.GetShape()[SHAPE_DIM2] : operand2.GetShape()[SHAPE_DIM3];
    OP_CHECK(true, { ASSERT(orgKa == orgKb) << "orgK mismatch: Ka: " << orgKa << ", Kb: " << orgKb << std::endl; });
    int64_t firstDimA = attrParam.transA ? orgKa : orgM;
    int64_t secondDimA = attrParam.transA ? orgM : orgKa;
    int64_t firstDimB = attrParam.transB ? orgN : orgKb;
    int64_t secondDimB = attrParam.transB ? orgKb : orgN;
    auto operand2D1 = Reshape(operand1, {batchSizeA1 * batchSizeA2 * firstDimA, secondDimA});
    auto operand2D2 = Reshape(operand2, {batchSizeB1 * batchSizeB2 * firstDimB, secondDimB});
    int64_t batchSize1 = std::max(batchSizeA1, batchSizeB1);
    int64_t batchSize2 = std::max(batchSizeA2, batchSizeB2);
    Tensor result(dataType, {batchSize1 * batchSize2 * orgM, orgN});
    if (attrParam.isCMatrixNZ) {
        result =
            Tensor(dataType, {batchSize1 * batchSize2 * orgM, orgN}, "BatchMatmulOutputNz", TileOpFormat::TILEOP_NZ);
    }
    CheckMatmulOperands(dataType, operand2D1, operand2D2, attrParam);

    int64_t strideA = batchSizeA2 == 1 ? 0 : firstDimA;
    int64_t strideB = batchSizeB2 == 1 ? 0 : firstDimB;
    int64_t offsetC = 0;
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int64_t i = 0; i < batchSize1; i++) {
        int64_t offsetA = batchSizeA1 == 1 ? 0 : i * batchSizeA2 * firstDimA;
        int64_t offsetB = batchSizeB1 == 1 ? 0 : i * batchSizeB2 * firstDimB;
        for (int64_t j = 0; j < batchSize2; j++) {
            auto tensorA = operand2D1.GetStorage()->View(curFunc, {firstDimA, secondDimA}, {offsetA, 0});
            auto tensorB = operand2D2.GetStorage()->View(curFunc, {firstDimB, secondDimB}, {offsetB, 0});
            auto tensorC = result.GetStorage()->View(curFunc, {orgM, orgN}, {offsetC, 0});
            AddAMulBNode(tensorA, tensorB, tensorC, nullptr, attrParam);
            offsetC += orgM;
            offsetA += strideA;
            offsetB += strideB;
        }
    }
    return Reshape(result, {batchSize1, batchSize2, orgM, orgN});
};

Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix, const bool isTransA,
    const bool isTransB, const bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isTransA, isTransB, isCMatrixNZ);
    auto vecTile = TileShape::Current().GetVecTile();
    if (vecTile.size() < SHAPE_DIM2) {
        const int32_t vecTileShape = 128;
        TileShape::Current().SetVecTile({vecTileShape, vecTileShape});
    }
    DECLARE_TRACER();
    OP_CHECK(true, {
        ASSERT(aMatrix.GetShape().size() == bMatrix.GetShape().size())
            << "Matrix dimension mismatch: a: " << aMatrix.GetShape().size() << ", b: " << bMatrix.GetShape().size()
            << std::endl;
    });
    Tensor res;
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        res = ABatchMulB4D(dataType, aMatrix, bMatrix, attrParam);
    } else {
        res = ABatchMulB3D(dataType, aMatrix, bMatrix, attrParam);
    }
    return res;
}
}  // namespace Matrix
}  // namespace tile_fwk
}  // namespace npu

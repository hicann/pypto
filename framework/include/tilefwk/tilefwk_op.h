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
 * \file tilefwk_op.h
 * \brief
 */

#pragma once

#include <array>
#include <sstream>

#include "tilefwk/tensor.h"
#include "tilefwk/element.h"

namespace npu::tile_fwk {
class SymbolicScalar;
class Element;
constexpr const int TILE_VEC_DIMS = 2;
constexpr const int TILE_CUBE_DIMS = 6;

enum class OpType {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};
enum class OutType {
    BOOL,
    BIT,
};

namespace experimental {
struct PrintHelper {
    SymbolicScalar cond;
    std::vector<Tensor> tensors;
    std::vector<SymbolicScalar> scalars;
    std::stringstream ss;

    template <typename T>
    void Append(T t) {
        if constexpr (std::is_same_v<T, Tensor>) {
            tensors.push_back(t);
            ss << "{T}";
        } else if constexpr (std::is_same_v<T, SymbolicScalar>) {
            scalars.push_back(t);
            ss << "{S}";
        } else {
            ss << t;
        }
    }
};

void Print(SymbolicScalar cond, const std::string &format, const std::vector<Tensor> &tensors,
    const std::vector<SymbolicScalar> &scalars);

template<bool isB, bool isTrans>
Tensor GatherInL1(const Tensor &src, const Tensor &offsets, const Tensor &blockTable, int blockSize, int size);
Tensor GatherInUB(const Tensor &params, const Tensor &indices, const Tensor &blockTable, int blockSize, int axis);
} // namespace experimental

template <typename... Args>
void Print(Args... args) {
    experimental::PrintHelper helper;
    (helper.Append(args), ...);
    experimental::Print(1, helper.ss.str(), helper.tensors, helper.scalars);
}

template <typename... Args>
void PrintIf(SymbolicScalar cond, Args... args) {
    experimental::PrintHelper helper;
    (helper.Append(args), ...);
    experimental::Print(cond, helper.ss.str(), helper.tensors, helper.scalars);
}

/**
 * \brief Dump a tensor to file
 *
 * \param cond Dump the tensor only `cond` evaluate result is none zero
 * \param operand tensor to dump
 * \param fname filename, {S} can be used as scalar placeholder
 * \param scalars scalars to dump
 */
void ToFile(const Tensor &operand, const std::string &fname, const std::vector<SymbolicScalar> &scalars = {}, SymbolicScalar cond = 1);

Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets);
Tensor View(const Tensor &operand, const DataType dstDataType);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newOffsets);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::initializer_list<SymbolicScalar> &newOffsets);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets);

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensors);
void Assemble(const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest);

struct AssembleItem {
    Tensor tensor;
    std::vector<SymbolicScalar> offsets;
};

void Assemble(const std::vector<AssembleItem> &items, Tensor &src, bool parallelInAssemble = false);

Tensor Reshape(const Tensor &operand, const std::vector<int64_t> &dstshape, const std::vector<SymbolicScalar> &validShape={}, const bool inplace=false);
Tensor Reshape(const Tensor &operand, const std::initializer_list<int64_t> &dstshape, const std::initializer_list<SymbolicScalar> &validShape={}, const bool inplace=false);
Tensor Reshape(const Tensor &operand, const std::vector<SymbolicScalar> &dstShape, const bool inplace);

void Reshape(const Tensor &operand, Tensor &dst);

Tensor Load(const Tensor &src, const Tensor &offsets);

Tensor Full(const Element &src, DataType dtype, const std::vector<int64_t> &dstShape,
    std::vector<SymbolicScalar> validShape = {});
Tensor Full(const SymbolicScalar &src, DataType dtype, const std::vector<int64_t> &dstShape,
    std::vector<SymbolicScalar> validShape = {});
Tensor Transpose(const Tensor &self, std::vector<int> perm);
Tensor Cast(const Tensor &self, DataType dstDataType, CastMode mode = CAST_NONE);

Tensor Exp(const Tensor &self);
Tensor Neg(const Tensor &self);
Tensor Round(const Tensor &self, const int &decimals = 0);
Tensor Rsqrt(const Tensor &self);
Tensor Relu(const Tensor &self);
Tensor BitwiseNot(const Tensor &self);
Tensor Sqrt(const Tensor &self);
Tensor Ceil(const Tensor &self);
Tensor Floor(const Tensor &self);
Tensor Trunc(const Tensor &self);
Tensor Reciprocal(const Tensor &operand);
Tensor Abs(const Tensor &self);
Tensor Ln(const Tensor &operand);
Tensor Hub(const Tensor &operand);
Tensor Sign(const Tensor &operand);

Tensor Duplicate(const Tensor &operand);
Tensor Gather(const Tensor &params, const Tensor &indices, int axis);
Tensor GatherElements(const Tensor &params, const Tensor &indices, int axis);

enum class ScatterMode {
    NONE,
    ADD,
    MULTIPLY,
    UNKNOWN,
};

/**
 * \brief Write the scalar value of src into self Tensor, with the write position specified by the indices Tensor.
 *
 * \param self : Tensor to write into.
 * \param indices : the index Tensor of element to be dispersed.
 * \param src : scalar value or tensor to be dispersed.
 * \param axis : axis to be indexed.
 * \param reduce : scatter reduction mode to be applied. Support NONE, ADD, MULTIPLY. NONE is default.
 * \return Tensor
 */
Tensor Scatter(const Tensor &self, const Tensor &indices, const Element &src, int axis,
    ScatterMode reduce = ScatterMode::NONE);
Tensor Scatter(const Tensor &self, const Tensor &indices, const Tensor &src, int axis,
    ScatterMode reduce = ScatterMode::NONE);
void IndexPut_(Tensor &self, const std::vector<Tensor> &indices, const Tensor &values, bool accumulate = false);
Tensor IndexAdd(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha = Element(DT_FP32, 1.0f));
Tensor RowSumExpand(const Tensor &operand);
Tensor RowMaxExpand(const Tensor &operand);

Tensor Sum(const Tensor &self, int axis = -1, bool keepDim=false);
Tensor Amax(const Tensor &self, int axis = -1, bool keepDim=false);
Tensor Amin(const Tensor &self, int axis = -1, bool keepDim=false);

Tensor Compact(const Tensor &operand);

Tensor Add(const Tensor &self, const Tensor &other);
Tensor Sub(const Tensor &self, const Tensor &other);
Tensor Div(const Tensor &self, const Tensor &other);
Tensor Mul(const Tensor &self, const Tensor &other);
Tensor Hypot(const Tensor &self, const Tensor &other);
Tensor Fmod(const Tensor &self, const Tensor &other);
Tensor Maximum(const Tensor &operand1, const Tensor &operand2);
Tensor Minimum(const Tensor &operand1, const Tensor &operand2);
Tensor BitwiseAnd(const Tensor &self, const Tensor &other);
Tensor BitwiseOr(const Tensor &self, const Tensor &other);
Tensor BitwiseXor(const Tensor &self, const Tensor &other);
Tensor Add(const Tensor &self, const Element &other);
Tensor Sub(const Tensor &self, const Element &other);
Tensor Div(const Tensor &self, const Element &other);
Tensor Mul(const Tensor &self, const Element &other);
Tensor Fmod(const Tensor &self, const Element &other);
Tensor BitwiseAnd(const Tensor &self, const Element &other);
Tensor BitwiseOr(const Tensor &self, const Element &other);
Tensor BitwiseXor(const Tensor &self, const Element &other);
Tensor Minimum(const Tensor &operand1, const Element &operand2);
Tensor Maximum(const Tensor &operand1, const Element &operand2);
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode);
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode);
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode);
Tensor Pow(const Tensor &self, const Tensor &other);
Tensor Pow(const Tensor &self, const Element &other);
Tensor CopySign(const Tensor &self, const Tensor &other);

Tensor BitwiseRightShift(const Tensor &self, const Tensor &other);
Tensor BitwiseRightShift(const Tensor &self, const Element &other);
Tensor BitwiseRightShift(const Element &self, const Tensor &other);
Tensor BitwiseLeftShift(const Tensor &self, const Tensor &other);
Tensor BitwiseLeftShift(const Tensor &self, const Element &other);
Tensor BitwiseLeftShift(const Element &self, const Tensor &other);

Tensor Where(const Tensor &condition, const Tensor &input, const Tensor &other);
Tensor Where(const Tensor &condition, const Tensor &input, const Element &other);
Tensor Where(const Tensor &condition, const Element &input, const Tensor &other);
Tensor Where(const Tensor &condition, const Element &input, const Element &other);

Tensor Unsqueeze(const Tensor &old, int unsqueezeDimNum);

Tensor TensorIndex(const Tensor &params, const Tensor &indices);
Tensor ScatterUpdate(const Tensor &dst, const Tensor &index, const Tensor &src, int axis = -2,
    std::string cacheMode = "PA_BNSD", int chunkSize = 1);

Tensor Expand(const Tensor &self, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape = {});

Tensor Sin(Tensor operand);
Tensor Cos(Tensor operand);
Tensor Softmax(const Tensor &operand);
Tensor RmsNorm(const Tensor &operand);
Tensor RmsNorm(const Tensor &operand, const Tensor &gamma, float epsilon = 1e-05f);
Tensor Cat(const std::vector<Tensor> &tensors, int axis);
Tensor NewCompact(const Tensor &operand);
Tensor Pad(const Tensor &old, const std::vector<int64_t> &newShape);
Tensor LogicalNot(const Tensor &self);
Tensor Range(const Element &start, const Element &end, const Element &step);
Tensor LogicalAnd(const Tensor &self, const Tensor &other);

Tensor Assign(const Tensor &operand);

// Implementation of `Tensor` type should be placed at first, so that it can be routed when only single input.
Tensor Clip(const Tensor &self, const Tensor &min = {}, const Tensor &max = {});
Tensor Clip(const Tensor &self, const Element &min = {}, const Element &max = {});

std::tuple<Tensor, Tensor> TopK(const Tensor &self, int k, int axis = -1, bool isLargest = true);
Tensor ArgSort(const Tensor &self, int axis = -1, bool descending = false);

/**
 * @brief Sort a tensor with shape (1, n) along the last dimension, n must be orders of 2.
 *        The vecTile (1, t), t must be orders of 2, maximum is 8K.
 * @param x The input tensor to be sorted, the indices are initialized to 0123...
 * @param descending If true, sorts in descending order; otherwise ascending order (default: true).
 * @return std::tuple<Tensor, Tensor> A tuple containing two tensors:
 *         - First tensor: The sorted data.
 *         - Second tensor: The corresponding indices.
 */
std::tuple<Tensor, Tensor> Sort(const Tensor &x, bool descending = true);

/**
 * @brief Sort a tensor & indices with shape (1, n) along the last dimension, n must be orders of 2.
 *        The vecTile (1, t), t must be orders of 2, maximum is 8K.
 * @param x The input tensor to be sorted.
 * @param idx The input indices corresponding to x.
 * @param descending If true, sorts in descending order; otherwise ascending order (default: true).
 * @return std::tuple<Tensor, Tensor> A tuple containing two tensors:
 *         - First tensor: The sorted data.
 *         - Second tensor: The corresponding indices.
 */
std::tuple<Tensor, Tensor> SortWithIndex(const Tensor &x, const Tensor &idx, bool descending = true);

Tensor SoftmaxNew(const Tensor &operand);
void SoftmaxDynamic(Tensor &input, Tensor &output);

Tensor RotateHalf(const Tensor &input);

// moe
Tensor Sigmoid(Tensor &input);

std::tuple<Tensor, Tensor> Quant(
    const Tensor &input, bool isSymmetry = true, bool hasSmoothFactor = false, const Tensor &smoothFactor = Tensor());

Tensor ScalarDivS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarAddS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarMaxS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarSubS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarMulS(const Tensor &operand, const Element &value, bool reverseOperand = false);

Tensor ScalarSub(const Tensor &operand1, const Tensor &operand2);
Tensor ScalarDiv(const Tensor &operand1, const Tensor &operand2);
Tensor CumSum(const Tensor &input, const int &axis);
Tensor TriU(const Tensor &input, const SymbolicScalar &diagonal);
Tensor TriL(const Tensor &input, const SymbolicScalar &diagonal);
struct PaTileShapeConfig {
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

enum class ReduceMode {
    ATOMIC_ADD,
};
// template <ReduceMode reduceMode>
Tensor Reduce(const std::vector<Tensor> &aggregation, const ReduceMode reduceMode);

Tensor Maxpool(const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &strides,
    const std::vector<int> &paddings);

enum class LogBaseType {
    LOG_E,
    LOG_2,
    LOG_10,
};
Tensor Log(const Tensor &self, LogBaseType base = LogBaseType::LOG_E);

Tensor OneHot(const Tensor &self, int numClasses);

struct IfaTileShapeConfig {
    int blockSize;
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

struct RoPETileShapeConfig {
    std::vector<int64_t> twoDimsTileShape;
    std::vector<int64_t> threeDimsTileShape;
    std::vector<int64_t> fourDimsTileShape;
    std::vector<int64_t> fiveDimsTileShape;
};

struct RoPETileShapeConfigNew {
    std::vector<int64_t> threeDimsTileShape;
    std::vector<int64_t> fourDimsTileShapeQ;
    std::vector<int64_t> fourDimsTileShapeK;
    std::vector<int64_t> fiveDimsTileShape;
};

void ApplyRotaryPosEmb(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin,
    const Tensor &positionIds, Tensor &qEmbed, Tensor &kEmbed, const int unsqueezeDim = 1,
    const RoPETileShapeConfig &ropeTileShapeConfig = {});

void ApplyRotaryPosEmbV2(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, Tensor &qEmbed,
    Tensor &kEmbed, const int unsqueezeDim = 2, const RoPETileShapeConfigNew &ropeTileShapeConfig = {});

void IncreFlashAttention(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    std::vector<std::vector<int>> &blockTable, std::vector<int> &actSeqs, float softmaxScale, Tensor &attentionOut,
    IfaTileShapeConfig &tileConfig);

void PageAttentionAddS(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1);

void PageAttentionAddSSingleOutput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1);

void PrologPost(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, Tensor &weightUV, Tensor &weightO, int blockSize, float softmaxScale,
    Tensor &postOut, PaTileShapeConfig &tileConfig);

namespace Matrix {

enum class ReLuType : int64_t
{
    NoReLu = 0,
    ReLu = 1
};

struct MatmulExtendParam {
    Tensor biasTensor{Tensor()};
    Tensor scaleTensor{Tensor()};
    float scaleValue{0.0f};
    ReLuType reluType{ReLuType::NoReLu};

    MatmulExtendParam(Tensor bias, Tensor scale, float scaleVal, ReLuType relu)
        : biasTensor(std::move(bias)),
          scaleTensor(std::move(scale)),
          scaleValue(scaleVal),
          reluType(relu) {}

    MatmulExtendParam() = default;
};

Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans = false,
    bool isBTrans = false, bool isCMatrixNZ = false);

Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const MatmulExtendParam &extendParam,
    bool isATrans = false, bool isBTrans = false, bool isCMatrixNZ = false);

Tensor MatmulMX(DataType outType, const Tensor &aMatrix, const Tensor &aScale, const Tensor &bMatrix,
    const Tensor &bScale, bool isATrans = false, bool isAScaleTrans = false, bool isBTrans = false,
    bool isBScaleTrans = false, bool isCMatrixNZ = false);

Tensor MatmulMX(DataType outType, const Tensor &aMatrix, const Tensor &aScale, const Tensor &bMatrix,
    const Tensor &bScale, const MatmulExtendParam &extendParam, bool isATrans = false, bool isAScaleTrans = false,
    bool isBTrans = false, bool isBScaleTrans = false, bool isCMatrixNZ = false);

Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans = false,
    bool isBTrans = false, bool isCMatrixNZ = false);

Tensor TransposedBatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix);

Tensor QuantMM(const Tensor &operand1, const Tensor &operand2, const Tensor &dequantScaleW);
} // namespace Matrix

namespace Distributed {
enum class DistReduceType {
    DIST_REDUCE_ADD,
    DIST_REDUCE_MAX,
    DIST_REDUCE_MIN,
};

enum class AtomicType {
    SET,
    ADD
};

struct MoeConfig {
    int32_t routedExpertNum{0};
    int32_t expertNumPerRank{0};
    int32_t rankNum{0};
};
void MoeDistributedDispatchV2(const Tensor& x, const Tensor& expertIds, const char* group,
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& expandX,
    Tensor& assistInfoForCombine, Tensor& expertTokenNums, Tensor& recvCounts);
void MoeDistributedDispatch(const Tensor& tokenTensor, const Tensor& tokenExpertTable, Tensor& expandX, Tensor& validCnt,
    Tensor& combineInfo, const char *group, const MoeConfig& moeConfig);
void AllGather(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out);
void AllGather(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out);
Tensor ShmemBarrier(const Tensor& predToken, Tensor& shmemSignal, const char* group, uint32_t worldSize);
Tensor ShmemDataSet(const Tensor& predToken, const Tensor& shmemData);
Tensor ShmemSignalSet(const Tensor& predToken, const Tensor& shmemSignal);
void ReduceScatter(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize,
    DistReduceType reduceType, Tensor& out);
void ReduceScatter(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData, Tensor& shmemSignal,
    DistReduceType reduceType, Tensor& out);
void OneShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out);
void OneShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out);
void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out);
void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out);
void MoeDistributedCombine(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out);
void MoeDistributedCombineV2(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out);
void CreateShmemData(const char* group, int64_t worldSize, DataType dataType,
    const Shape& shape, Tensor& shmemTensor, uint64_t memType = 0);
void CreateShmemSignal(const char* group, Tensor& shmemData, Tensor& shmemSignal);
Tensor ShmemPut(const Tensor& predToken, const Tensor& in, const Tensor& shmemData,
    AtomicType atomicType = AtomicType::SET);
Tensor ShmemPutUb2Gm(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy,
 	AtomicType atomicType = AtomicType::SET);
Tensor ShmemSignal(const Tensor& predToken, const Tensor& shmemSignal, AtomicType atomicType = AtomicType::SET);
Tensor WaitUntil(const Tensor& predToken, const Tensor& shmemSignal, int32_t expectedSum, bool resetSignal = false);
Tensor ShmemGet(const Tensor& predToken, const Tensor& shmemData, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET);
Tensor ShmemGetGm2Ub(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET);
} // namespace Distributed
std::tuple<Tensor, Tensor> TopKSort(const Tensor &x, int idxStart);
std::tuple<Tensor, Tensor> TopKSort(const Tensor &x, const SymbolicScalar &idxStart);
Tensor TopKExtract(const Tensor &x, int k, bool isIndex);
Tensor TopKMerge(const Tensor &x, int mergeSize);
Tensor Nop(const std::vector<Tensor> &inTensors);
} // namespace npu::tile_fwk

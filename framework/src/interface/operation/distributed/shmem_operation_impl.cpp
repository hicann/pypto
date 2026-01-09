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
 * \file shmem_operation_impl.cpp
 * \brief
*/

#include <type_traits>
#include "distributed_common.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk::Distributed {
std::pair<int32_t, int32_t> GetRankSizeAndTileCount()
{
    const TileShape& tileShape = TileShape::Current();

    auto rankShape = tileShape.GetDistTileRank();
    int32_t rankSize = rankShape[0] * rankShape[1] + rankShape[2];

    auto tileRow = tileShape.GetDistTileRow();
    auto tileCol = tileShape.GetDistTileCol();
    int32_t rowCount = tileRow[1] + (tileRow[2] != 0 ? 1 : 0);
    int32_t colCount = tileCol[1] + (tileCol[2] != 0 ? 1 : 0);
    int32_t tileCount = rowCount * colCount;

    return {rankSize, tileCount};
}

void ValidateGroup(const char* group)
{
    ASSERT(group != nullptr) << "\"group\" cannot be nullptr";
    int32_t groupLen = std::strlen(group);
    ASSERT((groupLen >= 1) && (groupLen < 128)) << "The length of \"group\" only supports [1, 128), but got "
        << groupLen;
}

void ValidateTilingSize(const VecTile &vecTile, const Tensor& in)
{
    int32_t expectedTileSize = in.GetShape().size();
    ASSERT(expectedTileSize == static_cast<int32_t>(vecTile.size())) <<
        "Invalid dim of tile shape: dim of tile shape must be equal to " << std::to_string(expectedTileSize) << ".";
    ASSERT(std::all_of(vecTile.tile.begin(), vecTile.tile.begin() + expectedTileSize, [](int64_t v) { return v > 0;}))
        << "Invalid vecTile set: each element of the tileSize must be > 0";
}

void ValidateParams(const Tensor &in, const Tensor &out, Shape shmemDataShape, DataType shmemDataType,
    bool checkShapeMatch = false, bool validateType = false, const std::unordered_set<DataType> &allowedTypes = {}) 
{
    ASSERT(in.GetShape().size() == 2UL) << "Invalid dimensional: Input dimensional must be 2, but got dimensional=" << in.GetShape().size();
    ASSERT(out.GetShape().size() == 2UL) << "Invalid dimensional: Output dimensional must be 2, but got dimensional=" << out.GetShape().size();
    ASSERT(out.GetDataType() == in.GetDataType()) << "The data type of \"out\" must be consistent with that of \"in\", "
        << "but the data type of \"out\" is "<< DataType2String(out.GetDataType()) << " and the data type of \"in\" is "
        << DataType2String(in.GetDataType()) << ".";
    ASSERT(in.Format() == out.Format()) << "Output tensor format dose not match input tensor fromat. "
        << "in format: " << std::to_string(in.Format())
        << "out format: " << std::to_string(out.Format()) << ".";
    int32_t inRow = in.GetShape(0);
    int32_t inCol = in.GetShape(1);
    int32_t outRow = out.GetShape(0);
    int32_t outCol = out.GetShape(1);
    ASSERT(inRow > 0 && inCol > 0) << "Input parameter error - the 'row' and 'col' dimensional of the input tensor must be greater than 0, "
        << "but got row=" << inRow << ", col=" << inCol;
    if (checkShapeMatch) {
        ASSERT((inRow == outRow) && (inCol == outCol)) << "Shape mismatch: Input and output dimensions must be the same, but got "
        << "Input shape: (" << inRow << "," << inCol << "), Output shape: (" << outRow << "," << outCol << ").";
    }
    if (validateType) {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (const auto& dtype : allowedTypes) {
            if (!first) {
                oss << ", ";
            }
            oss << DataType2CCEStr(dtype);
            first = false;
        }
        oss << "]";
        ASSERT(allowedTypes.count(in.GetDataType())) << "Invalid data type for input tensor. Expected: " << oss.str() <<
        ", but got: " << DataType2CCEStr(in.GetDataType());
    }
    int64_t shmemDataEleNum =
        std::accumulate(shmemDataShape.begin() + 1, shmemDataShape.end(), 1, std::multiplies<int64_t>());
    int64_t shmemSignalEleNum = shmemDataShape[0] * MAX_TILE_NUM * SHMEM_SIGNAL_STRIDE;
    uint64_t shmemSize = shmemDataEleNum * BytesOf(shmemDataType) + shmemSignalEleNum * BytesOf(DT_INT32);
    const uint64_t winSize = 1024 * 1024 * 200;
    ASSERT(shmemSize < winSize) << "Exceeds winSize limit. Maximum allowed: " << winSize << ", got: " << shmemSize;
}

Tensor ShmemPut(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, 
    AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{MAX_TILE_NUM, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT,
        {in.GetStorage(), shmemDataTile.GetStorage(), barrierDummy.GetStorage()}, {dummy});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummy;
}

Tensor ShmemPutUb2Gm(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, int tileCount,
    AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT_UB2GM,
        {in.GetStorage(), shmemDataTile.GetStorage(), barrierDummy.GetStorage()}, {dummy});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummy;
}

Tensor ShmemSignal(const Tensor &dummy, const Tensor &shmemSignalTile, AtomicType atomicType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto dummyOut = std::make_shared<LogicalTensor>(function, DT_INT32, dummy.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {dummy.GetStorage(), shmemSignalTile.GetStorage()},
        {dummyOut});
    DistOpAttr distOpAttr;
    distOpAttr.signalValue = 1;
    distOpAttr.atomicType = atomicType;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummyOut;
}

Tensor ShmemGet(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemDataTile.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape, shmemDataTile.Format());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET, {dummy.GetStorage(), shmemDataTile.GetStorage()},
        {tempOutTile});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return tempOutTile;
}

Tensor ShmemGetGm2Ub(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemDataTile.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {dummy.GetStorage(), shmemDataTile.GetStorage()},
        {tempOutTile});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return tempOutTile;
}

Tensor WaitUntil(const Tensor &dummyIn, const Tensor &shmemSignalTile, int32_t expectedSum, bool resetSignal = false)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{MAX_TILE_NUM, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {dummyIn.GetStorage(), shmemSignalTile.GetStorage()},
        {dummy});
    std::vector<int64_t> param = {static_cast<int64_t>(expectedSum), static_cast<int64_t>(SHMEM_SIGNAL_STRIDE), static_cast<int64_t>(resetSignal)};
    DistOpAttr distOpAttr;
    distOpAttr.aicpuOpParams = param;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummy;
}

void ShmemReduce(const Tensor &in, const Tensor &shmData, const Tensor &dummy, const Tensor &out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op = function.AddOperation(Opcode::OP_SHMEM_REDUCE,
        {in.GetStorage(), shmData.GetStorage(), dummy.GetStorage()}, {out.GetStorage()});
    DistOpAttr distOpAttr;
    // fp16 和 bf16 做reduce计算，默认转化为fp32
    if ((in.GetDataType() == DT_FP16) || (in.GetDataType() == DT_BF16)) {
        distOpAttr.fp32Mode = true;
    } else {
        distOpAttr.fp32Mode = false;
    }
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
}

void CreateShmemData(const char *group, int64_t worldSize, DataType dataType,
    const Shape &shape, Tensor &shmemTensor, uint64_t memType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape shmemShape{worldSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, memType,
        AlignUp(BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()), 512)));
}

void CreateShmemSignal(const char *group, Tensor &shmemData, Tensor &shmemSignal)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int64_t worldSize = shmemData.GetShape(0);
    Shape shmemShape{worldSize, worldSize};
    Shape shmemDataShape;
    shmemDataShape.assign(shmemData.GetShape().begin() + 1, shmemData.GetShape().end());
    shmemShape.insert(shmemShape.end(), shmemDataShape.begin(), shmemDataShape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shmemShape);
    shmemSignal = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemSignal, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
}
void ShmemBarrier(const Tensor& predToken, Tensor& shmemSignal, const char* group, Tensor& out)
{
    ValidateGroup(group);

    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    (void)tileCount;
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    auto shmemSignalTile = View(shmemSignal, {rankSize, 1, 1, 8}, std::vector<SymbolicScalar>{0, 0, 0, 0});
    auto shmemSignalOut = ShmemSignal(predToken, shmemSignalTile, AtomicType::ADD);
    auto shmemSignalLocal = View(shmemSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
    out = WaitUntil(shmemSignalOut, shmemSignalLocal, rankSize, true);
}

Tensor ShmemSet(const Tensor& predToken, const Tensor& shmemTensor)
{
    constexpr int32_t supportedDim = 4;
    ASSERT(shmemTensor.GetShape().size() == supportedDim) << "The dim of \"shmemTensor\" only supports " << supportedDim
        << ", but got " << shmemTensor.GetShape().size();

    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    function.AddOperation(Opcode::OP_SHMEM_SET, {predToken.GetStorage(), shmemTensor.GetStorage()}, {out});
    return out;
}

void AllGather(const Tensor &predToken, const Tensor &in, const char *group, uint32_t worldSize, Tensor &out)
{
    ASSERT(worldSize > 0) << "AllGather worldSize is invalud";
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);

    Shape shmemDataShape = {worldSize, row, col};
    Shape outShape = {row * worldSize, col};
    ASSERT(out.GetShape() == outShape) << "This shape of out is invalid";
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    Tensor shmemData;
    Tensor shmemSignal;
    const TileShape& tileShape = TileShape::Current();
    ValidateGroup(group);
    ValidateTilingSize(tileShape.GetVecTile(), in);
    ValidateParams(in, out, shmemDataShape, in.GetDataType());
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemData(group, worldSize, in.GetDataType(), shmemDataShape, shmemData);
        CreateShmemSignal(group, shmemData, shmemSignal);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, worldSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, thisRank, 0, 0});

        auto dummy = ShmemPut(in, shmemDataTile, predToken);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::SET);

        auto shmemDataLocal = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto shmemSignalLocal =
            View(shmemSignal, {1, 1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, thisRank, dynRankId, 0, 0});
        auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, 1);
        auto tempOutTile = ShmemGet(dummyLocal, shmemDataLocal);
        Assemble(tempOutTile, {dynRankId * row, 0}, out);
    }
}

void ReduceScatter(const Tensor &predToken, const Tensor& in, const char* group,
    uint32_t worldSize, DistReduceType reduceType, Tensor& out)
{
    (void)reduceType;
    ASSERT(worldSize > 0) << "ReduceScatter worldSize is invalid";
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ASSERT((row % worldSize) == 0);
    const int32_t rowOut = row / worldSize;
    Shape outShape = {rowOut, col};
    ASSERT(out.GetShape() == outShape) << "This shape of out is invalid";

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape shmemDataShape = {1, rowOut, col};
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    const TileShape& tileShape = TileShape::Current();
    ValidateGroup(group);
    ValidateTilingSize(tileShape.GetVecTile(), in);
    ValidateParams(in, out, shmemDataShape, shmemDataType, false, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemData(group, worldSize, shmemDataType, shmemDataShape, shmemData);
        CreateShmemSignal(group, shmemData, shmemSignal);
    }
    LOOP("RS", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, worldSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto dummy = ShmemPut(inTile, shmemDataTile, predToken, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);

        IF (dynRankId == thisRank) {
            auto shmemDataLocal = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto shmemSignalLocal =
                View(shmemSignal, {1, 1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0, 0});
            auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, worldSize);
            out = ShmemGet(dummyLocal, shmemDataLocal, in.GetDataType());
        }
    }
}

void AllReduceValidate(const Tensor& in, const Tensor& shmemData, const char* group, Tensor& out)
{
    ValidateGroup(group);
    ValidateParams(in, out, shmemData.GetShape(), shmemData.GetDataType(), true, true,
        {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetVecTile(), in);
}

void OneShotAllReduce(const Tensor& predToken, const Tensor& in, const Tensor& shmemData,
    const Tensor& shmemSignal, const char* group, uint32_t worldSize, Tensor& out)
{
    ASSERT(worldSize > 0) << "OneShotAllReduce worldSize is invalid";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row;
    AllReduceValidate(in, shmemData, group, out);
    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    LOOP("OneShotAllReduce", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, worldSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {1, 1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0, 0});
        auto dummy = ShmemPut(in, shmemDataTile, predToken, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);
        IF (thisRank == dynRankId) {
            auto dummyLocal = WaitUntil(dummySignal, shmemSignalTile, worldSize);
            out = ShmemGet(dummyLocal, shmemDataTile, in.GetDataType());
        }
    }
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, const Tensor& shmemData,
    const Tensor& shmemSignal, const char* group, uint32_t worldSize, Tensor& out)
{
    ASSERT(worldSize > 0) << "TwoShotAllReduce worldSize is invalid";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / worldSize;
    AllReduceValidate(in, shmemData, group, out);
    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    LOOP("TwoShotAllReduce", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, worldSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {worldSize, 1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{0, dynRankId, dynRankId, 0, 0});
        auto inTile = View(in, {rowPerRank, col}, std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        auto dummy = ShmemPut(inTile, shmemDataTile, predToken, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);
        auto waitSignalTile = View(shmemSignal, {1, 1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, dynRankId, 0, 0});
        auto dummyLocal = WaitUntil(dummySignal, waitSignalTile, worldSize);
        auto tmp = ShmemGet(dummyLocal, shmemDataTile, in.GetDataType());
        Assemble(tmp, {rowPerRank * dynRankId, 0}, out);
    }
}


template<AllReduceType allReduceType>
void CreateShmemTensorAndAllReduce(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out)
{
    ASSERT(worldSize > 0) << "AllReduce worldSize is invalid";
    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row;
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    Shape shmemDataShape = {1, rowPerRank, col};
    if constexpr (allReduceType == AllReduceType::TWO_SHOT) {
        rowPerRank = row / worldSize;
        ASSERT(row % worldSize == 0) << "Two_Shot_AllReduce mode constraint violated: row must be divisible by worldSize";
        shmemDataShape = {worldSize, rowPerRank, col};
    }
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemData(group, worldSize, shmemDataType, shmemDataShape, shmemData);
        CreateShmemSignal(group, shmemData, shmemSignal);
    }
    if constexpr (allReduceType == AllReduceType::ONE_SHOT) {
        OneShotAllReduce(predToken, in, shmemData, shmemSignal, group, worldSize, out);
    } else {
        TwoShotAllReduce(predToken, in, shmemData, shmemSignal, group, worldSize, out);
    }
}

void OneShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out)
{
    CreateShmemTensorAndAllReduce<AllReduceType::ONE_SHOT>(predToken, in, group, worldSize, out);
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, uint32_t worldSize, Tensor& out)
{
    CreateShmemTensorAndAllReduce<AllReduceType::TWO_SHOT>(predToken, in, group, worldSize, out);
}
}   // namespace npu::tile_fwk::Distributed

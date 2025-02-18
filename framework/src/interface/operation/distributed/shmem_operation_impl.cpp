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

void ValidateTilingSize(std::array<int32_t, MAX_DIST_DIM_SIZE> tilingStrategy, int32_t totalExpected, std::string desc)
{
    ASSERT(tilingStrategy[0] * tilingStrategy[1] + tilingStrategy[2] == totalExpected) << "Invalid tiling strategy of "
        << "the " << desc << " axis: expect tilingStrategy[0] * tilingStrategy[1] + tilingStrategy[2] == "
        << totalExpected << ", but got tilingStrategy[0]=" << tilingStrategy[0] << ", tilingStrategy[1]="
        << tilingStrategy[1] << ", tilingStrategy[2]=" << tilingStrategy[2];
}

void ValidateParams(const Tensor &in, const Tensor &out, Shape shmemDataShape, Shape shmemSignalShape, DataType shmemDataType,
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
    const int64_t maxTileCount = 32 * 32 + 1;
    const int64_t tileCount = shmemSignalShape[1];
    ASSERT(tileCount < maxTileCount) << "The tiling setting is invalid. The maximum number of tileCount allowed is 1024, "
        << "but got: " << tileCount;
    const uint64_t winSize = 1024 * 1024 * 200;
    const uint64_t shmemSize = shmemDataShape[0] * (shmemDataShape[1] * shmemDataShape[2] * BytesOf(shmemDataType) +
            shmemSignalShape[1] * shmemSignalShape[2] * BytesOf(DT_INT32));
    ASSERT(shmemSize < winSize) << "Exceeds winSize limit. Maximum allowed: " << winSize << ", got: " << shmemSize;
}

Tensor ShmemPut(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, int tileCount,
    AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
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

Tensor WaitUntil(const Tensor &dummyIn, const Tensor &shmemSignalTile, int32_t tileCount, int32_t hcclGroupIndex,
    int32_t expectedSum)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {dummyIn.GetStorage(), shmemSignalTile.GetStorage()},
        {dummy});
    std::vector<int64_t> param = {static_cast<int64_t>(hcclGroupIndex), static_cast<int64_t>(expectedSum)};
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

Tensor ShmemClearSignal(const Tensor &in, const Tensor &shmemSignalTile, const int32_t tileCount)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    function.AddOperation(Opcode::OP_SHMEM_CLEAR_SIGNAL, {in.GetStorage(), shmemSignalTile.GetStorage()}, {dummy});
    return dummy;
}

void CreateShmemTensor(Tensor &shmemTensor, int32_t rankSize, int32_t hcclGroupIndex, DataType dataType, const Shape &shape)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape{rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())));
}

Tensor Barrier(const Tensor &in, const char *group)
{
    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();

    Shape shmemSignalShape = {rankSize, tileCount, 8}; // shmemSignalShape 根据算子的具体情况设置
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Tensor shmemSignal;
    Tensor shmemBarrierSignal;
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
        CreateShmemTensor(shmemBarrierSignal, rankSize, hcclGroupIndex, DT_INT32, Shape{1, 1, 8});
    }
    LOOP("Barrier", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        auto shmemSignalTile =
            View(shmemSignal, {1, rankSize, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto clearSignalDummy = ShmemClearSignal(in, shmemSignalTile, tileCount);

        for (int32_t rank = 0; rank < rankSize; rank++) {
            auto shmemBarrierSignalTile =
                View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{rank, 0, 0, 0});
            ShmemSignal(clearSignalDummy, shmemBarrierSignalTile, AtomicType::ADD);
        }
        auto shmemBarrierSignalLocal =
            View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        barrierDummy = WaitUntil(clearSignalDummy, shmemBarrierSignalLocal, 1, hcclGroupIndex, rankSize);
    }
    return barrierDummy;
}

void ShmemAllGather(const Tensor &in, const Tensor &barrierDummy, const char *group, Tensor &out)
{
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();

    Shape shmemDataShape = {rankSize, row, col};
    Shape shmemSignalShape = {rankSize, tileCount, 8};
    Shape outShape = {row * rankSize, col};
    ASSERT(out.GetShape() == outShape) << "This shape of out is invalid";
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetDistTileRow(), row, "row");
    ValidateTilingSize(tileShape.GetDistTileCol(), col, "col");
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    Tensor shmemData;
    Tensor shmemSignal;
    ValidateGroup(group);
    ValidateParams(in, out, shmemDataShape, shmemSignalShape, in.GetDataType());
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, rankSize, hcclGroupIndex, in.GetDataType(), shmemDataShape);
        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});

        auto dummy = ShmemPut(in, shmemDataTile, barrierDummy, tileCount);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::SET);

        auto shmemDataLocal = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto shmemSignalLocal =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, 1);
        auto tempOutTile = ShmemGet(dummyLocal, shmemDataLocal);
        Assemble(tempOutTile, {dynRankId * row, 0}, out);
    }
}

void ShmemReduceScatter(const Tensor& in, const char* group, DistReduceType reduceType, Tensor& out)
{
    (void)reduceType;
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    ASSERT((row % rankSize) == 0);
    const int32_t rowOut = row / rankSize;
    Shape outShape = {rowOut, col};
    ASSERT(out.GetShape() == outShape) << "This shape of out is invalid";

    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetDistTileRow(), rowOut, "row");
    ValidateTilingSize(tileShape.GetDistTileCol(), col, "col");

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape shmemDataShape = {1, rowOut, col};
    Shape shmemSignalShape = {1, tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    ValidateGroup(group);
    ValidateParams(in, out, shmemDataShape, shmemSignalShape, shmemDataType, false, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, rankSize, hcclGroupIndex, shmemDataType, shmemDataShape);
        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("RS", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        Tensor fakeBarrierDummy(DT_INT32, {1, 1}, "fakeBarrierDummy");
        auto dummy = ShmemPut(inTile, shmemDataTile, fakeBarrierDummy, tileCount, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);

        IF (dynRankId == thisRank) {
            auto shmemDataLocal = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto shmemSignalLocal =
                View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, rankSize);
            out = ShmemGet(dummyLocal, shmemDataLocal, in.GetDataType());
        }
    }
}

void OneShotShmemAllReduce(const Tensor& in, const char* group, Tensor& out) {
     int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    const int32_t rowPerRank = row;
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    Shape shmemDataShape = {1, rowPerRank, col};
    Shape shmemSignalShape = {1, tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    ValidateGroup(group);
    ValidateParams(in, out, shmemDataShape, shmemSignalShape, shmemDataType, true, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetDistTileRow(), rowPerRank, "row");
    ValidateTilingSize(tileShape.GetDistTileCol(), col, "col");
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, rankSize, hcclGroupIndex, shmemDataType, shmemDataShape);
        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        Tensor fakeBarrierDummy(DT_INT32, {1, 1}, "fakeBarrierDummy");
        auto dummy = ShmemPut(in, shmemDataTile, fakeBarrierDummy, tileCount, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);
        IF (thisRank == dynRankId) {
            auto dummyLocal = WaitUntil(dummySignal, shmemSignalTile, tileCount, hcclGroupIndex, rankSize);
            out = ShmemGet(dummyLocal, shmemDataTile, in.GetDataType());
        }
    }
}

void TwoShotShmemAllReduce(const Tensor& in, const char* group, Tensor& out) {
     int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    const int32_t rowPerRank = row / rankSize;
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    Shape shmemDataShape = {rankSize, rowPerRank, col};
    Shape shmemSignalShape = {rankSize, tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    ValidateGroup(group);
    ASSERT(row % rankSize == 0) << "Two_Shot_AllReduce mode constraint violated: row must be divisible by rankSize";
    ValidateParams(in, out, shmemDataShape, shmemSignalShape, shmemDataType, true, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetDistTileRow(), rowPerRank, "row");
    ValidateTilingSize(tileShape.GetDistTileCol(), col, "col");
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, rankSize, hcclGroupIndex, shmemDataType, shmemDataShape);
        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
        auto inTile = View(in, {rowPerRank, col}, std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        Tensor fakeBarrierDummy(DT_INT32, {1, 1}, "fakeBarrierDummy");
        auto dummy = ShmemPut(inTile, shmemDataTile, fakeBarrierDummy, tileCount, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);
        auto dummyLocal = WaitUntil(dummySignal, shmemSignalTile, tileCount, hcclGroupIndex, rankSize);
        auto tmp = ShmemGet(dummyLocal, shmemDataTile, in.GetDataType());
        Assemble(tmp, {rowPerRank * dynRankId, 0}, out);
    }
}

Tensor MoeCombineSend(const Tensor& in, const Tensor& combineInfo, const Tensor& shmemData, const Tensor& shmemSignal,
    int32_t topK)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto dummyOut = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_MOE_COMBINE_SEND, {in.GetStorage(), combineInfo.GetStorage(),
        shmemData.GetStorage(), shmemSignal.GetStorage()}, {dummyOut});
    DistOpAttr distOpAttr;
    distOpAttr.topK = topK;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummyOut;
}

Tensor MoeCombineReceive(const Tensor& dummyIn, const Tensor& scale, const Tensor& shmemData, const Tensor& shmemSignal)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t batchSize = shmemData.GetShape(2);
    int32_t hiddenSize = shmemData.GetShape(3);
    auto out = std::make_shared<LogicalTensor>(function, shmemData.GetDataType(), Shape{batchSize, hiddenSize});
    function.AddOperation(Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE,
        {dummyIn.GetStorage(), scale.GetStorage(), shmemData.GetStorage(), shmemSignal.GetStorage()}, {out});
    return out;
}

void ShmemMoeCombine(const Tensor& in, const Tensor& combineInfo, const Tensor& scale, const char* group,
    int32_t rankSize, int32_t totalExpertNum, Tensor& out)
{
    ASSERT(in.GetShape().size() == 2) << "The dim of \"in\" only supports 2, but got " << in.GetShape().size();
    ASSERT(combineInfo.GetShape().size() == 2) << "The dim of \"combineInfo\" only supports 2, but got "
        << combineInfo.GetShape().size();
    ASSERT(scale.GetShape().size() == 2) << "The dim of \"scale\" only supports 2, but got " << scale.GetShape().size();
    ASSERT(out.GetShape().size() == 2) << "The dim of \"out\" only supports 2, but got " << out.GetShape().size();

    int32_t inRow = in.GetShape(0);
    int32_t inCol = in.GetShape(1);
    int32_t combineInfoRow = combineInfo.GetShape(0);
    int32_t combineInfoCol = combineInfo.GetShape(1);
    int32_t scaleRow = scale.GetShape(0);
    int32_t scaleCol = scale.GetShape(1);
    int32_t outRow = out.GetShape(0);
    int32_t outCol = out.GetShape(1);

    int32_t hiddenSize = inCol;
    int32_t batchSize = scaleRow;
    int32_t topK = scaleCol;
    int32_t expectedRow = std::min(topK * batchSize * rankSize, batchSize * totalExpertNum);

    ASSERT(inRow == expectedRow) << "The first axis of \"in\" must be the smaller value between topK (the second axis "
        << "of \"scale\") * batchSize (the first axis of \"scale\") * rankSize and batchSize * totalExpertNum, topK="
        << topK << ", batchSize=" << batchSize << ", rankSize=" << rankSize << ", totalExpertNum=" << totalExpertNum
        << ", the expected first axis of \"in\" should be " << expectedRow << " but got " << inRow;
    ASSERT(inCol == 5120) << "The second axis of \"in\" only supports 5120, but got " << hiddenSize;
    ASSERT(in.GetDataType() == DT_BF16) << "The data type of \"in\" only supports DT_BF16, but got "
        << DataType2String(in.GetDataType());
    ASSERT(in.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"in\" only supports ND, but got "
        << "NZ";

    ASSERT(combineInfoRow == inRow) << "The first axis of \"combineInfo\" must be consistent with that of \"in\", but "
        << "inRow=" << inRow << ", combineInfoRow=" << combineInfoRow;
    ASSERT(combineInfoCol == 3) << "The second axis of \"combineInfo\" must be 3, but got " << combineInfoCol;
    ASSERT(combineInfo.GetDataType() == DT_INT32) << "The data type of \"combineInfo\" only supports DT_INT32, but got "
        << DataType2String(combineInfo.GetDataType());
    ASSERT(combineInfo.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"combineInfo\" only "
        << "supports ND, but got NZ";

    ASSERT((scaleRow == 8) || (scaleRow == 256)) << "The first axis of \"scale\" only supports 8 or 256, but got "
        << scaleRow;
    ASSERT(scaleCol == 8) << "The second axis of \"scale\" only supports 8, but got " << scaleCol;
    ASSERT(scale.GetDataType() == DT_FP32) << "The data type of \"scale\" only supports DT_FP32, but got "
        << DataType2String(scale.GetDataType());
    ASSERT(scale.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"scale\" only supports ND, but "
        << "got NZ";

    ASSERT(outRow == scaleRow) << "The first axis of \"out\" must be consistent with that of \"scale\", but scaleRow="
        << scaleRow << ", outRow=" << outRow;
    ASSERT(outCol == inCol) << "The second axis of \"out\" must be consistent with that of \"in\", but inCol=" << inCol
        << ", outCol=" << outCol;
    ASSERT(out.GetDataType() == in.GetDataType()) << "The data type of \"out\" must be consistent with that of \"in\", "
        << "but the data type of \"in\" is "<< DataType2String(in.GetDataType()) << " and the data type of \"out\" is "
        << DataType2String(out.GetDataType());
    ASSERT(out.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"out\" only supports ND, but got "
        << "NZ";

    ValidateGroup(group);
    ASSERT(totalExpertNum == 160) << "totalExpertNum only supports 160, but got " << totalExpertNum;
    ASSERT((rankSize == 4) || (rankSize == 8)) << "rankSize only supports 4 or 8, but got " << rankSize;

    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    int32_t shmemDataRow = topK * batchSize;
    Shape shmemDataShape = {1, shmemDataRow, hiddenSize};
    int32_t shmemSignalCol = SAME_ADDR_BYTE_SIZE / BytesOf(DataType::DT_FP32) * batchSize;
    Shape shmemSignalShape = {shmemSignalCol};

    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, rankSize, hcclGroupIndex, in.GetDataType(), shmemDataShape);
        CreateShmemTensor(shmemSignal, rankSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("MoeCombine", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        TileShape::Current().SetDistTile({inRow / AIV_NUM, AIV_NUM, inRow % AIV_NUM}, {hiddenSize, 1, 0}, {0, 0, 0});
        auto sendDummy = MoeCombineSend(in, combineInfo, shmemData, shmemSignal, topK);

        auto shmemDataThisRank = View(shmemData, {1, 1, shmemDataRow, hiddenSize},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto shmemSignalThisRank = View(shmemSignal, {1, shmemSignalCol}, std::vector<SymbolicScalar>{thisRank, 0});
        TileShape::Current().SetDistTile(
            {batchSize / AIV_NUM, AIV_NUM, batchSize % AIV_NUM}, {hiddenSize, 1, 0}, {0, 0, 0});
        out = MoeCombineReceive(sendDummy, scale, shmemDataThisRank, shmemSignalThisRank);
    }
}
}   // namespace npu::tile_fwk::Distributed

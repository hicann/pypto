/**
 * Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_graph_base.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include "type.h"
#include "value.h"
#include "operation.h"

namespace pto {

class TileBaseOp : public Operation {
public:
    TileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : Operation(opcode, iops, oops) {}

    std::shared_ptr<TileValue> GetInOperand(size_t index) const;
    std::shared_ptr<TileValue> GetOutOperand(size_t index) const;
private:
};

class ElementWiseTileBaseOp : public TileBaseOp {
public:
    ElementWiseTileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class ElementWiseUnaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseUnaryTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class ElementWiseBinaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseBinaryTileBaseOp(Opcode opcode,
                                TileValuePtr lhs,
                                TileValuePtr rhs,
                                TileValuePtr out)
        : ElementWiseTileBaseOp(opcode, {lhs, rhs}, {out}) {
    }
};

class ElementWiseScalarMixBinaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseScalarMixBinaryTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)}, {ObjectCast<Value>(output)}) {}
    bool GetReverse() const { return reverse_; }
    void SetReverse(bool reverse) { reverse_ = reverse; }

private:
    bool reverse_ = false;
};

class WhereTTTileOp : public ElementWiseTileBaseOp {
public:
    WhereTTTileOp(Opcode opcode, TileValuePtr Condition, TileValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(Condition), ObjectCast<Value>(input), ObjectCast<Value>(other)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class WhereTSTileOp : public ElementWiseTileBaseOp {
public:
    WhereTSTileOp(Opcode opcode, TileValuePtr Condition, TileValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(Condition), ObjectCast<Value>(input), ObjectCast<Value>(other)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class WhereSTTileOp : public ElementWiseTileBaseOp {
public:
    WhereSTTileOp(Opcode opcode, TileValuePtr Condition, ScalarValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(Condition), ObjectCast<Value>(input), ObjectCast<Value>(other)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class WhereSSTileOp : public ElementWiseTileBaseOp {
public:
    WhereSSTileOp(Opcode opcode, TileValuePtr Condition, ScalarValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(Condition), ObjectCast<Value>(input), ObjectCast<Value>(other)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class SortTileOp : public TileBaseOp {
public:
    SortTileOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class TopKTileOp : public SortTileOp {
public:
    TopKTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output1, TileValuePtr output2)
        : SortTileOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output1), ObjectCast<Value>(output2)}) {}
};

class BitSortTileOp : public SortTileOp {
public:
    BitSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class MrgSortTileOp : public SortTileOp {
public:
    MrgSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class ArgSortTileOp : public SortTileOp {
public:
    ArgSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class ExtractSortTileOp : public SortTileOp {
public:
    ExtractSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class TiledMrgSortTileOp : public SortTileOp {
public:
    TiledMrgSortTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr input4,
                        TileValuePtr output, TileValuePtr temp)
        : SortTileOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2), ObjectCast<Value>(input3), ObjectCast<Value>(input4)},
                    {ObjectCast<Value>(output), ObjectCast<Value>(temp)}) {}
};

class ReduceTileBaseOp : public TileBaseOp {
public:
    ReduceTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class ReduceWithTempTileBaseOp : public TileBaseOp {
public:
    ReduceWithTempTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class ElementWiseBinaryWithTempTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseBinaryWithTempTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)},
                                        {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class ElementWiseUnaryWithTempTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseUnaryWithTempTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class CastTileBaseOp : public TileBaseOp {
public:
    CastTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class VecDupTileBaseOp : public TileBaseOp {
public:
    VecDupTileBaseOp(Opcode opcode, ScalarValuePtr Scalar, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(Scalar)}, {ObjectCast<Value>(output)}) {}
};

class RangeTileBaseOp : public TileBaseOp {
public:
    RangeTileBaseOp(Opcode opcode, ScalarValuePtr START, ScalarValuePtr STEP, ScalarValuePtr SIZE, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(START), ObjectCast<Value>(STEP),ObjectCast<Value>(SIZE)},
                             {ObjectCast<Value>(output)}) {}
};

class ScatterTileBaseOp : public TileBaseOp {
public:
    ScatterTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, TileValuePtr Src2, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(Src0), ObjectCast<Value>(Src1), ObjectCast<Value>(Src2)}, {ObjectCast<Value>(output)}) {}
};

class ScatetrElementsTileBaseOp : public TileBaseOp {
public:
    ScatetrElementsTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, ScalarValuePtr Scatter, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(Src0), ObjectCast<Value>(Src1), ObjectCast<Value>(Scatter)}, {ObjectCast<Value>(output)}) {}
};

class GatherTileBaseOp : public TileBaseOp {
public:
    GatherTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)}, {ObjectCast<Value>(output)}) {}
};

class ExpandTileBaseOp : public TileBaseOp {
public:
    ExpandTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class OnehotTileBaseOp : public TileBaseOp {
public:
    OnehotTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class ConcatTileBaseOp : public TileBaseOp {
public:
    // Concat通常输入是列表，这里假设用vector传入
    ConcatTileBaseOp(Opcode opcode, std::vector<ValuePtr> inputs, TileValuePtr output)
        : TileBaseOp(opcode, inputs, {ObjectCast<Value>(output)}) {}
};

class TransposeTileBaseOp : public TileBaseOp {
public:
    TransposeTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class PowTileBaseOp : public TileBaseOp {
public:
    PowTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)}, {ObjectCast<Value>(output)}) {}
};

class CumSumTileBaseOp : public TileBaseOp {
public:
    CumSumTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class CompareScalarTileBaseOp : public TileBaseOp {
public:
    CompareScalarTileBaseOp(Opcode opcode, TileValuePtr in1, ScalarValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ObjectCast<Value>(in1), ObjectCast<Value>(in2)}, {ObjectCast<Value>(output), ObjectCast<Value>(temp)}) {}
};

class CompareTileBaseOp : public TileBaseOp {
public:
    CompareTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ObjectCast<Value>(in1), ObjectCast<Value>(in2)}, {ObjectCast<Value>(output), ObjectCast<Value>(temp)}) {}
};

class BroadcastWithTempTileBaseOp : public TileBaseOp {
public:
    BroadcastWithTempTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)}, {ObjectCast<Value>(output), ObjectCast<Value>(TempTensor)}) {}
};

class BroadcastTileBaseOp : public TileBaseOp {
public:
    BroadcastTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(lhs), ObjectCast<Value>(rhs)}, {ObjectCast<Value>(output)}) {}
};

class DataCopyTileBaseOp : public TileBaseOp {
public:
    DataCopyTileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class ConvertTileOp : public DataCopyTileBaseOp {
public:
    ConvertTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class CopyInOutTileOp : public DataCopyTileBaseOp {
public:
    CopyInOutTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class DuplicateTileOp : public DataCopyTileBaseOp {
public:
    DuplicateTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class LoadTileOp : public DataCopyTileBaseOp {
public:
    LoadTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2)}, {ObjectCast<Value>(output)}) {}
};

class LoadStoreTileOp : public DataCopyTileBaseOp {
public:
    LoadStoreTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2)}, {ObjectCast<Value>(output)}) {}
};

class GatherInUBTileOp : public DataCopyTileBaseOp {
public:
    GatherInUBTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2), ObjectCast<Value>(input3)}, {ObjectCast<Value>(output)}) {}
};

class GatherInL1TileOp : public DataCopyTileBaseOp {
public:
    GatherInL1TileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2), ObjectCast<Value>(input3)}, {ObjectCast<Value>(output)}) {}
};

class AnyDataCopyTileOp : public DataCopyTileBaseOp {
public:
    AnyDataCopyTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}) {}
};

class IndexOutCastTileOp : public TileBaseOp {
public:
    IndexOutCastTileOp(Opcode opcode, TileValuePtr Src, TileValuePtr Index, TileValuePtr Dst, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(Src), ObjectCast<Value>(Index), ObjectCast<Value>(Dst)}, {ObjectCast<Value>(output)}) {}
};

class IndexAddTileOp : public TileBaseOp {
public:
    IndexAddTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, ScalarValuePtr Alpha, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input1), ObjectCast<Value>(input2), ObjectCast<Value>(input3), ObjectCast<Value>(Alpha)}, {ObjectCast<Value>(output)}) {}
};

class MatmulTileBaseOp : public TileBaseOp {
public:
    MatmulTileBaseOp(Opcode opcode, TileValuePtr input, std::vector<ScalarValuePtr> offsets, TileValuePtr output)
        : TileBaseOp(opcode, {ObjectCast<Value>(input)}, {ObjectCast<Value>(output)}),  offsets_(offsets) {}

    ScalarValuePtr GetOffset(size_t index) const;

private:
    std::vector<ScalarValuePtr> offsets_;
};

class MatmulMmadTileBaseOp : public TileBaseOp {
public:
    MatmulMmadTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out)
        : TileBaseOp(opcode, {lhs, rhs}, {out}) {
    }
};
class DataCopyInTileBaseOp : public DataCopyTileBaseOp {
public:
    DataCopyInTileBaseOp(Opcode opcode, TensorValuePtr src, std::vector<ScalarValuePtr> offset, TileValuePtr dst)
      : DataCopyTileBaseOp(opcode, ValueUtils::Join(ObjectCast<Value>(src), offset), ValueUtils::Join(ObjectCast<Value>(dst))) {}
};

class DataCopyOutTileBaseOp : public DataCopyTileBaseOp {
public:
    DataCopyOutTileBaseOp(Opcode opcode, TileValuePtr src, std::vector<ScalarValuePtr> offset, TensorValuePtr dst)
      : DataCopyTileBaseOp(opcode, ValueUtils::Join(ObjectCast<Value>(src), offset), ValueUtils::Join(ObjectCast<Value>(dst))) {}
};

class CustomTileBaseOp : public TileBaseOp {

};

} // namespace pto

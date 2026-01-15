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
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
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
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
    bool GetReverse() const { return reverse_; }
    void SetReverse(bool reverse) { reverse_ = reverse; }

private:
    bool reverse_ = false;
};

class WhereTTTileOp : public ElementWiseTileBaseOp {
public:
    WhereTTTileOp(Opcode opcode, TileValuePtr Condition, TileValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(Condition), ValueCast<Value>(input), ValueCast<Value>(other)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class WhereTSTileOp : public ElementWiseTileBaseOp {
public:
    WhereTSTileOp(Opcode opcode, TileValuePtr Condition, TileValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(Condition), ValueCast<Value>(input), ValueCast<Value>(other)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class WhereSTTileOp : public ElementWiseTileBaseOp {
public:
    WhereSTTileOp(Opcode opcode, TileValuePtr Condition, ScalarValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(Condition), ValueCast<Value>(input), ValueCast<Value>(other)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class WhereSSTileOp : public ElementWiseTileBaseOp {
public:
    WhereSSTileOp(Opcode opcode, TileValuePtr Condition, ScalarValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(Condition), ValueCast<Value>(input), ValueCast<Value>(other)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class SortTileOp : public TileBaseOp {
public:
    SortTileOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class TopKTileOp : public SortTileOp {
public:
    TopKTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output1, TileValuePtr output2)
        : SortTileOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output1), ValueCast<Value>(output2)}) {}
};

class BitSortTileOp : public SortTileOp {
public:
    BitSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class MrgSortTileOp : public SortTileOp {
public:
    MrgSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ArgSortTileOp : public SortTileOp {
public:
    ArgSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ExtractSortTileOp : public SortTileOp {
public:
    ExtractSortTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : SortTileOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class TiledMrgSortTileOp : public SortTileOp {
public:
    TiledMrgSortTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr input4,
                        TileValuePtr output, TileValuePtr temp)
        : SortTileOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2), ValueCast<Value>(input3), ValueCast<Value>(input4)},
                    {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class ReduceTileBaseOp : public TileBaseOp {
public:
    ReduceTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ReduceWithTempTileBaseOp : public TileBaseOp {
public:
    ReduceWithTempTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class ElementWiseBinaryWithTempTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseBinaryWithTempTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, 
                                        {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class ElementWiseUnaryWithTempTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseUnaryWithTempTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output, TileValuePtr TempTensor)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class CastTileBaseOp : public TileBaseOp {
public:
    CastTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class VecDupTileBaseOp : public TileBaseOp {
public:
    VecDupTileBaseOp(Opcode opcode, ScalarValuePtr Scalar, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Scalar)}, {ValueCast<Value>(output)}) {}
};

class RangeTileBaseOp : public TileBaseOp {
public:
    RangeTileBaseOp(Opcode opcode, ScalarValuePtr START, ScalarValuePtr STEP, ScalarValuePtr SIZE, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(START), ValueCast<Value>(STEP),ValueCast<Value>(SIZE)},
                                    {ValueCast<Value>(output)}) {}
};

class ScatterTileBaseOp : public TileBaseOp {
public:
    ScatterTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, TileValuePtr Src2, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Src0), ValueCast<Value>(Src1), ValueCast<Value>(Src2)}, {ValueCast<Value>(output)}) {}
};

class ScatetrElementsTileBaseOp : public TileBaseOp {
public:
    ScatetrElementsTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, ScalarValuePtr Scatter, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Src0), ValueCast<Value>(Src1), ValueCast<Value>(Scatter)}, {ValueCast<Value>(output)}) {}
};

class GatherTileBaseOp : public TileBaseOp {
public:
    GatherTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class ExpandTileBaseOp : public TileBaseOp {
public:
    ExpandTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class OnehotTileBaseOp : public TileBaseOp {
public:
    OnehotTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ConcatTileBaseOp : public TileBaseOp {
public:
    // Concat通常输入是列表，这里假设用vector传入
    ConcatTileBaseOp(Opcode opcode, std::vector<ValuePtr> inputs, TileValuePtr output)
        : TileBaseOp(opcode, inputs, {ValueCast<Value>(output)}) {}
};

class TransposeTileBaseOp : public TileBaseOp {
public:
    TransposeTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class PowTileBaseOp : public TileBaseOp {
public:
    PowTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class CumSumTileBaseOp : public TileBaseOp {
public:
    CumSumTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class CompareScalarTileBaseOp : public TileBaseOp {
public:
    CompareScalarTileBaseOp(Opcode opcode, TileValuePtr in1, ScalarValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ValueCast<Value>(in1), ValueCast<Value>(in2)}, {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class CompareTileBaseOp : public TileBaseOp {
public:
    CompareTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ValueCast<Value>(in1), ValueCast<Value>(in2)}, {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class BroadcastWithTempTileBaseOp : public TileBaseOp {
public:
    BroadcastWithTempTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, 
                                        {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class BroadcastTileBaseOp : public TileBaseOp {
public:
    BroadcastTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class DataCopyTileBaseOp : public TileBaseOp {
public:
    DataCopyTileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class ConvertTileOp : public DataCopyTileBaseOp {
public:
    ConvertTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class CopyInOutTileOp : public DataCopyTileBaseOp {
public:
    CopyInOutTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class DuplicateTileOp : public DataCopyTileBaseOp {
public:
    DuplicateTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class LoadTileOp : public DataCopyTileBaseOp {
public:
    LoadTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2)}, {ValueCast<Value>(output)}) {}
};

class LoadStoreTileOp : public DataCopyTileBaseOp {
public:
    LoadStoreTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2)}, {ValueCast<Value>(output)}) {}
};

class GatherInUBTileOp : public DataCopyTileBaseOp {
public:
    GatherInUBTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2), ValueCast<Value>(input3)}, {ValueCast<Value>(output)}) {}
};

class GatherInL1TileOp : public DataCopyTileBaseOp {
public:
    GatherInL1TileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2), ValueCast<Value>(input3)}, {ValueCast<Value>(output)}) {}
};

class AnyDataCopyTileOp : public DataCopyTileBaseOp {
public:
    AnyDataCopyTileOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : DataCopyTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class IndexOutCastTileOp : public TileBaseOp {
public:
    IndexOutCastTileOp(Opcode opcode, TileValuePtr Src, TileValuePtr Index, TileValuePtr Dst, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Src), ValueCast<Value>(Index), ValueCast<Value>(Dst)}, {ValueCast<Value>(output)}) {}
};

class IndexAddTileOp : public TileBaseOp {
public:
    IndexAddTileOp(Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, ScalarValuePtr Alpha, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input1), ValueCast<Value>(input2), ValueCast<Value>(input3), ValueCast<Value>(Alpha)}, {ValueCast<Value>(output)}) {}
};

class MatmulTileBaseOp : public TileBaseOp {
public:
    MatmulTileBaseOp(Opcode opcode, TileValuePtr input, std::vector<ScalarValuePtr> offsets, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}),  offsets_(offsets) {}

    ScalarValuePtr GetOffset(size_t index) const;

private:
    std::vector<ScalarValuePtr> offsets_;
};

class MatmulMmadTileBaseOp : public TileBaseOp {
public:
    MatmulMmadTileBaseOp(Opcode opcode,
                            TileValuePtr lhs,
                            TileValuePtr rhs,
                            TileValuePtr out)
        : TileBaseOp(opcode, {lhs, rhs}, {out}) {
    }
};

class SysBaseOp : public ScalarBaseOp {
public:
    std::string GetName() const;
private:
    std::string name_;
};

class CustomTileBaseOp : public TileBaseOp {

};

} // namespace pto

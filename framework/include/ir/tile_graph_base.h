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
                                ValuePtr lhs,
                                ValuePtr rhs,
                                ValuePtr out)
        : ElementWiseTileBaseOp(opcode, {lhs, rhs}, {out}) {
    }
};

class ElementWiseScalarMixBinaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseScalarMixBinaryTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class ReduceTileBaseOp : public TileBaseOp {
};

class BroadcastTileBaseOp : public TileBaseOp {
};

class DataCopyTileBaseOp : public TileBaseOp {
};

class MatmulTileBaseOp : public TileBaseOp {
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

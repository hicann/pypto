/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/expr.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

MakeTuple::MakeTuple(std::vector<ExprPtr> elements, Span span) : Expr(std::move(span)), elements_(std::move(elements))
{
    // Collect types from all element expressions
    std::vector<TypePtr> elementTypes;
    elementTypes.reserve(elements_.size());
    for (const auto& elem : elements_) {
        elementTypes.push_back(elem->GetType());
    }

    // Set result type to TupleType
    type_ = std::make_shared<TupleType>(std::move(elementTypes));
}

GetItemExpr::GetItemExpr(ExprPtr value, ExprPtr slice, Span span)
    : Expr(std::move(span)), value_(std::move(value)), slice_(std::move(slice))
{
    CHECK(value_) << "GetItemExpr requires a non-null value";
    CHECK(slice_) << "GetItemExpr requires a non-null slice";

    auto value_type = value_->GetType();
    if (auto tile_type = As<TileType>(value_type)) {
        type_ = tile_type;
        return;
    }

    if (auto tuple_type = As<TupleType>(value_type)) {
        auto const_idx = As<ConstInt>(slice_);
        CHECK(const_idx) << "GetItemExpr on a tuple requires slice to be a ConstInt, got " << slice_->TypeName();
        int index = static_cast<int>(const_idx->value_);
        CHECK(index >= 0 && index < static_cast<int>(tuple_type->types_.size()))
            << "GetItemExpr index " << index << " out of bounds for tuple with " << tuple_type->types_.size()
            << " elements";
        type_ = tuple_type->types_[index];
        return;
    }

    CHECK(false) << "GetItemExpr requires value to have TupleType or TileType, got " << value_type->TypeName();
}

} // namespace ir
} // namespace pypto

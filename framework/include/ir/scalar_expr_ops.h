/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scalar_expr_ops.h
 * \brief Higher-level scalar expression construction functions with type promotion and checking
 *
 * This header provides factory functions (Make*) for constructing scalar expressions
 * with automatic type promotion, type checking, and implicit casting.
 * For basic IR node class definitions, see scalar_expr.h.
 */

#pragma once
#include <memory>
#include <string>

#include "core/logging.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

// ========== Helper Functions for Operator Construction ==========

/**
 * \brief Get the dtype from a scalar expression or scalar var
 *
 * \param expr Expression to extract dtype from
 * \return DataType of the expression
 * \throws ValueError if expr is not a scalar expression or scalar var
 */
inline DataType GetScalarDtype(const ExprPtr& expr, const Span& span = Span::Unknown())
{
    // Note: Must use dynamic_pointer_cast here because this header is included before
    // the TypePtr overload of As<> is defined in kind_traits.h
    auto scalarType = std::dynamic_pointer_cast<const ScalarType>(expr->GetType());
    IRCHECK(scalarType) << "Expression must be Var with ScalarType, got " << expr->TypeName() << " with type "
                        << expr->GetType()->TypeName() << " at " << span.ToString();
    return scalarType->dtype_;
}

inline bool IsBoolDtype(const DataType& dtype) { return dtype == DataType::BOOL; }

enum class ScalarCategory {
    INT,
    FLOAT,
};

inline ScalarCategory GetNumericCategory(
    const DataType& dtype, const std::string& opName, const Span& span = Span::Unknown())
{
    if (dtype.IsFloat()) {
        return ScalarCategory::FLOAT;
    }
    if (dtype.IsInt()) {
        return ScalarCategory::INT;
    }
    IRCHECK(false) << "Operator '" << opName << "' requires numeric scalar dtype, got " << dtype.ToString() << " at "
                   << span.ToString();
    return ScalarCategory::INT; // unreachable, suppress compiler warning
}

inline DataType PromoteSameCategoryDtype(
    const DataType& leftDtype, const DataType& rightDtype, const std::string& opName,
    const Span& span = Span::Unknown())
{
    IRCHECK(!IsBoolDtype(leftDtype) && !IsBoolDtype(rightDtype))
        << "Operator '" << opName << "' does not accept bool dtype"
        << " at " << span.ToString();
    auto leftCategory = GetNumericCategory(leftDtype, opName, span);
    auto rightCategory = GetNumericCategory(rightDtype, opName, span);
    IRCHECK(leftCategory == rightCategory)
        << "Operator '" << opName << "' requires same numeric dtype category, got " << leftDtype.ToString() << " and "
        << rightDtype.ToString() << " at " << span.ToString();
    size_t leftBits = leftDtype.GetBit();
    size_t rightBits = rightDtype.GetBit();
    if (leftBits > rightBits) {
        return leftDtype;
    }
    if (rightBits > leftBits) {
        return rightDtype;
    }
    return leftDtype;
}

struct BinaryOperands {
    ExprPtr left;
    ExprPtr right;
    DataType dtype;
};

inline ExprPtr MaybeCast(const ExprPtr& expr, DataType targetDtype, const Span& span)
{
    DataType dtype = GetScalarDtype(expr, span);
    if (dtype == targetDtype) {
        return expr;
    }
    // INDEX and INT64 share the same int64_t representation in block DSL scalar expressions.
    // Keep the high-level IR clean so round-trip printing does not emit redundant casts.
    if ((dtype == DataType::INDEX && targetDtype == DataType::INT64) ||
        (dtype == DataType::INT64 && targetDtype == DataType::INDEX)) {
        return expr;
    }
    return std::make_shared<Cast>(expr, targetDtype, span);
}

inline BinaryOperands PromoteBinaryOperands(
    const ExprPtr& left, const ExprPtr& right, const std::string& opName, const Span& span)
{
    DataType leftDtype = GetScalarDtype(left, span);
    DataType rightDtype = GetScalarDtype(right, span);
    DataType promotedDtype = PromoteSameCategoryDtype(leftDtype, rightDtype, opName, span);
    return {MaybeCast(left, promotedDtype, span), MaybeCast(right, promotedDtype, span), promotedDtype};
}

inline BinaryOperands PromoteIntBinaryOperands(
    const ExprPtr& left, const ExprPtr& right, const std::string& opName, const Span& span)
{
    DataType leftDtype = GetScalarDtype(left, span);
    DataType rightDtype = GetScalarDtype(right, span);
    IRCHECK(leftDtype.IsInt() && rightDtype.IsInt())
        << "Operator '" << opName << "' requires integer dtype, got " << leftDtype.ToString() << " and "
        << rightDtype.ToString() << " at " << span.ToString();
    DataType promotedDtype = PromoteSameCategoryDtype(leftDtype, rightDtype, opName, span);
    return {MaybeCast(left, promotedDtype, span), MaybeCast(right, promotedDtype, span), promotedDtype};
}

// ========== Binary Operator Construction Functions ==========

inline ExprPtr MakeCast(const ExprPtr& operand, DataType dtype, const Span& span = Span::Unknown())
{
    return std::make_shared<Cast>(operand, dtype, span);
}

inline ExprPtr MakeAdd(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "add", span);
    return std::make_shared<Add>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeSub(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "sub", span);
    return std::make_shared<Sub>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeMul(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "mul", span);
    return std::make_shared<Mul>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloatDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "truediv", span);
    return std::make_shared<FloatDiv>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloorDiv(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "floordiv", span);
    return std::make_shared<FloorDiv>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeFloorMod(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "mod", span);
    return std::make_shared<FloorMod>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakePow(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "pow", span);
    return std::make_shared<Pow>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeEq(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "eq", span);
    return std::make_shared<Eq>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeNe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "ne", span);
    return std::make_shared<Ne>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeLt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "lt", span);
    return std::make_shared<Lt>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeLe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "le", span);
    return std::make_shared<Le>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeGt(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "gt", span);
    return std::make_shared<Gt>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeGe(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "ge", span);
    return std::make_shared<Ge>(operands.left, operands.right, DataType::BOOL, span);
}

inline ExprPtr MakeMin(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "min", span);
    return std::make_shared<Min>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeMax(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteBinaryOperands(left, right, "max", span);
    return std::make_shared<Max>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitAnd(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteIntBinaryOperands(left, right, "bit_and", span);
    return std::make_shared<BitAnd>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitOr(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteIntBinaryOperands(left, right, "bit_or", span);
    return std::make_shared<BitOr>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitXor(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteIntBinaryOperands(left, right, "bit_xor", span);
    return std::make_shared<BitXor>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitShiftLeft(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteIntBinaryOperands(left, right, "bit_shift_left", span);
    return std::make_shared<BitShiftLeft>(operands.left, operands.right, operands.dtype, span);
}

inline ExprPtr MakeBitShiftRight(const ExprPtr& left, const ExprPtr& right, const Span& span = Span::Unknown())
{
    auto operands = PromoteIntBinaryOperands(left, right, "bit_shift_right", span);
    return std::make_shared<BitShiftRight>(operands.left, operands.right, operands.dtype, span);
}

// ========== Unary Operator Construction Functions ==========

inline ExprPtr MakeNeg(const ExprPtr& operand, const Span& span = Span::Unknown())
{
    return std::make_shared<Neg>(operand, GetScalarDtype(operand, span), span);
}

inline ExprPtr MakeBitNot(const ExprPtr& operand, const Span& span = Span::Unknown())
{
    DataType dtype = GetScalarDtype(operand, span);
    IRCHECK(dtype.IsInt()) << "Operator 'bit_not' requires integer dtype, got " << dtype.ToString() << " at "
                           << span.ToString();
    return std::make_shared<BitNot>(operand, dtype, span);
}

} // namespace ir
} // namespace pypto

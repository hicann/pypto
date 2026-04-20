/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {
/**
 * @brief Memory reference variable for shaped types (tensor and tile)
 *
 * Represents a memory reference combining an allocation identity (base Ptr),
 * a byte offset within that allocation, and a size.
 *
 * - base_: VarPtr to the Ptr variable from tile.alloc/tensor.alloc (allocation identity)
 * - byte_offset_: byte offset from base (0 for root alloc, computed for views)
 * - size_: size in bytes of this memory region
 *
 * Aliasing is determined by comparing base_ pointers (SameAllocation) and
 * checking for overlapping byte ranges (MayAlias).
 */
class MemRef : public Expr {
public:
    MemorySpace memorySpace_; ///< Memory space of this MemRef, e.g. Global, Local, Constant
    ExprPtr offset_;          ///< Byte offset from base (0 for full alloc, view offset for views)
    uint64_t size_;           ///< Size in bytes of this MemRef

    /**
     * @brief Construct with explicit variable name. Used by deserialization and
     * address allocation where the name must be preserved exactly.
     */
    MemRef(MemorySpace memory_space, ExprPtr offset, uint64_t size, Span span = Span::Unknown());

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRef; }
    [[nodiscard]] std::string TypeName() const override { return "MemRef"; }

    /// Are two MemRefs from the same allocation? (compare base_ Ptr identity)
    static bool SameAllocation(const MemRefPtr& a, const MemRefPtr& b);

    /// Do two MemRefs potentially alias? (same base + overlapping byte ranges)
    static bool MayAlias(const MemRefPtr& a, const MemRefPtr& b);

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Expr::GetFieldDescriptors(),
            std::make_tuple(
                reflection::UsualField(&MemRef::memorySpace_, "memory_space"),
                reflection::UsualField(&MemRef::offset_, "offset"), reflection::UsualField(&MemRef::size_, "size")));
    }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

} // namespace ir
} // namespace pypto

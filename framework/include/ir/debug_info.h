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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pypto {
namespace ir {

class TupleType;
using TupleTypePtr = std::shared_ptr<const TupleType>;

enum class TupleTypeKind {
    TUPLE,
    NAMED_TUPLE,
    STRUCT,
};

struct TupleTypeInfo {
    TupleTypeKind kind = TupleTypeKind::TUPLE;
    std::optional<std::string> name;
    std::vector<std::string> fields;

    bool operator==(const TupleTypeInfo& other) const
    {
        return kind == other.kind && name == other.name && fields == other.fields;
    }
    bool operator!=(const TupleTypeInfo& other) const { return !(*this == other); }
};

/**
 * \brief Compilation-session side table for semantic tuple metadata.
 *
 * Tuple kind, name, and fields are *not* part of the tuple type's core semantics
 * (positional element types and order are). The parser records this metadata for
 * named tuples and structs. An absent entry denotes a plain positional tuple.
 *
 * The key is the `TupleType` pointer. TupleType is never structurally interned --
 * each `struct.create` allocates a fresh instance (`make_shared`),
 * so same-shape-different-name structs get distinct keys. IR passes copy `TypePtr`
 * by shared_ptr (types are immutable and never rebuilt via reflection), so the
 * pointer stays valid and identical while IR transformation passes rebuild the
 * surrounding nodes.
 *
 * The instance rides on `Program` so codegen can capture it at its entry point.
 */
class IRDebugInfo {
public:
    /// Record semantic metadata for a named tuple or struct type.
    void RegisterTupleTypeInfo(const TupleTypePtr& type, TupleTypeInfo info)
    {
        if (type == nullptr) {
            return;
        }
        tupleTypeInfos_[type.get()] = std::move(info);
    }

    /// Look up semantic tuple metadata by type pointer. Returns nullptr for a plain tuple.
    [[nodiscard]] const TupleTypeInfo* GetTupleTypeInfo(const TupleType* type) const
    {
        auto it = tupleTypeInfos_.find(type);
        return it == tupleTypeInfos_.end() ? nullptr : &it->second;
    }

    /// Merge another table's entries into this one (later registrations win).
    void Merge(const IRDebugInfo& other)
    {
        for (const auto& [type, info] : other.tupleTypeInfos_) {
            tupleTypeInfos_[type] = info;
        }
    }

private:
    std::unordered_map<const TupleType*, TupleTypeInfo> tupleTypeInfos_;
};

using IRDebugInfoPtr = std::shared_ptr<IRDebugInfo>;

} // namespace ir
} // namespace pypto

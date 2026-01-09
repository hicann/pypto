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
 * \file ir_builder_value.cpp
 * \brief
 */

#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>

namespace pto {

ValuePtr IRBuilder::AddToCompound(IRBuilderContext& ctx, ValuePtr v) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::AddToCompound: ctx.compound is null");
    if (!v) throw std::runtime_error("IRBuilder::AddToCompound: value is null");

    std::string key = v->GetName();
    ctx.compound->SetEnvVar(key, v);
    return v;
}

std::shared_ptr<TensorValue> IRBuilder::CreateTensor(
    IRBuilderContext& ctx,
    const std::vector<ScalarValuePtr>& shape, DataType dt, std::string name) {

    auto t = std::make_shared<TensorValue>(shape, dt, std::move(name));
    AddToCompound(ctx, t);
    return t;
}

std::shared_ptr<TileValue> IRBuilder::CreateTile(
    IRBuilderContext& ctx,
    const std::vector<uint64_t>& shape, DataType dt, std::string name) {

    auto t = std::make_shared<TileValue>(shape, dt, std::move(name));
    AddToCompound(ctx, t);
    return t;
}

std::shared_ptr<ScalarValue> IRBuilder::CreateScalar(IRBuilderContext& ctx, DataType dt, std::string name) {
    auto s = std::make_shared<ScalarValue>(dt, std::move(name), ScalarValueKind::Symbolic);
    AddToCompound(ctx, s);
    return s;
}

std::shared_ptr<ScalarValue> IRBuilder::CreateConst(IRBuilderContext& ctx, int64_t v, std::string name) {
    auto s = std::make_shared<ScalarValue>(v, std::move(name));
    AddToCompound(ctx, s);
    return s;
}

std::shared_ptr<ScalarValue> IRBuilder::CreateConst(IRBuilderContext& ctx, double v, std::string name) {
    auto s = std::make_shared<ScalarValue>(v, std::move(name));
    AddToCompound(ctx, s);
    return s;
}

} // namespace pto

/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/core.h"

#include <sstream>
#include <string>
#include <utility>

namespace pypto {
namespace ir {

Span::Span(std::string filename, int beginLine, int beginColumn, int endLine, int endColumn)
    : filename_(std::move(filename)),
      beginLine_(beginLine),
      beginColumn_(beginColumn),
      endLine_(endLine),
      endColumn_(endColumn)
{}

std::string Span::ToString() const
{
    std::ostringstream oss;
    oss << filename_ << ":" << beginLine_ << ":" << beginColumn_;
    return oss.str();
}

static Span kUnknownSpan = Span("", -1, -1, -1, -1);

bool Span::IsUnknown(const Span& span) { return &span == &kUnknownSpan; }

Span& Span::Unknown() { return kUnknownSpan; }

} // namespace ir
} // namespace pypto

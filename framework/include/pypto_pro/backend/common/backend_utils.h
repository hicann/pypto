/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_
#define PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/span.h"
#include "ir/type.h"

namespace pypto {

namespace codegen {
class CCECodegen;
} // namespace codegen

namespace backend {

namespace round_mode {

int FindIndex(const std::string& mode, const std::string& op_name);

} // namespace round_mode

namespace mutex_id {

std::vector<int> GetMutexIdsFromKwargs(const ir::CallPtr& op);

} // namespace mutex_id

namespace gather {

struct CompareAttrs {
    bool has_cmp_mode = false;
    int cmp_mode = 0;
    int offset = 0;
};

CompareAttrs GetCompareAttrs(const ir::CallPtr& op);

} // namespace gather

namespace cce {

std::string ComputeStrideBasedOffset(codegen::CCECodegen& codegen, const ir::MakeTuplePtr& offsets,
                                     const ir::TensorTypePtr& tensor_type);
bool IsNZTensorType(const ir::TensorTypePtr& tensor_type);
int64_t GetNZInnerCols(const ir::DataType& dtype,
                       const std::string& error_prefix = "CCE NZ tensor lowering does not support dtype ");
std::string GetCmpModeEnum(int cmp_type);
void ValidateStoreNZPreconditions(const std::string& op_name, const ir::ExprPtr& src_expr,
                                  const ir::MakeTuplePtr& offsets, const ir::TensorTypePtr& dst_tensor_type);

} // namespace cce

namespace debug_printf {

struct PrintfSegment {
    std::string format_segment;
    char conversion;
};

struct PrintfFormatParts {
    std::string prefix;
    std::string conversion_spec;
    std::string suffix;
};

std::string EscapeStringLiteral(const std::string& text);
std::string QuoteMlirStringLiteral(const std::string& text);
std::string FormatDebugLocation(const ir::Span& span);
std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& op_name);
bool IsSupportedPrintfConversion(char conversion);
size_t FindPrintfConversionIndex(const std::string& format_segment);
PrintfFormatParts SplitPrintfSegment(const std::string& format_segment);
std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format);

} // namespace debug_printf
} // namespace backend
} // namespace pypto

#endif // PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_

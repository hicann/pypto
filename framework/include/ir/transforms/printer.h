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
#pragma once
#include <string>

#include "ir/core.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
/**
 * @brief Print an IR node in Python syntax
 *
 * @param node IR node to print (Expr, Stmt, Function, or Program)
 * @param prefix Module prefix to use (default: "ir")
 * @param concise If true, omit intermediate type annotations (default: false)
 * @return Python-style string representation
 */
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix = "ir", bool concise = false);

/**
 * @brief Print a type in Python syntax
 *
 * @param type Type to print (ScalarType, TensorType, TupleType, etc.)
 * @param prefix Module prefix to use (default: "pl", can be "ir" for legacy)
 * @return Python-style string representation
 */
std::string PythonPrint(const TypePtr& type, const std::string& prefix = "ir");

} // namespace ir
} // namespace pypto

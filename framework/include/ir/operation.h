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
 * \file operation.h
 * \brief
 */

#pragma once

#include "ir/utils_defop.h"
#include "ir/utils.h"
#include "ir/value.h"
#include "ir/opcode.h"
#include "ir/operation_base.h"

#include <memory>
#include <ostream>
#include <string>

namespace pto {

#define DEFOP DEFOP_CLASS
#include "operation.def"
#undef DEFOP

} // namespace pto



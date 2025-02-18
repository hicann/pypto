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
 * \file test_codegen_common.h
 * \brief Unit test for codegen.
 */

#ifndef TEST_CODEGEN_COMMON_H
#define TEST_CODEGEN_COMMON_H

#include <iostream>

namespace npu::tile_fwk {
const std::string SUB_FUNC_SUFFIX = "_Unroll1_PATH0";
const std::string HIDDEN_FUNC_SUFFIX = "_hiddenfunc0";

} // namespace npu::tile_fwk

#endif // TEST_CODEGEN_COMMON_H
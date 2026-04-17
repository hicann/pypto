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
 * \file dump_stubs.cpp
 * \brief
 */

#include <dump/adump_api.h>

namespace Adx {

uint64_t AdumpGetDumpSwitch(DumpType type)
{
    (void)type;
    return 0;
}

int32_t AdumpDumpTensorV2(
    const std::string& opType,
    const std::string& opName,
    const std::vector<TensorInfoV2>& tensorInfos,
    aclrtStream stream)
{
    (void)opType;
    (void)opName;
    (void)tensorInfos;
    (void)stream;
    return 0;
}

} // namespace Adx
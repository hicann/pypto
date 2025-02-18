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
 * \file CommonTools.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/common/CommonType.h"


namespace CostModel {

const int PROCESS_ID_OFFSET = 10000;

inline uint64_t GetProcessID(CostModel::MachineType type, size_t sequence)
{
    return (static_cast<uint64_t>(type) * PROCESS_ID_OFFSET) + sequence;
}

inline int GetMachineType(CostModel::Pid pid)
{
    return (pid / PROCESS_ID_OFFSET);
}

inline int GetMachineSeq(CostModel::Pid pid)
{
    return (pid % PROCESS_ID_OFFSET);
}
}
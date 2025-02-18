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
 * \file Packet.cpp
 * \brief
 */

#include "cost_model/simulation/common/Packet.h"

namespace CostModel {
using namespace std;

std::string CachePacket::Dump() const
{
    std::stringstream oss;
    oss << "pid:" << dec << pid;
    oss << ", tid:" << tid;
    oss << ", Type:" << CacheRequestName(requestType);
    oss << ", addr:" << hex << addr;
    oss << ", size:" << dec << size << "(Bytes)";
    return oss.str();
}
}
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
 * \file SimulatorA2A3.cpp
 * \brief
 */

#include "SimulatorA2A3.h"
#include "cost_model/simulation_ca/A2A3/model/pipe.h"

namespace CostModel
{
    using namespace std;
    uint64_t SimulatorA2A3::Run(std::vector<std::string> program)
    {
        auto prog = GetProgram(program);
        auto pipe = std::make_shared<CostModelPipe>("cost", 0, 0);
        uint32_t baseLatency = 0;
        uint32_t rLatency = 307;
        uint32_t wLatency = 153;
        pipe->SetReadGmFactor(rLatency + baseLatency);
        pipe->SetWriteGmFactor(wLatency + baseLatency);
        uint64_t ret = pipe->GetOpCycle(prog);
        return ret;
    }
} // namespace CostModel


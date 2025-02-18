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
 * \file SimulatorA2A3.h
 * \brief
 */

#pragma once

#include "cost_model/simulation/arch/Simulator.h"

namespace CostModel
{
    class SimulatorA2A3 : public Simulator
    {
    public:
        uint64_t Run(std::vector<std::string> program) override;
        const int rGmLatency = 0;
        const int wGmLatency = 0;
    };
} // namespace CostModel

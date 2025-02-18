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
 * \file GenSimulation.cpp
 * \brief
 */

#include "GenSimulation.h"
#include <algorithm>
#include <iostream>
#include "cost_model/simulation_ca/A2A3/SimulatorA2A3.h"
#include "cost_model/simulation_ca/SimulatorAdaptor.h"

using namespace std;

uint64_t GetCceInput(const vector<string>& program)
{
    CostModel::SimulatorAdaptor sa;
    auto np = sa.Rewrite(program);

    CostModel::SimulatorA2A3 sm;
    return sm.Run(np);
}

int main()
{
    vector<string> program;
    string str;
    while(getline(cin, str)) {
        program.push_back(str);
    }

    cout<<GetCceInput(program)<<endl;
    return 0;
}
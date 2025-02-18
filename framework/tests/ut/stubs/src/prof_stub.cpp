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
 * \file prof_stub.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>

extern "C" {
    int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length);
    int32_t AdprofCheckFeatureIsOn(uint64_t feature);
}

int32_t AdprofCheckFeatureIsOn(uint64_t feature)
{
    (void)feature;
    std::cout << "AdprofCheckFeatureIsOn is called." << std::endl;
    return 1;
}

int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length)
{
    (void)agingFlag;
    (void)data;
    (void)length;
    std::cout << "AdprofReportAdditionalInfo is called." << std::endl;
    return 0;
}
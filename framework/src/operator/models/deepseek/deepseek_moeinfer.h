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
 * \file deepseek_moeinfer.h
 * \brief
 */

#pragma once
#ifndef FFN_DYNAMIC
#define FFN_DYNAMIC

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

void DynamicFFN(const Tensor &hiddenStates, const Tensor &ffnweight1, const Tensor &ffnweight2, const Tensor &ffnweight3,
                    Tensor &out, int BASIC_BATCH);
void DynamicFFNQuant(const Tensor &hiddenStatesQuant, const Tensor &hiddenStatesScale, const Tensor &ffnweight1, const Tensor &ffnweight2, const Tensor &ffnweight3,
                    const Tensor &ffnScale1, const Tensor &ffnScale2, const Tensor &ffnScale3,
                    Tensor &out, int BASIC_BATCH);

} // namespace npu::tile_fwk

#endif // FFN_DYNAMIC

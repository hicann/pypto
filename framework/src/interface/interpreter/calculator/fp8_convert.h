/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp8_convert.h
 * \brief FP8 format conversion utilities (E4M3, E5M2, E8M0) between FP8 and Float32.
 */

#pragma once

#include <torch/torch.h>
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {

// Convert FP8 (stored as uint8) to Float32. actualType specifies the FP8 format.
torch::Tensor Fp8ToFloat32(const torch::Tensor &self, DataType actualType);

// Convert Float32 to FP8 (returns uint8 tensor). actualType specifies the FP8 format.
torch::Tensor Float32ToFp8(const torch::Tensor &self, DataType actualType);

}  // namespace npu::tile_fwk

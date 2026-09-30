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
 * \file test_codegen_pad.h
 * \brief
 */

#pragma once

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "codegen/codegen.h"
#include "codegen/npu/litenpu/codegen_litenpu.h"
#include "test_codegen_common.h"

class TestCodeGenPad {
public:
    void test_pad_001();
    void test_pad_002();
    void test_pad_003();
    void test_pad_004();
    void test_pad_005();
    void test_pad_006();
    void test_pad_007();
    void test_pad_008();
    void test_pad_009();
    void test_pad_010();
    void test_pad_011();
    void test_pad_012();
    void test_pad_013();
    void test_pad_014();
    void test_pad_015();
    void test_pad_016();
    void test_pad_017();
    void test_pad_018();
    void test_pad_019();
    void test_pad_020();
    void test_pad_021();
    void test_pad_022();
    void test_pad_023();
    void test_pad_024();
    void test_pad_025();
    void test_pad_026();
    void test_pad_027();
    void test_pad_028();
    void test_pad_029();
    void test_pad_030();
    void test_pad_031();
    void test_pad_032();
    void test_pad_033();
    void test_pad_034();
    void test_pad_035();
    void test_pad_036();
    void test_pad_037();
    void test_pad_038();
    void test_pad_039();
    void test_pad_040();
    void test_pad_041();
    void test_pad_042();
    void test_pad_043();
    void test_pad_044();
    void test_pad_045();
    // ONNX (tiny_fp32_sim.onnx) cases - mirror python/tests/ut/kirin/common_pad.py ids 046~048.
    void test_pad_046();
    void test_pad_047();
    void test_pad_048();

    // Shared driver: trace one Pad op and run the lite codegen. Keeps each case
    // body a single line; the case table matches python/tests/ut/kirin/common_pad.py.
    static void RunPad(const std::string& name, npu::tile_fwk::DataType dtype, const npu::tile_fwk::Shape& inShape,
                       const npu::tile_fwk::Shape& outShape, const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& vecTile, const npu::tile_fwk::Element& value);

    static TestCodeGenPad& Instance();

private:
    TestCodeGenPad();
    ~TestCodeGenPad();
};

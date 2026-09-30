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
 * \file test_codegen_pad.cpp
 * \brief
 */

#include "include/test_codegen_pad.h"

using namespace npu::tile_fwk;

TestCodeGenPad::TestCodeGenPad() = default;
TestCodeGenPad::~TestCodeGenPad() = default;

TestCodeGenPad& TestCodeGenPad::Instance()
{
    static TestCodeGenPad instance;
    return instance;
}

// Shared driver: trace a single Pad op then run the lite codegen.
void TestCodeGenPad::RunPad(const std::string& name, DataType dtype, const Shape& inShape, const Shape& outShape,
                            const std::vector<int64_t>& padding, const std::vector<int64_t>& vecTile,
                            const Element& value)
{
    PROGRAM(name)
    {
        Tensor input(dtype, inShape, "input");
        auto output = Tensor(dtype, outShape, "output");
        FUNCTION(name)
        {
            TileShape::Current().SetVecTile(vecTile);
            output = Pad(input, padding, "constant", value);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

// ---------------- 1D ----------------
void TestCodeGenPad::test_pad_001()
{
    RunPad("PAD_001", DataType::DT_FP32, {160}, {168}, {0, 8}, {200}, Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_002()
{
    RunPad("PAD_002", DataType::DT_FP32, {160}, {200}, {0, 40}, {100}, Element(DataType::DT_FP32, 1.5));
}

void TestCodeGenPad::test_pad_003()
{
    RunPad("PAD_003", DataType::DT_FP16, {96}, {108}, {5, 7}, {64}, Element(DataType::DT_FP16, -2.0));
}

void TestCodeGenPad::test_pad_004()
{
    RunPad("PAD_004", DataType::DT_FP32, {100}, {110}, {10, 0}, {64}, Element(DataType::DT_FP32, 3.0));
}

// ---------------- 2D tile-split matrix ----------------
void TestCodeGenPad::test_pad_005()
{
    RunPad("PAD_005", DataType::DT_FP32, {24, 40}, {32, 48}, {0, 8, 0, 8}, {32, 48}, Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_006()
{
    RunPad("PAD_006", DataType::DT_FP32, {24, 40}, {32, 48}, {0, 8, 0, 8}, {32, 16}, Element(DataType::DT_FP32, 1.0));
}

void TestCodeGenPad::test_pad_007()
{
    RunPad("PAD_007", DataType::DT_FP16, {24, 40}, {32, 48}, {0, 8, 0, 8}, {16, 48}, Element(DataType::DT_FP16, -3.0));
}

void TestCodeGenPad::test_pad_008()
{
    RunPad("PAD_008", DataType::DT_FP16, {24, 40}, {32, 48}, {0, 8, 0, 8}, {16, 16}, Element(DataType::DT_FP16, 2.0));
}

// ---------------- 2D boundary ----------------
void TestCodeGenPad::test_pad_009()
{
    RunPad("PAD_009", DataType::DT_FP32, {32, 48}, {32, 48}, {0, 0, 0, 0}, {16, 16}, Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_010()
{
    RunPad("PAD_010", DataType::DT_FP32, {32, 48}, {47, 69}, {5, 16, 7, 8}, {16, 16}, Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_011()
{
    RunPad("PAD_011", DataType::DT_FP16, {20, 1}, {20, 45}, {0, 44, 0, 0}, {16, 16}, Element(DataType::DT_FP16, 0.0));
}

void TestCodeGenPad::test_pad_012()
{
    RunPad("PAD_012", DataType::DT_FP32, {64, 6}, {64, 8}, {1, 1, 0, 0}, {16, 8}, Element(DataType::DT_FP32, 5.0));
}

void TestCodeGenPad::test_pad_013()
{
    RunPad("PAD_013", DataType::DT_FP32, {16, 16}, {64, 64}, {0, 48, 0, 48}, {16, 16}, Element(DataType::DT_FP32, 7.0));
}

void TestCodeGenPad::test_pad_014()
{
    RunPad("PAD_014", DataType::DT_FP32, {24, 16}, {32, 16}, {0, 0, 8, 0}, {8, 16}, Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_015()
{
    RunPad("PAD_015", DataType::DT_FP32, {32, 48}, {32, 50}, {2, 0, 0, 0}, {16, 16}, Element(DataType::DT_FP32, 2.0));
}

// ---------------- 3D tile-split matrix ----------------
void TestCodeGenPad::test_pad_016()
{
    RunPad("PAD_016", DataType::DT_FP32, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {2, 32, 32},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_017()
{
    RunPad("PAD_017", DataType::DT_FP32, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {2, 32, 16},
           Element(DataType::DT_FP32, 1.5));
}

void TestCodeGenPad::test_pad_018()
{
    RunPad("PAD_018", DataType::DT_FP16, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {2, 8, 32},
           Element(DataType::DT_FP16, -3.0));
}

void TestCodeGenPad::test_pad_019()
{
    RunPad("PAD_019", DataType::DT_FP16, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {2, 8, 16},
           Element(DataType::DT_FP16, 2.0));
}

void TestCodeGenPad::test_pad_020()
{
    RunPad("PAD_020", DataType::DT_FP32, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {1, 32, 32},
           Element(DataType::DT_FP32, 0.5));
}

void TestCodeGenPad::test_pad_021()
{
    RunPad("PAD_021", DataType::DT_FP32, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {1, 32, 16},
           Element(DataType::DT_FP32, -1.0));
}

void TestCodeGenPad::test_pad_022()
{
    RunPad("PAD_022", DataType::DT_FP16, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {1, 8, 32},
           Element(DataType::DT_FP16, 4.0));
}

void TestCodeGenPad::test_pad_023()
{
    RunPad("PAD_023", DataType::DT_FP16, {2, 16, 16}, {2, 24, 24}, {0, 8, 0, 8}, {1, 8, 16},
           Element(DataType::DT_FP16, 6.0));
}

// ---------------- 3D boundary ----------------
void TestCodeGenPad::test_pad_024()
{
    RunPad("PAD_024", DataType::DT_FP32, {2, 16, 1}, {2, 16, 35}, {3, 31, 0, 0}, {1, 8, 8},
           Element(DataType::DT_FP32, 1.0));
}

void TestCodeGenPad::test_pad_025()
{
    RunPad("PAD_025", DataType::DT_FP16, {1, 64, 6}, {1, 64, 16}, {5, 5, 0, 0}, {1, 32, 16},
           Element(DataType::DT_FP16, 5.0));
}

void TestCodeGenPad::test_pad_026()
{
    RunPad("PAD_026", DataType::DT_FP32, {6, 32, 48}, {6, 47, 69}, {5, 16, 7, 8}, {3, 16, 16},
           Element(DataType::DT_FP32, 0.0));
}

// ---------------- 4D tile-split matrix ----------------
void TestCodeGenPad::test_pad_027()
{
    RunPad("PAD_027", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 2, 32, 32},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_028()
{
    RunPad("PAD_028", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 2, 32, 16},
           Element(DataType::DT_FP32, 1.0));
}

void TestCodeGenPad::test_pad_029()
{
    RunPad("PAD_029", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 2, 8, 32},
           Element(DataType::DT_FP32, -2.0));
}

void TestCodeGenPad::test_pad_030()
{
    RunPad("PAD_030", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 2, 8, 16},
           Element(DataType::DT_FP32, 3.0));
}

void TestCodeGenPad::test_pad_031()
{
    RunPad("PAD_031", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 1, 32, 32},
           Element(DataType::DT_FP16, 0.5));
}

void TestCodeGenPad::test_pad_032()
{
    RunPad("PAD_032", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 1, 32, 16},
           Element(DataType::DT_FP16, -1.0));
}

void TestCodeGenPad::test_pad_033()
{
    RunPad("PAD_033", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 1, 8, 32},
           Element(DataType::DT_FP16, 2.0));
}

void TestCodeGenPad::test_pad_034()
{
    RunPad("PAD_034", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {2, 1, 8, 16},
           Element(DataType::DT_FP16, 4.0));
}

void TestCodeGenPad::test_pad_035()
{
    RunPad("PAD_035", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 2, 32, 32},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_036()
{
    RunPad("PAD_036", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 2, 32, 16},
           Element(DataType::DT_FP32, 1.5));
}

void TestCodeGenPad::test_pad_037()
{
    RunPad("PAD_037", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 2, 8, 32},
           Element(DataType::DT_FP32, -3.0));
}

void TestCodeGenPad::test_pad_038()
{
    RunPad("PAD_038", DataType::DT_FP32, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 2, 8, 16},
           Element(DataType::DT_FP32, 2.5));
}

void TestCodeGenPad::test_pad_039()
{
    RunPad("PAD_039", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 1, 32, 32},
           Element(DataType::DT_FP16, 7.0));
}

void TestCodeGenPad::test_pad_040()
{
    RunPad("PAD_040", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 1, 32, 16},
           Element(DataType::DT_FP16, -5.0));
}

void TestCodeGenPad::test_pad_041()
{
    RunPad("PAD_041", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 1, 8, 32},
           Element(DataType::DT_FP16, 0.25));
}

void TestCodeGenPad::test_pad_042()
{
    RunPad("PAD_042", DataType::DT_FP16, {2, 2, 16, 16}, {2, 2, 24, 24}, {0, 8, 0, 8}, {1, 1, 8, 16},
           Element(DataType::DT_FP16, 9.0));
}

// ---------------- 4D boundary ----------------
void TestCodeGenPad::test_pad_043()
{
    RunPad("PAD_043", DataType::DT_FP16, {1, 4096, 1, 4}, {1, 4096, 1, 16}, {6, 6, 0, 0}, {1, 64, 1, 16},
           Element(DataType::DT_FP16, 5.0));
}

void TestCodeGenPad::test_pad_044()
{
    RunPad("PAD_044", DataType::DT_FP32, {4, 2, 32, 48}, {4, 2, 47, 69}, {5, 16, 7, 8}, {2, 1, 16, 16},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_045()
{
    RunPad("PAD_045", DataType::DT_FP32, {1, 1, 4, 4}, {1, 1, 64, 64}, {0, 60, 0, 60}, {1, 1, 16, 16},
           Element(DataType::DT_FP32, 7.0));
}

// ---------------- ONNX model cases (tiny_fp32_sim.onnx, mirrors common_pad.py 046~048) ----------------
void TestCodeGenPad::test_pad_046()
{
    RunPad("PAD_046", DataType::DT_FP32, {1, 4, 77, 64}, {1, 4, 256, 64}, {0, 0, 0, 179}, {1, 1, 64, 64},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_047()
{
    RunPad("PAD_047", DataType::DT_FP32, {1, 4, 165, 64}, {1, 4, 256, 64}, {0, 0, 13, 78}, {1, 1, 64, 64},
           Element(DataType::DT_FP32, 0.0));
}

void TestCodeGenPad::test_pad_048()
{
    RunPad("PAD_048", DataType::DT_FP32, {1, 4, 142, 64}, {1, 4, 256, 64}, {0, 0, 114, 0}, {1, 1, 64, 64},
           Element(DataType::DT_FP32, 0.0));
}

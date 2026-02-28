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
 * \file test_dynamic_ops.cpp
 * \brief
 */

#include "test_cost_macro.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/interpreter/calc.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::calc;

class DynamicOpsTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        ProgramData::GetInstance().Reset();
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    }
};

TEST_F(DynamicOpsTest, FmodFp32) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int s = 32;
    int n = 2;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 2.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Fmod(t0a, t1a);
            auto t2b = Fmod(t0b, t1b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            out = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, FmodSFp32) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_FP32, {b, s}, "self");
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 5.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 1.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 1.0),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Element src(DT_FP32, 2.0);
            auto t0 = View(self, {b, s}, {0, 0});
            out = Fmod(self, src);
        }
    }
}

TEST_F(DynamicOpsTest, Assemble) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int s = 32;
    int n = 2;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 3.0),
    });

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Add(t0a, t1a);
            auto t2b = Add(t0b, t1b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            out = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, AssembleFp16) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int s = 32;
    int n = 2;
    int m = 1;
    Tensor t0(DT_FP16, {n * s, m * s}, "t0");
    Tensor t1(DT_FP16, {n * s, m * s}, "t1");
    Tensor out(DT_FP16, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t0, 1.0),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, 3.0),
    });

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(3)) {
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Add(t0a, t1a);
            auto t2b = Add(t0b, t1b);
            ToFile(t2b, "t2b_%d.bin", {i});
            PrintIf(i == 1,"t2b=", t2b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            out = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, Ceil) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t n = 4;
    
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, n}, "outValue");

    std::vector<float> inputData = {
        1.2f,  2.0f,  3.9f, -1.1f,
        -2.9f, 5.5f, -0.1f, 7.0f
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
    });

    FUNCTION("main", {self}, {outValue}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            outValue = Ceil(t0);
        }
    }
}

TEST_F(DynamicOpsTest, Floor) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t n = 4;
    
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, n}, "outValue");

    std::vector<float> inputData = {
        1.2f,  2.0f,  3.9f, -1.1f,
        -2.9f, 5.5f, -0.1f, 7.0f
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
    });

    FUNCTION("main", {self}, {outValue}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            outValue = Floor(t0);
        }
    }
}

TEST_F(DynamicOpsTest, Trunc) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t n = 4;
    
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, n}, "outValue");

    std::vector<float> inputData = {
        1.2f,  2.0f,  3.9f, -1.1f,
        -2.9f, 5.5f, -0.1f, 7.0f
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
    });

    FUNCTION("main", {self}, {outValue}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            outValue = Trunc(t0);
        }
    }
}

TEST_F(DynamicOpsTest, PassVerifyWithoutGoldens) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int s = 32;
    int m = 1;
    int n = 2;
    Tensor t0(DT_FP16, {n * s, m * s}, "t0");
    Tensor t1(DT_FP16, {n * s, m * s}, "t1");
    Tensor output(DT_FP16, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t0, 1.0),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(output, 0),
    });

    FUNCTION("main", {t0, t1}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(3)) {
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Add(t0a, t1a);
            auto t2b = Add(t0b, t1b);
            PrintIf(i == 1,"t2b=", t2b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            output = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, OpsElementWise) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 2;
    int m = 2;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor t5(DT_INT32, {n, s, m * s}, "t5");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    float t0Data = 10.0;
    float t1Data = 20.0;
    float t2Data = 30.0;
    float t3Data = 1.0;
    float t4Data = 50.0;

    std::vector<int> t5Data(n * s * m * s, 1);
    t5Data[n * s * m * s - 1] = 3; // 3: loop cnt

    float r0Data = 0;
    int loopCount = 3;
    int condThreshold = 2;

    for (int i = 0; i < loopCount; i++) {
        if (i == 0) {
            r0Data = t0Data + t1Data;
        }  else {
            r0Data = r0Data + t1Data; // +t0, +t1
            if (i < condThreshold) {
                r0Data = r0Data + t2Data; // +t2 * 5
            } else {
                r0Data = r0Data * t3Data; // +t3 * 2
            }
            r0Data = r0Data + t4Data;
        }
    }
    for (int i = 0; i < loopCount; i++) {
        r0Data = r0Data + t1Data; // +t1
        if (i < condThreshold) {
            r0Data = r0Data + t2Data; // +t2 * 5
        } else {
            r0Data = r0Data * t3Data; // +t3 * 2
        }
        r0Data = r0Data + t4Data;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, t0Data),
        RawTensorData::CreateConstantTensor<float>(t1, t1Data),
        RawTensorData::CreateConstantTensor<float>(t2, t2Data),
        RawTensorData::CreateConstantTensor<float>(t3, t3Data),
        RawTensorData::CreateConstantTensor<float>(t4, t4Data),
        RawTensorData::CreateTensor<int>(t5, t5Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, r0Data),
    });

    FUNCTION("main", {t0, t1, t2, t3, t4, t5}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(npu::tile_fwk::GetTensorData(t5, {n - 1, s - 1, m * s - 1}))) {
            IF (i == 0) {
                out = Add(t0, t1); // +t0, +t1
            } ELSE {
                out = Add(out, t1); // +t1 * 7
                IF(i < condThreshold) {
                    out = Add(out, t2); // +t2 * 5
                }
                ELSE {
                    out = Mul(out, t3); // +t3 * 2
                }
                out = Add(out, t4);
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(loopCount)) {
            out = Add(out, t1); // +t1
            IF(i < condThreshold) {
                out = Add(out, t2); // +t2 * 5
            }
            ELSE {
                out = Mul(out, t3); // +t3 * 2
            }
            out = Add(out, t4);
        }
    }
}

TEST_F_WITH_COST(DynamicOpsTest, Cube, 98) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int n = 4;
    int k = 1024;
    int m = 576;

    using EltType = float;
    DataType eltType = DT_FP32;
    Tensor t0(eltType, {n, k}, "t0");
    Tensor t1(eltType, {k, m}, "t1");
    Tensor t2(eltType, {n, m}, "t2");

    std::vector<EltType> t0Data(n * k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            t0Data[i * k + j] = (i / 32) * 4 + (j / 64);
        }
    }
    std::vector<EltType> t1Data(k * m);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            t1Data[i * m + j] = (i / 64) * 4 + (j / 64);
        }
    }
    std::vector<EltType> t2Data(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            EltType sum = 0;
            for (int x = 0; x < k; x++) {
                sum += t0Data[i * k + x] * t1Data[x * m + j];
            }
            t2Data[i * m + j] = sum;
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<EltType>(t0, t0Data),
        RawTensorData::CreateTensor<EltType>(t1, t1Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<EltType>(t2, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<EltType>(t2, t2Data),
    });

    FUNCTION("main", {t0, t1}, {t2}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            t2 = Matrix::Matmul(eltType, t0, t1); // int32
        }
    }
}

TEST_F(DynamicOpsTest, Cmps) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 16;
    Tensor self(DT_FP32, {b, s}, "self");
    Element elem(DT_FP32, 4.0f);
    Tensor out(DT_BOOL, {b, s}, "out");
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 4.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<bool>(out, false),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<bool>(out, true),
    });
    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = Compare(t0, elem,
                          static_cast<OpType>(CmpOperationType::EQ),
                          static_cast<OutType>(CmpModeType::BOOL));
        }
    }
}


TEST_F(DynamicOpsTest, ElementScalar) {
    auto floatElement = Element(DT_BF16, 2.0);
    auto intElement = Element(DT_INT32, static_cast<long>(2));

    auto floatRes = floatElement + floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 4.0);

    floatRes = floatElement - floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 0.0);

    floatRes = floatElement * floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 4.0);

    floatRes = floatElement / floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 1.0);

    auto intRes = intElement % intElement;
    EXPECT_EQ(intRes.GetSignedData(), 0);

    intRes = intElement + intElement;
    EXPECT_EQ(intRes.GetSignedData(), 4);

    intRes = intElement - intElement;
    EXPECT_EQ(intRes.GetSignedData(), 0);

    intRes = intElement * intElement;
    EXPECT_EQ(intRes.GetSignedData(), 4);

    intRes = intElement / intElement;
    EXPECT_EQ(intRes.GetSignedData(), 1);

    auto BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement == floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement != floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement < floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement <= floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement >= floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement == intElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = intElement != intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement < intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement <= intElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = intElement > intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement >= intElement;
    EXPECT_EQ(BoolRes, true);
}

TEST_F(DynamicOpsTest, MatmulAcc) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    Tensor t0(DT_FP32, {128, 128}, "t0");
    Tensor t1(DT_FP32, {128, 128}, "t1");
    Tensor t2(DT_FP32, {64, 64}, "t2");
    Tensor out(DT_FP32, {64, 64}, "out");

    auto d0 = RawTensorData::CreateConstantTensor<float>(t0, 1.0f);
    auto d1 = RawTensorData::CreateConstantTensor<float>(t1, 1.0f);
    auto d2 = RawTensorData::CreateConstantTensor<float>(t2, 1.0f);
    auto out0 = RawTensorData::CreateConstantTensor<float>(out, 1.0f);
    auto golden = RawTensorData::CreateConstantTensor<float>(out, 65.0f);
    ProgramData::GetInstance().PrepareData({d0, d1, d2}, {out0}, {golden});

    FUNCTION("main", {t0, t1, t2}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto v0 = View(t0, {64, 64}, {64, 64});
            auto v1 = View(t1, {64, 64}, {64, 64});
            auto m0 = Matrix::Matmul(DT_FP32, v0, v1, false, true);
            out = Add(m0, t2);
        }
    }
}

TEST_F(DynamicOpsTest, GetTensorData) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor out(DT_FP32, {64, 64}, "out");

    auto t0Data = RawTensorData::CreateConstantTensor<float>(t0, 1.0f);
    auto outData = RawTensorData::CreateConstantTensor<float>(out, 0.0f);
    auto golden = RawTensorData::CreateConstantTensor<float>(out, 2.0f);

    ProgramData::GetInstance().PrepareData({t0Data}, {outData}, {golden});

    FUNCTION("main", {t0}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(2)) {
            auto v = Full(Element(DT_INT32, 32), DT_INT32, {16, 16});
            auto index = GetTensorData(v, {0, 0});
            Print("i=", i, " index=", index, " v=", v);
            auto d = Add(t0, t0);
            Assemble(d, {index * i, 0}, out);
            Assemble(d, {index * i, 32}, out);
        }
    }
}

static auto Random(DataType t, const std::vector<int64_t> &shape) {
    auto data = std::make_shared<LogicalTensorData>(std::make_shared<RawTensorData>(t, shape));
    calc::Random(data);
    return data;
}

static void TestMatmul(DataType inType, DataType outType) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);

    Tensor t0(inType, {64, 256}, "t0");
    Tensor t1(inType, {256, 64}, "t1");
    Tensor out(outType, {64, 64}, "out");

    auto d0 = Random(inType, t0.GetShape());
    auto d1 = Random(inType, t1.GetShape());
    auto out0 = Random(outType, out.GetShape());
    auto golden = Random(outType, out.GetShape());
    calc::MatMul(golden, d0, d1);

    ProgramData::GetInstance().PrepareData({d0->GetData(), d1->GetData()}, {out0->GetData()}, {golden->GetData()});

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            out = Matrix::Matmul(outType, t0, t1, false, false);
        }
    }
}

TEST_F(DynamicOpsTest, MatmulFP16FP16) {
    TestMatmul(DT_FP16, DT_FP16);
}

TEST_F(DynamicOpsTest, MatmulBF16BF16) {
    TestMatmul(DT_BF16, DT_BF16);
}

TEST_F(DynamicOpsTest, MatmulFP16FP32) {
    TestMatmul(DT_FP16, DT_FP32);
}

TEST_F(DynamicOpsTest, MatmulBF16FP32) {
    TestMatmul(DT_BF16, DT_FP32);
}

TEST_F(DynamicOpsTest, MatmulFP32FP32) {
    TestMatmul(DT_FP32, DT_FP32);
}

TEST_F(DynamicOpsTest, MatMulPertensor) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    Tensor t0(DT_INT8, {128, 256}, "t0");
    Tensor t1(DT_INT8, {128, 256}, "t1");
    Tensor out(DT_FP16, {128, 128}, "out");

    auto d0 = RawTensorData::CreateConstantTensor<int8_t>(t0, 1);
    auto logicTensor0 = LogicalTensorData::Create(*d0);
    auto d1 = RawTensorData::CreateConstantTensor<int8_t>(t1, 1);
    auto logicTensor1 = LogicalTensorData::Create(*d1);
    auto out0 = Random(DT_FP16, out.GetShape());
    auto golden = Random(DT_FP16, out.GetShape());
    float scaleValue = 2.0;
    uint32_t scaleValueTmp = 0;
    memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &scaleValue, sizeof(scaleValue));
    calc::MatMul(golden, logicTensor0, logicTensor1,
        {false, true, 0, scaleValueTmp, 1, nullptr, nullptr});

    ProgramData::GetInstance().PrepareData({logicTensor0->GetData(), logicTensor1->GetData()},
        {out0->GetData()}, {golden->GetData()});

    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64}, true, false);
    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Matrix::MatmulExtendParam pm;
            pm.scaleValue = scaleValueTmp;
            pm.reluType = Matrix::ReLuType::ReLu;
            out = Matrix::Matmul(DT_FP16, t0, t1, pm, false, true, false);
        }
    }
}

TEST_F(DynamicOpsTest, MatMulPerchannel) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    Tensor t0(DT_INT8, {128, 128}, "t0");
    Tensor t1(DT_INT8, {128, 128}, "t1");
    Tensor out(DT_FP16, {128, 128}, "out");
    Tensor scaleTensor(DT_UINT64, {1, 128}, "scale");

    auto d0 = RawTensorData::CreateConstantTensor<int8_t>(t0, 1);
    auto logicTensor0 = LogicalTensorData::Create(*d0);
    auto d1 = RawTensorData::CreateConstantTensor<int8_t>(t1, 1);
    auto logicTensor1 = LogicalTensorData::Create(*d1);
    auto out0 = Random(DT_FP16, out.GetShape());
    auto golden = Random(DT_FP16, out.GetShape());
    float scaleValue = 2.0;
    uint32_t scaleValueTmp = 0;
    memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &scaleValue, sizeof(scaleValue));
    auto scaleTensorRaw =
        RawTensorData::CreateConstantTensor<uint64_t>(scaleTensor, scaleValueTmp);
    auto logicScale = LogicalTensorData::Create(*scaleTensorRaw);
    auto logicScaleData = Trans(logicScale);
    calc::MatMul(golden, logicTensor0, logicTensor1,
        {false, true, 0, 0, 0, &logicScaleData, nullptr});

    ProgramData::GetInstance().PrepareData({logicTensor0->GetData(), logicTensor1->GetData(),
        logicScale->GetData()}, {out0->GetData()}, {golden->GetData()});

    FUNCTION("main", {t0, t1, scaleTensor}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Matrix::MatmulExtendParam pm;
            pm.scaleTensor = scaleTensor;
            out = Matrix::Matmul(DT_FP16, t0, t1, pm, false, true, false);
        }
    }
}

TEST_F(DynamicOpsTest, MatMulBias) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    Tensor t0(DT_FP16, {256, 64}, "t0");
    Tensor t1(DT_FP16, {64, 256}, "t1");
    Tensor out(DT_FP16, {256, 256}, "out");
    Tensor biasTensor(DT_FP16, {1, 256}, "bias");

    auto d0 = Random(DT_FP16, t0.GetShape());
    auto d1 = Random(DT_FP16, t1.GetShape());
    auto out0 = Random(DT_FP16, out.GetShape());
    auto golden = Random(DT_FP16, out.GetShape());
    auto logicBias = Random(DT_FP16, biasTensor.GetShape());
    auto logicBiasData = Trans(logicBias);
    calc::MatMul(golden, d0, d1,
        {false, false, 0, 0, 0, nullptr, &logicBiasData});

    ProgramData::GetInstance().PrepareData({d0->GetData(), d1->GetData(),
        logicBias->GetData()}, {out0->GetData()}, {golden->GetData()});

    FUNCTION("main", {t0, t1, biasTensor}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Matrix::MatmulExtendParam pm;
            pm.biasTensor = biasTensor;
            out = Matrix::Matmul(DT_FP16, t0, t1, pm, false, false, false);
        }
    }
}

TEST_F(DynamicOpsTest, MatMulL0CToL1Fixpipe) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    Tensor t0(DT_INT8, {64, 64}, "t0");
    Tensor t1(DT_INT8, {64, 64}, "t1");
    Tensor l0c2L1Tensor(DT_FP16, {64, 64}, "l0c2L1");
    Tensor scaleTensor(DT_UINT64, {1, 64}, "scale");
    Tensor out(DT_FP16, {64, 64}, "out");

    auto d0 = RawTensorData::CreateConstantTensor<int8_t>(t0, 1);
    auto logicTensor0 = LogicalTensorData::Create(*d0);
    auto d1 = RawTensorData::CreateConstantTensor<int8_t>(t1, 1);
    auto logicTensor1 = LogicalTensorData::Create(*d1);
    auto l0c2L1Data = Random(DT_FP16, l0c2L1Tensor.GetShape());
    auto out0 = Random(DT_FP16, out.GetShape());
    auto golden = Random(DT_FP16, out.GetShape());
    float scaleValue = 2.0;
    uint32_t scaleValueTmp = 0;
    memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &scaleValue, sizeof(scaleValue));
    auto scaleTensorRaw =
        RawTensorData::CreateConstantTensor<uint64_t>(scaleTensor, scaleValueTmp);
    auto logicScale = LogicalTensorData::Create(*scaleTensorRaw);
    auto logicScaleData = Trans(logicScale);
    calc::MatMul(golden, logicTensor0, logicTensor1,
        {false, false, 0, 0, 0, &logicScaleData, nullptr});
    calc::MatMul(golden, golden, l0c2L1Data);

    ProgramData::GetInstance().PrepareData({logicTensor0->GetData(), logicTensor1->GetData(),
        l0c2L1Data->GetData(), logicScale->GetData()}, {out0->GetData()}, {golden->GetData()});

    FUNCTION("main", {t0, t1, l0c2L1Tensor, scaleTensor}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Matrix::MatmulExtendParam pm;
            pm.scaleTensor = scaleTensor;
            Tensor tensorTmp = Matrix::Matmul(DT_FP16, t0, t1, pm, false, false, false);
            out = Matrix::Matmul(DT_FP16, tensorTmp, l0c2L1Tensor, false, false);
        }
    }
}

TEST_F(DynamicOpsTest, GatherInL1) {
    Tensor param(DT_FP16, {4, 16}, "t0");
    Tensor indices(DT_INT32, {1, 4}, "t1");
    Tensor pageTable(DT_INT32, {1, 2}, "t1");
    Tensor out(DT_FP16, {4, 16}, "t1");
 	 
    auto paramData = Random(DT_FP16, param.GetShape());
    auto indicesRaw = RawTensorData::CreateTensor<int32_t>(indices, {0, 1, 1, 0});
    auto indicesData = LogicalTensorData::Create(*indicesRaw);
    auto pageTableRaw = RawTensorData::CreateTensor<int32_t>(pageTable, {0, 1});
    auto pageTableData = LogicalTensorData::Create(*pageTableRaw);
    auto out0 = Random(DT_FP16, out.GetShape());
    auto golden = Random(DT_FP16, out.GetShape());
    int64_t blockSize = 2;
    int hidden_dim = 16;
 	calc::GatherInL1(golden, paramData, indicesData, pageTableData, blockSize);
    ProgramData::GetInstance().PrepareData({paramData->GetData(), indicesData->GetData(),
        pageTableData->GetData()}, {out0->GetData()}, {golden->GetData()});

    FUNCTION("test", {param, indices, pageTable}, {out}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1)) {
            (void)sIdx;
            TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {128, 128});

            std::vector<SymbolicScalar> srcValidShape = {param.GetShape()[0], param.GetShape()[1]};
            Tensor dynSrc = View(param, param.GetShape(), srcValidShape, {0, 0});
            std::vector<SymbolicScalar> offsetsValidShape = {indices.GetShape()[0], indices.GetShape()[1]};
            Tensor dynOffsets = View(indices, indices.GetShape(), offsetsValidShape, {0, 0});
            out = experimental::GatherInL1<false, false>(dynSrc, dynOffsets, pageTable, blockSize, hidden_dim);
        }
    }
}

TEST_F(DynamicOpsTest, Round) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t n = 4;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, n}, "outValue");

    std::vector<float> inputData = {1.2f, 2.0f, 3.9f, -1.1f, -2.9f, 5.5f, -0.1f, 7.0f};

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
    });

    int decimals = 0;
    FUNCTION("main", {self}, {outValue}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            outValue = Round(t0, decimals);
        }
    }
}

TEST_F(DynamicOpsTest, TriU) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 8;
    int64_t s = 8;
    int64_t diagonal = 0;
    Tensor input(DT_INT8, {b, s}, "input");
    Tensor out(DT_INT8, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int8_t>(input, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int8_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int8_t>(out, 2),
    });

    FUNCTION("main", {input}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(input, {b, s}, {0, 0});
            out = TriU(t0, diagonal);
        }
    }
}

TEST_F(DynamicOpsTest, Gcd) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 8;
    int64_t s = 8;
    Tensor input1(DT_INT32, {b, s}, "input1");
    Tensor input2(DT_INT32, {b, s}, "input2");
    Tensor out(DT_INT32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input1, 1),
    });
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input2, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });

    FUNCTION("main", {input1, input2}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t1 = View(input1, {b, s}, {0, 0});
            auto t2 = View(input2, {b, s}, {0, 0});
            out = Gcd(t1, t2);
        }
    }
}

TEST_F(DynamicOpsTest, GcdBrc) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 8;
    int64_t s = 8;
    Tensor input1(DT_INT32, {b, s}, "input1");
    Tensor input2(DT_INT32, {b, 1}, "input2");
    Tensor out(DT_INT32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input1, 1),
    });
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input2, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });

    FUNCTION("main", {input1, input2}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t1 = View(input1, {b, s}, {0, 0});
            auto t2 = View(input2, {b, 1}, {0, 0});
            out = Gcd(t1, t2);
        }
    }
}

TEST_F(DynamicOpsTest, Gcds) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 8;
    int64_t s = 8;
    Element alpha = Element(DT_INT32, b);
    Tensor input1(DT_INT32, {b, s}, "input1");
    Tensor out(DT_INT32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input1, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int32_t>(out, 2),
    });

    FUNCTION("main", {input1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t1 = View(input1, {b, s}, {0, 0});
            out = Gcd(t1, alpha);
        }
    }
}

TEST_F(DynamicOpsTest, GatherElement) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t s = 8;
    Tensor source(DT_FP32, {b, s}, "source");
    Tensor index(DT_INT64, {b, s}, "index");
    int axis = 0;
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(source, 1.0),
        RawTensorData::CreateConstantTensor<int64_t>(index, 0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 2.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 1.0),
    });

    FUNCTION("main", {source, index}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(source, {b, s}, {0, 0});
            auto t1 = View(index, {b, s}, {0, 0});
            out = GatherElements(t0, t1, axis);
        }
    }
}

TEST_F(DynamicOpsTest, IndexAdd) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 2;
    int64_t s = 8;
    Tensor self(DT_FP32, {b, s}, "self");
    Tensor source(DT_FP32, {b, s}, "source");
    Tensor index(DT_INT32, {b}, "index");
    int axis = 0;
    Element alpha(DT_FP32, 2.0);
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 1.0),
        RawTensorData::CreateConstantTensor<float>(source, 1.0),
        RawTensorData::CreateConstantTensor<int32_t>(index, 0)
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 3.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 3.0),
    });

    FUNCTION("main", {self, source, index}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            auto t1 = View(source, {b, s}, {0, 0});
            auto t2 = View(index, {b}, {0});
            out = IndexAdd(t0, t1, t2, axis, alpha);
        }
    }
}

TEST_F(DynamicOpsTest, ScatterElement) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_FP32, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Element src(DT_FP32, 2.0);
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 1.0),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 2.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 2.0),
    });

    FUNCTION("main", {self, idx}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            auto t1 = View(idx, {b, s}, {0, 0});
            out = Scatter(t0, t1, src, 0);
        }
    }
}

static void Scatter(Tensor &self, Tensor &idx, Tensor &src, Tensor &out, int b, int s) {
    FUNCTION("main", {self, idx, src}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto a = View(self, {b, s}, {0, 0});
            auto d = View(src, {b, s}, {0, 0});
            auto c = View(idx, {b, s}, {0, 0});
            out = Scatter(a, c, d, 0);
        }
    }
}

TEST_F(DynamicOpsTest, ScatterINT8) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_INT8, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_INT8, {b, s}, "src");
    Tensor out(DT_INT8, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int8_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<int8_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int8_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int8_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ScatterUINT8) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_UINT8, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_UINT8, {b, s}, "src");
    Tensor out(DT_UINT8, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<uint8_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<uint8_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<uint8_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<uint8_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ScatterINT16) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_INT16, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_INT16, {b, s}, "src");
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<int16_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ScatterUINT16) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_UINT16, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_UINT16, {b, s}, "src");
    Tensor out(DT_UINT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<uint16_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<uint16_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<uint16_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<uint16_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ScatterUINT32) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_UINT32, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_UINT32, {b, s}, "src");
    Tensor out(DT_UINT32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<uint32_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<uint32_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<uint32_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<uint32_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ScatterUINT64) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_UINT64, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_UINT64, {b, s}, "src");
    Tensor out(DT_UINT64, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<uint64_t>(self, 1),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<uint64_t>(src, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<uint64_t>(out, 2),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<uint64_t>(out, 2),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, Scatter) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t s = 8;
    Tensor self(DT_FP32, {b, s}, "self");
    Tensor idx(DT_INT64, {b, s}, "idx");
    Tensor src(DT_FP32, {b, s}, "src");
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 1.0),
        RawTensorData::CreateConstantTensor<int64_t>(idx, 0),
        RawTensorData::CreateConstantTensor<float>(src, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 2.0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 2.0),
    });
    Scatter(self, idx, src, out, b, s);
}

TEST_F(DynamicOpsTest, ReduceMax) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 16;
    int64_t n = 16;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {1, n}, "outValue");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 1.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
    });

    FUNCTION("main", {self}, {outValue}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            outValue = Amax(t0, 0, true);
        }
    }
}

TEST_F(DynamicOpsTest, Topk) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t n = 64;
    int k = 32;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, k}, "outValue");
    Tensor outIndex(DT_INT32, {b, k}, "outIndex");

    std::vector<float> inputData = {
        31.0f, 15.0f, 27.0f, 8.0f, 19.0f, 3.0f, 23.0f, 11.0f,
        7.0f, 28.0f, 16.0f, 2.0f, 24.0f, 9.0f, 30.0f, 14.0f,
        22.0f, 5.0f, 18.0f, 1.0f, 26.0f, 10.0f, 29.0f, 13.0f,
        6.0f, 20.0f, 12.0f, 25.0f, 4.0f, 21.0f, 0.0f, 17.0f,

        31.0f, 15.0f, 27.0f, 8.0f, 19.0f, 3.0f, 23.0f, 11.0f,
        7.0f, 28.0f, 16.0f, 2.0f, 24.0f, 9.0f, 30.0f, 14.0f,
        22.0f, 5.0f, 18.0f, 1.0f, 26.0f, 10.0f, 29.0f, 13.0f,
        6.0f, 20.0f, 12.0f, 25.0f, 4.0f, 21.0f, 0.0f, 17.0f,
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
        RawTensorData::CreateConstantTensor<float>(outIndex, 0.0f),
    });

    FUNCTION("main", {self}, {outValue, outIndex}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            std::tie(outValue, outIndex) = TopK(t0, k, 1, true);
        }
    }
}

TEST_F(DynamicOpsTest, TopKSort) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t n = 32;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor outValue(DT_FP32, {b, n * 2}, "outValue");
    Tensor outTemp(DT_FP32, {b, n * 2}, "outTemp");

    std::vector<float> inputData = {
        31.0f, 15.0f, 27.0f, 8.0f, 19.0f, 3.0f, 23.0f, 11.0f,
        7.0f, 28.0f, 16.0f, 2.0f, 24.0f, 9.0f, 30.0f, 14.0f,
        22.0f, 5.0f, 18.0f, 1.0f, 26.0f, 10.0f, 29.0f, 13.0f,
        6.0f, 20.0f, 12.0f, 25.0f, 4.0f, 21.0f, 0.0f, 17.0f
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outValue, 0.0f),
        RawTensorData::CreateConstantTensor<float>(outTemp, 0.0f),
    });

    FUNCTION("main", {self}, {outValue, outTemp}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            std::tie(outValue, outTemp) = TopKSort(t0, 0);
        }
    }
}

TEST_F(DynamicOpsTest, TopKMerge) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t n = 32;  // 32 elements = 16 packs
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor out(DT_FP32, {b, n}, "out");

    // Pre-sorted pack data: two groups of 8 packs each
    std::vector<float> packData = {
        // First 8 packs (sorted descending)
        30.0f, 0.0f, 28.0f, 1.0f, 26.0f, 2.0f, 24.0f, 3.0f,
        22.0f, 4.0f, 20.0f, 5.0f, 18.0f, 6.0f, 16.0f, 7.0f,
        // Second 8 packs (sorted descending)
        31.0f, 8.0f, 29.0f, 9.0f, 27.0f, 10.0f, 25.0f, 11.0f,
        23.0f, 12.0f, 21.0f, 13.0f, 19.0f, 14.0f, 17.0f, 15.0f
    };

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, packData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            out = TopKMerge(t0, 8);
        }
    }
}

TEST_F(DynamicOpsTest, TopKExtractValues) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t n = 32;  // 16 packs
    int64_t k = 8;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor out(DT_FP32, {1, k}, "out");

    // Pack data with values and indices
    std::vector<float> packData = {
        100.0f, 5.0f, 95.0f, 12.0f, 90.0f, 3.0f, 85.0f, 18.0f,
        80.0f, 7.0f, 75.0f, 21.0f, 70.0f, 1.0f, 65.0f, 14.0f,
        60.0f, 9.0f, 55.0f, 25.0f, 50.0f, 2.0f, 45.0f, 16.0f,
        40.0f, 11.0f, 35.0f, 28.0f, 30.0f, 4.0f, 25.0f, 19.0f
    };

    std::vector<float> expectedValues = {100.0f, 95.0f, 90.0f, 85.0f, 80.0f, 75.0f, 70.0f, 65.0f};

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, packData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor(out, expectedValues),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            out = TopKExtract(t0, k, false);
        }
    }
}

TEST_F(DynamicOpsTest, TopKExtractIndices) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t b = 1;
    int64_t n = 32;  // 16 packs
    int64_t k = 8;
    Tensor self(DT_FP32, {b, n}, "self");
    Tensor out(DT_INT32, {1, k}, "out");

    // Pack data with values and indices
    std::vector<float> packData = {
        100.0f, 5.0f, 95.0f, 12.0f, 90.0f, 3.0f, 85.0f, 18.0f,
        80.0f, 7.0f, 75.0f, 21.0f, 70.0f, 1.0f, 65.0f, 14.0f,
        60.0f, 9.0f, 55.0f, 25.0f, 50.0f, 2.0f, 45.0f, 16.0f,
        40.0f, 11.0f, 35.0f, 28.0f, 30.0f, 4.0f, 25.0f, 19.0f
    };

    std::vector<int32_t> expectedIndices = {5, 12, 3, 18, 7, 21, 1, 14};

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor(self, packData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor(out, expectedIndices),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, n}, {0, 0});
            out = TopKExtract(t0, k, true);
        }
    }
}

TEST_F(DynamicOpsTest, BitwiseRightShift) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_INT16, {b, s}, "self");
    Tensor other(DT_INT16, {b, s}, "other");
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(self, 4),
        RawTensorData::CreateConstantTensor<int16_t>(other, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 2),
    });

    FUNCTION("main", {self, other}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = BitwiseRightShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, BitwiseLeftShift) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_INT16, {b, s}, "self");
    Tensor other(DT_INT16, {b, s}, "other");
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(self, 4),
        RawTensorData::CreateConstantTensor<int16_t>(other, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 8),
    });

    FUNCTION("main", {self, other}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = BitwiseLeftShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, BitwiseRightShifts) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_INT16, {b, s}, "self");
    Element other(DT_INT16, 1);
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(self, 4),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 2),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = BitwiseRightShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, BitwiseLeftShifts) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_INT16, {b, s}, "self");
    Element other(DT_INT16, 1);
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(self, 4),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 8),
    });

    FUNCTION("main", {self}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = BitwiseLeftShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, SBitwiseRightShift) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Element self(DT_INT16, 4);
    Tensor other(DT_INT16, {b, s}, "self");
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(other, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 2),
    });

    FUNCTION("main", {other}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(other, {b, s}, {0, 0});
            out = BitwiseRightShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, SBitwiseLeftShift) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Element self(DT_INT16, 4);
    Tensor other(DT_INT16, {b, s}, "self");
    Tensor out(DT_INT16, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int16_t>(other, 1),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int16_t>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int16_t>(out, 8),
    });

    FUNCTION("main", {other}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(other, {b, s}, {0, 0});
            out = BitwiseLeftShift(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, CopySign) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    int64_t b = 8;
    int64_t s = 8;
    Tensor self(DT_FP32, {b, s}, "self");
    Tensor other(DT_FP32, {b, s}, "other");
    Tensor out(DT_FP32, {b, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(self, 4),
        RawTensorData::CreateConstantTensor<float>(other, 4),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 4),
    });

    FUNCTION("main", {self, other}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = View(self, {b, s}, {0, 0});
            out = CopySign(self, other);
        }
    }
}

TEST_F(DynamicOpsTest, Range) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    int64_t size = 5;  
    Element start(DT_INT32, 1);
    Element end(DT_INT32, 10);
    Element step(DT_INT32, 2);
    
    Tensor out(DT_INT32, {size}, "out");
    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(out, 0),
    });

    std::vector<int32_t> expected_data = {1, 3, 5, 7, 9};
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(out, expected_data),
    });

    FUNCTION("main", {}, {out}) {
        out = Range(start, end, step);
    }
}
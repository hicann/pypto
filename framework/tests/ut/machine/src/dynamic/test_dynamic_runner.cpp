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
 * \file test_dynamic_runner.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "machine/runtime/device_runner.h"
#include "machine/runtime/machine_agent.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#define private public
using namespace npu::tile_fwk;
class TestDynamicDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        Platform::Instance().ObtainPlatformInfo();
    }

    void TearDown() override {}
};

TEST_F(TestDynamicDeviceRunner, TestInitArgs) {
    auto &runner = DeviceRunner::Get();
    [[maybe_unused]]DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 2;
    args.nrValidAic = args.nrAic;
    runner.InitDynamicArgs(args);
    runner.DumpAiCoreExecutionTimeData();
    runner.DumpAiCorePmuData();
    runner.SynchronizeDeviceToHostProfData();
}

TEST_F(TestDynamicDeviceRunner, TestDynamicRun) {
    auto &runner = npu::tile_fwk::DeviceRunner::Get();
    [[maybe_unused]]DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 2;
    runner.InitDynamicArgs(args);
    [[maybe_unused]]npu::tile_fwk::AstKernelArgs taskArgs;
    runner.args_.nrAic = 2;
    runner.args_.nrAiv = 2;
    int ret = runner.DynamicRun(0, 0, 0, 0, &taskArgs, 2);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestDynamicDeviceRunner, TestDynMachineAgent) {
    npu::tile_fwk::MachinePipe machinePipe;
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    config::SetBuildStatic(true);
    FUNCTION("ADD", {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");
    auto task_1 = std::make_shared<MachineTask>(0, function);
    DeviceAgentTask agentTask1(task_1);
    machinePipe.PipeProc(&agentTask1);

    function->SetFunctionType(FunctionType::DYNAMIC_LOOP);
    auto task_2 = std::make_shared<MachineTask>(0, function);
    DeviceAgentTask agentTask2(task_2);
    machinePipe.PipeProc(&agentTask2);

    function->SetFunctionType(FunctionType::INVALID);
    auto task_3 = std::make_shared<MachineTask>(0, function);
    DeviceAgentTask agentTask3(task_3);
    machinePipe.PipeProc(&agentTask3);
}

TEST_F(TestDynamicDeviceRunner, TestRegisterDynamicKernel) {
    [[maybe_unused]]rtBinHandle staticHdl_;
    npu::tile_fwk::DeviceRunner runner;
    runner.RegisterKernelBin(&staticHdl_);
}

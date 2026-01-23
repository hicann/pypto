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
#include "machine/runtime/device_launcher.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/dump/kernel_dump_utils.h"
#include "interface/program/program.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/runtime/dump_device_perf.h"
#define private public
using namespace npu::tile_fwk;

 extern "C" uint32_t DynPyptoKernelServerNull(void *targ);
class TestDynamicDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
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
    [[maybe_unused]]npu::tile_fwk::DeviceKernelArgs taskArgs;
    std::vector<uint8_t> tensorInfo(sizeof(dynamic::AiCpuArgs));
    taskArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo.data());
    taskArgs.outputs = 0;
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

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_null) {
    DeviceKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuSoLen = 2;
    pyptoKernelArgs.cfgdata = static_cast<int64_t *>(static_cast<void *>(&devKernelArgs));
    auto ret = DynPyptoKernelServerNull(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_kernel_dump) {
    const std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A0");
    Tensor inputB(DT_FP32, shape, "B1");
    Tensor output(DT_FP32, shape, "C0");

    config::SetBuildStatic(true);
    FUNCTION("ADD0", {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD0");
    auto task_1 = std::make_shared<MachineTask>(0, function);
    auto deviceMachineTask = std::make_shared<MachineTask>(task_1->GetTaskId(), task_1->GetFunction());
    auto deviceAgentTask = std::make_shared<DeviceAgentTask>(deviceMachineTask);

    KernelDumpUtils kernelDump;
    std::string jsonDir = "/tmp/pypto/";
    std::string kerneName = "pypto_add";
    kernelDump.DumpJsonFile(deviceAgentTask.get(), kerneName,jsonDir);
    std::vector<JsonInfo> binJsonPath;
    std::string jsonFilePath = jsonDir + kerneName + ".json";
    std::string binFileName = "add_bin";
    kernelDump.WriteFatbinJson(binJsonPath, jsonFilePath, binFileName);
    auto ret = IsPathExist(jsonFilePath);
    EXPECT_EQ(ret, false);
}

TEST_F(TestDynamicDeviceRunner, test_dump_device_perf) {
    DeviceArgs devKernelArgs;
    devKernelArgs.nrAic = 1;
    devKernelArgs.nrAiv = 2;
    devKernelArgs.nrValidAic = 1;
    devKernelArgs.nrAicpu = 3;
    std::vector<void *> perfData;
    Metrics *metr = static_cast<Metrics*>(malloc(sizeof(Metrics) + sizeof(TaskStat)));
    TaskStat taskStat;
    taskStat.execEnd =1;
    metr->taskCount = 1;
    metr->tasks[0] = taskStat;
    metr->perfTrace[0][0] = 1;
    metr->taskCount = 1;

    for (uint64_t i = 0; i < devKernelArgs.nrAic + devKernelArgs.nrAiv; i++) {
        perfData.push_back(static_cast<void*>(metr));
    }
    npu::tile_fwk::dynamic::DumpAicoreTaskExectInfo(devKernelArgs, perfData);
    free(metr);
    std::string jsonPath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    EXPECT_EQ(IsPathExist(jsonPath), true);
}

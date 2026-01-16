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
 * \file test_dev_func_runner.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <thread>
#include <cstdint>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/host/device_agent_task.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/runtime/machine_agent.h"
#include "machine/runtime/device_launcher.h"
#include "cost_model/simulation/backend.h"
#include "machine/runtime/host_prof.h"
#include "machine/runtime/distributed_context.h"

using namespace npu::tile_fwk::dynamic;

namespace npu::tile_fwk {

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {}

    bool IsDevice() { return !isTest_; }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = AllocDev(size, cachedDevAddrHolder);
        if (isTest_)
            memcpy_s(devPtr, size, data, size);
        else
            rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        return devPtr;
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        if (isTest_)
            memcpy_s(data, size, devPtr, size);
        else
            rtMemcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = nullptr;
        if (isTest_)
            devPtr = machine::GetRuntimeHostAgent()->AllocHostAddr(size);
        else
            machine::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }

    uint8_t *AllocZero(uint64_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = AllocDev(size, nullptr);
        if (isTest_)
            memset(devPtr, 0, size);
        else
            rtMemset(devPtr, size, 0, size);
        return devPtr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedHolder) {
        (void)cachedHolder;
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T), nullptr);
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size(), nullptr);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData &tensorData) {
        CopyFromDev(tensorData.data(), tensorData.GetDevPtr(), tensorData.size());
    }

    uint64_t GetL2Offset() {
        return machine::GetRA()->GetL2Offset();
    }

    bool isTest_{true};
};

extern "C" int DynTileFwkBackendKernelServer(void *targ);
extern "C" int DynTileFwkBackendKernelServerInit(void *targ);
extern "C" int PyptoKernelCtrlServer(void *targ);

class DevFuncRunner : public DeviceLauncher {
public:
    static void Run(Function *function, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto runner = DevFuncRunner(function, config);
        DeviceRunner::Get().GetHostProfInstance().SetProfFunction(function);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

    // Run with incast/outcast from ProgramData
    static void Run(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        auto runner = DevFuncRunner(function, config);
        DeviceRunner::Get().GetHostProfInstance().SetProfFunction(function);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

private:
    DevFuncRunner(Function *function, const DeviceLauncherConfig &config) : function_(function), config_(config) {
        if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
            DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary);
        }
    }
    void RunDynamic(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (function_ == nullptr || function_->GetDyndevAttribute() == nullptr) {
            return;
        }
        KernelLaunchPrecheck(inputs, outputs);
        DevAscendProgram *functionDevProg = reinterpret_cast<DevAscendProgram *>(function_->GetDyndevAttribute()->devProgBinary.data());
        if (config_.controlFlowCache) {
            functionDevProg->controlFlowCache.isRecording = true;
        }
        RunModel(inputs, outputs);
        if (functionDevProg->controlFlowCache.isRecording) {
            functionDevProg->controlFlowCache.isRecording = false;

            uint64_t contextWorkspaceAddr = functionDevProg->controlFlowCache.contextWorkspaceAddr;

            functionDevProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
            functionDevProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr);
            functionDevProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(functionDevProg), 0);
            functionDevProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
            functionDevProg->controlFlowCache.TaskAddrRelocProgram(reinterpret_cast<uint64_t>(functionDevProg), 0);
            functionDevProg->ResetFromLaunch();
            functionDevProg->controlFlowCache.isActivated = true;
        }
        if (config_.onBoard) {
            RunOnBoard(inputs, outputs);
        }
    }
    static void RunStatic() {
        if (gDeviceAgentTaskPtr == nullptr) {
            return;
        }
        MachineAgent::AgentProc(gDeviceAgentTaskPtr.get());
        CheckDeviceId();
        MachinePipe::PipeProc(gDeviceAgentTaskPtr.get());
    }

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (!config_.runModel) {
            return;
        }
        DeviceKernelArgs kArgs;
        DeviceLauncherConfigFillDeviceInfo(config_);
        DeviceInitDistributedContextToHost(function_->GetDyndevAttribute()->commGroupNames,
 	        function_->GetDyndevAttribute()->devProgBinary);
        DeviceInitTilingData(MemoryHelper(true), kArgs, function_->GetDyndevAttribute()->devProgBinary, config_, nullptr);
        for (int i = 0; i < (config_.controlFlowCache ? 1 : config_.repeatNum); i++) {
            InitKernelInOuts(kArgs, inputs, outputs, true, {}, false);
            std::cout << "!!! Run CostModel " << i << "\n";
            RunCostModel(&kArgs);
            std::cout << "!!! Run TestModel " << i << "\n";
            RunTestMode(&kArgs);
        }
        RunDynCostModel();
    }

    bool IsDumpTensorEnable() const {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(function_).data()));
        return devProg->memBudget.debug.dumpTensor != 0;
    }

    static void DumpDevDataBinary(std::ostream &os, const uint8_t *hostData, uint64_t size, const uint8_t *devptr) {
        /*
         * Format:
         *   8 bytes: address on device
         *   8 bytes: data block size
         *   n bytes: data block
         */
        uint64_t header[] = {
            reinterpret_cast<uint64_t>(devptr),
            size,
        };
        os.write(reinterpret_cast<const char *>(header), sizeof(header));
        if (hostData != nullptr) {
            os.write(reinterpret_cast<const char *>(hostData), size);
        } else {
            static constexpr uint64_t THROUGHPUT = UINT64_C(1024) * 1024 * 1024;
            std::vector<uint8_t> buf;
            buf.reserve(std::min(THROUGHPUT, size));
            for (uint64_t offset = 0; offset < size; offset += THROUGHPUT) {
                uint64_t blockSize = std::min(THROUGHPUT, size - offset);
                rtMemcpy(buf.data(), buf.capacity(), devptr + offset, blockSize, RT_MEMCPY_DEVICE_TO_HOST);
                os.write(reinterpret_cast<const char *>(buf.data()), blockSize);
            }
        }
    }

    void DumpTensorContents(const DeviceKernelArgs &kArgs,
                            const std::vector<RawTensorDataPtr> &inputs,
                            const std::vector<RawTensorDataPtr> &outputs) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(function_).data()));
        uint8_t *dumpTensorWsPtr = reinterpret_cast<uint8_t *>(kArgs.workspace) + devProg->memBudget.Total() - devProg->memBudget.debug.dumpTensor;
        uint64_t dumpTensorWsUsed = 0;
        rtMemcpy(&dumpTensorWsUsed, sizeof(uint64_t), dumpTensorWsPtr, sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST);
        ALOG_ERROR_F("[DumpTensor] dumpTensorWsPtr=%p, memory used=%lu\n", dumpTensorWsPtr, dumpTensorWsUsed);

        std::string path = config::LogTopFolder() + "/dump_tensor.txt";
        std::ofstream fout(path, std::ios::out | std::ios::binary);

        auto printIODevAddrs = [&](const std::vector<RawTensorDataPtr> &ptrs) {
            uint64_t ptrNum = ptrs.size();
            fout.write(reinterpret_cast<const char *>(&ptrNum), sizeof(ptrNum));
            int idx = 0;
            for (auto &ptr : ptrs) {
                uint64_t devPtr = ptr ? reinterpret_cast<uint64_t>(ptr->GetDevPtr()) : 0;
                ALOG_ERROR_F("[DumpTensor] devPtr %d = %lu\n", idx++, devPtr);
                fout.write(reinterpret_cast<const char *>(&devPtr), sizeof(devPtr));
            }
        };

        // write input/output devAddr list
        ALOG_ERROR_F("[DumpTensor] #inputs=%zu\n", inputs.size());
        printIODevAddrs(inputs);
        ALOG_ERROR_F("[DumpTensor] #outputs=%zu\n", outputs.size());
        printIODevAddrs(outputs);

        DumpDevDataBinary(fout, nullptr, dumpTensorWsUsed, dumpTensorWsPtr);
        for (auto &input : inputs) {
            if (input) {
                DumpDevDataBinary(fout, input->data(), input->GetDataSize(), input->GetDevPtr());
            }
        }
        for (auto &output : outputs) {
            if (output) {
                DumpDevDataBinary(fout, output->data(), output->GetDataSize(), output->GetDevPtr());
            }
        }
        fout.close();
    }

    void RunOnBoard(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        std::cout << "!!! Kernel Launch " << "\n";
        int rc = aclInit(nullptr);
        if (rc != 0 && rc != ACL_ERROR_REPEAT_INITIALIZE) {
            ALOG_ERROR_F("Acl init failed!!!");
            return;
        }
        CheckDeviceId();
        DeviceKernelArgs kArgs;
        DeviceLauncherConfigFillDeviceInfo(config_);
        DeviceInitDistributedContext(function_->GetDyndevAttribute()->commGroupNames,
 	        function_->GetDyndevAttribute()->devProgBinary);
        DeviceInitTilingData(MemoryHelper(false), kArgs, function_->GetDyndevAttribute()->devProgBinary, config_, nullptr);
        auto aicpuStream = machine::GetRA()->GetScheStream();
        auto aicoreStream = machine::GetRA()->GetStream();
        auto ctrlStream = config_.cpuSeparate ? machine::GetRA()->GetCtrlStream() : nullptr;
        for (int i = 0; i < config_.repeatNum; i++) {
            InitKernelInOuts(kArgs, inputs, outputs, false, function_->GetDyndevAttribute()->disableL2List,
                config_.isGETensorList);
            rc = DeviceRunner::Get().DynamicRun(aicpuStream, ctrlStream, aicoreStream, 0, &kArgs, config_.blockdim, config_.aicpuNum);
            EXPECT_EQ(rc, 0);
            DeviceRunner::Get().SynchronizeDeviceToHostProfData();
        }
        CopyFromDev(MemoryHelper(false), outputs);
        if (outputs.size() == 0 || HasInplaceArgs(function_)) {
            CopyFromDev(MemoryHelper(false), inputs);
        }
        if (IsDumpTensorEnable()) {
            DumpTensorContents(kArgs, inputs, outputs);
        }
    }

    void RunCostModel(DeviceKernelArgs *kArgs) {
        if (!config::GetPlatformConfig(KEY_ENABLE_DYN_COST_MODEL, true)) {
            return;
        }
        Function *function = Program::GetInstance().GetLastFunction();
        if (function == nullptr) {
            return;
        }
        config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::LEAF_FUNCTION);
        CostModelAgent costModelAgent;
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
        CostModel::ModelData* modelData = new CostModel::ModelData();
        auto attr = function->GetDyndevAttribute();
        modelData->functionTime.resize(attr->devLeafIndex2Hash.size(), 0);
        for (const auto& [index, hash] : attr->devLeafIndex2Hash) {
            auto time = costModelAgent.GetLeafFunctionTimeCost(hash);
            DEV_INFO("devLeafIndex2Hash, %d -> %lu: %lu\n", index, hash, time);
            modelData->functionTime[index] = time;
        }
        kArgs->costmodeldata = modelData;
    }

    void RunDynCostModel()
    {
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
            return;
        }
        config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::NORMAL);
        CostModelAgent costModelAgent;

        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
        costModelAgent.SubmitTopo(path);
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
        ALOG_DEBUG_F("Finish Run DynCostMode which topo path is: %s", path.c_str());
    }

    void RunTestMode(DeviceKernelArgs *kArgs) {
        (void) kArgs;
        std::thread aicpus[DEVICE_MAX_AICPU_NUM];
        std::atomic<int> idx{0};
        auto *devProg = (DevAscendProgram *)(kArgs->cfgdata);
        auto rc0 = DynTileFwkBackendKernelServerInit(kArgs);
        EXPECT_EQ(rc0, 0);
        int threadNum = static_cast<int>(devProg->devArgs.nrAicpu);
        threadNum = (devProg->devArgs.enableCtrl == 1) ? threadNum : threadNum + 1;
        for (int i = 0; i < threadNum; i++) {
            aicpus[i] = std::thread([&]() {
                int tidx = idx++;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tidx, &cpuset);
                char name[64];
                sprintf(name, "aicput%d", tidx);
                std::cout << "start thread: " << name << std::endl;
                pthread_setname_np(pthread_self(), name);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                auto rc = 0;
                if ((devProg->devArgs.enableCtrl == 0) && (uint32_t)tidx == devProg->devArgs.scheCpuNum) {
                    rc = PyptoKernelCtrlServer(kArgs);
                } else {
                    rc = DynTileFwkBackendKernelServer(kArgs);
                }
                EXPECT_EQ(rc, 0);
            });
        }

        for (int i = 0; i < threadNum; i++) {
            if (aicpus[i].joinable()) {
                aicpus[i].join();
            }
        }
    }

    void InitKernelInOuts(DeviceKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputTensors,
        const std::vector<RawTensorDataPtr> &outputTensors, bool isTest, const std::vector<uint8_t>& disableL2List,
        bool isGETensorList) {
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        std::tie(inputList, outputList) = BuildInputOutputFromHost(MemoryHelper(isTest), inputTensors, outputTensors);
        DeviceInitKernelInOuts(MemoryHelper(isTest), kArgs, inputList, outputList, disableL2List, isGETensorList);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
    }

    void KernelLaunchPrecheck(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        auto checkInouts = [&](std::vector<std::reference_wrapper<const Tensor>> &tensorList,
                               const std::vector<RawTensorDataPtr> &dataList) {
            EXPECT_EQ(tensorList.size(), dataList.size()) << "argument num not match !!!!";
            for (size_t i = 0; i < tensorList.size(); i++) {
                auto &t = tensorList[i].get();
                auto &d = dataList[i];
                if (d) {
                    EXPECT_EQ(t.GetDataType(), d->GetDataType());
                    auto rawShape = t.GetStorage()->GetRawTensor()->GetDynRawShape();
                    auto shape = d->GetShape();
                    for (size_t k = 0; k < rawShape.size(); k++) {
                        if (rawShape[k].IsImmediate()) {
                            EXPECT_EQ(rawShape[k].Concrete(), shape[k]);
                        }
                    }
                }
            }
        };

        checkInouts(function_->GetDyndevAttribute()->startArgsInputTensorList, inputs);
        checkInouts(function_->GetDyndevAttribute()->startArgsOutputTensorList, outputs);
    }

private:
    Function *function_;
    DeviceLauncherConfig config_;
};
}

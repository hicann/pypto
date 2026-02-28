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
 * \file cost_model_launcher.h
 * \brief
 */

#pragma once

#include <thread>
#include <cstdint>
#include <unistd.h>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/host/device_agent_task.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/runtime/machine_agent.h"
#include "machine/runtime/device_launcher.h"
#include "cost_model/simulation/backend.h"
#include "machine/runtime/host_prof.h"
#include "tilefwk/tilefwk_log.h"


namespace npu::tile_fwk::dynamic {
class HostAgentStub {
public:
    HostAgentStub(HostAgentStub &other) = delete;

    void operator=(const HostAgentStub &other) = delete;

    static HostAgentStub *GetAgent() {
        static HostAgentStub inst;
        return &inst;
    }

    uint8_t* AllocHostAddr(uint64_t size) {
        if (size == 0) {
            SIMULATION_LOGE("malloc size is 0!");
            return nullptr;
        }
        auto hostPtr = (uint8_t *)malloc(size);
        allocatedHostAddr.emplace_back(hostPtr);
        return hostPtr;
    }

    void Finalize() {
        if (hostInited) {
            DestroyMemory();
        }
    }

    ~HostAgentStub() { Finalize(); }

protected:
    HostAgentStub() {
        Init();
    }

    void DestroyMemory() {
        for (uint8_t *addr : allocatedHostAddr) {
            free(addr);
        }
    }

private:
    void Init() {
        hostInited = true;
    }

private:
    bool hostInited{false};

    std::vector<uint8_t *> allocatedHostAddr;
};

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {}

    bool IsDevice() { return !isTest_; }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        auto ptr = npu::tile_fwk::dynamic::HostAgentStub::GetAgent()->AllocHostAddr(size);
        memcpy_s(ptr, size, data, size);
        return ptr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T), nullptr);
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size(), nullptr);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        memcpy_s(data, size, devPtr, size);
    }

    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = machine::GetRuntimeHostAgent()->AllocHostAddr(size);
        return devPtr;
    }

    uint8_t *AllocZero(uint64_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = AllocDev(size, nullptr);
        memset_s(devPtr, size, 0, size);
        return devPtr;
    }

    void CopyFromDev(RawTensorData &t) {
        CopyFromDev(t.data(), t.GetDevPtr(), t.size());
    }

    uint64_t GetL2Offset() {
        return 0;
    }

    bool isTest_{true};
};

class AiCorePvModelImpl : public CostModel::AiCoreModel {
private:
    std::shared_ptr<CostModel::DynPvModel> pv_;
    std::unordered_map<int, uint64_t> funcdata_;
    std::mutex mtx_;

public:
    explicit AiCorePvModelImpl(std::shared_ptr<CostModel::DynPvModel> pv) : pv_(pv) {
    }

    void InitData(int coreIdx, int64_t funcdata) {
        std::lock_guard<std::mutex> lock(mtx_);
        funcdata_[coreIdx] = funcdata;
    }

    void SendTask(int coreIdx, uint64_t taskId) {
        auto funcdata = funcdata_[coreIdx];
        DynFuncHeader *header = reinterpret_cast<DynFuncHeader*>(funcdata);
        DynFuncData *data = reinterpret_cast<DynFuncData*>(header + 1);
        pv_->Run(data, coreIdx, FuncID(taskId), TaskID(taskId));
    }
};

extern "C" int DynTileFwkBackendKernelServer(void *targ);
extern "C" int PyptoKernelCtrlServer(void *targ);

class CostModelLauncher : public DeviceLauncher {
public:
    static void CostModelRunOnce(Function *function, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

    // Run with incast/outcast from ProgramData
    static void CostModelRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

private:
    CostModelLauncher(Function *function, const DeviceLauncherConfig &config) : function_(function), config_(config) {
    }

    void RunDynamic(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (function_ == nullptr || function_->GetDyndevAttribute() == nullptr) {
            return;
        }

        DevAscendProgram *functionDevProg = reinterpret_cast<DevAscendProgram *>(function_->GetDyndevAttribute()->devProgBinary.data());
        if (config_.controlFlowCache) {
            functionDevProg->controlFlowCache.isRecording = true;
        }
        RunModel(inputs, outputs);
    }

    static void RunStatic() {
    }

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (!config_.runModel) {
            return;
        }
        DeviceKernelArgs kArgs;
        config_.onBoard = false;
        auto dynAttr = function_->GetDyndevAttribute();
        DeviceLauncherConfigFillDeviceInfo(config_);
        DeviceInitDistributedContext(MemoryHelper(true), dynAttr->commGroupNames, kArgs);
        DeviceInitTilingData(MemoryHelper(true), kArgs, dynAttr->devProgBinary, nullptr, config_, nullptr);
        InitKernelInOuts(kArgs, inputs, outputs, true);
        RunCostModel(&kArgs);
        SIMULATION_LOGI("Run TestModel");
        RunTestMode(&kArgs, DEVICE_MAX_AICPU_NUM);
        SIMULATION_LOGI("Run DynCostModel");
        RunDynCostModel();
        SIMULATION_LOGI("Run PvModel");
        RunPvModel(kArgs, inputs, outputs);
    }

    bool IsDumpTensorEnable() const {
        return GetDevProg(function_)->memBudget.debug.dumpTensor != 0;
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
                os.write(reinterpret_cast<const char *>(buf.data()), blockSize);
            }
        }
    }

    void DumpTensorContents(const DeviceKernelArgs &kArgs,
                            const std::vector<RawTensorDataPtr> &inputs,
                            const std::vector<RawTensorDataPtr> &outputs) {
        auto *devProg = GetDevProg(function_);
        uint8_t *dumpTensorWsPtr = reinterpret_cast<uint8_t *>(kArgs.workspace) + devProg->memBudget.tensor.Total() + devProg->memBudget.metadata.Total();
        uint64_t dumpTensorWsUsed = 0;
        SIMULATION_LOGE("[DumpTensor] dumpTensorWsPtr=%p, memory used=%lu\n", dumpTensorWsPtr, dumpTensorWsUsed);

        std::string path = config::LogTopFolder() + "/dump_tensor.txt";
        std::ofstream fout(path, std::ios::out | std::ios::binary);

        auto printIODevAddrs = [&](const std::vector<RawTensorDataPtr> &ptrs) {
            uint64_t ptrNum = ptrs.size();
            fout.write(reinterpret_cast<const char *>(&ptrNum), sizeof(ptrNum));
            int idx = 0;
            for (auto &ptr : ptrs) {
                uint64_t devPtr = ptr ? reinterpret_cast<uint64_t>(ptr->GetDevPtr()) : 0;
                SIMULATION_LOGE("[DumpTensor] devPtr %d = %lu\n", idx++, devPtr);
                fout.write(reinterpret_cast<const char *>(&devPtr), sizeof(devPtr));
            }
        };

        // write input/output devAddr list
        SIMULATION_LOGE("[DumpTensor] #inputs=%zu\n", inputs.size());
        printIODevAddrs(inputs);
        SIMULATION_LOGE("[DumpTensor] #outputs=%zu\n", outputs.size());
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
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        CostModelAgent costModelAgent;

        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
        costModelAgent.SubmitTopo(path);
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
    }

    void RunPvModel(DeviceKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs)
    {
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM || std::getenv("ASCEND_HOME_PATH") == nullptr) {
            return;
        }
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        try {
            pv_ = CostModel::PvModelFactory::CreateDyn();
            pv_->InitPv();
        } catch (const std::runtime_error &e) {
            SIMULATION_LOGE("pv init fail.");
            return;
        }

        model_ = std::make_shared<AiCorePvModelImpl>(pv_);
        const int maxCpuNum = 6;
        pv_->Codegen(function_);
        BuildPvKernelArgs(kArgs, inputs, outputs);
        RunTestMode(&kArgs, maxCpuNum);
        SetDevPtr(inputs, outputs);
        CopyFromDev(inputs, outputs);
    }

    void BuildPvKernelArgs(DeviceKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs)
    {
        MemoryHelper devMem{true};
        auto buildInouts = [&](auto &tensorList, DevTensorData* tensorData) {
            for (auto &t : tensorList) {
                if (t) {
                    auto addrs = reinterpret_cast<uint64_t>(pv_->CopyTensorToDev((uint8_t *)t->data(), t->size()));
                    DevAscendTensorDataCreator::Init(tensorData, addrs, t->GetShape().data(), t->GetShape().size());
                } else {
                    std::vector<int> shape;
                    DevAscendTensorDataCreator::Init(tensorData, 0UL, shape.data(), shape.size());
                }
                tensorData++;
            }
            return;
        };

        std::vector<uint8_t> &devProgData = function_->GetDyndevAttribute()->devProgBinary;
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));

        devProg->devArgs.nrAicpu = 6;
        devProg->devArgs.nrValidAic = 24;
        devProg->devArgs.runtimeDataRingBufferAddr = (uint64_t)pv_->AllocWorkspaceDev(DEV_ARGS_SIZE);
        devProg->workspaceSize = devProg->memBudget.Total();
        devProg->devArgs.scheCpuNum = 1;
        AssignMetaAddr(devMem, kArgs, devProg, nullptr);
        for (auto &input: inputs) {
            if (input)
                input->SetDevPtr(nullptr);
        }
        for (auto &output: outputs) {
            if (output)
                output->SetDevPtr(nullptr);
        }
        size_t tensorSize = (inputs.size() + outputs.size()) * sizeof(DevTensorData) + 2 * sizeof(uint64_t);
        std::vector<uint8_t> tensorInfo(tensorSize);
        auto data = reinterpret_cast<uint64_t*>(tensorInfo.data());
        *data = inputs.size();
        data++;
        *data = outputs.size();
        data++;
        auto dataPtr = reinterpret_cast<DevTensorData*>(data);
        buildInouts(inputs, dataPtr);
        dataPtr += inputs.size();
        buildInouts(outputs, dataPtr);
        kArgs.inputs = (int64_t *)pv_->CopyToDev(tensorInfo.data(), tensorSize);
        kArgs.outputs = kArgs.inputs + 1;
        kArgs.workspace = (int64_t *)pv_->AllocWorkspaceDev(devProg->workspaceSize);
        kArgs.cfgdata = (int64_t *)pv_->CopyToDev(devProgData.data(), devProgData.size());
        kArgs.aicoreModel = model_.get();
    }

    void SetDevPtr(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs)
    {
        auto setDevPtr = [&](auto &tensorList) {
            for (uint i = 0; i < tensorList.size(); i++) {
                int index = pv_->GetOutIndex(i, tensorList.size());
                tensorList[i]->SetDevPtr(reinterpret_cast<uint8_t *>(pv_->GetDataHostPtr(index)));
            }
        };
        setDevPtr(inputs);
        setDevPtr(outputs);
    }

    void CopyFromDev(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs)
    {
        auto copyFromDev = [&](auto &tensorList) {
            for (auto &tensor : tensorList) {
                if (tensor)
                    pv_->CopyFromDev(tensor->data(), tensor->GetDevPtr(), tensor->size());
            }
        };

        copyFromDev(inputs);
        copyFromDev(outputs);
    }

    void RunTestMode(DeviceKernelArgs *kArgs, int maxCpuNum) {
        (void) kArgs;
        std::vector<std::thread> aicpus(maxCpuNum);
        std::atomic<int> idx{0};
        auto *devProg = (DevAscendProgram *)(kArgs->cfgdata);
        size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        auto deviceTaskCtrlPoolAddr = devProg->devArgs.runtimeDataRingBufferAddr + sizeof(RuntimeDataRingBufferHead) + DEV_ARGS_SIZE;
        (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
        int threadNum = static_cast<int>(devProg->devArgs.nrAicpu);
        threadNum = (devProg->devArgs.enableCtrl == 1) ? threadNum : threadNum + 1;
        for (int i = 0; i < threadNum; i++) {
            aicpus[i] = std::thread([&]() {
                int tidx = idx++;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tidx, &cpuset);
                std::string name = "aicput" + std::to_string(tidx);
                pthread_setname_np(pthread_self(), name.c_str());
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                if ((devProg->devArgs.enableCtrl == 0) && (uint32_t)tidx == devProg->devArgs.scheCpuNum) {
                    (void)PyptoKernelCtrlServer(kArgs);
                } else {
                    (void)DynTileFwkBackendKernelServer(kArgs);
                }
            });
        }

        for (int i = 0; i < threadNum; i++) {
            if (aicpus[i].joinable()) {
                aicpus[i].join();
            }
        }
    }

    void InitKernelInOuts(DeviceKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputTensors,
        const std::vector<RawTensorDataPtr> &outputTensors, bool isTest) {
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        std::tie(inputList, outputList) = BuildInputOutputFromHost(MemoryHelper(isTest), inputTensors, outputTensors);
        DeviceInitKernelInOuts(MemoryHelper(isTest), kArgs, inputList, outputList, {});
        SIMULATION_LOGI("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
    }

private:
    Function *function_;
    DeviceLauncherConfig config_;
    std::shared_ptr<CostModel::DynPvModel> pv_;
    std::shared_ptr<CostModel::AiCoreModel> model_;
}; // CostModelLauncher
} // npu::tile_fwk::dynamic

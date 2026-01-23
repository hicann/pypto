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
 * \file test_cost_model.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <thread>
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"
#include "machine/runtime/device_runner.h"
#include "machine/device/dynamic/device_common.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "tilefwk/core_func_data.h"
#include "runtime.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace CostModel;

extern "C" int DynTileFwkBackendKernelServer(void *targ);

struct MemoryH {
    MemoryH(bool isTest) : isTest_(isTest) {}

    uint8_t *CopyToDev(uint8_t *data, uint64_t size) {
        uint8_t *devPtr = AllocDev(size);
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

    template <typename T>
    T *CopyToDev(std::vector<T> data) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size());
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    uint8_t *AllocZero(uint64_t size) {
        uint8_t *devPtr = AllocDev(size);
        if (isTest_)
            memset(devPtr, 0, size);
        else
            rtMemset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t *AllocDev(size_t size) {
        uint8_t *devPtr = nullptr;
        if (isTest_)
            devPtr = (uint8_t *)malloc(size);
        else
            machine::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }

    void CopyFromDev(RawTensorData &t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    bool isTest_{true};
};

class AiCorePvModelImpl : public AiCoreModel
{
private:
    std::shared_ptr<DynPvModel> pv_;
    std::unordered_map<int, uint64_t> funcdata_;
    std::mutex mtx_;

public:
    explicit AiCorePvModelImpl(std::shared_ptr<DynPvModel> pv) : pv_(pv) {
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

class CostModelDynFuncRunner {
public:
    CostModelDynFuncRunner(Function *func): func_(func), devProg_(func->GetDyndevAttribute()->devProgBinary) {
        pv_ = CostModel::PvModelFactory::CreateDyn();
        model_ = std::make_shared<AiCorePvModelImpl>(pv_);
    }

    static void Run(Function* func,
        const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            auto runner = CostModelDynFuncRunner(func);
            runner.RunModel(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void Run(Function* func) {
        auto runner = CostModelDynFuncRunner(func);
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        runner.RunModel(inputs, outputs);
    }

private:

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            auto funcop = func_->GetDyndevAttribute();
            KernelLaunchPrecheck(funcop);
            pv_->Codegen(func_);

            for (int i = 0; i < 1; i++) {
                DeviceKernelArgs kArgs = BuildKernelArgs(inputs, outputs);
                std::cout << "!!! Run CostModel " << i << "\n";
                RunTestMode(&kArgs);
            }
    }

    bool HasInplaceArgs() {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        return devProg->outputInplaceSlotList.size() != 0;
    }

    void AssignMetaAddr(DevAscendProgram *devProg, MemoryH &h) {
        uint64_t generalSize = devProg->memBudget.metadata.general;
        uint64_t stitchPoolSize = devProg->memBudget.metadata.stitchPool;
        size_t shmSize = DEVICE_SHM_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum +
            generalSize + stitchPoolSize;
        uint64_t shmAddr = (uint64_t)h.AllocDev(shmSize);
        devProg->devArgs.startArgsAddr = shmAddr;
        shmAddr += DEV_ARGS_SIZE;
        devProg->devArgs.taskCtrl = shmAddr;
        shmAddr += DEVICE_TASK_CTRL_SIZE;
        devProg->devArgs.taskQueue = shmAddr;
        shmAddr += DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        devProg->devArgs.generalAddr = shmAddr;
        shmAddr += generalSize;
        devProg->devArgs.stitchPoolAddr = shmAddr;
        return;
    }

    void InitTilingData(DeviceKernelArgs *kArgs, bool isTest) {
        MemoryH h{isTest};
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t *>(devProg_.data()));
        devProg->devArgs.nrAic = 25;
        devProg->devArgs.nrAiv = 50;
        devProg->devArgs.nrAicpu = 6;
        devProg->devArgs.nrValidAic = 24;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->workspaceSize = devProg->memBudget.Total();
        std::cout << devProg->workspaceSize << std::endl;
        devProg->l2CacheOffset = machine::GetRA()->GetL2Offset();
        devProg->l2CacheOffset = machine::GetRA()->GetL2Offset();
        AssignMetaAddr(devProg, h);
        kArgs->workspace = (int64_t *)h.AllocDev(devProg->workspaceSize);
        kArgs->cfgdata = (int64_t *)h.CopyToDev(devProg_);
        kArgs->machineConfig = devProg->devArgs.machineConfig;
        kArgs->toSubMachineConfig = devProg->devArgs.toSubMachineConfig;
        return;
    }

    void RunTestMode(DeviceKernelArgs *kArgs) {
        (void) kArgs;
        InitTilingData(kArgs, true);
        constexpr int threadNum = 6;
        std::thread aicpus[threadNum];
        std::atomic<int> idx{0};
        for (int i = 0; i < threadNum; i++) {
            aicpus[i] = std::thread([&]() {
                int tidx = idx++;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tidx, &cpuset);
                constexpr int nameLen = 64;
                char name[nameLen];
                sprintf_s(name, sizeof(name), "aicput%d", tidx);
                pthread_setname_np(pthread_self(), name);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                auto rc = DynTileFwkBackendKernelServer(kArgs);
                EXPECT_EQ(rc, 0);
            });
        }

        for (int i = 0; i < threadNum; i++) {
            aicpus[i].join();
        }
    }

    void CopyFromDev(const std::vector<RawTensorDataPtr> &outputs) {
        for (auto &output : outputs) {
            if (output)
                pv_->CopyFromDev(output->data(), output->GetDevPtr(), output->size());
        }
    }

    DeviceKernelArgs BuildKernelArgs(const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs) {
        DeviceKernelArgs kArgs;

        auto buildInouts = [&](auto &tensorList) {
            std::vector<DevTensorData> geTensors;
            for (auto &t : tensorList) {
                if (t) {
                    auto addrs = pv_->CopyTensorToDev((uint8_t*)t->data(), t->size());
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create((uint64_t)addrs, t->GetShape()));
                } else {
                    std::vector<int> shape;
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create(0UL, shape));
                }
            }
            auto outs = DevAscendTensorDataCreator::Encode(geTensors);
            return (int64_t*)pv_->CopyToDev((uint8_t *)outs.data(), outs.size()*sizeof(int64_t));
        };

        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        devProg->devArgs.nrAic = 25;
        devProg->devArgs.nrAiv = 50;
        devProg->devArgs.nrAicpu = 6;
        devProg->devArgs.nrValidAic = 24;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->devArgs.startArgsAddr = (uint64_t)pv_->AllocWorkspaceDev(DEV_ARGS_SIZE);
        for (auto &input: inputs) {
            if (input)
                input->SetDevPtr(nullptr);
        }
        for (auto &output: outputs) {
            if (output)
                output->SetDevPtr(nullptr);
        }

        kArgs.inputs = buildInouts(inputs);
        kArgs.outputs = buildInouts(outputs);
        kArgs.workspace = (int64_t *)pv_->AllocWorkspaceDev(devProg->memBudget.Total());
        kArgs.cfgdata = (int64_t *)pv_->CopyToDev(devProg_.data(), devProg_.size());
        kArgs.machineConfig  = devProg->devArgs.machineConfig;
        kArgs.aicoreModel = model_.get();
        return kArgs;
    }

    void KernelLaunchPrecheck(std::shared_ptr<DyndevFunctionAttribute> funcop) {
        auto checkInouts = [&](std::vector<std::reference_wrapper<const Tensor>> &tensorList,
                               const std::vector<RawTensorDataPtr> &dataList) {
            for (size_t i = 0; i < tensorList.size(); i++) {
                auto &t = tensorList[i].get();
                auto &d = dataList[i];
                if (d) {
                    EXPECT_EQ(t.GetDataType(), d->GetDataType());
                    EXPECT_EQ(t.GetShape(), d->GetShape());
                }
            }
        };

        checkInouts(funcop->startArgsInputTensorList, ProgramData::GetInstance().GetInputDataList());
        checkInouts(funcop->startArgsOutputTensorList, ProgramData::GetInstance().GetOutputDataList());
    }

private:
    Function *func_;
    const std::vector<uint8_t> &devProg_;
    std::shared_ptr<DynPvModel> pv_;
    std::shared_ptr<AiCoreModel> model_;
};

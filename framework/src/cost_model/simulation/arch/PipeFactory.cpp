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
 * \file PipeFactory.cpp
 * \brief
 */

#include "PipeFactory.h"

#include <dlfcn.h>

#include "TileAllocPipeImpl.h"
#include "CallPipeImpl.h"
#include "PipeSimulatorFast.h"
#include "cost_model/simulation/arch/A2A3/L2CacheImplA2A3.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"

namespace CostModel
{
    UnifiedPipeMachinePtr CreatePipeSimulator(const std::string& simulator) {
        std::string soPath = "libtile_fwk_simulation_ca.so";
        void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("can not load library: " + std::string(dlerror()));
        }

        // 获取工厂函数符号
        using CreateFunc = UnifiedPipeMachinePtr(*)();
        std::string funcName = "CreatePipeSimulator" + simulator;
        auto createFunc = (CreateFunc)(dlsym(handle, funcName.c_str()));
        if (!createFunc) {
            dlclose(handle);
            throw std::runtime_error("can not find the factory func: " + std::string(dlerror()));
        }

        // 创建对象并返回
        return createFunc();
    }

    UnifiedPipeMachinePtr PipeFactory::Create(CorePipeType pipeType, std::string archType, int accLevel)
    {
        if (IsTileAlloc(pipeType)) {
            return CreateTileAllocPipeImpl();
        } else if (pipeType == CorePipeType::PIPE_CALL) {
            return CreateCallPipeImpl();
        } else if (pipeType == CorePipeType::PIPE_MTE_IN || pipeType == CorePipeType::PIPE_MTE1 ||
                   pipeType == CorePipeType::PIPE_MTE_OUT) {
            if (archType == "A2A3") {
                if (accLevel == 1) {
                    return CreatePipeSimulatorFast<PostSimulatorA2A3>();
                }
                else {
                    return CreatePipeSimulator("SimulatorA2A3");
                }
            } else {
                throw std::invalid_argument("unknown arch type " + archType);
            }
        } else if (pipeType == CorePipeType::PIPE_VECTOR_ALU || pipeType == CorePipeType::PIPE_S) {
            if (archType == "A2A3") {
                if (accLevel == 1) {
                    return CreatePipeSimulatorFast<PostSimulatorA2A3>();
                }
                else {
                    return CreatePipeSimulator("SimulatorA2A3");
                }
            } else {
                throw std::invalid_argument("unknown arch type " + archType);
            }
        } else if (pipeType == CorePipeType::PIPE_CUBE) {
            if (archType == "A2A3") {
                if (accLevel == 1) {
                    return CreatePipeSimulatorFast<PostSimulatorA2A3>();
                }
                else {
                    return CreatePipeSimulator("SimulatorA2A3");
                }
            } else {
                throw std::invalid_argument("unknown arch type " + archType);
            }
        } else {
            throw std::invalid_argument("unknown pipe type " + CorePipeName(pipeType));
        }
        return nullptr;
    }

    std::unique_ptr<CacheMachineImpl> PipeFactory::CreateCache(CostModel::CacheType type, std::string archType) {
        if (type == CacheType::L2CACHE) {
            if (archType == "A2A3") {
                return std::make_unique<L2CacheImplA2A3>();
            } else {
                throw std::invalid_argument("unknown arch type " + archType);
            }
        } else {
            throw std::invalid_argument("unknown pipe type " + CacheName(type));
        }
    }
}

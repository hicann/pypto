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
 * \file PvModelFactory.cpp
 * \brief
 */

#include <dlfcn.h>
#include "PvModelFactory.h"

namespace CostModel
{
    std::shared_ptr<PvModel> PvModelFactory::Create()
    {
        std::string arch = "A2A3";
        std::string soPath = "libtile_fwk_simulation_pv.so";
        void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("can not load library");
        }

        // 获取工厂函数符号
        using CreateFunc = std::shared_ptr<PvModel>(*)();
        std::string funcName = "CreatePvModelImpl" + arch;
        auto createFunc = (CreateFunc)(dlsym(handle, funcName.c_str()));

        // 创建对象并返回
        return createFunc();
    }

    std::shared_ptr<DynPvModel> PvModelFactory::CreateDyn()
    {
        std::string soPath = "libtile_fwk_simulation_pv.so";
        void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("can not load library: ");
        }

        // 获取工厂函数符号
        using CreateFunc = std::shared_ptr<DynPvModel>(*)();
        std::string funcName = "CreateDynPvModelImpl";
        auto createFunc = (CreateFunc)(dlsym(handle, funcName.c_str()));

        // 创建对象并返回
        return createFunc();
    }
}

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
 * \file PvModel.h
 * \brief
 */

#pragma once

#include <string>
#include "interface/function/function.h"
#include "PvData.h"
#include "tilefwk/core_func_data.h"

namespace CostModel
{
    class PvModel
    {
    public:
        virtual ~PvModel() = default;
        virtual void Submit(npu::tile_fwk::Function *func, PvData *data, int level, std::string dir) = 0;
        virtual void Run(int esgId, int psgId) = 0;
    };

    class DynPvModel 
    {
    public:
        virtual ~DynPvModel() = default;
        virtual void Codegen(npu::tile_fwk::Function *func) = 0;
        virtual void InitPv() = 0;
        virtual uint8_t* AllocWorkspaceDev(uint64_t size) = 0;
        virtual uint8_t* CopyToDev(const uint8_t *data, uint64_t size) = 0;
        virtual uint8_t* CopyTensorToDev(const uint8_t *data, uint64_t size) = 0;
        virtual void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) = 0;
        virtual void Run(npu::tile_fwk::DynFuncData *funcdata, int coreId, int funcId, int taskId) = 0;
        virtual uint64_t *GetDataHostPtr(int index) = 0;
        virtual int GetOutIndex(int index, int out_size) = 0;
    };
} // namespace CostModel

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
 * \file test_dev_encode_compute_determinism.cpp
 * \brief UT for encode compute determinism: atomic_add write type.
 */

#include <gtest/gtest.h>
#include <vector>

#define private public
#define protected public

#include "machine/utils/dynamic/dev_encode_program.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

constexpr int kOutcastWrwrTile = 32;

void SetupEncodeDeterminismTest()
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    ConfigManagerNg::SetGlobalConfig("compute_determinism_level", static_cast<int64_t>(0));
    TileShape::Current().SetVecTile(kOutcastWrwrTile, kOutcastWrwrTile);
    TileShape::Current().SetCubeTile({kOutcastWrwrTile, kOutcastWrwrTile}, {kOutcastWrwrTile, kOutcastWrwrTile},
                                     {kOutcastWrwrTile, kOutcastWrwrTile});
}

DevAscendProgram* EncodeAndGetDevProg()
{
    Function* func = Program::GetInstance().GetLastFunction();
    EXPECT_NE(func, nullptr);
    if (func == nullptr) {
        return nullptr;
    }
    auto dynAttr = func->GetDyndevAttribute();
    EXPECT_NE(dynAttr, nullptr);
    if (dynAttr == nullptr) {
        return nullptr;
    }
    EXPECT_FALSE(dynAttr->devProgBinary.empty());
    if (dynAttr->devProgBinary.empty()) {
        return nullptr;
    }
    auto* devProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    if (devProg == nullptr) {
        return nullptr;
    }
    // After encode, RelocProgram(this, 0) stores relative offsets; restore before At()/list access.
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg));
    return devProg;
}

void BuildTwoTileAtomicAddKernel()
{
    Tensor a(DT_FP32, {kOutcastWrwrTile, kOutcastWrwrTile}, "a");
    Tensor b(DT_FP32, {kOutcastWrwrTile, kOutcastWrwrTile}, "b");
    Tensor out(DT_FP32, {2 * kOutcastWrwrTile, kOutcastWrwrTile}, "out");
    FUNCTION("atomic_add_compute_determinism", {a, b}, {out})
    {
        LOOP("loop_atomic", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            AtomicRMW(Add(a, b), {0, 0}, out, AtomicRMWMode::ADD);
            AtomicRMW(Add(a, b), {kOutcastWrwrTile, 0}, out, AtomicRMWMode::ADD);
        }
    }
}

bool CollectOutcastWriteTypes(DevAscendProgram* devProg, std::vector<CellMatchOpType>& writeTypes)
{
    writeTypes.clear();
    if (devProg == nullptr) {
        return false;
    }
    for (int fi = 0; fi < static_cast<int>(devProg->GetFunctionSize()); ++fi) {
        DevAscendFunction* func = devProg->GetFunction(fi);
        if (func == nullptr) {
            continue;
        }
        for (size_t oi = 0; oi < func->GetOutcastSize(); ++oi) {
            auto& outcast = func->GetOutcast(static_cast<int>(oi));
            std::vector<CellMatchOpType> cur;
            for (size_t k = 0; k < outcast.producerConsumerList.size(); ++k) {
                const auto& use = func->At(outcast.producerConsumerList, k);
                if (use.opType != CellMatchOpType::READ) {
                    cur.push_back(use.opType);
                }
            }
            if (cur.size() >= 2) {
                writeTypes.swap(cur);
                return true;
            }
        }
    }
    return false;
}

} // namespace

TEST(DevEncodeComputeDeterminismTest, AtomicAdd_ComputeDeterminismOff_AtomicWrite)
{
    SetupEncodeDeterminismTest();
    ConfigManagerNg::SetGlobalConfig("compute_determinism_level", static_cast<int64_t>(0));
    BuildTwoTileAtomicAddKernel();

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);

    std::vector<CellMatchOpType> writeTypes;
    ASSERT_TRUE(CollectOutcastWriteTypes(devProg, writeTypes));
    for (auto writeType : writeTypes) {
        EXPECT_EQ(writeType, CellMatchOpType::ATOMIC_WRITE);
    }
}

TEST(DevEncodeComputeDeterminismTest, AtomicAdd_ComputeDeterminismOn_NormalWrite)
{
    SetupEncodeDeterminismTest();
    ConfigManagerNg::SetGlobalConfig("compute_determinism_level", static_cast<int64_t>(1));
    BuildTwoTileAtomicAddKernel();

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);

    std::vector<CellMatchOpType> writeTypes;
    ASSERT_TRUE(CollectOutcastWriteTypes(devProg, writeTypes));
    for (auto writeType : writeTypes) {
        EXPECT_EQ(writeType, CellMatchOpType::NORMAL_WRITE);
    }
}

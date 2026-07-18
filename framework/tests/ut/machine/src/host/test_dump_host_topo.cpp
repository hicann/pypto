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
 * \file test_dump_host_topo.cpp
 * \brief UT for machine/host/dump_host_topo.cpp
 */

#include <gtest/gtest.h>
#include "machine/host/dump_host_topo.h"
#include "interface/configs/config_manager.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::topo_dump;

TEST(DumpHostTopoTest, WritersDisabledWhenDumpOff)
{
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));

    StaticTopoCsvWriter staticWriter;
    EXPECT_FALSE(staticWriter.Enabled());

    SlotCellTableCsvWriter slotWriterTrue(true);
    EXPECT_FALSE(slotWriterTrue.Enabled());

    SlotCellTableCsvWriter slotWriterFalse(false);
    EXPECT_FALSE(slotWriterFalse.Enabled());
}

TEST(DumpHostTopoTest, DumpSlotMappingAndWriteOps_NoCrashWhenDisabled)
{
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));

    TensorSlotManager slotManager;
    std::unordered_map<int, int> slotIdxMapping;
    IncastOutcastLink inoutLink;
    DumpSlotMapping(slotManager, slotIdxMapping, inoutLink);

    StaticTopoCsvWriter staticWriter;
    dynamic::DevAscendFunction* func = nullptr;
    staticWriter.WriteFunction(0, *func);

    SlotCellTableCsvWriter slotWriter(true);
    dynamic::DevCellMatchTableDesc desc;
    slotWriter.WritePartial(0, desc, 0);
    slotWriter.WriteFullCover(0, 0, 0, desc);

    SUCCEED();
}

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
 * \file PvModelConfig.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include "PvModelConfig.h"

namespace CostModel {
void PvModelSystemConfigA2A3::Dump(std::string path) {
    std::ofstream outFile(path);

    // 检查文件是否成功打开
    if (!outFile.is_open()) {
        return;
    }

    std::string config = R"!!!(
title = "PV Config"
[Project]
    name = "A2A3"

[LOG]
    disable_list            = [ "log" ]
    enable_list             = [ "instr_log", "instr_popped_log" ]
    # trace: 0, debug: 1, info: 2, warn: 3, error: 4, critical: 5, off: 6
    file_print_level        = 1
    screen_print_level      = 3
    flush_level             = 1
    rotating_file_size      = 134217728     # 0x8000000     # ~130MB
    rotating_file_number    = 2
    core_enable_mask        = ["0xffffffff"]

[ARCH]
    cube_core_num = 1
    vec_core_num = 2
    inorder_acc = 1
    wait_flag_dev_en = 0
    max_sim_time = 30000000

[BIU]
    atomic_switch = 1
    sc_mte_pcie_win_size = 18
    sc_mte_pcie_win_base = 0

[UB_Buffer]
    total_size = 196608
    wrap_en = 1

[L0A_Buf]
    total_size = 65536
    wrap_en = 0

[L0B_Buf]
    total_size = 65536
    wrap_en = 0

[L0A_Wino_Buf]
    total_size = 65536
    wrap_en = 0

[L0B_Wino_Buf]
    total_size = 65536
    wrap_en = 0

[L0C_Buf]
    total_size = 131072
    wrap_en = 1

[L1_Buf]
    total_size = 524288
    wrap_en = 0

[FIX_Buf]
    total_size = 6144
    wrap_en = 0

[BT_Buf]
    total_size = 1024
    wrap_en = 0

[SCALAR_BUF]
    total_size              = 16384
    wrap_en                 = 0
    start_address           = 262144
    sys_va_base_config      = 1           # 0: config by model spr    1: config by spec
    sys_va_base             = 0           # sys va base adress
    stack_phy_base_config   = 1           # 0: config by model spr    1: config by spec
    stack_phy_base          = 34603008    # (stack_va_base_addr[48:25] == sys_va_addr[48:25]) == > stack in ub  0x2100000

[SMASK]
    total_size              = 256
    wrap_en                 = 0
)!!!";

    outFile << config;

    outFile.close();
    return;
}

void PvModelCaseConfigBase::SetTitle(std::string title)
{
    title_ = title;
}

void PvModelCaseConfigBase::SetCoreType(uint64_t coreType)
{
    subcoreId_ = coreType;
}

void PvModelCaseConfigBase::SetBin(uint64_t addr, std::string path) {
    binAddr_ = addr;
    binPath_ = path;
}

void PvModelCaseConfigBase::AddInputArg(uint64_t addr, uint64_t size, std::string path) {
    inputArgs_.emplace_back(std::make_tuple(addr, size, path));
}

void PvModelCaseConfigBase::AddOutputArg(uint64_t addr, uint64_t size, std::string path) {
    outputArgs_.emplace_back(std::make_tuple(addr, size, path));
}

void PvModelCaseConfigA2A3::Dump(std::string path) {
    std::fstream file(path, std::ios::out);

    if (!file.is_open()) {
        std::cerr << "[PVMODEL]open config file error: " << path << std::endl;
        return;
    }

    file << "title = \"" << title_ << "\"" << std::endl;
    file << "log_open_value = 0xffffffff" << std::endl;
    file << "path = \"./\"" << std::endl;
    file << "hbm_para_addr = 0xffff8000" << std::endl;
    file << "chip_version = 6" << std::endl;
    file << "subcore_id = " << subcoreId_ << std::endl;
    file << "block_idx = 0" << std::endl;
    file << std::endl;

    file << "[BIN]" << std::endl;
    file << "name = \"" << binPath_ << "\"" << std::endl;
    file << "addr = " << "0x" << std::hex << binAddr_ << std::endl;
    file << std::endl;

    constexpr int pathIdx = 2;
    constexpr int addrIdx = 0;
    constexpr int sizeIdx = 1;
    int paraOffset = 0;
    for (const auto &arg : inputArgs_) {
        file << "[[input_para_array]]" << std::endl;
        file << "name = \"" << std::get<pathIdx>(arg) << "\"" << std::endl;
        file << "addr = " << "0x" << std::hex << std::get<addrIdx>(arg) << std::endl;
        file << "size = " << "0x" << std::hex << std::get<sizeIdx>(arg) << std::endl;
        file << "valid = 1" << std::endl;
        file << "para_offset = " << std::dec << paraOffset << std::endl;
        paraOffset++;
        file << std::endl;
    }

    for (const auto &arg : outputArgs_) {
        file << "[[output_para_array]]" << std::endl;
        file << "name = \"" << std::get<pathIdx>(arg) << "\"" << std::endl;
        file << "addr = " << "0x" << std::hex << std::get<addrIdx>(arg) << std::endl;
        file << "size = " << "0x" << std::hex << std::get<sizeIdx>(arg) << std::endl;
        file << "valid = 1" << std::endl;
        file << "para_offset = " << std::dec << paraOffset << std::endl;
        paraOffset++;
        file << std::endl;
    }

    file.close();
}
} // namespace CostModel

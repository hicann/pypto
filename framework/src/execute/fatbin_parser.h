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
 * \file fatbin_parser.h
 * \brief
 */

#ifndef FATBIN_PARSER_H_
#define FATBIN_PARSER_H_
#include <map>
#include <vector>
#include <string>
#include "machine/dump/kernel_dump_utils.h"

namespace npu::tile_fwk {
class FatbinParser {
 public:
    static bool ParseFatbin(const std::string &bin_file_path, const uint64_t &config_key, size_t &subkernl_index,
                            std::vector<uint8_t> &op_binary_bin, std::vector<uint8_t> &kernel_bin);

 private:
    static bool MatchSubkernel(const std::vector<char> &fatbin, const uint64_t &config_key,
                               size_t &subkernl_index, std::vector<uint8_t> &subkernel);

    static bool ParseSubkernel(const std::vector<uint8_t> &subkernel, std::vector<uint8_t> &op_binary_bin,
                               std::vector<uint8_t> &kernel_bin);
};
}
#endif
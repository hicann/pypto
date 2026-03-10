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
 * \file platform_parser.h
 * \brief
 */
#ifndef PLATFORM_PARSER_H_
#define PLATFORM_PARSER_H_

#include <map>
#include <set>
#include <vector>
#include <fstream>
#include <unordered_map>
#include "tilefwk/file.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu {
namespace tile_fwk {
class INIParser : public PlatformParser {
public:
    INIParser() = default;
    ~INIParser() = default;
    bool Initialize(const std::string &iniFilePath); 
    bool GetStringVal(const std::string& column, const std::string& key, std::string& val) const override;
private:
    bool ReadINIFile(const std::string& filepath);
    std::map<std::string, std::map<std::string, std::string>> data_;
};
} // namespace tile_fwk
} // namepsace npu 
#endif
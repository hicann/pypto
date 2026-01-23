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
 * \file ini_parser.h
 * \brief
 */
#ifndef INI_PARSER_H_
#define INI_PARSER_H_
#include <fstream>
#include "interface/utils/common.h"
namespace npu {
namespace tile_fwk {

class INIParser {
  public:
    INIParser() = default;
    ~INIParser() = default;
    Status Initialize(const std::string &iniFilePath); 
    Status GetStringVal(const std::string& column, const std::string& key, std::string& val);
    Status GetSizeVal(const std::string& column, const std::string& key, size_t& val);

    Status GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion);
    Status GetCoreVersion(std::unordered_map<std::string, std::string>& curVersion);
    Status GetDataPath(std::vector<std::vector<std::string>>& dataPath);
  private:
    Status ReadINIFile(const std::string& filepath);
    bool FilterCCECVersion(const std::string& key, std::string &coreType);
    bool FilterDirections(const std::string& value, std::string &part);
    bool FilterDataPath(const std::string& part, std::string &from, std::string &to);

    std::map<std::string, std::map<std::string, std::string>> data_;
};
} // namespace tile_fwk
} // namepsace npu 
#endif
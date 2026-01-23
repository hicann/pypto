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
 * \file config.h
 * \brief
 */

#pragma once
#include <iostream>
#include <variant>
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager_ng.h"


namespace npu::tile_fwk {
    
/* Rundata KEYS */
constexpr const char *KEY_RUNTYPE = "runtype";
constexpr const char *KEY_PTO_CONFIG_FILE = "pto_config_file";
constexpr const char *KEY_COMPUTE_GRAPH_PATH = "compute_graph_path";
constexpr const char *KEY_SWIM_GRAPH_PATH = "swim_graph_path";
constexpr const char *KEY_AICPU_PERF_GRAPH_PATH = "aicpu_perf_path";
constexpr const char *KEY_FLOW_VERIFY_PATH = "flow_verify_path";
constexpr const char *KEY_PROGRAM_PATH = "program_file";

struct ConfigStorage;

struct PrintOptions {
    int edgeItems;
    int precision;
    int threshold;
    int linewidth;
};

struct SemanticLabel {
    std::string label;
    std::string filename;
    int lineno;

    SemanticLabel(const std::string &tlabel, const char *tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno) {}
    SemanticLabel(const std::string &tlabel, const std::string &tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno) {}
};

namespace config {
FunctionType GetFunctionType();

std::shared_ptr<SemanticLabel> GetSemanticLabel();
void SetSemanticLabel(std::shared_ptr<SemanticLabel> label);


PrintOptions &GetPrintOptions();

void SetRunDataOption(const std::string &key, const std::string &value);

using ValueType = std::variant<bool, int64_t, std::string, std::vector<int64_t>,
                               std::vector<std::string>, std::map<int64_t, int64_t>>;

} // namespace config
} // namespace npu::tile_fwk

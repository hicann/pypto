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

constexpr const char *SG_PARALLEL_NUM = "pg_parallel_lower_bound";
constexpr const char *SG_PG_UPPER_BOUND = "pg_upper_bound";
constexpr const char *SG_PG_LOWER_BOUND = "pg_lower_bound";
constexpr const char *SG_SET_SCOPE = "sg_set_scope";
constexpr const char *CUBE_L1_REUSE_MODE = "cube_l1_reuse_mode";
constexpr const char *CUBE_L1_REUSE_SETTING = "cube_l1_reuse_setting";
constexpr const char *CUBE_NBUFFER_MODE = "cube_nbuffer_mode";
constexpr const char *CUBE_NBUFFER_SETTING = "cube_nbuffer_setting";
constexpr const char *MG_COPYIN_UPPER_BOUND = "mg_copyin_upper_bound";
constexpr const char *OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method";
constexpr const char *VEC_NBUFFER_MODE = "vec_nbuffer_mode";
constexpr const char *VEC_NBUFFER_SETTING = "vec_nbuffer_setting";
constexpr const char *SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num";
constexpr const char *MG_VEC_PARALLEL_LB = "mg_vec_parallel_lb";
constexpr const char *PG_SKIP_PARTITION = "pg_skip_partition";
constexpr const char *DB_TYPE = "db_type";
constexpr const char *COPYOUT_RESOLVE_COALESCING = "copyout_resolve_coalescing";
constexpr const char *COMPILE_STAGE = "compile_stage";
constexpr const char *SUPPORT_DYNAMIC_ALIGNED = "support_dynamic_aligned";

//runtime
constexpr const char *DEVICE_SCHED_MODE = "device_sched_mode";
constexpr const char *STITCH_FUNCTION_INNER_MEMORY = "stitch_function_inner_memory";
constexpr const char *STITCH_FUNCTION_OUTCAST_MEMORY = "stitch_function_outcast_memory";
constexpr const char *STITCH_FUNCTION_NUM_INITIAL = "stitch_function_num_initial";
constexpr const char *STITCH_FUNCTION_NUM_STEP = "stitch_function_num_step";
constexpr const char *STITCH_FUNCTION_SIZE = "stitch_function_size";
constexpr const char *STITCH_CFGCACHE_SIZE = "stitch_cfgcache_size";
constexpr const char *CFG_RUN_MODE = "run_mode";
const int64_t CFG_RUN_MODE_NPU = 0;
const int64_t CFG_RUN_MODE_SIM = 1;

//debug
constexpr const char *CFG_COMPILE_DBEUG_MODE = "compile_debug_mode";
constexpr const char *CFG_RUNTIME_DBEUG_MODE = "runtime_debug_mode";
const int64_t CFG_DEBUG_NONE = 0;
const int64_t CFG_DEBUG_ALL = 1;
const int64_t CFG_DEBUG_NO_DEVICE_TENSOR_DEPEND = 2;

/* Rundata KEYS */
constexpr const char *KEY_RUNTYPE = "runtype";
constexpr const char *KEY_PTO_CONFIG_FILE = "pto_config_file";
constexpr const char *KEY_COMPUTE_GRAPH_PATH = "compute_graph_path";
constexpr const char *KEY_SWIM_GRAPH_PATH = "swim_graph_path";
constexpr const char *KEY_FLOW_VERIFY_PATH = "flow_verify_path";
constexpr const char *KEY_PROGRAM_PATH = "program_file";


/* flow virifer tools KEYs */
const std::string KEY_ENABLE_PASS_VERIFY = "enable_pass_verify";
const std::string KEY_PASS_VERIFY_SAVE_TENSOR = "pass_verify_save_tensor";
const std::string KEY_PASS_VERIFY_SAVE_TENSOR_DIR = "pass_verify_save_tensor_dir";
const std::string KEY_PASS_VERIFY_FILTER = "pass_verify_pass_filter";

constexpr const int ALL_COMPLETE = 0;
constexpr const int GEN_KERNEL_CODE = 1;
constexpr const int HOST_COMPILE_END = 2;
constexpr const int GEN_TENSOR_GRAPH = 3;

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

std::shared_ptr<ConfigScope> Duplicate();
void Restore(std::shared_ptr<ConfigScope> config);

PrintOptions &GetPrintOptions();

void SetRunDataOption(const std::string &key, const std::string &value);

using ValueType = std::variant<bool, int64_t, std::string, std::vector<int64_t>,
                               std::vector<std::string>, std::map<int64_t, int64_t>>;

} // namespace config
} // namespace npu::tile_fwk

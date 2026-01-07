# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

project(PTO_Framework
        LANGUAGES C CXX
        VERSION 1.0.0
)


########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(PTO_FWK_SRC_ROOT         "${CMAKE_CURRENT_SOURCE_DIR}"   REALPATH)
get_filename_component(PTO_FWK_BIN_ROOT         "${CMAKE_CURRENT_BINARY_DIR}"   REALPATH)
get_filename_component(PTO_FWK_BIN_OUTPUT_ROOT  "${PTO_FWK_BIN_ROOT}/output"    REALPATH)  # 二进制 BUILD 阶段输出路径

message(STATUS "PTO_FWK_SRC_ROOT=${PTO_FWK_SRC_ROOT}")
message(STATUS "PTO_FWK_BIN_ROOT=${PTO_FWK_BIN_ROOT}")

# 公开编译开关(build)
option(BUILD_OPEN_PROJECT "Build in open scenario." ON)
option(BUILD_WITH_CANN "Build with CANN." ON)  # Need to install CANN package and set corresponding environment variables.
option(BUILD_WITH_CANN_SUB "Build with CANN Sub package" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_GCOV "Enable GNU Coverage Instrumentation Tool" OFF)

# 公开编译开关(Feature 特性级开关)
option(ENABLE_FEATURE_PYTHON_FRONT_END "Enable Python Front-end" ON)
option(ENABLE_FEATURE_INSTALL_EXPERIMENT "Enable experiment header or binary file install" OFF)
option(ENABLE_TORCH_VERIFIER "Flow verify with libtorch" OFF)

# 公开编译开关(tests)
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL "Auto execute tests executable parallel" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT "Auto execute tests executable parallel timeout" OFF)
option(ENABLE_TESTS_EXECUTE_CHANGED_FILE "Specify tests changed file" OFF)
option(ENABLE_UTEST "Enable UTest" OFF)
option(ENABLE_UTEST_MODULE "Enable UTest Module compile and execute" OFF)
option(ENABLE_STEST_GOLDEN_PATH "Specific STest Golden Path" OFF)
option(ENABLE_STEST_GOLDEN_PATH_CLEAN "Clean STest Golden" OFF)
option(ENABLE_STEST_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id" OFF)
option(ENABLE_STEST_DUMP_JSON "Enable STest Dump Json" OFF)
option(ENABLE_STEST_INTERPRETER_CONFIG "Enable STest Interpreter Config" OFF)
option(ENABLE_STEST_BINARY_CACHE "Enable STest Binary Cache" OFF)
option(ENABLE_STEST "Enable STest" OFF)
option(ENABLE_STEST_GROUP "Enable STest Group" OFF)
option(ENABLE_STEST_DISTRIBUTED "Enable STest(Distributed)" OFF)

# 公开编译开关(stest.tools)
option(ENABLE_STEST_TOOLS_CASE_FILE "Specify STest Tools Case file path" OFF)
option(ENABLE_STEST_TOOLS_INTERCEPT "Enable STest Tools Intercept if has any failed case" OFF)
option(ENABLE_STEST_TOOLS_OUTPUT_CLEAN "Clean STest Tools Output Directory" OFF)
option(ENABLE_STEST_TOOLS_PROF "Enable STest Tools(Profiling)" OFF)
option(ENABLE_STEST_TOOLS_PROF_LEVEL "Specify STest Tools Profiling Level" OFF)
option(ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT "Specify STest Tools Profiling Warn-up cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_TRY_CNT "Specify STest Tools Profiling Try cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_MAX_CNT "Specify STest Tools Profiling Max cnt" OFF)

# 缺省编译配置, 编译函数
find_package(Threads REQUIRED)
include(ExternalProject)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/func.cmake)
include(cmake/config.cmake)
include(cmake/intf.cmake)
include(cmake/third_party/nlohmann_json/nlohmann_json.cmake)
include(cmake/third_party/libboundscheck/libboundscheck.cmake)
include(cmake/third_party/googletest/googletest.cmake)


########################################################################################################################
# 编译流程
########################################################################################################################

add_subdirectory(framework)
add_subdirectory(python)


########################################################################################################################
# 安装流程
########################################################################################################################

if (ENABLE_FEATURE_PYTHON_FRONT_END AND BUILD_OPEN_PROJECT)
    # whl 包名配置
    set(PyPTO_WhlName "pypto")

    # 安装 二进制文件
    set(InstallExport ${PROJECT_NAME}Targets)
    set(InstallTargets
            tile_fwk_simulation_platform
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_compiler
            tile_fwk_runtime
            tile_fwk_runtime_stub
            tile_fwk_simulation
            tile_fwk_simulation_ca
    )
    if (BUILD_WITH_CANN)
        install(TARGETS tile_fwk_server LIBRARY DESTINATION "${PyPTO_WhlName}/lib")
        install(TARGETS pypto_ctrl_server ARCHIVE DESTINATION "${PyPTO_WhlName}/lib")
    endif ()
    # 不判断 c_sec_shared 存在, 不存在时属异常场景, 应在构建阶段发现
    get_target_property(cSecSharedFile c_sec_shared IMPORTED_LOCATION)
    install(FILES ${cSecSharedFile}
            DESTINATION "${PyPTO_WhlName}/lib"
            PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
    if (ENABLE_TORCH_VERIFIER)
        list(APPEND InstallTargets tile_fwk_calculator)
    endif()
    PTO_Fwk_InstallBinaries(
            EXPORT      ${InstallExport}
            WHL_NAME    ${PyPTO_WhlName}
            TARGETS     ${InstallTargets}
    )
    install(TARGETS pypto_impl LIBRARY DESTINATION "${PyPTO_WhlName}" COMPONENT pypto_impl_lib)

    # 安装 配置文件
    install(FILES
                framework/src/interface/configs/tile_fwk_config.json
                framework/src/interface/configs/tile_fwk_types.json
                framework/src/interface/configs/tile_fwk_config_ng.json
                framework/src/interface/configs/tile_fwk_config_schema.json
                framework/src/cost_model/simulation_platform/platform_config/A2A3.ini
            DESTINATION ${PyPTO_WhlName}/lib/configs
    )

    # 安装 脚本文件
    install(FILES
                tools/profiling/draw_swim_lane.py
                framework/src/cost_model/simulation/scripts/print_swim_lane.py
                framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py
                framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py
                tools/profiling/function_json_convert.py
                tools/profiling/parse_pipe_time_trace.py
            DESTINATION ${PyPTO_WhlName}/lib/scripts
    )

    # 安装 头文件
    if (ENABLE_FEATURE_INSTALL_EXPERIMENT)
        install(DIRECTORY framework/include/
                DESTINATION ${PyPTO_WhlName}/include
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
    endif ()
    install(DIRECTORY framework/src/interface/tileop/
            DESTINATION ${PyPTO_WhlName}/lib/include/tileop
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )
    install(DIRECTORY framework/src/cost_model/simulation_ca/mock/
            DESTINATION ${PyPTO_WhlName}/lib/include/mock
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )
    install(DIRECTORY framework/src/interface/machine/device/tilefwk/
            DESTINATION ${PyPTO_WhlName}/lib/include/tilefwk
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )

    # 生成和安装配置文件
    if (ENABLE_FEATURE_INSTALL_EXPERIMENT)
        PTO_Fwk_InstallCMakeConfig(
                EXPORT              ${InstallExport}
                WHL_NAME            ${PyPTO_WhlName}
                CMAKE_PARENT_DIR    ${PyPTO_WhlName}/lib
                TARGETS             ${InstallTargets}
        )
    endif ()
endif ()

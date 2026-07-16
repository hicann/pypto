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

project(tile_fwk_server LANGUAGES C CXX)

########################################################################################################################
# 基础配置
########################################################################################################################
add_subdirectory(stub)

file(GLOB _Device_Utils_Srcs
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_cell_match_dump.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_workspace.cpp
)

add_subdirectory(dynamic)

# 缺省编译配置, 编译函数
if (BUILD_WITH_CANN)
    set(CMAKE_C_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")
endif ()

########################################################################################################################
# 编译流程
########################################################################################################################

file(GLOB_RECURSE _Device_Srcs *.cpp)
file(GLOB_RECURSE _Stub_Srcs stub/*.cpp)
list(REMOVE_ITEM _Device_Srcs ${_Stub_Srcs})

add_library(tile_fwk_server SHARED)
target_sources(tile_fwk_server PRIVATE ${_Device_Srcs} ${_Device_Utils_Srcs})
target_compile_options(tile_fwk_server
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -shared
            -fPIC
            -D__DEVICE__
            -Werror
            -fno-common
            -fstack-protector-strong
)
target_include_directories(tile_fwk_server
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/include
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/toolchain>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
)
target_link_directories(tile_fwk_server
        PRIVATE
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_server
        PRIVATE
            -Wl,--no-as-needed
            c_sec
            slog_stub
            -ldl
            $<$<TARGET_EXISTS:intf_pub>:$<BUILD_INTERFACE:intf_pub>>
)
set_target_properties(tile_fwk_server PROPERTIES OUTPUT_NAME tilefwk_backend_server)

if (NOT ENABLE_FEATURE_PYTHON_FRONT_END)
    install(TARGETS tile_fwk_server LIBRARY DESTINATION lib OPTIONAL)
endif ()

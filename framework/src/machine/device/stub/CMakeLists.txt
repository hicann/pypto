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

if (BUILD_WITH_CANN)
    set(CMAKE_C_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")
endif ()

add_library(slog_stub SHARED)

target_sources(slog_stub
        PRIVATE
            slog_stub.cpp)

target_compile_options(slog_stub
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
)
target_include_directories(slog_stub
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/include
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
        PRIVATE  # 使用 CANN 场景依赖
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
        PRIVATE  # 使用 CANN 整包
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/toolchain>
        PRIVATE  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
)
target_link_directories(slog_stub
        PRIVATE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(slog_stub
        PRIVATE
            -Wl,--no-as-needed
            c_sec
            -ldl
)
set_target_properties(slog_stub PROPERTIES OUTPUT_NAME slog)
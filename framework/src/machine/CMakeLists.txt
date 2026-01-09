# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


if (BUILD_WITH_CANN)
    add_subdirectory(kernel)
    add_subdirectory(device)
endif ()
add_subdirectory(runtime)

file(GLOB_RECURSE _Compiler_Srcs
        host/*.cpp
        compile/*.cpp
        dump/*.cpp
        platform/*.cpp
        utils/*.cpp
        cache_manager/*.cpp)

add_library(tile_fwk_compiler SHARED)
target_sources(tile_fwk_compiler PRIVATE ${_Compiler_Srcs})
target_include_directories(tile_fwk_compiler
        PUBLIC  # 编译 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/inc>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>>
        PUBLIC  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(tile_fwk_compiler
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:PROCESSOR_SUBPATH="${CMAKE_SYSTEM_PROCESSOR}-linux">>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
)
target_link_directories(tile_fwk_compiler
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>>
)
target_link_libraries(tile_fwk_compiler
        PRIVATE
            $<$<BOOL:${ENABLE_UTEST}>:tile_fwk_interface>
            $<$<BOOL:${ENABLE_UTEST}>:tile_fwk_codegen>
            json
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            c_sec
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_compiler)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_compiler)

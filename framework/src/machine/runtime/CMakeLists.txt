# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _Runtime_Srcs *.cpp)
file(GLOB_RECURSE _Device_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/*.cpp)
add_library(tile_fwk_runtime SHARED)
file(GLOB_RECURSE _Simulation_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/simulation/*.cpp)
target_sources(tile_fwk_runtime PRIVATE ${_Runtime_Srcs} ${_Device_Srcs} ${_Simulation_Srcs})
target_include_directories(tile_fwk_runtime
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_compile_definitions(tile_fwk_runtime
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
)
target_link_libraries(tile_fwk_runtime
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_adapter>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_interface>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_compiler>>
            $<BUILD_INTERFACE:json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            c_sec
)

if (BUILD_WITH_CANN)
    add_dependencies(tile_fwk_runtime tile_fwk_server)
endif ()

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_runtime)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_runtime)

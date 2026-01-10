# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


file(GLOB_RECURSE _Ca_Model_Srcs *.cpp)
#PTO_Fwk_Debug_Source_List(NAME "_Ca_Model_Srcs" LIST ${_Ca_Model_Srcs} DETAIL ON)

add_library(tile_fwk_simulation_ca SHARED)
target_sources(tile_fwk_simulation_ca PRIVATE ${_Ca_Model_Srcs})
target_include_directories(tile_fwk_simulation_ca
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_link_libraries(tile_fwk_simulation_ca
        PUBLIC
            $<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_codegen>
            $<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_simulation>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            c_sec
            json
)
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_simulation_ca)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_simulation_ca)

if (BUILD_WITH_CANN AND NOT ENABLE_FEATURE_PYTHON_FRONT_END)
    # 新增CCE前仿二进制
    add_executable(get_cce_simulation)
    target_sources(get_cce_simulation PRIVATE tools/GenSimulation.cpp)
    target_include_directories(get_cce_simulation
            PRIVATE
                ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
    )
    target_link_libraries(get_cce_simulation
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                tile_fwk_simulation_ca
                tile_fwk_interface
                tile_fwk_codegen
                tile_fwk_simulation
                dl
                Threads::Threads
    )
endif ()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_STestNamePrefix}_operator)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_simulation tile_fwk_simulation_ca tile_fwk_simulation_pv
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/print_swim_lane.py         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/draw_swim_lane.py                                                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/function_json_convert.py                                         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMENT "Soft link of ${_Target} dependence scripts file has been created at ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts"
)

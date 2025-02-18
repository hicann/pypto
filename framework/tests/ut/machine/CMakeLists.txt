# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_machine)

file(GLOB_RECURSE _Sources src/*.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/aicore_prof.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/device_machine.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/distributed/comm_wait_flag.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/distributed/shmem_wait_until.cpp)
set(_Include "include")

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${_Include}
        PUBLIC_LINK_LIBRARIES tile_fwk_compiler tile_fwk_runtime tile_fwk_interface dl
        GTEST_FILTER_LIST ${_CaseFilterList}
)
target_compile_definitions(${_Target}
        PRIVATE
            NO_NEON_SUPPORT
)
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stubs/compiler/data/platform_config/Ascend910B1.ini
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/
        COMMENT "Copy ${_Target} dependence config file to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/"
)

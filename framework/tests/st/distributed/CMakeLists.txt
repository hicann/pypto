# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ops)

add_subdirectory(framework)

# 用例可执行程序(编译)
set(_Target ${PTO_Fwk_STestNamePrefix}_distributed)
file(GLOB_RECURSE _Sources src/*.cpp)
set(_PrivateLinkLibraries
        tile_fwk_interface
        tile_fwk_codegen
        tile_fwk_compiler
        tile_fwk_runtime
        tile_fwk_simulation
        ${PTO_Fwk_STestNamePrefix}_utils
        ${PTO_Fwk_STestNamePrefix}_distributed_ops
        ${PTO_Fwk_STestNamePrefix}_distributed_framework
)
PTO_Fwk_GTest_AddExe(
        TARGET                  ${_Target}
        SOURCES                 ${_Sources}
        PRIVATE_LINK_LIBRARIES  ${_PrivateLinkLibraries}
)

# PyPTO, 依赖 CI 任务拆分, 先通过依赖确保上板顺序
if (TARGET ${PTO_Fwk_STestNamePrefix}_python)
    message(STATUS "Add Dependencies ${_Target} -> ${PTO_Fwk_STestNamePrefix}_python")
    add_dependencies(${_Target} ${PTO_Fwk_STestNamePrefix}_python)
endif ()

# 用例可执行程序(执行)
set(_DistributedGTestCaseFilterConfig 
        "TestAllgather/DistributedTest.TestAllgather/0" "4"
        "TestAllreduce/DistributedTest.TestAllreduce/0" "4"
        "TestAllreduce/DistributedTest.TestAllreduce/1" "4"
        "TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/0" "4"
        "TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine/1" "4"
        "TestReducescatter/DistributedTest.TestReducescatter/0" "4"
        "TestReducescatter/DistributedTest.TestReducescatter/1" "4"
        "TestReducescatter/DistributedTest.TestReducescatter/2" "4"
        "TestAllgather_AttnPost_Reducescatter/DistributedTest.TestAllgather_AttnPost_Reducescatter/0" "4"
        "TestAllreduce_Add_Allreduce/DistributedTest.TestAllreduce_Add_Allreduce/0" "4"
)
set(_LdLibrariesExt
        tile_fwk_compiler
        ${PTO_Fwk_STestNamePrefix}_distributed_framework
)
set(_EnvSetupExt
        "TILE_FWK_STEST_GOLDEN_PATH=${ENABLE_STEST_GOLDEN_PATH}"
)
PTO_Fwk_STest_Distributed_RunExe(
       TARGET                  ${_Target}
       LD_LIBRARIES_EXT        ${_LdLibrariesExt}
       ENV_LINES_EXT           ${_EnvSetupExt}
       GTEST_FILTER_CONFIG     ${_DistributedGTestCaseFilterConfig}
       GOLDEN_SCRIPT_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/ops/script
)


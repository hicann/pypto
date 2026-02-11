# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB _Interface_Srcs
        # 前端框架
        schema/*.cpp
        configs/*.cpp
        tensor/*.cpp
        program/*.cpp
        function/*.cpp
        interpreter/*.cpp
        utils/*.cpp
        ir/*.cpp
        ir/builder/*.cpp
        ir/serializer/*.cpp
        plugin/*.cpp

        # Operation
        operation/*.cpp
        operation/vector/*.cpp
        operation/distributed/*.cpp

        # Machine
        cache/*.cpp
        machine/*.cpp
        machine/host/*.cpp
        registry/*.cpp
)
#PTO_Fwk_Debug_Source_List(NAME "_Interface_Srcs" LIST ${_Interface_Srcs} DETAIL OFF)

add_library(tile_fwk_interface SHARED)
target_sources(tile_fwk_interface PRIVATE ${_Interface_Srcs})
target_compile_definitions(tile_fwk_interface
        PRIVATE
            $<$<BOOL:${ENABLE_TESTS}>:SRCPATH="${PTO_FWK_SRC_ROOT}">
            $<$<BOOL:${ENABLE_FEATURE_PYTHON_FRONT_END}>:ENABLE_FEATURE_PYTHON_FRONT_END>
)
target_include_directories(tile_fwk_interface
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(tile_fwk_interface
        PUBLIC
            $<BUILD_INTERFACE:json>
            $<BUILD_INTERFACE:c_sec>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub

            dl
            Threads::Threads
            tile_fwk_simulation_platform
)
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_interface)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_interface)

if (ENABLE_TORCH_VERIFIER)
    add_library(tile_fwk_calculator SHARED)
    target_sources(tile_fwk_calculator
            PRIVATE
                interpreter/calculator/calc_torch.cpp
                interpreter/calculator/fp8_convert.cpp
    )
    target_compile_options(tile_fwk_calculator
            PRIVATE
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:--coverage -fprofile-arcs -ftest-coverage>>
                -D_GLIBCXX_USE_CXX11_ABI=${PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI}
                -D__DEVICE__
            PUBLIC
                -DENABLE_VERIFIER=1
    )
    target_include_directories(tile_fwk_calculator
            SYSTEM PRIVATE
                ${PY3_MOD_TORCH_ROOT_PATH}/include
                ${PY3_MOD_TORCH_ROOT_PATH}/include/torch/csrc/api/include
                ${PTO_FWK_SRC_ROOT}/framework/include
                ${PTO_FWK_SRC_ROOT}/framework/src/
                ${PTO_FWK_SRC_ROOT}/framework/src/interface
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
    )
    target_link_directories(tile_fwk_calculator
            PUBLIC
                ${PY3_MOD_TORCH_ROOT_PATH}/lib
                ${PY3_MOD_TORCH_ROOT_PATH}/../torch.libs
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
    )
    target_link_libraries(tile_fwk_calculator
            PRIVATE
                json
                c_sec
                torch
                tile_fwk_intf_pub
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:gcov>>
    )
    target_link_options(tile_fwk_calculator
            PRIVATE
                -Wl,-no-as-needed,${PY3_MOD_TORCH_ROOT_PATH}/lib/libtorch_cpu.so -Wl,-as-needed
                -Wl,-no-as-needed,${PY3_MOD_TORCH_ROOT_PATH}/lib/libc10.so -Wl,-as-needed
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:-fprofile-arcs -ftest-coverage>>
    )
endif () ## ENABLE_TORCH_VERIFIER

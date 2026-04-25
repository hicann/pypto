# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _Adapter_Srcs
        api/*.cpp
        manager/*.cpp
        stubs/*.cpp)

add_library(tile_fwk_adapter SHARED)
target_sources(tile_fwk_adapter PRIVATE ${_Adapter_Srcs})
target_include_directories(tile_fwk_adapter
        PUBLIC  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(tile_fwk_adapter
        PUBLIC
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_WITH_CANN}>,$<NOT:$<BOOL:${ENABLE_UTEST}>>>:BUILD_WITH_CANN>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_MOBILE}>:BUILD_WITH_CANN_MOBILE>>
)
target_link_directories(tile_fwk_adapter
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:/usr/local/Ascend/driver/lib64/driver>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>>
)
target_link_libraries(tile_fwk_adapter
        PRIVATE
            $<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            c_sec
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_adapter)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_adapter)

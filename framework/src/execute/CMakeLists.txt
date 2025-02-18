# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _Executor_Srcs *.cpp)
add_library(tile_fwk_executor SHARED)
target_sources(tile_fwk_executor PRIVATE ${_Executor_Srcs})
target_include_directories(tile_fwk_executor
        PUBLIC  # 编译 CANN 场景依赖
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/metadef/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/metadef/pkg_inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/metadef/inc/external>
        PUBLIC  # 使用 CANN 场景依赖
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>
        PUBLIC  # 使用 CANN 整包
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/toolchain>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${PTO_FWK_SRC_ROOT}/third_party/metadef/inc>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${PTO_FWK_SRC_ROOT}/third_party/metadef/pkg_inc>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${PTO_FWK_SRC_ROOT}/third_party/metadef/inc/external>
        PUBLIC  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            include
)
target_link_directories(tile_fwk_executor
        PUBLIC
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_executor
        PUBLIC
            graph
            register
            runtime
            c_sec
            json
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            unified_dlog
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_executor)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_executor)

# 安装 二进制文件
install(TARGETS tile_fwk_executor LIBRARY DESTINATION lib OPTIONAL)
# 安装 头文件
install(DIRECTORY include/ DESTINATION lib/tilefwk/)

# 设置包信息
set(CPACK_PACKAGE_NAME "pypto-executor")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION "pypto-executor")
include(CPack)

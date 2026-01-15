# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_utils)
add_library(${_Target} INTERFACE)
target_include_directories(${_Target}
        INTERFACE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../utils>
)
target_compile_definitions(${_Target}
        INTERFACE
            $<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>
)
target_compile_options(${_Target}
        INTERFACE
           -fno-access-control
)
target_link_libraries(${_Target}
        INTERFACE
            intf_pub_cxx17
            tile_fwk_intf_pub
)

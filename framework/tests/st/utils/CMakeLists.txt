# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_STestNamePrefix}_utils)
add_library(${_Target} INTERFACE)
target_include_directories(${_Target}
        INTERFACE
            include
            ${PTO_FWK_SRC_ROOT}/framework/src/machine/runtime
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
        INTERFACE  # 编译 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc/toolchain>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/slog/inc/toolchain>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc/runtime>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc/external>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc>>
        INTERFACE  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
        INTERFACE  # 使用 CANN 整包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/base>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
        INTERFACE  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
)
target_compile_definitions(${_Target}
        INTERFACE
            $<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>
            $<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>
)
target_link_directories(${_Target}
        INTERFACE
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
            $<$<BOOL:${BUILD_WITH_CANN}>:/usr/local/Ascend/driver/lib64/driver>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>
)
target_link_libraries(${_Target}
        INTERFACE
            ${PTO_Fwk_STestNamePrefix}_intf_pub
            $<$<BOOL:${BUILD_WITH_CANN}>:ascendcl>
            $<$<BOOL:${BUILD_WITH_CANN}>:runtime>
            $<$<BOOL:${BUILD_WITH_CANN}>:hccl>
)

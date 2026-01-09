# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

pybind11_add_module(pypto_impl
        MODULE
            pybind11.cpp
            bindings/enum.cpp
            bindings/element.cpp
            bindings/tensor.cpp
            bindings/symbolic_scalar.cpp
            bindings/controller.cpp
            bindings/operation.cpp
            bindings/runtime.cpp
            bindings/cost_model.cpp
            bindings/pass_config.cpp
            bindings/function.cpp
            bindings/ir.cpp
)
target_include_directories(pypto_impl
        PRIVATE  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
        PRIVATE  # 使用 CANN 整包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/base>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
        PRIVATE  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
)
target_compile_definitions(pypto_impl
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
)

# 所依赖的本项目 so 均在 pypto 包 __init__.py 中加载, 此处不再 link
target_link_libraries(pypto_impl
        PRIVATE
            json
            c_sec
            intf_pub_cxx17
            tile_fwk_intf_pub
)
if (TARGET c_sec_shared)
    add_dependencies(pypto_impl c_sec_shared)
endif ()

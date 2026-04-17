# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_WITH_CANN)
    # stub ascendcl
    add_library(ascendcl SHARED)
    target_sources(ascendcl
            PRIVATE
                src/acl_stubs.cpp
    )
    target_link_libraries(ascendcl
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
    )
    target_include_directories(ascendcl
            PRIVATE
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
    )

    # stub runtime
    add_library(runtime SHARED src/runtime_stubs.cpp)
    target_link_libraries(runtime
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                c_sec
    )
    target_include_directories(runtime
            PRIVATE
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
    )

    # stub profiling
    add_library(profapi SHARED src/host_prof_stubs.cpp)
    target_link_libraries(profapi
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                c_sec
    )
    target_include_directories(profapi
            PUBLIC
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
    )

    # stub hccl
    add_library(hccl_stubs SHARED)
    target_sources(hccl_stubs
            PRIVATE
                src/hccl_stubs.cpp
    )
    target_link_libraries(hccl_stubs
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                c_sec
    )
    target_include_directories(hccl_stubs
            PRIVATE
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl>>
 
    )

    # stub dump
    add_library(dump_stubs SHARED)
    target_sources(dump_stubs
            PRIVATE
                src/dump_stubs.cpp
    )
    target_link_libraries(dump_stubs
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                c_sec
    )
    target_include_directories(dump_stubs
            PRIVATE
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
    )

    # 追加 Stubs 相关配置, 需保证:
    # 1. stubs 下相关代码编译至可执行文件内, 保证用例内直调 Stubs 内相关函数走进 Stubs;
    # 2. stubs 下相关 target 创建, 保证业务代码(如 tile_fwk_compiler) 链接正常;
    set(_Target ${PTO_Fwk_UTestNamePrefix}_stubs)

    set(_Sources
            src/acl_stubs.cpp
            src/host_prof_stubs.cpp
            src/prof_stub.cpp
            src/runtime_stubs.cpp
            src/hccl_stubs.cpp
            src/dump_stubs.cpp
    )

    add_library(${_Target} STATIC)
    target_sources(${_Target} PRIVATE ${_Sources})
    target_include_directories(${_Target}
            PUBLIC
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl>>
    )
    target_link_libraries(${_Target}
            PUBLIC
                c_sec
    )
endif()

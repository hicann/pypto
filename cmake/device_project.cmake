# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(add_pypto_device_project component)
    if (NOT TOPLEVEL_PROJECT)
        return()
    endif ()
    if (DEFINED ENABLE_BUILD_DEVICE AND NOT ENABLE_BUILD_DEVICE)
        return()
    endif ()

    set(_DeviceCMakeArgs)
    if (ASCEND_CANN_PACKAGE_PATH)
        list(APPEND _DeviceCMakeArgs "-D" "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
    else ()
        list(APPEND _DeviceCMakeArgs "-D" "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}")
    endif ()
    if (HI_PYTHON)
        list(APPEND _DeviceCMakeArgs "-D" "HI_PYTHON=${HI_PYTHON}")
    endif ()

    set(_DeviceGeneratorArgs)
    set(_DeviceBuildArgs)
    if (CMAKE_GENERATOR STREQUAL "Ninja")
        list(APPEND _DeviceGeneratorArgs CMAKE_GENERATOR "Unix Makefiles")
        list(APPEND _DeviceBuildArgs BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>)
    endif ()

    include(ExternalProject)
    ExternalProject_Add(cann_device
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/cmake/device
        BINARY_DIR ${CMAKE_BINARY_DIR}/device_build
        ${_DeviceGeneratorArgs}
        CMAKE_ARGS
            ${_DeviceCMakeArgs}
            -D TOOLCHAIN_DIR=${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc
            -D CMAKE_TOOLCHAIN_FILE=${CANN_CMAKE_DIR}/toolchain/aarch64-hcc-toolchain.cmake
            -D CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}
            -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -D ENABLE_SIGN=${ENABLE_SIGN}
            -D CUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}
            -D VERSION_INFO=${VERSION_INFO}
            -D ENABLE_OPEN_SRC=TRUE
            -D BUILD_OPEN_PROJECT=TRUE
        ${_DeviceBuildArgs}
        INSTALL_COMMAND ${CMAKE_CPACK_COMMAND}
        BUILD_ALWAYS TRUE
    )
    install(FILES
        ${CMAKE_BINARY_DIR}/device_build/device-${component}.tar.gz
        DESTINATION .
        COMPONENT ${component}
    )
endfunction()

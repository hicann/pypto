# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT BUILD_OPEN_PROJECT)
    return()
endif ()

if (BUILD_WITH_CANN AND DEFINED ENV{LD_LIBRARY_PATH})
    set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
    string(REPLACE ":" ";" LIBRARY_PATHS "${LD_LIBRARY_PATH}")
    if (NOT LIBRARY_PATHS)
        message(FATAL_ERROR "BUILD_WITH_CANN but ENV{LD_LIBRARY_PATH} empty.")
        return()
    endif ()
    find_library(c_sec_LIBRARY
            NAMES c_sec
            PATHS ${LIBRARY_PATHS}
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
    )
    if (NOT c_sec_LIBRARY)
        message(WARNING "Can't find c_sec from ENV{LD_LIBRARY_PATH}=$ENV{LD_LIBRARY_PATH}")
        return()
    endif ()
    get_filename_component(c_sec_LIBRARY "${c_sec_LIBRARY}" REALPATH)
    add_library(c_sec_shared SHARED IMPORTED)
    set_target_properties(c_sec_shared PROPERTIES
            IMPORTED_LOCATION ${c_sec_LIBRARY}
    )
    message(STATUS "Use c_sec from binary, c_sec_shared: ${c_sec_LIBRARY}")
    return()
endif ()

function(TryAdd_c_sec)
    cmake_parse_arguments(
            ARG
            "SKIP_CHECK"
            "PREFIX"
            "DEPENDS"
            ""
            ${ARGN}
    )
    if ((EXISTS "${ARG_PREFIX}/lib/libc_sec.so" AND EXISTS "${ARG_PREFIX}/include/securec.h" AND EXISTS "${ARG_PREFIX}/include/securectype.h") OR ARG_SKIP_CHECK)
        add_library(c_sec_shared SHARED IMPORTED)
        set_target_properties(c_sec_shared PROPERTIES
                IMPORTED_LOCATION ${ARG_PREFIX}/lib/libc_sec.so
        )
        add_library(c_sec INTERFACE)
        set_target_properties(c_sec PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ARG_PREFIX}/include"
                INTERFACE_LINK_LIBRARIES "c_sec_shared"
        )
        if (ARG_DEPENDS)
            add_dependencies(c_sec ${ARG_DEPENDS})
        endif ()
    endif ()
endfunction()

# 免重入
if (TARGET c_sec)
    return()
endif ()

# 异常拦截
if (NOT PYPTO_THIRD_PARTY_PATH)
    # 不再从环境上查找, 因当前部分软件编译过程需要添加 -D_GLIBCXX_USE_CXX11_ABI=0, 可能与环境上已安装软件冲突.
    set(_Msg
            "Failed to get c_sec source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR ${_Msg})
endif ()

set(_TargetVersion "1.1.16")

# 直接查找制品, 若找到则直接退出
get_filename_component(_TargetTarGzFile "${PYPTO_THIRD_PARTY_PATH}/libboundscheck-v${_TargetVersion}.tar.gz" REALPATH)
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
TryAdd_c_sec(PREFIX ${_TargetInstallPrefix})
if (TARGET c_sec)
    message(STATUS "Use c_sec from binary, c_sec_Install_Prefix=${_TargetInstallPrefix}")
    return()
endif ()

# 触发编译
get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/libboundscheck-v${_TargetVersion}" REALPATH)
get_filename_component(_TargetBinaryDir "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/build/libboundscheck-v${_TargetVersion}" REALPATH)
PTO_Fwk_CleanEmptyDir(DIR ${_TargetSourceDir})

set(_ExtArgs)
if (NOT EXISTS ${_TargetSourceDir})
    list(APPEND _ExtArgs
            URL "https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz"
            URL_HASH SHA256=aee8368ef04a42a499edd5bfebce529e7f32dd138bfed383d316e48af4e45d2c
            DOWNLOAD_DIR ${PYPTO_THIRD_PARTY_PATH}
    )
endif ()
ExternalProject_Add(ExternalProject_c_sec   ${_ExtArgs}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/libboundscheck-v${_TargetVersion}
        SOURCE_DIR ${_TargetSourceDir}
        BINARY_DIR ${_TargetBinaryDir}
        INSTALL_DIR ${_TargetInstallPrefix}
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            -S ${CMAKE_CURRENT_LIST_DIR}
            -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            # 编译器相关配置
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            # 具体软件相关变量
            -Dlibboundscheck_SRC_DIR=${_TargetSourceDir}
        BUILD_ALWAYS FALSE
        EXCLUDE_FROM_ALL TRUE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        TLS_VERIFY OFF
        BUILD_BYPRODUCTS
            ${_TargetInstallPrefix}/include/securec.h
            ${_TargetInstallPrefix}/include/securectype.h
            ${_TargetInstallPrefix}/lib/libc_sec.so
)
TryAdd_c_sec(PREFIX ${_TargetInstallPrefix} DEPENDS ExternalProject_c_sec SKIP_CHECK)
message(STATUS "Use c_sec from source: ${_TargetSourceDir}")

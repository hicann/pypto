# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT BUILD_OPEN_PROJECT OR NOT ENABLE_TESTS)
    return()
endif ()

set(_TargetVersion 1.14.0)

# 免重入
if (GTest_FOUND)
    return()
endif ()

# 异常拦截
if (NOT PYPTO_THIRD_PARTY_PATH)
    # 不再从环境上查找, 因当前部分软件编译过程需要添加 -D_GLIBCXX_USE_CXX11_ABI=0, 可能与环境上已安装软件冲突.
    set(_Msg
            "Failed to get GTest source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR ${_Msg})
endif ()

# 直接查找制品, 若找到则直接退出
get_filename_component(_TargetTarGzFile "${PYPTO_THIRD_PARTY_PATH}/googletest-${_TargetVersion}.tar.gz" REALPATH)
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
find_package(GTest ${_TargetVersion} EXACT CONFIG PATHS ${_TargetInstallPrefix} NO_DEFAULT_PATH)
if (NOT GTest_FOUND)
    # 兼容部分镜像直接存放 gtest 安装结果的情况
    get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/gtest" REALPATH)
    find_package(GTest ${_TargetVersion} EXACT CONFIG PATHS ${_TargetSourceDir} NO_DEFAULT_PATH)
endif ()
if (GTest_FOUND)
    message(STATUS "Use GTest from binary, GTest_DIR=${GTest_DIR}")
    return()
endif ()

# 触发编译
get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/googletest-${_TargetVersion}" REALPATH)
get_filename_component(_TargetBinaryDir "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/build/googletest-${_TargetVersion}" REALPATH)
PTO_Fwk_CleanEmptyDir(DIR ${_TargetSourceDir})

set(_ExtArgs)
if (NOT EXISTS ${_TargetSourceDir})
    list(APPEND _ExtArgs
            URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz"
            URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
            DOWNLOAD_DIR ${PYPTO_THIRD_PARTY_PATH}
    )
endif ()

ExternalProject_Add(ExternalProject_GTest   ${_ExtArgs}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/googletest-${_TargetVersion}
        SOURCE_DIR ${_TargetSourceDir}
        BINARY_DIR ${_TargetBinaryDir}
        INSTALL_DIR ${_TargetInstallPrefix}
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -S <SOURCE_DIR>
            -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            # 编译器相关配置
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            # 具体软件相关变量
            -DBUILD_GMOCK=OFF
            -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON  # -fPIC
        BUILD_ALWAYS FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        TLS_VERIFY OFF
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS
            ${_TargetInstallPrefix}/lib/cmake/GTest/
            ${_TargetInstallPrefix}/${CMAKE_INSTALL_LIBDIR}/libgtest.a
)
file(MAKE_DIRECTORY ${_TargetInstallPrefix}/include)
add_library(gtest_static STATIC IMPORTED)
set_target_properties(gtest_static PROPERTIES
        IMPORTED_LOCATION "${_TargetInstallPrefix}/${CMAKE_INSTALL_LIBDIR}/libgtest.a"
)
add_library(GTest::gtest INTERFACE IMPORTED)
set_target_properties(GTest::gtest PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_TargetInstallPrefix}/include"
        INTERFACE_LINK_LIBRARIES "Threads::Threads;gtest_static"
)
add_dependencies(GTest::gtest ExternalProject_GTest)
message(STATUS "Use GTest from source: ${_TargetSourceDir}")

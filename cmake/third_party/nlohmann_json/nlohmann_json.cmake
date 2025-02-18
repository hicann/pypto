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

set(_TargetNameAlias "json")
set(_TargetVersion 3.11.3)

# 免重入
if (TARGET ${_TargetNameAlias})
    return()
endif ()

# 异常拦截
if (NOT PYPTO_THIRD_PARTY_PATH)
    # 不再从环境上查找, 因当前部分软件编译过程需要添加 -D_GLIBCXX_USE_CXX11_ABI=0, 可能与环境上已安装软件冲突.
    set(_Msg
            "Failed to get nlohmann_json source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR ${_Msg})
endif ()

# 直接查找制品, 若找到则直接退出
get_filename_component(_TargetTarGzFile "${PYPTO_THIRD_PARTY_PATH}/json-${_TargetVersion}.tar.gz" REALPATH)
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
find_package(nlohmann_json ${_TargetVersion} EXACT CONFIG PATHS ${_TargetInstallPrefix} NO_DEFAULT_PATH)
if (NOT nlohmann_json_FOUND)
    # 兼容部分镜像直接存放 json 安装结果的情况
    get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/json" REALPATH)
    if (NOT EXISTS ${_TargetSourceDir})
        get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/json-${_TargetVersion}" REALPATH)
    endif ()
    find_package(nlohmann_json ${_TargetVersion} EXACT CONFIG PATHS ${_TargetSourceDir} NO_DEFAULT_PATH)
endif ()
if (nlohmann_json_FOUND)
    message(STATUS "Use nlohmann_json from binary, nlohmann_json_DIR=${nlohmann_json_DIR}")
    # 重命名目标
    if (NOT TARGET json)
        get_target_property(_JsonInc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        add_library(json INTERFACE)
        set_target_properties(json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_JsonInc}")
    endif ()
    return()
endif ()

# 触发编译
get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/json" REALPATH)
if (NOT EXISTS ${_TargetSourceDir})
    get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/json-${_TargetVersion}" REALPATH)
endif ()
get_filename_component(_TargetBinaryDir "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/build/json-${_TargetVersion}" REALPATH)
PTO_Fwk_CleanEmptyDir(DIR ${_TargetSourceDir})

set(_ExtArgs)
if (NOT EXISTS ${_TargetSourceDir})
    list(APPEND _ExtArgs
            URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz"
            URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
            DOWNLOAD_DIR ${PYPTO_THIRD_PARTY_PATH}
    )
endif ()

ExternalProject_Add(ExternalProject_nlohmann_json   ${_ExtArgs}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/json-${_TargetVersion}
        SOURCE_DIR ${_TargetSourceDir}
        BINARY_DIR ${_TargetBinaryDir}
        INSTALL_DIR ${_TargetInstallPrefix}
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -S <SOURCE_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            # 编译器相关配置
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            # 具体软件相关变量
            -DJSON_MultipleHeaders=ON
            -DJSON_BuildTests=OFF
        BUILD_ALWAYS FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        TLS_VERIFY OFF
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS
            ${_TargetInstallPrefix}/include/nlohmann/json.hpp
)
if (NOT TARGET json)
    add_library(json INTERFACE)
    set_target_properties(json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_TargetInstallPrefix}/include")
    add_dependencies(json ExternalProject_nlohmann_json)
    message(STATUS "Use nlohmann_json from source: ${_TargetSourceDir}")
endif ()

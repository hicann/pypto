# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ENABLE_FEATURE_PACKING_WHL_INTO_RUN)
    set(CANN_CMAKE_TAG "master-026")          # tag 名
    set(CANN_CMAKE_DIR_NAME "cann-cmake")     # 压缩包解压后的目录名

    if (PYPTO_THIRD_PARTY_PATH AND IS_DIRECTORY "${PYPTO_THIRD_PARTY_PATH}/${CANN_CMAKE_DIR_NAME}")
        include("${PYPTO_THIRD_PARTY_PATH}/${CANN_CMAKE_DIR_NAME}/function/prepare.cmake")
    else ()
        include(FetchContent)
        get_filename_component(CANN_CMAKE_SOURCE_DIR "${PYPTO_THIRD_PARTY_PATH}/${CANN_CMAKE_DIR_NAME}" ABSOLUTE)

        if (PYPTO_THIRD_PARTY_PATH AND EXISTS "${PYPTO_THIRD_PARTY_PATH}/${CANN_CMAKE_TAG}.tar.gz")
            FetchContent_Declare(
                ${CANN_CMAKE_DIR_NAME}
                SOURCE_DIR "${CANN_CMAKE_SOURCE_DIR}"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                URL "${PYPTO_THIRD_PARTY_PATH}/${CANN_CMAKE_TAG}.tar.gz"
                URL_HASH SHA256=b0db2d4d0d2e94bd0dd961d88dc16b6b042fbacb4de4acb9530128c98e637fca
            )
        else ()
            FetchContent_Declare(
                ${CANN_CMAKE_DIR_NAME}
                SOURCE_DIR "${CANN_CMAKE_SOURCE_DIR}"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                URL "https://raw.gitcode.com/cann/cmake/archive/refs/heads/${CANN_CMAKE_TAG}.tar.gz"
                URL_HASH SHA256=b0db2d4d0d2e94bd0dd961d88dc16b6b042fbacb4de4acb9530128c98e637fca
            )
        endif ()
        FetchContent_GetProperties(${CANN_CMAKE_DIR_NAME})
        if (NOT ${CANN_CMAKE_DIR_NAME}_POPULATED)
            FetchContent_Populate(${CANN_CMAKE_DIR_NAME})
        endif ()
        message(STATUS "${CANN_CMAKE_DIR_NAME}_SOURCE_DIR=${${CANN_CMAKE_DIR_NAME}_SOURCE_DIR}")
        include("${${CANN_CMAKE_DIR_NAME}_SOURCE_DIR}/function/prepare.cmake")
    endif ()
endif ()

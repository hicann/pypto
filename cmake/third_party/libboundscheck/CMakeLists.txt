# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(libboundscheck)

file(GLOB_RECURSE SRC_LIST
        "${libboundscheck_SRC_DIR}/src/vsprintf_s.c"
        "${libboundscheck_SRC_DIR}/src/wmemmove_s.c"
        "${libboundscheck_SRC_DIR}/src/strncat_s.c"
        "${libboundscheck_SRC_DIR}/src/vsnprintf_s.c"
        "${libboundscheck_SRC_DIR}/src/fwscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/scanf_s.c"
        "${libboundscheck_SRC_DIR}/src/strcat_s.c"
        "${libboundscheck_SRC_DIR}/src/sscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/secureprintoutput_w.c"
        "${libboundscheck_SRC_DIR}/src/wmemcpy_s.c"
        "${libboundscheck_SRC_DIR}/src/wcsncat_s.c"
        "${libboundscheck_SRC_DIR}/src/secureprintoutput_a.c"
        "${libboundscheck_SRC_DIR}/src/secureinput_w.c"
        "${libboundscheck_SRC_DIR}/src/memcpy_s.c"
        "${libboundscheck_SRC_DIR}/src/fscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/vswscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/secureinput_a.c"
        "${libboundscheck_SRC_DIR}/src/memmove_s.c"
        "${libboundscheck_SRC_DIR}/src/swscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/snprintf_s.c"
        "${libboundscheck_SRC_DIR}/src/vscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/vswprintf_s.c"
        "${libboundscheck_SRC_DIR}/src/wcscpy_s.c"
        "${libboundscheck_SRC_DIR}/src/vfwscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/memset_s.c"
        "${libboundscheck_SRC_DIR}/src/wscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/vwscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/strtok_s.c"
        "${libboundscheck_SRC_DIR}/src/wcsncpy_s.c"
        "${libboundscheck_SRC_DIR}/src/vfscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/vsscanf_s.c"
        "${libboundscheck_SRC_DIR}/src/wcstok_s.c"
        "${libboundscheck_SRC_DIR}/src/securecutil.c"
        "${libboundscheck_SRC_DIR}/src/gets_s.c"
        "${libboundscheck_SRC_DIR}/src/swprintf_s.c"
        "${libboundscheck_SRC_DIR}/src/strcpy_s.c"
        "${libboundscheck_SRC_DIR}/src/wcscat_s.c"
        "${libboundscheck_SRC_DIR}/src/strncpy_s.c"
)

include_directories(${libboundscheck_SRC_DIR}/include)
include_directories(${libboundscheck_SRC_DIR}/src)


add_library(shared_c_sec SHARED
        ${SRC_LIST}
        $<$<NOT:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>>:${libboundscheck_SRC_DIR}/src/sprintf_s.c>
)

target_compile_options(shared_c_sec PRIVATE
        $<$<NOT:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>>:-Wall>
        $<$<NOT:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>>:-Werror>
        $<$<NOT:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>>:-O1>
)

target_compile_definitions(shared_c_sec PRIVATE
        NDEBUG
        SECUREC_SUPPORT_STRTOLD=1
        $<$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0>
)

set_target_properties(shared_c_sec
        PROPERTIES
        OUTPUT_NAME $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>,libc_sec,c_sec>
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
)

install(TARGETS shared_c_sec OPTIONAL
        DESTINATION lib
)

install(FILES
        "${libboundscheck_SRC_DIR}/include/securec.h"
        "${libboundscheck_SRC_DIR}/include/securectype.h"
        DESTINATION include
)

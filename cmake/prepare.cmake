# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#[[
因下载 cann-cmake 同样需要 PYPTO_THIRD_PARTY_PATH 定义, 且下载流程需在整体定义 project() 之前执行.
故将其放在本文件中定义, 并在 project() 之前调用.
]]

if (PYPTO_THIRD_PARTY_PATH)
    get_filename_component(PYPTO_THIRD_PARTY_PATH "${PYPTO_THIRD_PARTY_PATH}" REALPATH)
elseif (DEFINED ENV{PYPTO_THIRD_PARTY_PATH})
    get_filename_component(PYPTO_THIRD_PARTY_PATH "$ENV{PYPTO_THIRD_PARTY_PATH}" REALPATH)
else ()
    get_filename_component(PYPTO_THIRD_PARTY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party_path" REALPATH)
    set(_Msg
            "PYPTO_THIRD_PARTY_PATH is not specified, ${PYPTO_THIRD_PARTY_PATH} will be used as its default value. "
            "It is necessary to confirm that the relevant software already exists in this path or that the network "
            "can be accessed normally so that CMake can automatically download the corresponding software."
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(WARNING "${_Msg}")
endif ()
message(STATUS "PYPTO_THIRD_PARTY_PATH=${PYPTO_THIRD_PARTY_PATH}")

# cann/cmake 内部函数依赖该变量（superbuild 可覆盖，独立构建回退到 PYPTO_THIRD_PARTY_PATH）
if (NOT DEFINED CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH "${PYPTO_THIRD_PARTY_PATH}")
endif()

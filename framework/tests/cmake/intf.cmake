# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_intf_pub)
add_library(${_Target} INTERFACE)
target_compile_options(${_Target}
        INTERFACE
            -fno-access-control
)
target_link_libraries(${_Target}
        INTERFACE
            intf_pub_cxx17
            tile_fwk_intf_pub
)

set(_Target ${PTO_Fwk_STestNamePrefix}_intf_pub)
add_library(${_Target} INTERFACE)
target_compile_options(${_Target}
        INTERFACE
            -fno-access-control
)
target_compile_definitions(${_Target}
        INTERFACE
            $<$<BOOL:${ENABLE_STEST_BINARY_CACHE}>:ENABLE_STEST_BINARY_CACHE>
)
target_compile_definitions(${_Target}
        INTERFACE
            $<$<BOOL:${ENABLE_STEST_DUMP_JSON}>:ENABLE_STEST_DUMP_JSON>
)
target_link_libraries(${_Target}
        INTERFACE
            intf_pub_cxx17
            tile_fwk_intf_pub
)

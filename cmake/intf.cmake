# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(tile_fwk_intf_pub INTERFACE)
target_include_directories(tile_fwk_intf_pub
        INTERFACE   # 源码依赖
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>
)
target_compile_options(tile_fwk_intf_pub
        INTERFACE
            # 安全编译选项
            $<$<CONFIG:Release>:-O2 -D_FORTIFY_SOURCE=2>
            $<$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>,$<BOOL:${ENABLE_GCOV}>>:-Og>
            # 基础要求选项
            $<$<CONFIG:Debug>:-g>
            # 告警增强选项
            -Wextra
            -Wundef
            -Wunused
            -Wcast-qual
            -Wpointer-arith
            -Wdate-time
            -Wunused-macros
            -Wfloat-equal
            -Wformat=2
            -Wshadow
            -Wsign-compare
            -Wunused-macros
            -Wvla
            -Wdisabled-optimization
            -Wempty-body
            -Wignored-qualifiers
            $<$<CXX_COMPILER_ID:GNU>:-Wimplicit-fallthrough=3>
            -Wtype-limits
            -Wshift-negative-value
            -Wswitch-default
            $<$<CXX_COMPILER_ID:GNU>:$<$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>,$<BOOL:${ENABLE_GCOV}>>:--param max-gcse-memory=1000000000>>
            -Wframe-larger-than=$<IF:$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>>,131072,32768>
            -Woverloaded-virtual
            -Wnon-virtual-dtor
            $<$<CXX_COMPILER_ID:GNU>:-Wshift-overflow=2>
            -Wshift-count-overflow
            -Wwrite-strings
            -Wmissing-format-attribute
            -Wformat-nonliteral
            -Wdelete-non-virtual-dtor
            $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
            $<$<CXX_COMPILER_ID:GNU>:-Wtrampolines>
            $<$<CXX_COMPILER_ID:GNU>:-Wsized-deallocation>
            $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
            $<$<CXX_COMPILER_ID:GNU>:-Wsuggest-attribute=format>
            $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
            $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
            # -Wmissing-include-dirs
            $<$<CXX_COMPILER_ID:GNU>:-Wformat-signedness>
            $<$<CXX_COMPILER_ID:GNU>:-Wreturn-local-addr>
            -Wredundant-decls
            -Wfloat-conversion
            $<$<CXX_COMPILER_ID:Clang>:-Wno-tautological-unsigned-enum-zero-compare>
            -fno-common
            -fno-strict-aliasing
            # 放在最后
            -Wno-return-type
            -Wno-array-bounds
            $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable>
            -Wno-unused-variable
            -Wno-unused-parameter
            -Wno-unused-result
            # Clang
            $<$<CXX_COMPILER_ID:Clang>:-Wno-mismatched-tags>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-non-pod-varargs>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-const-variable>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-private-field>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-uninitialized>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-lambda-capture>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-braced-scalar-init>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-frame-larger-than=>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-variable>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-missing-braces>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-cast-qual>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-shadow>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unsequenced>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-function>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-return-type-c-linkage>
            -Werror
            # 依赖分析选项
            $<$<CXX_COMPILER_ID:GNU>:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:-MMD>>
)
target_link_directories(tile_fwk_intf_pub
        INTERFACE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_options(tile_fwk_intf_pub
        INTERFACE
            # 安全编译选项
            -Wl,-z,relro
            -Wl,-z,now
            -Wl,-z,noexecstack
            $<$<CONFIG:Release>:-s>
)

if (BUILD_OPEN_PROJECT)
    add_library(intf_pub_cxx17 INTERFACE)
    target_compile_options(intf_pub_cxx17
            INTERFACE
                # 安全编译选项
                -fPIC
                $<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>>
                $<$<CXX_COMPILER_ID:GNU>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>>
                $<$<CXX_COMPILER_ID:Clang>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},10.0.0>,-fstack-protector-strong,-fstack-protector-all>>
                # 基础要求选项
                -Wall
                # GCOV
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:--coverage -fprofile-arcs -ftest-coverage>>
                # ASAN
                $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-address-use-after-scope -fsanitize=leak>
                # UBSAN
                # 在 Clang 编译器场景下 使能 -fsanitize=undefined 会默认开启基本所有的 UBSAN 检查项, 只有以下检查项不会开启
                #   float-divide-by-zero, unsigned-integer-overflow, implicit-conversion, local-bounds 及 nullability-* 类检查.
                # 故在 Clang 编译器使能 UBSAN 场景下, 需开启 -fsanitize=undefined 使能时仍未开启的对应检查项
                # 在 GNU 编译器场景下, 官方文档并未对使能 -fsanitize=undefined 时开启的默认检查项范围进行说明, 故手工开启常用基本检查项, 避免能力遗漏
                $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined -fsanitize=float-divide-by-zero -fno-sanitize=alignment>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:-fsanitize=unsigned-integer-overflow>>    # GNU 不支持这些检查项
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:$<$<VERSION_GREATER_EQUAL:${CMAKE_C_COMPILER_VERSION},10.0.0>:-fsanitize=implicit-conversion>>>    # GNU 不支持这些检查项, Clang高版本才支持这些检查项
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=shift>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=integer-divide-by-zero>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=signed-integer-overflow>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=float-divide-by-zero>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=float-cast-overflow>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=bool>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=enum>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=vptr>>
                # ASAN/UBSAN 公共
                $<$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>>:-fno-omit-frame-pointer -fsanitize-recover=all>
    )
    target_compile_definitions(intf_pub_cxx17
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>    # 必须设置, 以保证与 CANN 包内其他 C++ 二进制兼容
    )
    target_link_libraries(intf_pub_cxx17
            INTERFACE
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:gcov>>
    )
    target_link_options(intf_pub_cxx17
            INTERFACE
                # GCOV
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:-fprofile-arcs -ftest-coverage>>
                # ASAN
                $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
                # UBSAN
                $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
    )
endif()

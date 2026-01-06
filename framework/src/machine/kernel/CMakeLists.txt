# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
        set(BISHENG_PROGRAM bisheng)
        set(BISHENG_LD ld.lld)
else ()
        set(BISHENG_PROGRAM ${CCEC_PATH}/bisheng)
        set(BISHENG_LD ${CCEC_PATH}/ld.lld)
endif ()

set(AI_CORE_FLAGS
    "-c -O3 -g -x cce -Wall -Werror -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -D__MIX__ \
    -D__STATIC_KERNEL__ \
    -D__TILINGKEY__=0   \
    -D__OPTYPE__=tilefwk   \
    -I${PTO_FWK_SRC_ROOT}/framework/include \
    -I${PTO_FWK_SRC_ROOT}/framework/src \
    -I${PTO_FWK_SRC_ROOT}/framework/src/interface \
    -I${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device"
)

separate_arguments(AI_CORE_FLAGS)

add_custom_command(
        OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel
        COMMENT "Build core, create ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel"
)

add_custom_target(tile_fwk_core ALL DEPENDS aicore.ascpp ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel
        COMMAND ${BISHENG_PROGRAM} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube -o kernel_aic.o ${CMAKE_CURRENT_SOURCE_DIR}/aicore.ascpp
        COMMAND ${BISHENG_PROGRAM} ${AI_CORE_FLAGS} -D__AIV__ --cce-aicore-arch=dav-c220-vec  -o kernel_aiv.o ${CMAKE_CURRENT_SOURCE_DIR}/aicore.ascpp
        COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -o  kernel.o kernel_aic.o kernel_aiv.o
        COMMAND ld -r -b binary -o kernel_host.o kernel.o
        COMMENT "Build core, output to ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
        WORKING_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel
        BYPRODUCTS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel/kernel_host.o ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel/kernel.o
)

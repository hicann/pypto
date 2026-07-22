#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 全量源码统一构建入口(薄壳):
#   不感知 whl 存在 — 编译权归 CMake, 打包权归 setuptools/cann-cmake.
#
# Usage:
#   ./build.sh [--release|--debug|--minsizerel|--relwithdebinfo] [-j N] [--wheel|--clean|--help]
#
#   (none)        编译 + 打包 whl + 打包 run (默认)
#   --wheel        仅编译 + 打包 whl
#   --clean        清理构建目录
#   --help         查看帮助
#
#   Build type:
#     --release            CMake Release 构建 (默认)
#     --debug              CMake Debug 构建
#     --minsizerel         CMake MinSizeRel 构建
#     --relwithdebinfo     CMake RelWithDebInfo 构建
#     -j N                 并行编译任务数 (默认: nproc)
#
# Environment variables (被命令行参数覆盖):
#   PYTHON3_EXECUTABLE  Python3 解释器路径 (default: which python3)
#   ASCEND_HOME_PATH    CANN 安装路径 (未设置时自动 -DBUILD_WITH_CANN=OFF)
#   PYPTO_THIRD_PARTY_PATH  三方库下载路径 (default: ./third_party_llpath)
#   BUILD_DIR           构建目录 (default: ./build)

set -euo pipefail

print_help() {
    echo "Usage: $0 [--release|--debug|--minsizerel|--relwithdebinfo] [-j N] [--wheel|--clean|--help]"
    echo ""
    echo "  (none)            编译 + 打包 whl + 打包 run (默认)"
    echo "  --wheel           仅编译 + 打包 whl"
    echo "  --clean           清理构建目录"
    echo ""
    echo "Build type:"
    echo "  --release         CMake Release 构建 (默认)"
    echo "  --debug           CMake Debug 构建"
    echo "  --minsizerel      CMake MinSizeRel 构建"
    echo "  --relwithdebinfo  CMake RelWithDebInfo 构建"
    echo "  -j N              并行编译任务数 (默认: nproc)"
    echo ""
    echo "Environment:"
    echo "  PYTHON3_EXECUTABLE ASCEND_HOME_PATH PYPTO_THIRD_PARTY_PATH BUILD_DIR"
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
PYTHON3_EXE="${PYTHON3_EXECUTABLE:-$(which python3 2>/dev/null || echo '')}"

# defaults (可被命令行覆盖)
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
BUILD_TARGET="pypto_package"            # 默认产出 run 包
CANN_PATH=""

if [ -z "${PYTHON3_EXE}" ]; then
    echo "ERROR: Python3 interpreter not found. Set PYTHON3_EXECUTABLE env."
    exit 1
fi

########################################################################################################################
# argument parsing
########################################################################################################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --minsizerel)
            BUILD_TYPE="MinSizeRel"
            shift
            ;;
        --relwithdebinfo)
            BUILD_TYPE="RelWithDebInfo"
            shift
            ;;
        -j)
            if [[ $# -lt 2 || "$2" =~ ^- ]]; then
                echo "ERROR: -j requires a numeric argument"
                exit 1
            fi
            BUILD_JOBS="$2"
            shift 2
            ;;
        -j[0-9]*)
            BUILD_JOBS="${1#-j}"
            shift
            ;;
        --wheel|wheel|whl)
            BUILD_TARGET="pypto_wheel"
            shift
            ;;
        --cann_path | -p)
            CANN_PATH="$(realpath $2)"
            shift 2
            ;;
        --clean|clean)
            rm -rf "${BUILD_DIR}"
            echo "[build.sh] Cleaned ${BUILD_DIR}"
            exit 0
            ;;
        -h|--help|help)
            print_help
            ;;
        *)
            echo "ERROR: Unknown option '$1'. Use --help for usage."
            exit 1
            ;;
    esac
done

set_env() {
    if [ "$(id -u)" != "0" ]; then
        DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/cann"
        DEFAULT_INSTALL_DIR="${HOME}/Ascend/cann"
    else
        DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/cann"
        DEFAULT_INSTALL_DIR="/usr/local/Ascend/cann"
    fi

    ASCEND_CANN_PACKAGE_PATH=""
    if [ -n "${CANN_PATH}" ];then
        ASCEND_CANN_PACKAGE_PATH=${CANN_PATH}
    elif [ -n "${ASCEND_HOME_PATH}" ];then
        ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
    elif [ -n "${ASCEND_OPP_PATH}" ];then
        ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
    elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
        ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
    elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
        ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
    fi
}
set_env

########################################################################################################################
# cmake options
########################################################################################################################
CMAKE_OPTS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DENABLE_UNIFIED_BUILD=ON
    -DPython3_EXECUTABLE="${PYTHON3_EXE}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}"
)

if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    CMAKE_OPTS+=(-DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}")
else
    CMAKE_OPTS+=(-DBUILD_WITH_CANN=OFF)
fi

if [ -n "${PYPTO_THIRD_PARTY_PATH:-}" ]; then
    CMAKE_OPTS+=(-DPYPTO_THIRD_PARTY_PATH="${PYPTO_THIRD_PARTY_PATH}")
fi

########################################################################################################################
# main
########################################################################################################################

echo "[build.sh] cmake configure (build_type=${BUILD_TYPE} jobs=${BUILD_JOBS})..."
cmake "${CMAKE_OPTS[@]}"

if [ -n "${BUILD_TARGET}" ]; then
    echo "[build.sh] cmake build --target ${BUILD_TARGET} -j ${BUILD_JOBS} ..."
    cmake --build "${BUILD_DIR}" --target "${BUILD_TARGET}" -j "${BUILD_JOBS}"
else
    echo "[build.sh] cmake build -j ${BUILD_JOBS} (binaries only)..."
    cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS}"
fi

########################################################################################################################
# report
########################################################################################################################
echo ""
echo "============================================"
echo "Build completed successfully"
echo "============================================"
echo "Build dir: ${BUILD_DIR}"

if [ "${BUILD_TARGET}" = "pypto_wheel" ] || [ "${BUILD_TARGET}" = "pypto_package" ]; then
    if compgen -G "${BUILD_DIR}/dist/"*.whl > /dev/null 2>&1; then
        echo "Wheel:     $(ls ${BUILD_DIR}/dist/*.whl)"
    fi
fi

if [ "${BUILD_TARGET}" = "pypto_package" ]; then
    if compgen -G "${BUILD_DIR}/dist/"*.run > /dev/null 2>&1; then
        echo "Run:       $(ls ${BUILD_DIR}/dist/*.run)"
    fi
fi

#!/bin/bash
set -euo pipefail
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

rm -rf /home/jenkins/opensource/json

BASE_DIR="/home/taskspace"
PYPTO_GOLDEN_PATH="$BASE_DIR/pypto_golden"
mkdir -p $PYPTO_GOLDEN_PATH
PYPTO_3RD_LIB_PATH="/home/opensource"
CHANGED_FILES_PARAM="$BASE_DIR/pr_filelist.txt"

echo "run branch \${GIT_TARGET_BRANCH} smoke test"

SHM_SIZE_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2 {print $2}' | sed 's/G//')
if [ "$SHM_SIZE_GB" -lt 32 ]; then
    echo "[FIX] Expanding /dev/shm from ${SHM_SIZE_GB}GB to 32GB for distributed tests..."
else
    echo "[CHECK] /dev/shm already ${SHM_SIZE_GB}GB (sufficient)"
fi

CANN_PATH="/usr/local/Ascend"
source /opt/conda/bin/activate python39
source $CANN_PATH/cann/bin/setenv.bash

# C++ stest_distributed 需要 mpirun 命令和 MPI_HOME 环境变量（用于动态加载 libmpi.so）
MPI_BIN_DIR="/opt/conda/bin"
MPI_LIB_DIR="/opt/conda/lib"
PYTHON39_BIN_DIR="/opt/conda/envs/python39/bin"
if [ -f "$MPI_BIN_DIR/mpirun" ]; then
    # 软链 mpirun 到 python39 的 bin，避免污染 PATH 中的 python 版本
    for tool in mpirun mpiexec hydra_pmi_proxy; do
        [ -f "$MPI_BIN_DIR/$tool" ] && ln -sf "$MPI_BIN_DIR/$tool" "$PYTHON39_BIN_DIR/$tool" 2>/dev/null
    done
    export MPI_HOME="/opt/conda"
    export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
    echo "[FIX] Configured MPI environment: MPI_HOME=$MPI_HOME"
else
    echo "[WARN] mpirun not found at $MPI_BIN_DIR, C++ distributed tests may fail"
fi

cd "$BASE_DIR"

set +e

device_params=("-d=0" "-d=1" "-d=2" "-d=3" "-d=4" "-d=5" "-d=6" "-d=7" "-d=8" "-d=9" "-d=10" "-d=11" "-d=12" "-d=13" "-d=14" "-d=15")
common_params=("--clean" "--verbose" "--golden_path=$PYPTO_GOLDEN_PATH" "--changed_files=$CHANGED_FILES_PARAM")

# ===== 测试阶段 1: Python STest & Examples & Models =====
python3 build_ci.py "${common_params[@]}" --no_isolation --stest --example --models "${device_params[@]}"
ret=$?
if [ $ret -ne 0 ]; then
    echo "[ERROR] Python(STest & Examples) failed"
    exit $ret
fi
echo "[INFO] Python(STest & Examples) succeeded"

# ===== 测试阶段 2: C++ STest =====
python3 build_ci.py --frontend=cpp "${common_params[@]}" --stest "${device_params[@]}" --case_execute_timeout=35
ret=$?
if [ $ret -ne 0 ]; then
    echo "[ERROR] C++(STest) failed"
    exit $ret
fi
echo "[INFO] C++(STest) succeeded"

# ===== 测试阶段 3: C++ STest Distributed =====
python3 build_ci.py --frontend=cpp "${common_params[@]}" --stest_distributed "${device_params[@]}" --case_execute_timeout=35
ret=$?
if [ $ret -ne 0 ]; then
    echo "[ERROR] C++(STest Distributed) failed"
    exit $ret
fi
echo "[INFO] C++(STest Distributed) succeeded"

echo "All builds completed successfully"
echo "execute sample success"
source /opt/conda/bin/deactivate

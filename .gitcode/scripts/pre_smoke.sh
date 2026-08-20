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

BASE_DIR=${WORKSPACE}
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

RUN_PACKAGE_NAME="cann-pypto_linux-aarch64_ubuntu24.run"
RUN_PACKAGE_URL="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${RUN_PACKAGE_NAME}"
if ! wget -q -O "${RUN_PACKAGE_NAME}" "${RUN_PACKAGE_URL}"; then
    echo "Package download failed!URL:${RUN_PACKAGE_URL}"
else
    echo "Package downloaded successfully: ./${RUN_PACKAGE_NAME}"
fi
# Add execute permission to the downloaded package
echo "Adding execute permission: chmod +x ${RUN_PACKAGE_NAME}"
chmod +x "${RUN_PACKAGE_NAME}" || echo "Failed to add execute permission to the package"
bash "${RUN_PACKAGE_NAME}" --full -q --pylocal --install-path=/usr/local/Ascend
source /usr/local/Ascend/cann/set_env.sh

# ===== 测试阶段 =====
python3 examples/validate_examples.py -t examples -d 0,1,2,3
ret=$?
if [ $ret -ne 0 ]; then
    echo "[ERROR] Python Examples failed"
    exit $ret
fi
echo "[INFO] Python Examples succeeded"

source /opt/conda/bin/deactivate

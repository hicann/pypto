#!/bin/bash
set -euo pipefail

cd "${WORKSPACE}"
export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
rm -rf /home/jenkins/opensource/json

if [[ "${arch}" == "arm64" ]]; then
    ARCH_SUFFIX="aarch64"
else
    ARCH_SUFFIX="x86_64"
fi

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
gcc --version

python3 -m pip install build
python3 -m pip install --upgrade packaging==24.2

if [[ "${GIT_TARGET_BRANCH}" == "master" ]]; then
    echo "package_type=run" >> "${ATOMGIT_OUTPUT}"
else
    echo "package_type=whl" >> "${ATOMGIT_OUTPUT}"
fi

if [ "${GIT_TARGET_BRANCH}" = "br_0.1.1_20260313_beta" ]; then
    wget https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/br_0313/package/cann-pto-isa_9.0.0_linux-${ARCH_SUFFIX}.run
elif [ "${GIT_TARGET_BRANCH}" = "0.2.0" ]; then
    wget https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/package/pto/9.0.0/20260330/${ARCH_SUFFIX}/cann-pto-isa_9.0.0_linux-${ARCH_SUFFIX}.run
elif [ "${GIT_TARGET_BRANCH}" = "9.1.0" ]; then
    wget https://opencann-obs.obs.cn-north-4.myhuaweicloud.com/pto-isa/9.1.0/cann-pto-isa_linux-${ARCH_SUFFIX}.run
else
    wget https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-${ARCH_SUFFIX}.run
fi
chmod +x *.run
echo "y" | su - jenkins -c "cd ${WORKSPACE} && bash *.run --full --quiet --install-path=/home/jenkins/Ascend"
source /home/jenkins/Ascend/cann/bin/setenv.bash

if [[ "${arch}" == "x64" ]]; then
    if [[ "${task_name}" == *_ubuntu24 ]]; then
        python3 build_ci.py --clean --timeout=400 --no_isolation --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --compile_dependency_check --verbose
    else
        python3 build_ci.py --clean --timeout=300 --no_isolation --plat_name=manylinux2014 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --compile_dependency_check --verbose
    fi
else
    if [[ "${task_name}" == *_ubuntu24 ]]; then
        python3 build_ci.py --clean --build_type=Release --timeout=900 --no_isolation --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16
    else
        python3 build_ci.py --clean --build_type=Release --timeout=900 --no_isolation --plat_name=manylinux2014  --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16
    fi
fi

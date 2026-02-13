

说明：本文描述如何快速使用示例 Dockerfile 创建可运行 PyPTO 的 Docker 容器。  
在使用 Docker 容器前，请务必**完成主机 NPU 硬件部署，以及 NPU 驱动和固件安装**，可参考文档  
[Environment README](../docs/install/prepare_environment.md)。  
建议 Docker 版本：**v27.2.1 及以上**。

## 版本说明

当前提供两类 Dockerfile：

- **版本 1：安装 CANN 包的 Dockerfile**
- **版本 2：不安装 CANN 包的 Dockerfile**

两类 Dockerfile 都会安装 PyPTO 运行所依赖的软件包，只是是否在镜像内完成 CANN 相关环境的安装不同。

---

## 版本 1：安装 CANN 包的 Dockerfile

### 支持的环境信息

当前示例 Dockerfile 构建镜像支持的环境信息如下：

```
#**************docker info*******************#
# os: ubuntu22.04, openeuler24.03
# arch: x86_64, aarch64
# python: 3.11
# cann env
# cann_verison: 8.5.0 
# torch: 2.6.0
# torch_npu: 2.6.0
# device_type: A2, A3
#**************docker info*******************#
```

示例 Dockerfile 以 **Ubuntu** 为基础编写，不同操作系统镜像之间存在轻微差异，请根据实际操作系统进行适配调整。  

在使用前，请根据 **操作系统 + 硬件类型** 指定 `CANN_VERSION`：  

- **Ubuntu + A3**：`ARG CANN_VERSION=8.5.0-a3-ubuntu22.04-py3.11`  
- **Ubuntu + A2**：`ARG CANN_VERSION=8.5.0-910b-ubuntu22.04-py3.11`  
- **openEuler + A3**：`ARG CANN_VERSION=8.5.0-a3-openeuler24.03-py3.11`  
- **openEuler + A2**：`ARG CANN_VERSION=8.5.0-910b-openeuler24.03-py3.11`  

根据 CPU 架构指定 `TARGETPLATFORM`：  

- **x86_64**：`ARG TARGETPLATFORM=linux/amd64`  
- **aarch64**：`ARG TARGETPLATFORM=linux/arm64`  

<span style="font-size:12px;">*若上述信息与实际硬件及驱动不匹配，将导致 CANN 包安装失败，从而导致镜像构建失败。*</span>

### 示例 Dockerfile（版本 1）


```dockerfile
# step1: 指定 CANN 基础镜像版本
ARG CANN_VERSION=8.5.0-a3-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:$CANN_VERSION

# 指定目标平台架构
ARG TARGETPLATFORM=linux/amd64

# [Optional] 设置 HTTP/HTTPS 代理（按需配置）
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 工作目录
WORKDIR /tmp

# step2: 安装 PyPTO 项目构建/运行所需依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gdb gawk wget curl tar lcov openssl ca-certificates \
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache \
    python-is-python3 python3-pip python3-venv ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 安装指定版本 torch / torch-npu（CPU 源 + NPU 插件）
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir torch-npu==2.6.0

# [Optional] step3: 在镜像中安装由 PyPTO 提供的 CANN 包（按需开启）
# 以下内容默认注释，如需要可取消注释并根据网络环境及仓库地址调整。
#
# WORKDIR /mount_home
# RUN git clone https://gitcode.com/cann/pypto.git
# WORKDIR /mount_home/pypto
# ARG CANN_VERSION
# RUN if echo "${CANN_VERSION}" | grep -iq "910b"; then \
#         DEVICE_TYPE="a2"; \
#     elif echo "${CANN_VERSION}" | grep -iq "a3"; then \
#         DEVICE_TYPE="a3"; \
#     else \
#         echo "ERROR: Unsupported CANN_VERSION format: ${CANN_VERSION}" 1>&2 && \
#         echo "Version should contain '910b' or 'a3' (case-insensitive)" 1>&2 && \
#         exit 1; \
#     fi && \
#     echo "DEVICE_TYPE=${DEVICE_TYPE}" && \
#     chmod +x tools/prepare_env.sh && \
#     bash tools/prepare_env.sh --type=cann --device-type=${DEVICE_TYPE} --install-path=/usr/local/Ascend/CANN_pypto --quiet
#
# # Note: 设置环境变量，容器登录自动生效
# RUN \
#     CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/CANN_pypto/ascend-toolkit/latest/set_env.sh" && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc
#
# ENTRYPOINT ["/bin/bash", "-c", "\
#     source /usr/local/Ascend/CANN_pypto/ascend-toolkit/latest/set_env.sh && \
#     exec \"$@\"", "--"]

# step4: 安装 cann-pto-isa
ARG PTO_ISA_INSTALL_PATH=/usr/local/Ascend
ENV PTO_ISA_INSTALL_PATH=$PTO_ISA_INSTALL_PATH
WORKDIR /tmp
RUN set -e; \
    ARCH="unknown"; \
    URL_SUFFIX=""; \
    case "${TARGETPLATFORM}" in \
        "linux/amd64") \
            ARCH="x86_64"; \
            URL_SUFFIX="ubuntu_x86/cann-pto-isa_linux-x86_64.run"; \
            ;; \
        "linux/arm64") \
            ARCH="aarch64"; \
            URL_SUFFIX="ubuntu_aarch64/cann-pto-isa_linux-aarch64.run"; \
            ;; \
        *) \
            echo "ERROR: Unsupported or undefined TARGETPLATFORM: ${TARGETPLATFORM}"; \
            echo "Please set TARGETPLATFORM to 'linux/amd64' or 'linux/arm64' during build."; \
            exit 1; \
            ;; \
    esac; \
    echo "Target platform: ${TARGETPLATFORM}, architecture: ${ARCH}"; \
    PACKAGE_URL="http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/release_version/${URL_SUFFIX}"; \
    PACKAGE_NAME="cann-pto-isa_8.5.0_linux-${ARCH}.run"; \
    echo "Downloading package from: ${PACKAGE_URL}"; \
    wget --quiet --no-check-certificate -O "${PACKAGE_NAME}" "${PACKAGE_URL}"; \
    chmod +x "${PACKAGE_NAME}"; \
    echo "Installing ${PACKAGE_NAME} to ${PTO_ISA_INSTALL_PATH}"; \
    ./"${PACKAGE_NAME}" --quiet --full --install-path="${PTO_ISA_INSTALL_PATH}"; \
    # rm -f "${PACKAGE_NAME}"; \
    echo "cann-pto-isa installation completed."

# step5: [Optional] 设置默认代理（仅当需要统一代理时启用）
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考 Ascend 社区提供的基础镜像：  
[`https://quay.io/repository/ascend/cann`](https://quay.io/repository/ascend/cann)

---

## 版本 2：不安装 CANN 包的 Dockerfile

### 支持的环境信息

```
#**************docker info*******************#
# os: ubuntu22.04, openeuler22.03
# arch:  x86_64, aarch64
# python: 3.11
# cann env: none
# torch: 2.6.0
# torch_npu: 2.6.0
# device_type: A2, A3
#**************docker info*******************#
```

Dockerfile 使用方式说明：  

- 使用 **Ubuntu 22.04**：`ARG PY_VERSION=3.11-ubuntu22.04`  
- 使用 **openEuler 22.03**：`ARG PY_VERSION=3.11-openeuler22.03`  

### 示例 Dockerfile（版本 2）

```dockerfile
ARG PY_VERSION=3.11-ubuntu22.04
FROM quay.io/ascend/python:$PY_VERSION

# [Optional] 设置 HTTP/HTTPS 代理
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 安装系统依赖并清理 APT 缓存索引
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gdb gawk wget curl tar lcov openssl ca-certificates \
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache \
    python-is-python3 python3-pip python3-venv ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

# [Optional] 配置 pip 源（按需开启其中一项）
# RUN pip config set global.index-url http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

# 安装 Python 依赖
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 升级 setuptools，满足 pypto 要求
RUN pip install --no-cache-dir --upgrade \
    setuptools
# pypto 依赖: setuptools>=77.0.3

# 安装 torch / torch-npu
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0

# [Optional] 设置默认代理，便于容器内访问外网
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他 Python / OS 组合的镜像，可参考：  
[`https://quay.io/repository/ascend/python`](https://quay.io/repository/ascend/python)

---

## 使用指导

### 1. 构建镜像

在本地准备好对应版本的 Dockerfile（例如保存为 `Dockerfile` 或 `dockerfile`），执行镜像构建命令：

```bash
docker build -t <镜像名:版本> -f ./dockerfile .
# 示例：
# docker build -t pyptox86/a3:latest -f /home/dockerfiles/Cann83rc2/dockerfile .
```

### 2. 基于镜像创建容器

仅有镜像无法直接作为开发环境使用，需要基于该镜像创建容器。示例命令如下：

```bash
sudo docker run -u root -itd --name <容器名> --ipc=host --net=host --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -w /mount_home \
    <镜像名:版本> \
    /bin/bash
```

示例：

```bash
sudo docker run -u root -itd --name pypto_x86a3 --ipc=host --net=host --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -w /mount_home \
    pyptox86/a3:latest \
    /bin/bash
```

### 3. 启动并进入容器

```bash
# 启动容器
docker start <容器名>

# 进入容器
docker exec -it <容器名> /bin/bash
```

示例：

```bash
docker start pypto_x86a3
docker exec -it pypto_x86a3 /bin/bash
```

### 4. 在容器内拉取并安装 PyPTO

进入容器后，执行以下步骤：

1. **克隆代码仓库**

   ```bash
   git clone https://gitcode.com/cann/pypto.git
   ```

2. **基于源码编译并安装 `pypto.whl`**

   ```bash
   cd pypto
   python3 -m pip install -e . --verbose
   ```

   可根据实际需求对安装参数、依赖源等进行适当调整与适配。

3. **验证**

   完成以上步骤后，即可在容器内运行 PyPTO 相关用例。

<span style="font-size:12px;">*注：出于兼容性考虑，当前 Docker 环境中编译构建得到的 `whl` 包建议仅在对应 Docker 容器内使用。*</span>
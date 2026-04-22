# 环境部署

在使用PyPTO开发或运行算子之前，请您先参考下面步骤完成基础环境搭建和PyPTO安装。

PyPTO支持在具备NPU硬件的**真实环境**和仅有CPU硬件的**仿真环境**中运行：

| 环境类型 | 硬件要求 | 运行模式 |
|:-----|:------|:------|
| 真实环境 | 配备CPU及NPU硬件 | 支持在NPU上执行计算，也可以通过CPU仿真获取预估性能和执行计算 |
| 仿真环境 | 仅有CPU硬件 | 支持通过CPU仿真，获取预估性能和执行计算 |

**说明:**
- NPU：指昇腾AI处理器，目前仅支持如下产品型号：
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品
- 支持的系统：PyPTO支持在OpenEuler、Ubuntu等主流Linux发行版上编译和运行

## 环境准备

本项目提供多种搭建昇腾环境的方式，请按需选择。

> **说明**：本文提到的编译态和运行态含义如下，请根据实际情况选择。
>
> - 编译态：针对仅编译PyPTO不运行的场景，只需安装CANN toolkit包。
> - 运行态：针对运行PyPTO的场景（编译运行或纯运行），需安装驱动与固件、CANN toolkit包、CANN ops包。

| 安装方式 | 使用说明 | 使用场景 |
| ----- | ------ | ------ |
| WebIDE | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新商发版CANN包**。 | 适用于没有昇腾设备的开发者。 |
| 主机安装（自动安装/手动安装） | 在宿主机上自行准备环境，可选择脚本自动安装部分软件包，或完全手动安装。 | 适用于有昇腾设备，希望在本机直接搭建环境的开发者。 |
| Docker | Docker镜像是一种高效部署方式，已预集成运行所需依赖。<br>当前支持Atlas A2/A3系列产品，OS支持Ubuntu和OpenEuler。 | 适用于有昇腾设备，需要快速搭建环境的开发者。 |

### 方式1：WebIDE环境

对于无昇腾设备的开发者，可直接使用WebIDE开发平台，即"**算子一站式开发平台**"，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新商发版CANN包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考[LINK](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击"`云开发`"按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。
![image.png](https://raw.gitcode.com/user-images/assets/8766299/6b3edd4d-1c7b-496a-82ef-278f9c67113f/image.png 'image.png')

2. 根据页面提示创建并启动云开发环境，单击"`连接 > WebIDE`"进入算子一站式开发平台，开源项目的源码资源默认在`/mnt/workspace`目录下。
![image.png](https://raw.gitcode.com/user-images/assets/8766299/648af2a7-1319-4518-ac97-9b2c2ccb1d17/image.png 'image.png')

### 方式2：主机安装（自动安装/手动安装）

对于有昇腾设备的开发者，若您希望直接在宿主机上搭建昇腾环境，可参考本章节。该章节提供两种软件包安装方式：自动安装和手动安装。

#### 安装方式总览

自动安装与手动安装的区别如下，请先根据自身场景选择：

| 安装子方式 | 覆盖内容 | 适用场景 |
|:---|:---|:---|
| 自动安装 | 通过`tools/prepare_env.sh`自动下载并安装CANN toolkit包、CANN ops包、pto-isa。 | 适用于脚本支持的系统环境，希望尽量减少手工操作的场景。 |
| 手动安装 | 手动下载并安装驱动、固件、CANN包，并手动获取pto-isa源码。 | 适用于脚本不支持的系统环境、需要自定义版本或希望完全控制安装过程的场景。 |

> **说明**：
>
> - 自动安装当前覆盖场景有限同时版本固定，若系统环境不受脚本支持，请自行适配脚本或直接使用手动安装方式。
> - 本章节中的“自动安装”仅针对软件包准备过程，Python依赖、PyTorch及编译依赖仍需按下文要求准备。

#### 需要先手动完成的内容

在选择自动安装或手动安装之前，请先确认以下事项：

1. **安装驱动与固件**

    若计划在真实NPU环境中运行PyPTO，必须先完成驱动与固件安装。详细指导请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"安装NPU驱动和固件"章节。

    > **说明**：
    >
    > - 驱动与固件是运行态依赖，若仅编译PyPTO或仅进行性能仿真，可不安装。
    > - `prepare_env.sh`的`--with-install-driver`参数仅用于下载驱动与固件安装包，不会自动执行安装，驱动与固件仍需您根据官方指导手动安装。

2. **明确前提依赖准备范围**

    无论最终选择自动安装还是手动安装，下述Python依赖、PyTorch/Ascend Extension for PyTorch以及编译依赖，均需根据实际场景自行准备。

#### 前提条件

1. **安装Python依赖**

    - Python：版本 >= 3.9
        - **重要**：若后续需要通过源码编译安装PyPTO，还需安装Python的Development组件（常称为`python3-dev`）。

    - 安装Python依赖包：

        依赖的pip包及对应版本在`python/requirements.txt`中描述，可以使用如下命令完成安装：

        ```bash
        # 进入pypto项目源码根目录
        cd pypto

        # 安装相关pip包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - PyTorch及Ascend Extension for PyTorch：
        - **顺序说明**：请务必参考下文"软件包安装"章节完成对应工具包安装后，再安装`Ascend Extension for PyTorch`。
        - 请根据实际环境的Python版本单独安装，请参考[Ascend Extension for PyTorch文档中心]（https://hiascend.com/document/redirect/pytorchuserguide）中的《软件安装》手册。
        - **重要**：需确保`PyTorch`、`Ascend Extension for PyTorch`与`PyPTO`三者的Python版本一致。
        - **仿真环境说明**：在仿真环境中可跳过`Ascend Extension for PyTorch`的安装，但仍需安装`PyTorch`。

2. **安装编译依赖**

    若不需要编译PyPTO，可跳过本步骤。

    **安装编译工具：**

    - cmake >= 3.16.3
    - make
    - g++ >= 7.3.1

    **准备第三方开源软件源码包**

    PyPTO编译过程依赖以下第三方开源软件源码包，若您的环境可正常访问[cann-src-third-party](https://gitcode.com/cann-src-third-party)，
    这些软件的源码包会在编译时自动下载和编译，否则请手动准备：

    | 软件包                 | 版本      |
    |:--------------------|:--------|
    | JSON for Modern C++ | v3.11.3 |
    | libboundscheck      | v1.1.16 |

    手工准备第三方开源源码包的方法:

    方法一：手工下载

    ```bash
    # 创建并进入用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    mkdir -p <path-to-your-thirdparty> && cd $_

    # 下载 JSON for Modern C++ 三方库
    wget https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz

    # 下载 libboundscheck 三方库
    wget https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz
    ```

    方法二：通过辅助脚本下载

    ```bash
    # 创建用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    mkdir -p <path-to-your-thirdparty>

    # 执行辅助脚本
    # 如果未指定--download-path参数，脚本会将所需三方依赖下载到pypto同级目录的pypto_download/third_party_packages路径下
    # 如果指定了--download-path参数，脚本会将所需三方依赖下载到path-to-your-thirdparty/third_party_packages路径下
    bash tools/prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty]
    ```

#### 软件包安装

> PyPTO支持两种仿真模式：
>
> - **性能仿真**：仅评估程序运行性能，无需安装CANN、NPU驱动与固件。
> - **精度仿真**：模拟真实NPU的执行逻辑，获取运算结果，必须依赖CANN工具包。
>
> 因此：
> - 若仅编译和运行PyPTO**性能仿真**，可跳过本节。
> - 若需编译和运行**精度仿真**，或计划在**真实NPU环境**中编译运行PyPTO，必须安装如下软件包。

##### 自动安装（脚本安装）

若当前系统环境受脚本支持，可使用项目`tools`目录下的`prepare_env.sh`进行自动安装。

脚本可覆盖的内容如下：

- 自动下载并安装CANN toolkit包。
- 自动下载并安装CANN ops包。
- 自动下载并安装pto-isa。

示例命令如下：

```bash
# 自动安装 CANN toolkit / ops / pto-isa
bash tools/prepare_env.sh --type=cann --device-type=a2

# 如需额外下载驱动与固件安装包，可增加如下参数
bash tools/prepare_env.sh --type=cann --device-type=a2 --with-install-driver=true
```

| 参数                    | 类型   | 是否必须 | 说明                                       |
|:----------------------|:-----|:-----|:-----------------------------------------|
| --type                | str  | 是    | 脚本安装类型，可选：cann, third_party, all |
| --device-type         | str  | 是    | 指定NPU型号，可选：a2, a3              |
| --install-path        | str  | 否    | 指定CANN包安装路径                            |
| --download-path       | str  | 否    | 指定CANN包以及三方依赖包下载路径                     |
| --with-install-driver | bool | 否    | 指定是否额外下载NPU驱动和固件安装包，默认为false；不会自动执行驱动和固件安装 |
| --help                | -    | 否    | 查看命令参数帮助信息                               |

##### 手动安装

若自动安装不适用，可按以下步骤逐项手动完成安装。

1. **安装CANN包**

    请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

    - 安装CANN toolkit包

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

    - 安装CANN ops包（运行态依赖）

        ops包是运行态依赖，若仅编译算子，可不安装此包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        - \$\{cann\_version\}：表示CANN包版本号。
        - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
        - \$\{soc\_name\}：表示NPU型号名称。
        - \$\{install\_path\}：表示指定安装路径，ops包需与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

2. **获取pto-isa源码**

    > 方法一：安装CANN pto-isa包
    > 根据实际环境下载对应的安装包，下载链接如下(如果浏览器不支持自动下载，请选择右键，"链接另存为...")：
    > - x86：[cann-pto-isa_linux-x86_64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-x86_64.run)
    > - aarch64：[cann-pto-isa_linux-aarch64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run)
    >
    > ```bash
    > # 安装命令
    > bash ./cann-pto-isa_linux-*.run --full
    > ```
    >
    > 方法二：下载源码方式
    >
    > ```bash
    > # 创建用于存放第三方开源软件源码包的目录path-to-your-pto-isa
    > mkdir -p ${path-to-your-pto-isa}
    > git clone https://gitcode.com/cann/pto-isa.git
    > # 设置环境变量
    > export PTO_TILE_LIB_CODE_PATH="${path-to-your-pto-isa}/pto-isa"
    > # 检查目录是否存在
    > ls ${PTO_TILE_LIB_CODE_PATH}/include/pto/
    > ```
    >
    > - \$\{path-to-your-pto-isa\}：存放pto-isa源码的路径。

#### 环境变量配置

安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定路径安装
source ${install_path}/ascend-toolkit/set_env.sh
```

### 方式3：Docker安装

对于有昇腾设备的开发者，若您想快速搭建昇腾环境，可使用Docker镜像部署。

> **说明**：
>
> - 使用Docker前，请务必**完成宿主机NPU驱动和固件安装**，请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"安装NPU驱动和固件"章节。
> - 建议Docker版本：**v27.2.1及以上**。
> - 镜像文件比较大，下载需要一定时间，请您耐心等待。关于docker命令的选项介绍可通过`docker --help`查询。

当前提供两类Dockerfile，均安装PyPTO运行所需依赖，区别在于是否在镜像内预装CANN包：

| Dockerfile版本 | 说明 |
|:---|:---|
| 版本1：安装CANN包 | 基于Ascend CANN基础镜像，已预集成CANN包。适用于希望开箱即用的场景。 |
| 版本2：不安装CANN包 | 仅安装基础依赖，CANN包需在容器内单独安装。适用于需要自定义CANN版本的场景。 |

#### 版本1：安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler24.03，架构支持x86_64和aarch64，Python 3.11，CANN 8.5.0，支持A2/A3。

在使用前，请根据**操作系统 + 硬件类型**指定`CANN_VERSION`：

- **Ubuntu + A3**：`ARG CANN_VERSION=8.5.0-a3-ubuntu22.04-py3.11`
- **Ubuntu + A2**：`ARG CANN_VERSION=8.5.0-910b-ubuntu22.04-py3.11`
- **openEuler + A3**：`ARG CANN_VERSION=8.5.0-a3-openeuler24.03-py3.11`
- **openEuler + A2**：`ARG CANN_VERSION=8.5.0-910b-openeuler24.03-py3.11`

根据CPU架构指定`TARGETPLATFORM`：

- **x86_64**：`ARG TARGETPLATFORM=linux/amd64`
- **aarch64**：`ARG TARGETPLATFORM=linux/arm64`

> **说明**：若上述信息与实际硬件及驱动不匹配，将导致CANN包安装失败，从而导致镜像构建失败。

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
#     CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/CANN_pypto/ascend-toolkit/set_env.sh" && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
#     echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc
#
# ENTRYPOINT ["/bin/bash", "-c", "\
#     source /usr/local/Ascend/CANN_pypto/ascend-toolkit/set_env.sh && \
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
    echo "cann-pto-isa installation completed."

# step5: [Optional] 设置默认代理（仅当需要统一代理时启用）
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考Ascend社区提供的基础镜像：[https://quay.io/repository/ascend/cann](https://quay.io/repository/ascend/cann)

#### 版本2：不安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler22.03，架构支持x86_64和aarch64，Python 3.11，支持A2/A3。

根据操作系统指定`PY_VERSION`：

- **Ubuntu 22.04**：`ARG PY_VERSION=3.11-ubuntu22.04`
- **openEuler 22.03**：`ARG PY_VERSION=3.11-openeuler22.03`

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

# 安装 Python 依赖
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 升级 setuptools，满足 pypto 要求
RUN pip install --no-cache-dir --upgrade setuptools

# 安装 torch / torch-npu
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0

# [Optional] 设置默认代理，便于容器内访问外网
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他Python/OS组合的镜像，可参考：[https://quay.io/repository/ascend/python](https://quay.io/repository/ascend/python)

> **说明**：PyPTO考虑到会在国内外都有部署使用，因此提供的参考Dockerfile基于国内外更常用的quay.io。若在国内存在访问quay.io较慢的情况，可通过配置Docker代理解决：
>
> ```bash
> # 配置信任证书
> mkdir -p /etc/systemd/system/docker.service.d/
> tee -a /etc/docker/daemon.json > /dev/null << 'EOF'
> {
>   "insecure-registries":["quay.io", "cdn01.quay.io"]
> }
> EOF
>
> # 配置Docker代理
> tee -a /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null << 'EOF'
> [Service]
> Environment="HTTP_PROXY=<代理地址>"
> Environment="HTTPS_PROXY=<代理地址>"
> EOF
>
> systemctl daemon-reexec
> systemctl daemon-reload
> systemctl restart docker.service
> ```

#### 构建镜像

在本地准备好对应版本的Dockerfile（例如保存为`Dockerfile`），执行镜像构建命令：

```bash
docker build -t <镜像名:版本> -f ./Dockerfile .
# 示例：
# docker build -t pyptox86/a3:latest -f ./Dockerfile .
```

#### 创建并启动容器

仅有镜像无法直接作为开发环境使用，需要基于该镜像创建容器。为确保容器能够正确访问NPU硬件和相关驱动，需在启动时映射宿主机设备和驱动目录。示例命令如下：

```bash
sudo docker run -u root -itd --name <容器名> --ipc=host --net=host --privileged \
    --device=/dev/davinci0 \
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

| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `-u root` | 以root用户运行容器。 | - |
| `--name <容器名>` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--ipc=host` | 使用宿主机的IPC命名空间，便于进程间通信。 | - |
| `--net=host` | 使用宿主机网络，便于网络访问。 | - |
| `--privileged` | 赋予容器特权模式，确保设备访问权限。 | - |
| `--device=/dev/davinci0` | 将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行`npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。如需映射多张卡，增加多个`--device`参数即可。|
| `--device=/dev/davinci_manager` | 映射NPU设备管理接口。 | - |
| `--device=/dev/devmm_svm` | 映射设备内存管理接口。 | - |
| `--device=/dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro` | 挂载宿主机的NPU驱动库到容器内（只读）。 | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info:ro` | 挂载CANN软件安装信息文件（只读）。 | - |
| `-w /mount_home` | 指定容器内工作目录。 | - |
| `-itd` | `-i`（交互式）、`-t`（分配伪终端）、`-d`（后台运行）的组合参数。 | - |
| `<镜像名:版本>` | 指定要运行的Docker镜像。 | 请确保与`docker build`时指定的镜像名和标签一致。 |
| `/bin/bash` | 容器启动后立即执行的命令。 | - |

启动并进入容器：

```bash
# 启动容器
docker start <容器名>

# 进入容器
docker exec -it <容器名> /bin/bash
```

> **说明**：出于兼容性考虑，当前Docker环境中编译构建得到的`whl`包建议仅在对应Docker容器内使用。
## 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

- **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN版本**

    ```bash
    # 查看CANN toolkit包版本信息（默认路径安装），WebIDE场景下将/usr/local替换为/home/developer
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # 查看CANN ops包版本信息（默认路径安装），WebIDE场景下将/usr/local替换为/home/developer
    cat /usr/
环境准备完成后，请参考[PyPTO安装](#pypto安装)章节完成PyPTO的安装。
## 可选安装

### PyPTO Toolkit插件

如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：

1. 单击[Link](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/pypto-toolkit-1.1.0.vsix)，下载`.vsix`插件文件。

2. 打开Visual Studio Code，进入"扩展"选项卡界面，单击右上角的"..."，选择"从VSIX安装..."。
 ![vscode_install](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/images/vscode_install.png)

3. 选择已下载的`.vsix`插件文件，完成安装。

### MPI依赖

PyPTO的分布式用例依赖MPI，推荐版本 >= 3.2.1。

**方式一：通过系统包管理器安装**

```bash
# Ubuntu/Debian系统
apt-get update && apt-get install -y mpich

# CentOS/RHEL系统
yum install -y mpich
```

**方式二：通过源码编译安装**

```bash
# 以3.2.1版本为例
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

安装完成后设置环境变量：

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

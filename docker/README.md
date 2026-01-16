说明：本文描述如何快速创建运行PyPTO的docker容器，在使用docker容器前请**完成主机NPU硬件部署、NPU驱动及固件安装**，参考文档[Environment README](../docs/context/prepare_environment.md)。docker推荐v27.2.1及以上。

## 版本说明

当前提供两类dockerfile，第一类是完成cann包环境安装的版本，第二类是不涉及cann包环境安装的版本。两类版本都安装了Pypto运行所依赖的软件包。

### 版本1：安装cann包的dockerfile

当前示例dockerfile构建镜像支持的环境信息如下：

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

示例dockerfile基于ubuntu操作系统进行编写，不同操作系统略有差异请根据实际使用进行调整。
使用前请根据操作系统及硬件类型指定 CANN_VERSION:<br>
Ubuntu+A3 :ARG CANN_VERSION=8.5.0.alpha001-a3-ubuntu22.04-py3.11;	<br>
Ubuntu+A2 :ARG CANN_VERSION=8.5.0.alpha001-910b-ubuntu22.04-py3.11;<br>
openEuler+A2 :ARG CANN_VERSION=8.5.0.alpha001-910b-openeuler24.03-py3.11;<br>
根据CPU架构指定 TARGETPLATFORM：<br>
x86_64: ARG TARGETPLATFORM=linux/amd64;<br>
aarch64:ARG TARGETPLATFORM=linux/arm64;<br>
<span style="font-size:12px;">*(若指定信息与硬件驱动不匹配，会导致CANN包安装失败，导致镜像无法构建）*</span><br>
示例dockerfile内容如下：

```
# step1 check your version
ARG CANN_VERSION=8.5.0.alpha002-a3-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:$CANN_VERSION
ARG TARGETPLATFORM=linux/arm64
   # [Optional] set proxy
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1
   
WORKDIR /tmp
# step2 extra utils, for PyPTO project
RUN pip install --no-cache-dir \
    wheel tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado
RUN pip install --no-cache-dir --upgrade \
    setuptools
   # pypto wants `setuptools>=77.0.3`
   #set  torch-npu&npu version
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0
# step3 install cann packages provided by pypto
WORKDIR /mount_home
RUN git clone https://gitcode.com/cann/pypto.git
WORKDIR /mount_home/pypto
ARG CANN_VERSION
RUN if echo "${CANN_VERSION}" | grep -iq "910b"; then \
        DEVICE_TYPE="a2"; \
    elif echo "${CANN_VERSION}" | grep -iq "a3"; then \
        DEVICE_TYPE="a3"; \
    else \
        echo "ERROR: Unsupported CANN_VERSION format: ${CANN_VERSION}" 1>&2 && \
        echo "Version should contain '910b' or 'a3' (case-insensitive)" 1>&2 && \
        exit 1; \
    fi && \
    echo "DEVICE_TYPE=${DEVICE_TYPE}" && \
    chmod +x tools/prepare_env.sh && \
    bash tools/prepare_env.sh --type=cann --device-type=${DEVICE_TYPE} --install-path=/usr/local/Ascend/CANN_pypto --quiet
# Note: Set environment variables
RUN \
    CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/CANN_pypto/ascend-toolkit/latest/set_env.sh" && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc
ENTRYPOINT ["/bin/bash", "-c", "\
    source /usr/local/Ascend/CANN_pypto/ascend-toolkit/latest/set_env.sh && \
    exec \"$@\"", "--"]
#step 4 [Optional] set default proxy
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考[https://quay.io/repository/ascend/cann](https://quay.io/repository/ascend/cann)，Ascend社区提供了丰富的基础镜像。

### 版本2：不安装cann包的dockerfile

支持的镜像信息如下：

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

dockerfile内容如下：<br>
使用Ubuntu22.04 : ARG PY_VERSION=3.11-ubuntu22.04<br>
使用openeuler ：ARG PY_VERSION=3.11-openeuler22.03

```
ARG PY_VERSION=3.11-ubuntu22.04
FROM quay.io/ascend/python:$PY_VERSION

# [Optional] set proxy
 ARG PROXY=""
 ENV https_proxy=$PROXY
 ENV http_proxy=$PROXY
 ENV GIT_SSL_NO_VERIFY=1
# install system dependencies and clean index cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim wget curl unzip tar lcov openssl ca-certificates\
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev unzip pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache python-is-python3 python3-pip python3-venv ninja-build python3-dev \
    && rm -rf /var/lib/apt/list/*     # clean apt index cache
# # [Optional] set pip proxy
# RUN pip config set global.index-url http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

RUN pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py

RUN pip install --no-cache-dir \
    wheel tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado
RUN pip install --no-cache-dir --upgrade \
    setuptools
# pypto wants `setuptools>=77.0.3`
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.6.0

# [Optional] set default proxy
 ENV PROXY=""
 ENV https_proxy=$PROXY
 ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考[https://quay.io/repository/ascend/python](https://quay.io/repository/ascend/python)，Ascend社区提供了丰富的基础镜像。

## 使用指导

### 构建镜像

这步是基于本地创建的dockerfile，构建docker镜像。构建镜像的命令如下：

```
docker build -t <镜像名：版本> -f ./dockerfile .
exp:  docker build -t pyptox86/a3:latest -f /home/dockerfiles/Cann83rc2/dockerfile .
```

### 构建容器

上一步骤构建的镜像不能直接作为开发环境使用，需基于该镜像生成使用的容器，构建容器的命令如下：

```
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
<镜像名：版本> \
/bin/bash
```

exp:

```
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

### 启动容器

```
docker start <容器名>
docker exec -it <容器名>  /bin/bash
```

exp:

```
docker start pypto_x86a3
docker exec -it pypto_x86a3 /bin/bash
```

进入容器拉取代码：
git clone [https://gitcode.com/cann/pypto.git](https://gitcode.com/cann/pypto.git)。<br>
基于源码编译与安装pypto.whl包：
进入Pypto源码目录，执行命令python3 -m pip install -e . --verbose进行安装。使用时根据需要进行修改适配<br>
完成以上命令后，即可运行Pypto相关用例。<br>
*<span style="font-size:12px;">注：考虑兼容性问题，当前docker环境编译构建的whl包建议仅在docker容器内使用。</span>*

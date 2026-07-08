# Docker环境部署

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

## 版本1：安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler24.03，架构支持x86_64和aarch64，Python 3.11，CANN 9.0.0，支持 Ascend 950PR、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品。

在使用前，请根据**操作系统 + 硬件类型**指定`CANN_VERSION`：

- **Ubuntu + A3**：`ARG CANN_VERSION=9.0.0-a3-ubuntu22.04-py3.11`
- **Ubuntu + A2**：`ARG CANN_VERSION=9.0.0-910-ubuntu22.04-py3.11`
- **openEuler + A3**：`ARG CANN_VERSION=9.0.0-a3-openeuler24.03-py3.11`
- **openEuler + A2**：`ARG CANN_VERSION=9.0.0-910-openeuler24.03-py3.11`

根据CPU架构指定`TARGETPLATFORM`：

- **x86_64**：`ARG TARGETPLATFORM=linux/amd64`
- **aarch64**：`ARG TARGETPLATFORM=linux/arm64`

> **说明**：若上述信息与实际硬件及驱动不匹配，将导致CANN包安装失败，从而导致镜像构建失败。

```dockerfile
# step1: 指定CANN基础镜像版本
ARG CANN_VERSION=9.0.0-a3-ubuntu22.04-py3.11
FROM quay.io/ascend/cann:$CANN_VERSION

# 指定目标平台架构
ARG TARGETPLATFORM=linux/amd64

# [Optional] 设置HTTP/HTTPS代理（按需配置）
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 工作目录
WORKDIR /tmp

# step2: 安装PyPTO项目构建/运行所需依赖
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

# 安装指定版本torch / TorchNPU（CPU源 + NPU插件）
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir torch-npu==2.8.0.post4

# step3: [Optional] 设置默认代理（仅当需要统一代理时启用）
ENV PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
```

若希望构建其他环境版本的镜像，可参考Ascend社区提供的基础镜像：[https://quay.io/repository/ascend/cann](https://quay.io/repository/ascend/cann)

## 版本2：不安装CANN包的Dockerfile

支持环境信息：OS支持Ubuntu22.04、OpenEuler22.03，架构支持x86_64和aarch64，Python 3.11，支持 Ascend 950PR、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品。

根据操作系统指定`PY_VERSION`：

- **Ubuntu 22.04**：`ARG PY_VERSION=3.11-ubuntu22.04`
- **openEuler 22.03**：`ARG PY_VERSION=3.11-openeuler22.03`

```dockerfile
ARG PY_VERSION=3.11-ubuntu22.04
FROM quay.io/ascend/python:$PY_VERSION

# [Optional] 设置HTTP/HTTPS代理
ARG PROXY=""
ENV https_proxy=$PROXY
ENV http_proxy=$PROXY
ENV GIT_SSL_NO_VERIFY=1

# 安装系统依赖并清理APT缓存索引
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gdb gawk wget curl tar lcov openssl ca-certificates \
    gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev \
    libssl-dev libffi-dev libbz2-dev libxslt1-dev pciutils \
    net-tools openssh-client libblas-dev gfortran libblas3 llvm ccache \
    python-is-python3 python3-pip python3-venv ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil>=5.9.0 protobuf scipy requests absl-py \
    tomli pybind11 pybind11-stubgen pytest pytest-forked pytest-xdist \
    tabulate pandas matplotlib build ml_dtypes jinja2 cloudpickle tornado

# 升级setuptools，满足pypto要求
RUN pip install --no-cache-dir --upgrade setuptools

# 安装torch / TorchNPU
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-npu==2.8.0.post4

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

## 构建镜像

在本地准备好对应版本的Dockerfile（例如保存为`Dockerfile`），执行镜像构建命令：

```bash
docker build -t <镜像名:版本> -f ./Dockerfile .
# 示例：
# docker build -t pyptox86/a3:latest -f ./Dockerfile .
```

## 创建并启动容器

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

# 环境部署

在使用PyPTO开发或运行算子之前，请您先参考下面步骤完成基础环境搭建。完成后请继续参考[PyPTO安装](./build_and_install.md)文档进行安装。

PyPTO支持在具备NPU硬件的**真实环境**和仅有CPU硬件的**仿真环境**中运行：

| 环境类型 | 硬件要求 | 运行模式 |
|:-----|:------|:------|
| 真实环境 | 配备CPU及NPU硬件 | 支持在NPU上执行计算，也可以通过CPU仿真获取预估性能和执行计算 |
| 仿真环境 | 仅有CPU硬件 | 支持通过CPU仿真，获取预估性能和执行计算 |

**说明:**

- NPU：指昇腾AI处理器，目前仅支持如下产品型号：
    - Ascend 950PR
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
| Docker | Docker镜像是一种高效部署方式，已预集成运行所需依赖。<br>当前支持Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品、暂不支持950PR, OS支持Ubuntu和OpenEuler。 | 适用于有昇腾设备，需要快速搭建环境的开发者。 |

### 方式1：WebIDE环境

对于无昇腾设备的开发者，可直接使用WebIDE开发平台，即"**算子一站式开发平台**"，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新商发版CANN包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考[CANN介绍](https://gitcode.com/org/cann/discussions/54)。

#### 1-进入开源项目

单击"`云开发`"按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

 ![创建云开发环境](../tutorials/figures/webide1.png)

#### 2-连接WebIDE

根据页面提示信息创建并启动云开发环境，单击"`连接 > WebIDE`"进入算子一站式开发平台，开源项目的源码资源默认在`/mnt/workspace`目录下。

 ![启动并连接WebIDE](../tutorials/figures/webide2.png)

#### 3-安装pto-isa

**方法一：基于run包安装**

> **说明**：若后续考虑通过PyPI方式安装PyPTO，不需要单独安装`pto-isa`。`pto-isa`版本已与CANN包版本匹配，并在安装CANN包时完成安装，可跳过本章节。

根据实际环境下载对应的安装包，下载链接如下（如果浏览器不支持自动下载，请选择右键，"链接另存为..."）：

- x86：[cann-pto-isa_linux-x86_64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-x86_64.run)
- aarch64：[cann-pto-isa_linux-aarch64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run)

```bash
# 安装命令
bash ./cann-pto-isa_linux-*.run --full
```

**方法二：基于源码安装**

```bash
# 创建用于存放第三方开源软件源码包的目录path-to-your-pto-isa
mkdir -p ${path-to-your-pto-isa}
git clone https://gitcode.com/cann/pto-isa.git
# 设置环境变量
export PTO_TILE_LIB_CODE_PATH="${path-to-your-pto-isa}/pto-isa"
# 检查目录是否存在
ls ${PTO_TILE_LIB_CODE_PATH}/include/pto/
```

- \$\{path-to-your-pto-isa\}：存放`pto-isa`源码的路径。

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

    若计划在真实NPU环境中运行PyPTO，必须先完成驱动与固件安装，并确保Ascend HDK版本为25.5.0及以上。详细指导请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"安装NPU驱动和固件"章节。

    > **重要**：
    >
    > - 支持版本：Ascend HDK 25.5.0及以上。
    > - 低于支持版本的HDK环境不在PyPTO验证和支持范围内，运行异常算子时可能导致NPU状态异常，进而影响后续算子执行，出现AIC超时等问题；严重情况下可能需要重启设备或主机后恢复。
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
        - 请根据实际环境的Python版本单独安装，请参考[Ascend Extension for PyTorch文档中心](https://hiascend.com/document/redirect/pytorchuserguide)中的《软件安装》手册。
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

    # 下载JSON for Modern C++ 三方库
    wget https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz

    # 下载libboundscheck三方库
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
>
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
# 自动安装CANN toolkit / ops / pto-isa
bash tools/prepare_env.sh --type=cann --device-type=a2

# 如需额外下载驱动与固件安装包，可增加如下参数
bash tools/prepare_env.sh --type=cann --device-type=a2 --with-install-driver=true
```

| 参数                    | 类型   | 是否必须 | 说明                                       |
|:----------------------|:-----|:-----|:-----------------------------------------|
| --type                | str  | 是    | 脚本安装类型，可选：cann, all |
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

2. **安装pto-isa**

    具体方法请参考[安装pto-isa](#3-安装pto-isa)。

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

Docker安装相关内容请参考：

- [Docker环境部署](./docker_environment.md)

在阅读Docker文档前，请先确保已完成宿主机NPU驱动和固件安装。

`pto-isa`安装具体方法请参考[安装pto-isa](#3-安装pto-isa)。

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

环境准备完成后，请参考[PyPTO安装](./build_and_install.md)文档完成PyPTO的安装。

## 可选安装

### PyPTO Toolkit插件

如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：

1. 单击[PyPTO_Toolkit](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/pypto-toolkit-1.1.0.vsix)，下载`.vsix`插件文件。

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

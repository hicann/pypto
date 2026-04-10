# 环境准备

PyPTO支持在具备NPU硬件的**真实环境**和仅有CPU硬件的**仿真环境**中运行，具体对比如下:

| 环境类型 | 硬件要求                   | 运行模式                                     |
|:-----|:-----------------------|:-----------------------------------------|
| 真实环境 | 配备CPU及NPU硬件 | 支持在NPU上执行计算，也可以通过CPU仿真获取预估性能和执行计算 |
| 仿真环境 | 仅有CPU硬件      | 支持通过CPU仿真，获取预估性能和执行计算                    |


**说明:**
- NPU：指昇腾AI处理器，目前仅支持如下产品型号：

    - Atlas A3 训练系列产品/Atlas A3 推理系列产品
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品
- 支持的系统：PyPTO支持在OpenEuler、Ubuntu等主流Linux发行版上编译和运行

## 前提条件

在使用PyPTO前，请确保已安装基础依赖。

1. **安装Python依赖**

    - Python：版本 >= 3.9
        - **重要**：若后续需要通过源码编译安装PyPTO，还需安装 Python 的 Development 组件（常称为 `python3-dev`）。

    - 安装Python依赖包：

        依赖的pip包及对应版本在`python/requirements.txt`中描述，可以使用如下命令完成安装：

        ```bash
        # 进入pypto项目源码根目录
        cd pypto

        # 安装相关pip包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - PyTorch及Ascend Extension for PyTorch：
        - 请根据实际环境的Python版本单独安装，参考[Ascend Extension for PyTorch安装说明](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)。
        - **重要**：需确保`PyTorch`、`Ascend Extension for PyTorch` 与`PyPTO`三者的Python版本一致。
        - **仿真环境说明**：在仿真环境中可跳过`Ascend Extension for PyTorch`的安装，但仍需安装`PyTorch`。
        - **顺序说明**：请参考下文 “安装工具包” 章节完成对应工具包安装后，再安装 `Ascend Extension for PyTorch`。

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
    >
    > ```bash
    > # 创建并进入用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    > mkdir -p <path-to-your-thirdparty> && cd $_
    >
    > # 下载 JSON for Modern C++ 三方库
    > wget https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz
    >
    > # 下载 libboundscheck 三方库
    > wget https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz
    > ```

    方法二：通过辅助脚本下载
    >
    > ```bash
    > # 创建用于存放第三方开源软件源码包的目录path-to-your-thirdparty
    > mkdir -p <path-to-your-thirdparty>
    >
    > # 执行辅助脚本
    > # 如果未指定`--download-path`参数，脚本会将所需三方依赖下载到pypto同级目录的`pypto_download/third_party_packages`路径下
    > # 如果指定了`--download-path`参数，脚本会将所需三方依赖下载到`path-to-your-thirdparty/third_party_packages`路径下
    > bash tools/prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty]
    > ```

## 软件包安装

> PyPTO支持两种仿真模式：
>
> - **性能仿真**：仅评估程序运行性能，无需安装CANN、NPU驱动与固件。
> - **精度仿真**：模拟真实NPU的执行逻辑，获取运算结果，必须依赖CANN工具包。
>
> 因此：
> - 若仅编译和运行PyPTO**性能仿真**，可跳过本节。
> - 若需编译和运行**精度仿真**，或计划在**真实NPU环境**中编译运行PyPTO并使用其在NPU上执行计算的能力时，必须安装如下软件包：

### 安装驱动与固件

   详细安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

    CANN-8.5.0 社区版:

    - 推荐版本：Ascend HDK 25.5.0
    - 支持版本：Ascend HDK 25.5.0

    CANN-8.5.0 商发版:

    - 推荐版本：Ascend HDK 25.5.0
    - 支持版本：Ascend HDK 25.5.0、Ascend HDK 25.3.0、Ascend HDK 25.2.0


### 安装工具包

工具包的安装提供了脚本安装和手动安装两种方式, 工具包需安装在同一路径下。

#### 使用安装脚本

toolkit包、ops包、PTO-inst包的下载与安装可通过项目tools目录下prepare_env.sh一键执行，命令如下，若遇到不支持系统，请参考该文件自行适配

```
bash tools/prepare_env.sh --type=cann --device-type=a2
```

| 全写                    | 类型   | 是否必须 | 说明                                       |
|:----------------------|:-----|:-----|:-----------------------------------------|
| --type                | str  | 是    | 脚本安装类型，可选：deps, cann, third_party, all |
| --device-type         | str  | 是    | 指定 NPU 型号，可选：a2, a3              |
| --install-path        | str  | 否    | 指定 CANN 包安装路径                            |
| --download-path       | str  | 否    | 指定 CANN 包以及三方依赖包下载路径                     |
| --with-install-driver | bool | 否    | 指定是否下载 NPU 驱动和固件包，默认为 false             |
| --help                | -    | 否    | 查看命令参数帮助信息                               |

#### 手动安装

1. **安装CANN toolkit包**

    根据实际环境下载对应的安装包，下载链接如下:
    - x86：[Ascend-cann-toolkit_8.5.0_linux-x86_64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/x86_64/Ascend-cann-toolkit_8.5.0_linux-x86_64.run)
    - aarch64：[Ascend-cann-toolkit_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/aarch64/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)

    ```bash
    # 安装命令
    bash ./Ascend-cann-toolkit_8.5.0_linux-*.run --install --force
    ```

2. **安装CANN ops包**

    根据实际环境和硬件类型(支持A2/A3)，下载对应的安装包，下载链接如下：
    - A2、x86：[CANN_A2-OPS-8.5.0.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/x86_64/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run)
    - A2、aarch64：[CANN_A2-OPS-8.5.0.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/aarch64/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run)
    - A3、x86：[CANN_A3-OPS-8.5.0.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/x86_64/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run)
    - A3、aarch64：[CANN_A3-OPS-8.5.0.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0_release/aarch64/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run)

    ```bash
    # 安装命令
    bash ./Ascend-cann-*-ops_8.5.0_linux-*.run --install --force
    ```

3. **获取pto-isa源码**

    > 方法一：安装CANN pto-isa包
    > 根据实际环境下载对应的安装包，下载链接如下(如果浏览器不支持自动下载，请选择右键，"链接另存为...")：
    > - x86：[cann-pto-isa_linux-x86_64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-x86_64.run)
    > - aarch64：[cann-pto-isa_linux-aarch64.run](https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run)

    > ```bash
    > # 安装命令
    > bash ./cann-pto-isa_linux-*.run --full
    > ```

    > 方法二：下载源码方式

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

## 安装PyPTO Toolkit插件（可选）

 如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：

 1. 单击[Link](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/pypto-toolkit-1.1.0.vsix)，下载.vsix插件文件。

 2. 打开Visual Studio Code，进入“扩展”选项卡界面，单击右上角的“...”，选择“从VSIX安装...”。
  ![vscode_install](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/images/vscode_install.png)

 3. 选择已下载的.vsix插件文件，完成安装。

## 安装MPI依赖（可选）

  PyPTO的分布式用例依赖MPI，推荐版本 >= 3.2.1：

 ### 软件包安装

    ```bash
    # 以3.2.1版本为例
    version='3.2.1'
    wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
    tar -xzf mpich-${version}.tar.gz
    cd mpich-${version}
    ./configure --prefix=/usr/local/mpich --disable-fortran
    make && make install
    ```

 ### 设置环境变量

    ```bash
    export MPI_HOME=/usr/local/mpich
    export PATH=${MPI_HOME}/bin:${PATH}
    ```

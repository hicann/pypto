# 环境准备

PyPTO 支持在具备 Ascend-NPU 硬件的**真实环境**和仅有 CPU 硬件的**仿真环境**中运行, 具体对比如下:

| 环境类型 | 硬件要求                   | 运行模式                                     |
|:-----|:-----------------------|:-----------------------------------------|
| 真实环境 | 配备 CPU 及 Ascend-NPU 硬件 | 支持在 Ascend-NPU 上执行计算, 也可以通过 CPU 仿真获取预估性能 |
| 仿真环境 | 仅有 CPU 硬件              | 仅支持通过 CPU 仿真获取预估性能                       |

**说明:**
- Ascend-NPU: 指 AscendAI 加速器（目前支持 A2/A3）
- 支持的系统: PyPTO 支持在 OpenEuler、Ubuntu 等主流 Linux 发行版上编译和运行

## 前提条件

在使用 PyPTO 前, 请确保已安装以基础依赖.

1. **安装 Python 依赖**

    - Python: 版本 >= 3.9
    - PyTorch 及 Ascend Extension for PyTorch
        - 请根据实际环境的 Python 版本单独安装, 参考 [Ascend Extension for PyTorch 安装说明](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)
        - **重要**: 需确保 `PyTorch`、`Ascend Extension for PyTorch` 与 `PyPTO` 三者的 Python 版本一致
        - **仿真环境说明**: 在仿真环境中可跳过 `Ascend Extension for PyTorch` 的安装, 但仍需安装 `PyTorch`

2. **安装编译依赖**

    若不需要编译 PyPTO, 可跳过本步骤.

    **安装编译工具:**

    - cmake >= 3.16.3
    - make
    - gcc >= 7.3.1

    **安装 Python 依赖包:**

    依赖的 pip 包及对应版本在 `python/requirements.txt` 中描述, 可以使用如下命令完成安装:

    ```bash
    # 进入 pypto 项目源码根目录
    cd pypto

    # 安装相关 pip 包依赖
    python3 -m pip install -r python/requirements.txt
    ```

    **准备第三方开源软件源码包**

    PyPTO 编译过程依赖以下第三方开源软件源码包, 若您的环境可正常访问 [cann-src-third-party](https://gitcode.com/cann-src-third-party),
    这些软件的源码包会在编译时自动下载和编译, 否则请手动准备:

    | 软件包                 | 版本      | 下载地址                                                                                                                    |
    |:--------------------|:--------|:------------------------------------------------------------------------------------------------------------------------|
    | JSON for Modern C++ | v3.11.3 | [下载链接](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/third_party_deps/json-3.11.3.tar.gz)                      |
    | libboundscheck      | v1.1.16 | [下载链接](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/third_party_deps/libboundscheck-v1.1.16.tar.gz) |

    手工准备第三方开源源码包的方法:

    > 方法一: 手工下载
    >
    > ```bash
    > # 创建用于存放第三方开源软件源码包的目录 path-to-your-thirdparty
    > mkdir -p <path-to-your-thirdparty>
    >
    > # 将上述三方库源码压缩包, 下载到本地并上传到开发环境对应的 `path-to-your-thirdparty` 目录中
    > ```

    > 方法二: 通过辅助脚本下载
    >
    > ```bash
    > # 创建用于存放第三方开源软件源码包的目录 path-to-your-thirdparty
    > mkdir -p <path-to-your-thirdparty>
    >
    > # 执行辅助脚本
    > # 如果未指定 `--download-path` 参数, 脚本会将所需三方依赖下载到 pypto 同级目录的 `pypto_download/third_party_packages` 路径下
    > # 如果指定了 `--download-path` 参数, 脚本会将所需三方依赖下载到 `path-to-your-thirdparty/third_party_packages` 路径下
    > bash tools/prepare_env.sh --type=third_party [--download-path=path-to-your-thirdparty]
    > ```

## 软件包安装

若仅在**仿真环境**中编译和运行 PyPTO , 可跳过本节.
在**真实环境**中编译运行 PyPTO 并使用其在 Ascend-NPU 上执行计算的能力时, 必须安装如下软件包:

1. **安装驱动与固件**

   详细安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》.

    - 推荐版本: Ascend HDK 25.3.0
    - 支持版本: Ascend HDK 25.3.0、Ascend HDK 25.2.0

2. **安装CANN toolkit包**

    根据实际环境下载对应的安装包, 下载链接如下:
    - x86: [Ascend-cann-toolkit_8.5.0_linux-x86_64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/x86/Ascend-cann-toolkit_8.5.0_linux-x86_64.run)
    - aarch64: [Ascend-cann-toolkit_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/aarch64/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)

    ```bash
    # 确保安装包有可执行权限
    chmod +x Ascend-cann-toolkit_8.5.0_linux-${arch}.run

    # 安装命令
    ./Ascend-cann-toolkit_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
    ```

    **参数说明**:
    - \$\{arch\}: 表示CPU架构, 如aarch64、x86_64.
    - \$\{install\_path\}: 表示指定安装路径, 默认安装在`/usr/local/Ascend`目录.

3. **安装CANN ops包**

    根据实际环境和硬件类型(支持A2/A3), 下载对应的安装包, 下载链接如下:
    - A2、x86: [CANN_A2-OPS-8.5.0.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/x86/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run)
    - A2、aarch64: [CANN_A2-OPS-8.5.0.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/aarch64/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run)
    - A3、x86: [CANN_A3-OPS-8.5.0.x86](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/x86/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run)
    - A3、aarch64: [CANN_A3-OPS-8.5.0.aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/8.5.0/aarch64/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run)
    ```
    # 确保安装包有可执行权限
    chmod +x Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
    ```

    - \$\{device_type\}: NPU 型号, 当前支持 A2、A3.
    - \$\{arch\}: CPU架构, 如aarch64、x86_64.
    - \$\{install-path\}: 表示制定安装路径, 默认安装在`/usr/local/Ascend`目录.

4. **安装CANN pto-isa包**

    根据实际环境下载对应的安装包, 下载链接如下:
    - x86: [cann-pto-isa_8.5.0_linux-x86_64.run](http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/202601/20260112/ubuntu_x86/cann-pto-isa_8.5.0_linux-x86_64.run)
    - aarch64: [cann-pto-isa_8.5.0_linux-aarch64.run](http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/202601/20260112/ubuntu_aarch64/cann-pto-isa_8.5.0_linux-aarch64.run)
    ```
    # 确保安装包有可执行权限
    chmod +x cann-pto-isa_8.5.0_linux-${arch}.run
    # 安装命令
    ./cann-pto-isa_8.5.0_linux-${arch}.run --full --install-path=${install_path}
    ```

    - \$\{arch\}: CPU架构, 如aarch64、x86_64.
    - \$\{install-path\}: 表示制定安装路径, 默认安装在`/usr/local/Ascend`目录.

5. **环境变量配置**

    ```bash
    # 默认路径安装, 以root用户为例（非root用户, 将/usr/local替换为${HOME}）
    source /usr/local/Ascend/cann/set_env.sh

    # 指定路径安装
    source ${install_path}/cann/set_env.sh
    ```

### 安装脚本

上述流程2 ~ 4中, toolkit包、ops包、PTO-inst包的下载与安装可通过项目tools目录下prepare_env.sh一键执行, 命令如下, 若遇到不支持系统, 请参考该文件自行适配
```
bash tools/prepare_env.sh --type=cann --device-type=a2
```
| 全写                    | 类型   | 是否必须 | 说明                                       |
|:----------------------|:-----|:-----|:-----------------------------------------|
| --type                | str  | 是    | 脚本安装类型, 可选[deps, cann, third_party, all] |
| --device-type         | str  | 是    | 指定 NPU 型号, 可选 [a2, a3]               |
| --install-path        | str  | 否    | 指定 CANN 包安装路径                            |
| --download-path       | str  | 否    | 指定 CANN 包以及三方依赖包下载路径                     |
| --with-install-driver | bool | 否    | 指定是否下载 NPU 驱动和固件包, 默认为 false             |
| --help                | -    | 否    | 查看命令参数帮助信息                               |
  
## 安装PyPTO Toolkit插件（可选）

 如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：

 1. 单击[Link](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/pypto-toolkit-1.1.0.vsix)，下载.vsix插件文件。

 2. 打开Visual Studio Code，进入“扩展”选项卡界面，单击右上角的“...”，选择“从VSIX安装...”。
  ![vscode_install](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/devkit/images/vscode_install.png)

 3. 选择已下载的.vsix插件文件，完成安装。
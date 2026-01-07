# 编译安装

## 前提条件

- **环境准备**: 编译安装 PyPTO 项目前, 请先参考[环境准备](prepare_environment.md)完成基础环境搭建.

## 通过 PyPI 安装 
 
PyPTO 已发布至 [PyPI](https://pypi.org/), 若不涉及对 PyPTO 源码的修改, 可以直接使用 `pip` 命令安装: 

```bash 
# 由 PyPI 源下载并安装 
python3 -m pip install pypto 
``` 
 
## 通过源码编译安装

### 环境自检

如果您的开发环境可以正常访问 [cann-src-third-party](https://gitcode.com/cann-src-third-party), PyPTO 编译所需的第三方开源软件将在编译过程中自动下载及编译.
如果无法访问, 请参考[环境准备](prepare_environment.md)中 “准备第三方开源软件源码包” 的相关章节完成源码包准备, 并在编译前设置如下环境变量:

```bash
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

### 常规安装

此方式适用于生产环境或代码稳定后使用. 编译安装后, 对 Python 源码的修改不会反映到已安装的 'pypto' 包中. 对应命令如下:

```bash
# (可选) 若开发环境无法访问 cann-src-third-party, 需设置
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装
python3 -m pip install . --verbose
```

**参数说明**:
- `--verbose`: 输出安装流程的基础详细信息(如下载的包版本、安装路径、依赖解析结果等).

**高级配置**: 

以下功能依赖 `pip` 支持 `--config-setting` 参数, 如需使用, 请确保 `pip` 版本不低于 **22.1**. 

1. 调整编译类型

   默认情况下, 常规安装编译出的 C++ 层二进制文件为 `Release` 类型. 若需进行调试, 可通过 `pip` 的 `--config-setting` 参数指定不同的编译类型: 

   ```bash
   # 通过 --config-setting 参数指定 C++ 层二进制编译为 Debug 类型
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug'
   ```

2. 开启 C++ 编译其详细输出模式

   ```bash
   # 额外开启 C++ 编译器详细输出模式(便于定位 C++ 编译问题)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug --cmake-verbose'
   ```

3. 指定 CMake Generator 类型

   ```bash
   # 指定 CMake Generator 类型(Ninja, Ninja 需提前安装)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-generator=Ninja'
   
   # 指定 CMake Generator 类型(Unix Makefiles)
   python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-generator="Unix Makefiles"'
   ```

### 可编辑安装

此方式适用于开发调试阶段. 该模式会在 `site-packages` 目录中创建指向本地源码的软链接, 对 Python 源码的修改会即时生效, 无需重新安装. 对应命令如下:

```bash
# (可选) 设置 PYPTO_THIRD_PARTY_PATH, 若开发环境无法访问 cann-src-third-party
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 执行编译及安装(可编辑模式)
python3 -m pip install -e . --verbose
```

**参数说明**:
- `-e`: 即 `--editable` 的简写形式, 标识采用可编辑安装模式；
- `--verbose`: 会输出安装流程的基础详细信息(如下载的包版本, 安装路径, 依赖解析结果等)；

**高级配置**: 

PyPTO 使用 `setuptools` 作为其编译打包工具. 需要注意的是, 当前版本的 `setuptools` 尚不支持直接接收 `pip` 命令通过 `--config-setting` 参数传递的配置. 
如需指定特定的 C++ 编译选项, 您需要将相关配置预先设置在 `PYPTO_BUILD_EXT_ARGS` 环境变量中. 该环境变量的值将在编译过程(`setup.py`)中被自动识别并使用.

示例: 以下示例演示了如何配置环境变量以编译 Debug 版本的 C++ 二进制文件并开启编译器的详细输出模式, 然后进行安装. 

```bash
# 设置编译参数: 指定编译类型为 Debug, 并开启编译器详细输出(便于诊断问题)
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose'

# 其他配置方式参考: 指定编译类型为 Debug, 并开启编译器详细输出(便于诊断问题), 指定 CMake Generator 为 Unix Makefiles
# export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose --cmake-generator="Unix Makefiles"'

# 执行编译及安装(可编辑模式)
python3 -m pip install -e . --verbose
```

## 通过 Docker 镜像安装

为了方便快速搭建环境, 同样提供已完成 PyPTO 运行环境搭建的 Docker 镜像, 详细使用请参考 [Docker README](../../docker/README.md).


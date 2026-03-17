---
name: pypto-environment-setup
description: "PyPTO 环境安装与环境问题修复，包括CANN、torch_npu、编译工具链、第三方依赖和PyPTO编译运行等。Triggers: PyPTO environment setup, CANN install, torch_npu, NPU environment, Ascend toolkit, compile PyPTO, build PyPTO, NPU driver, prepare_env, diagnose environment, fix import error, torch_npu import fail, DT_FP8E8M0, pto-isa, ASCEND_HOME_PATH, npu-smi, softmax verify, pip dependency conflict"
---

# PyPTO Environment Setup

## 约定

```bash
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
```


- `$SKILL_DIR`：由 agent 运行时自动注入的环境变量。指向当前 skill 的根目录（即本 `pypto-environment-setup/` 目录）。文档中所有 `$SKILL_DIR/scripts/...` 的引用均依赖此变量。手动执行时需自行设置，例如：`export SKILL_DIR=/path/to/pypto-environment-setup`。
- **默认版本**：CANN 8.5.0 + PyTorch 2.6.0 + torch_npu 2.6.0.post3

## ⛔ 隐私保护

> ⚠️ **禁止在屏幕、日志、错误信息中打印 `GITCODE_TOKEN` 环境变量**
>
> 克隆私有仓库时使用 Token 认证，请确保 Token 仅存储在安全位置（环境变量或配置文件），不要在终端输出中暴露。

## 工作流程

> ⚠️ **环境保护**：任何安装前必须先检测当前状态，向用户展示已有组件及版本，获得确认后才执行变更。

### 步骤 1：环境检测

```bash
# 步骤 1.1：检查 PYPTO_REPO 环境变量
echo "PYPTO_REPO: ${PYPTO_REPO:-未设置}"

# 步骤 1.2：检查当前目录是否为 PyPTO 仓库
[ -f "pyproject.toml" ] && [ -d "framework" ] && echo "✓ 当前目录是 PyPTO 仓库"
```

**PyPTO 仓库查找逻辑**（按优先级自动执行）：
1. 环境变量 `$PYPTO_REPO`
2. 当前目录（含 `pyproject.toml` + `framework/`）
3. `$PWD` 下搜索（最多 3 层），找到多个时提示用户选择
4. 未找到 → 自动克隆：

```bash
git clone https://gitcode.com/cann/pypto.git "$PWD/pypto"
export PYPTO_REPO="$PWD/pypto"
```

**运行环境诊断**：
```bash
# 步骤 1.3：进入 skill 目录运行诊断脚本
cd ${SKILL_DIR:-.opencode/skills/pypto-environment-setup}
python3 scripts/diagnose_env.py --checklist
```

**通过标准**：清单所有项 ✅ OK。⚠️/❌ 项需修复后再继续。

### 步骤 2：决策分支

- 0 个问题 → 跳至步骤 4 验证
- 有问题 → 向用户展示问题清单，获得确认后进入步骤 3
- 用户拒绝 → 停止，报告当前状态

### 步骤 3：按类别修复

只修复缺失项，不改动已正常组件。

**⚠️ 重要：NPU 环境 + CANN 缺失时，必须按顺序执行以下 5 步，全部完成后才能进入步骤 4**

| 问题类别 | 修复操作 | 失败回滚 |
|---------|---------|---------|
| **NPU 环境 + CANN 缺失** | **按顺序执行以下 5 步**：<br><br>**步骤 3.1：安装编译依赖**<br>`cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2\|a3>`<br><br>**步骤 3.2：下载安装第三方源码包**<br>`bash tools/prepare_env.sh --quiet --type=third_party`<br><br>**步骤 3.3：下载cann包**<br>`script -q -c "bash tools/prepare_env.sh --quiet --type=cann --only-download --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.download.log`<br><br>**步骤 3.4：安装 CANN**<br>`script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.install.log`<br><br>**步骤 3.5：验证 CANN 安装**<br>检查 CANN 安装日志，未完成则安装（见下方代码块）<br><br>⚠️ **全部 5 步完成后才能进入步骤 4** | 检查网络连通性和目录写入权限；见 `troubleshooting.md` |
| **编译工具链缺失** (cmake/gcc/make/g++/ninja/pip3) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=deps` | 回退到手动 `apt-get install`；检查 apt 源配置 |
| **第三方源码包缺失** (json/libboundscheck) | `cd $PYPTO_REPO && bash tools/prepare_env.sh --quiet --type=third_party` | 检查网络对 cann-src-third-party 的可达性 |

**CANN 安装检查脚本**：
```bash
cd ${PYPTO_REPO:-$PWD}
# 检查 CANN 安装日志，未完成则安装
if ! grep -q "Successfully installed CANN packages." prepare_env.cann.install.log 2>/dev/null; then
    echo "CANN 未安装完成，正在安装..."
    script -q -c "bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2\|a3> --install-path=$ASCEND_INSTALL_PATH" prepare_env.cann.install.log
else
    echo "CANN 已安装完成"
fi
```

> `--device-type` 由步骤 1 检测自动确定（910B→a2，910C→a3）。
> 手动安装/编译细节见 [📋 prepare_environment.md](references/prepare_environment.md)。
> 遇到报错见 [🔧 troubleshooting.md](references/troubleshooting.md)。

> ⚠️ **torch_npu 版本兼容性**：CANN 8.5.0 必须配套使用 `torch_npu==2.6.0.post3`，其他版本可能导致 HCCL 符号不兼容。安装命令：
> ```bash
> pip install torch==2.6.0 torch-npu==2.6.0.post3
> ```

### 步骤 4：验证和编译（必须执行）

任何安装/修复后必须通过 softmax 验证。

#### 步骤 4.1：加载 CANN 环境
```bash
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh
```

#### 步骤 4.2：检查可用的 NPU 卡 (务必在加载cann环境后检查)
```bash
npu-smi info
```

#### 步骤 4.3：设置环境变量
```bash
# 设置 NPU 设备 ID（根据步骤 4.2 选择空闲卡）
export TILE_FWK_DEVICE_ID=0


# 设置 PTO-ISA 路径
arch=$(uname -m)   # 常见值：x86_64 或 aarch64

# 3. 设置 PTO-ISA 库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/${arch}-linux

# 将环境变量写入当前目录文件
cat > env_setup.sh << "EOF"
#!/bin/bash
# 自动生成的环境配置文件
# 加载 Ascend 基础环境（根据实际安装路径调整）
source ${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/ascend-toolkit/set_env.sh

# 动态获取当前架构（确保生成的脚本在不同机器上仍正确）
arch=$(uname -m)
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/${arch}-linux

echo "env_setup.sh 加载完成：TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID}, PTO_TILE_LIB_CODE_PATH=${PTO_TILE_LIB_CODE_PATH}"
EOF
```
> - \$\{arch\}：CPU架构，如aarch64、x86_64.

#### 步骤 4.4：安装 torch 和 torch_npu
```bash
pip install torch==2.6.0 torch-npu==2.6.0.post3
```

#### 步骤 4.5：编译安装 PyPTO

```bash
# Source 环境变量文件
source env_setup.sh

# 如果有pypto先删除本地pypto
pip uninstall pypto -y
cd ${PYPTO_REPO:-$PWD}
rm -rf build_out
python3 -m pip install . --verbose
```

#### 步骤 4.6：运行测试验证
```bash
# Source 环境变量文件
source env_setup.sh

# NPU 模式（有 NPU 环境时必须使用）
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu

# SIM 模式（非 NPU 环境时使用）
# python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode sim

```

⚠️ **注意**：NPU 环境必须使用 NPU 模式通过验证。

**通过标准**：退出码 `0`，输出 `Softmax test passed`。

**失败时**：重新运行步骤 1 诊断 → 对照 [🔧 troubleshooting.md](references/troubleshooting.md) 排查。

### 步骤 5：完成报告

环境配置完成后，运行诊断并整理报告。

**运行诊断**：
```bash
cd ${PYPTO_REPO:-$PWD}
python3 scripts/diagnose_env.py --checklist
```

**报告模板**：
```
=====================================
PyPTO 环境配置
=====================================
CANN 版本:  8.5.0
NPU 芯片:   Ascend910 (A2/A3)
Python:     3.10.x
torch:      2.6.x
torch_npu:  2.6.0.post3
pypto:      ✅ 已安装
=====================================

验证结果:   Softmax（NPU 模式）✅ 通过

过程问题总结：
  - <问题> -> <解决方案>

⚠️ `TILE_FWK_DEVICE_ID` 需根据 `npu-smi info` 输出修改。
## 📚 参考文件

| 文件 | 内容 |
|------|----------|
| [📋 prepare_environment.md](references/prepare_environment.md) | 安装 CANN/torch_npu/pto-isa、编译 PyPTO |
| [🔧 troubleshooting.md](references/troubleshooting.md) | 安装/导入/运行报错 |
## 外部参考

- PyPTO: https://gitcode.com/cann/pypto
- CANN 安装指南: https://www.hiascend.com/document/redirect/CannCommunityInstSoftware
- Ascend Extension for PyTorch: https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html

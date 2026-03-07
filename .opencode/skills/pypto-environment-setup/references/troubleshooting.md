# 故障排除

## 通用排查步骤

1. 运行诊断：`python3 scripts/diagnose_env.py --pretty`
2. 确认 Python：`which python3`
3. 确认 CANN：`echo $ASCEND_HOME_PATH`（为空则需加载，见 [prepare_environment.md](prepare_environment.md) § "CANN 环境加载"）
4. conda 用户确认已激活正确环境
---

## 🔧 环境加载/变量问题

### npu-smi 运行失败：libc_sec.so not found

原因：未加载 CANN 环境变量。

修复：加载 CANN 环境后重试（见 [prepare_environment.md](prepare_environment.md) § "CANN 环境加载"），然后 `npu-smi info`。
### torch_npu 导入失败：libhccl.so / libatb.so / libascend_hal.so

原因分两类：
1. `set_env.sh` 未加载 → 加载 CANN 环境即可
2. CANN 安装不完整（缺 ops 包）→ 即使 source 了仍缺 so

```bash
# 确认 so 是否存在
test -n "${ASCEND_HOME_PATH:-}" && ls -la "${ASCEND_HOME_PATH}/aarch64-linux/lib64/libhccl.so" 2>/dev/null || echo "missing"

# 不存在：按芯片型号重装 ops
cd "$PYPTO_REPO" && bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2|a3> --install-path=$ASCEND_INSTALL_PATH

# 若缺 libatb.so，额外加载 NNAL/ATB
test -f "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/nnal/atb/set_env.sh" && source "${ASCEND_INSTALL_PATH:-/usr/local/Ascend}/nnal/atb/set_env.sh"

python3 -c "import torch_npu; print('ok')"
```

### PYTHONPATH 导致 pypto import 异常

原因：`PYTHONPATH` 包含 pypto 源码父目录，Python 优先找到未编译的源码。

```bash
unset PYTHONPATH
python3 -c "import pypto; print(pypto.__file__)"
```

---

## 📦 安装/编译问题

### pypto 导入失败：DT_FP8E8M0 缺失

原因：Python 侧代码与已编译的 `pypto_impl` 扩展版本不一致。

```bash
cd "$PYPTO_REPO"
python3 -m pip uninstall -y pypto || true
python3 build_ci.py -f python3 --clean --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall -q

# 验证
python3 -c "from pypto.pypto_impl import DataType; print('DT_FP8E8M0=', DataType.DT_FP8E8M0)"
```

仍失败：排查是否混用多个 pypto 路径/解释器 — `which python3` + `python3 -c "import pypto; print(pypto.__file__)"`.


### ModuleNotFoundError: No module named 'pypto'

原因：pypto 未编译安装。

```bash
cd ${PYPTO_REPO:-$PWD}
python3 build_ci.py -f python3 --clean --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall -q

# 验证
python3 -c "import pypto; print('✓ pypto 安装成功')"
```

### PTO ISA 编译/头文件错误

原因：`PTO_TILE_LIB_CODE_PATH` 指向错误或 pto-isa 版本过旧。

```bash
# 验证头文件目录
ls "${PTO_TILE_LIB_CODE_PATH:-/usr/local/Ascend/cann/aarch64-linux}/include/pto/comm/pto_comm_inst.hpp" 2>/dev/null && echo "✓ OK" || echo "✗ 缺失"
```

**方式 1：使用 CANN 安装目录（推荐）**
```bash
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

**方式 2：使用 pto-isa 源码目录（需最新版本）**
```bash
# 更新到最新版本
cd /path/to/pto-isa
git pull origin master

# 验证 comm 目录存在
ls include/pto/comm/pto_comm_inst.hpp

export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

### pto-isa 版本不匹配：缺少 pto::TROWEXPANDADD / pto::TROWEXPANDMAX

原因：pto-isa 头文件过旧或与 PyPTO 分支不匹配。

修复：切换到源码方式（见 [prepare_environment.md](prepare_environment.md) § "pto-isa 获取"），然后清理重编译：
```bash
cd "${PYPTO_REPO}/examples/02_intermediate/operators/softmax"
rm -rf output/ 2>/dev/null || true
python3 softmax.py --run_mode npu
# 无 NPU 环境：python3 softmax.py --run_mode sim
```


---

## 🔄 版本/依赖冲突

### undefined symbol / ABI 不匹配

原因：torch 与 torch_npu 版本组合不在兼容矩阵内。

修复：按兼容矩阵重装（见 [prepare_environment.md](prepare_environment.md) § "版本兼容"）。
### pip 依赖冲突：ResolutionImpossible

原因：torch 版本冲突。以兼容矩阵为准，必要时创建新 venv 重装。

### conda 中 torch 存在但 import 失败

原因：torch 在 conda env 中，但使用了系统 python。

```bash
pip show torch | grep Location
which python3
```

修复：激活正确的 conda 环境。

---

## ⬇️ 下载问题

### 下载/安装包损坏：Unexpected archive size

```bash
mkdir -p /tmp/pypto_download && cd /tmp/pypto_download
rm -f Ascend-cann-*.run cann-pto-isa_*.run 2>/dev/null || true
wget --progress=bar --timeout=600 --tries=10 -O <file.run> <url>
```

### prepare_env.sh --quiet 仍卡住

原因：`.run` 安装包交互提示未被抑制。

临时绕过：
```bash
PTO_RUN=${PTO_RUN:-$PYPTO_REPO/../pypto_download/cann_packages/cann-pto-isa_linux-aarch64.run}
chmod +x "$PTO_RUN"
"$PTO_RUN" --quiet --full --install-path=$ASCEND_INSTALL_PATH
```

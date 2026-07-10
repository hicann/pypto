---
name: pypto-simulation
description: PyPTO Soc CAModel 仿真执行技能。用于在 CPU 上模拟昇腾 AI 处理器行为，生成核内流水报告。触发词：Soc CAModel 仿真、核内流水、流水报告。
---

# PyPTO Soc CAModel 仿真执行

## 概述

Soc CAModel 仿真在 CPU 上模拟昇腾 AI 处理器，无需 NPU 硬件即可分析核内流水。

**关键特点：**
- 仿真速度比真实 NPU 慢 100-1000 倍
- 需充足资源：CPU >16核，内存 >32GB
- 生成 Chrome Tracing 流水图

---

## 核心流程（3 步）

### 步骤 1：检查并安装 PyPTO【必须】

> **执行目录：pypto 仓库主目录**

```bash
cd /path/to/pypto
pip show pypto
```

**检查点：**
- ✅ 输出包含 `Name: pypto` 和 `Version: x.x.x` → PyPTO 已安装，跳过安装
- ❌ 输出为空或报错 → PyPTO 未安装，需执行安装

**如 PyPTO 未安装，在主目录下执行：**

```bash
cd /path/to/pypto
# 编译产出 run
python3 build_ci.py --clean --py_abi=37 --plat_name=manylinux2014 --no_isolation --whl_into_run
# 安装到当前环境
bash build_out/cann-pypto_*.run --full -q --pylocal
```

安装完成后重新检查：
```bash
pip show pypto
```

---

### 步骤 2：执行仿真【必须】

> **执行目录：pypto 仓库主目录**

```bash
cd /path/to/pypto
cannsim record 'export ACCURACY_LEVEL=2 && python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 --gen-report -n 0
```

**检查点：**
- 输出显示 `Simulation SUCCESS`
- 无 `Segment fault` 或 `Aborted` 错误
- 主目录下生成 `cannsim_*/` 目录

**执行完成后，告知用户可选步骤 3：**
> 
> **可选步骤 3：** 如需全部核（32核）的完整流水报告，请确认后执行：
> ```bash
> cannsim report -e cannsim_* -o report --core-id all
> ```

---
**查看报告：**
- Chrome 浏览器 → `chrome://tracing` → Load `trace_core*.json`

---

## 实际案例：hello_world

> **所有命令均在 pypto 仓库主目录执行**

```bash
# 步骤1：检查PyPTO安装（必须）
$ cd /path/to/pypto
$ pip show pypto
Name: pypto
Version: 0.2.1

# 如果未安装：
$ python3 build_ci.py --clean --py_abi=37 --plat_name=manylinux2014 --no_isolation --whl_into_run
$ bash build_out/cann-pypto_*.run --full -q --pylocal

# 步骤2：执行仿真（必须）
$ cd /path/to/pypto
$ cannsim record 'export ACCURACY_LEVEL=2 && python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 --gen-report -n 0
Simulation SUCCESS ✓ run time 78.4s

# → 已生成单核报告 trace_core0.json

# 步骤3：生成全部核报告（可选，需用户确认）
$ cd /path/to/pypto
$ cannsim report -e cannsim_* -o report --core-id all
Json Saved: trace_core0.json ... trace_core31.json ✓
```

### 输出结果

```
cannsim_*/
├── cannsim.log              # 仿真日志
├── instr.bin                # 指令二进制
├── log_ca/                  # 指令日志目录
└── report/
    └── trace_core0.json     # 单核流水图（步骤2自动生成）

# 步骤3执行后追加：
report/
├── trace_core0.json
├── trace_core1.json
└── ... (trace_core31.json，共32个核)
```

---

## 关键注意事项

### 步骤执行顺序

- **必须严格按 1→2→3 顺序执行**
- **所有步骤均在 pypto 仓库主目录下执行**
- 步骤 3 为可选，其他步骤均为必须
- 每个步骤完成后需检查对应的检查点

### 可选步骤执行规范

步骤 3（生成全部核报告）需遵循以下规范：
- **不得主动执行**，需等待用户确认
- 步骤 2 完成后，告知用户已生成单核报告，并说明可选步骤 3
- 用户明确要求时，方可执行 `cannsim report --core-id all`

---

## 常见问题

### Q1：找不到算子文件

**原因：** 执行目录错误，未在 pypto 主目录下执行

**解决：** 必须在 pypto 主目录下执行所有命令
```bash
cd /path/to/pypto
cannsim record 'export ACCURACY_LEVEL=2 && python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 --gen-report -n 0
```

### Q2：报告生成超时

**原因：** 32核日志处理慢

**解决：** 使用 `--core-id 0` 只生成单核报告

---

## 命令速查

```bash
# 执行仿真
cannsim record 'export ACCURACY_LEVEL=2 && python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 --gen-report -n 0

# 生成完整报告（全部核）
cannsim report -e cannsim_* -o report --core-id all

# 生成报告（单核）
cannsim report -e cannsim_* -o report --core-id 0

# 查看输出目录
ls cannsim_*/

# 查看报告
ls cannsim_*/report/
```

---

## 触发词

- Soc CAModel 仿真、核内流水、流水报告、性能仿真

---

## 相关技能

- `pypto-environment-setup`：环境准备
- `tune-incore`：核内流水优化（基于 Soc CAModel 报告）
- `tune-swimlane`：泳道图分析优化
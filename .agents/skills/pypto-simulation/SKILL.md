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

## 核心流程（5 步）

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
python3 -m pip install . --verbose
```

安装完成后重新检查：
```bash
pip show pypto
```

---

### 步骤 2：配置仿真精度【必须】

> **执行目录：pypto 仓库主目录**

在算子文件中添加配置（在 `import pypto` 之后立即添加）：

```python
import pypto
pypto.set_global_config("simulation.accuracy_level", 2)  # 必须添加
```

**操作说明：**
- 算子文件路径如：`examples/00_hello_world/hello_world.py`
- 使用编辑工具在文件中添加上述配置

**检查点：** 算子文件中已包含该配置语句

---

### 步骤 3：创建仿真脚本【必须】

> **执行目录：pypto 仓库主目录**

脚本名与用例名保持一致，在主目录下创建：

```bash
cd /path/to/pypto
cat > hello_world.sh << 'EOF'
#!/bin/bash
python3 examples/00_hello_world/hello_world.py --run_mode sim
EOF
chmod +x hello_world.sh
```

**检查点：**
- 在主目录下执行 `ls -l hello_world.sh` 显示脚本存在
- 脚本有执行权限（显示 `-rwxr-xr-x`）

---

### 步骤 4：执行仿真【必须】

> **执行目录：pypto 仓库主目录**

```bash
cd /path/to/pypto
export TASK_QUEUE_ENABLE=0
cannsim record ./hello_world.sh -s Ascend950 --gen-report
```

**检查点：**
- 输出显示 `Simulation SUCCESS`
- 无 `Segment fault` 或 `Aborted` 错误
- 主目录下生成 `cannsim_<timestamp>_<script_name>/` 目录
- 默认已生成单核报告 `trace_core0.json`

**执行完成后，告知用户可选步骤：**
> ✅ 仿真已完成，已生成单核报告 `trace_core0.json`
> 
> **可选步骤 5：** 如需全部核（32核）的完整流水报告，请确认后执行：
> ```bash
> cannsim report -e cannsim_<timestamp>_hello_world.sh -o report --core-id all
> ```

---

### 步骤 5：生成全部核报告【可选】

> **执行目录：pypto 仓库主目录**
> ⚠️ 此步骤需用户确认后再执行

**触发条件：** 用户明确要求生成全部核报告

**执行命令：**
```bash
cd /path/to/pypto
cannsim report -e cannsim_<timestamp>_hello_world.sh -o report --core-id all
```

**输出：** `trace_core0.json` ... `trace_core31.json`（共32个核）

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
$ python3 -m pip install . --verbose

# 步骤2：配置仿真精度（必须）
# 在 hello_world.py 中添加：
import pypto
pypto.set_global_config("simulation.accuracy_level", 2)

# 步骤3：创建仿真脚本（必须）
$ cd /path/to/pypto
$ cat > hello_world.sh << 'EOF'
#!/bin/bash
python3 examples/00_hello_world/hello_world.py --run_mode sim
EOF
$ chmod +x hello_world.sh

# 步骤4：执行仿真（必须）
$ cd /path/to/pypto
$ export TASK_QUEUE_ENABLE=0
$ cannsim record ./hello_world.sh -s Ascend950 --gen-report
Simulation SUCCESS ✓ run time 78.4s

# → 已生成单核报告 trace_core0.json

# 步骤5：生成全部核报告（可选，需用户确认）
$ cd /path/to/pypto
$ cannsim report -e cannsim_20260519200437_hello_world.sh -o report --core-id all
Json Saved: trace_core0.json ... trace_core31.json ✓
```

### 输出结果

```
cannsim_20260519200437_hello_world.sh/
├── cannsim.log              # 仿真日志
├── instr.bin                # 指令二进制
├── log_ca/                  # 指令日志目录
└── report/
    └── trace_core0.json     # 单核流水图（步骤4自动生成）

# 步骤5执行后追加：
report/
├── trace_core0.json
├── trace_core1.json
└── ... (trace_core31.json，共32个核)
```

---

## 关键注意事项

### 步骤执行顺序

- **必须严格按 1→2→3→4→5 顺序执行**
- **所有步骤均在 pypto 仓库主目录下执行**
- 步骤 5 为可选，其他步骤均为必须
- 每个步骤完成后需检查对应的检查点

### 环境变量（步骤4）

执行仿真前必须设置：
```bash
export TASK_QUEUE_ENABLE=0
```

不设置会导致仿真崩溃或日志不完整。

### 仿真精度配置（步骤2）

**此步骤为必须执行**，否则：
- 仿真可能无法正常工作
- 核内流水报告详细度不足
- 无法进行有效的性能分析

配置位置：在 `import pypto` 之后立即添加
```python
import pypto
pypto.set_global_config("simulation.accuracy_level", 2)
```

### 可选步骤执行规范

步骤 5（生成全部核报告）需遵循以下规范：
- **不得主动执行**，需等待用户确认
- 步骤 4 完成后，告知用户已生成单核报告，并说明可选步骤 5
- 用户明确要求时，方可执行 `cannsim report --core-id all`

---

## 常见问题

### Q1：Segment fault

**原因：** 缺少 `export TASK_QUEUE_ENABLE=0`

**解决：**
```bash
export TASK_QUEUE_ENABLE=0
cannsim record ./hello_world.sh -s Ascend950 --gen-report
```

### Q2：找不到算子文件

**原因：** 执行目录错误，未在 pypto 主目录下执行

**解决：** 必须在 pypto 主目录下执行所有命令
```bash
cd /path/to/pypto
export TASK_QUEUE_ENABLE=0
cannsim record ./hello_world.sh -s Ascend950 --gen-report
```

### Q3：报告生成超时

**原因：** 32核日志处理慢

**解决：** 使用 `--core-id 0` 只生成单核报告

### Q4：日志目录为空

**原因：** Segment fault 导致日志未写入

**解决：** 检查环境变量 `export TASK_QUEUE_ENABLE=0`

---

## 命令速查

```bash
# 执行仿真
export TASK_QUEUE_ENABLE=0
cannsim record ./hello_world.sh -s Ascend950 --gen-report

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
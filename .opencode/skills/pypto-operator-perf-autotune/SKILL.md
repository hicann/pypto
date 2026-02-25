---
name: pypto-operator-perf-autotune
description: PyPTO 算子性能分析和自动调优技能。用于生成泳道图、分析性能数据、查看性能统计和提供优化建议。当用户需要分析 PyPTO 算子的性能、生成性能报告或进行性能调优时使用此技能。
---

# PyPTO 算子性能分析和自动调优

## 概述

此技能提供 PyPTO 算子性能分析和自动调优的完整工作流程，包括泳道图生成、性能数据分析、性能统计查看和优化建议。

## 工作流程

### 步骤 1：启用性能数据采集

在算子实现文件中，修改 `@pypto.frontend.jit` 装饰器，添加 `debug_options` 参数：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": mode},
    debug_options={"runtime_debug_mode": 1}
)
def kernel_function(
    x: pypto.Tensor(shape, pypto.DT_FP32),
) -> pypto.Tensor(shape, pypto.DT_FP32):
    # 算子实现
    return result
```

**⚠️ 重要提示**：
- 性能调优任务结束时，将修改的开关还原

### 步骤 2：重新编译并运行

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/cann/pypto/pto_isa/pto-isa/

# 编译 whl 包
python3 build_ci.py -f python3 --disable_auto_execute

# 运行算子（生成泳道图数据）
python3 custom/operator_name/operator.py --run-mode npu
```

执行后会在 `output/output_*/` 目录下生成泳道图数据文件：
- `merged_swimlane.json` - 泳道图数据文件
- `machine_runtime_operator_trace.json` - 性能追踪文件
- `bubble_analysis.log` - 气泡分析报告

### 步骤 3：查看泳道图

#### 方法一：使用 PyPTO Toolkit（推荐）

1. 在 VS Code 中安装 PyPTO Toolkit 插件
2. 点击侧边栏的 PyPTO Toolkit 图标
3. 在运行结果界面打开 `merged_swimlane.json` 文件
4. 或右键单击泳道图文件，选择 "PyPTO Toolkit：打开文件"

#### 方法二：使用 Perfetto

1. 打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. 点击左侧的 "Open trace file"
3. 上传 `machine_runtime_operator_trace.json` 文件
4. 查看性能分析结果

### 步骤 4：分析性能数据

#### 4.1 气泡分析报告 (bubble_analysis.log)

气泡分析报告显示线程等待时间和任务调度信息：

```
[AIV_48] Execute task num:1
    Core Total Work Time: 4.86us
    Total Wait Time: 0.0us
    Wait Schedule Time: 0.0us
    Wait Predecessor Time: 0.0us
```

**关键指标：**
- Core Total Work Time: 核心总工作时间
- Total Wait Time: 总等待时间
- Wait Schedule Time: 等待调度时间（气泡）
- Wait Predecessor Time: 等待前驱时间

#### 4.2 泳道图数据 (merged_swimlane.json)

泳道图数据包含任务执行信息和内存使用情况：

```json
{
  "args": {
    "event-hint": "Task:[0-0-0], rootHash:xxx, callOpMagic:xxx, leafHash:xxx",
    "ioperand-hint": "[{10: {'shape': [2, 4, 4, 4], 'dtype': 7, 'rawmagic': 13, 'mem_usage': 512}}]",
    "ooperand-hint": "[{12: {'shape': [2, 4, 4, 4], 'dtype': 7, 'rawmagic': 15, 'mem_usage': 512}}]",
    "execution-hint": "Average Execution Time: 4.35us\nMax Execution Time: 4.86us\nMin Execution Time: 3.84us"
  },
  "dur": 4.86us
}
```

**关键指标：**
- dur: 任务执行时间
- execution-hint: 执行时间统计
- ioperand-hint: 输入张量信息
- ooperand-hint: 输出张量信息

#### 4.3 性能追踪数据 (machine_runtime_operator_trace.json)

性能追踪数据包含 AICPU 控制流程详细信息：

```json
{
  "name": "DEV_TASK_BUILD(0)",
  "cat": "AICPU-CTRL",
  "dur": 64.42us
}
```

**关键阶段：**
- BEGIN: 开始阶段
- INIT: 初始化阶段
- CORE_HAND_SHAKE: 核握手
- DEV_TASK_BUILD: 设备任务构建
- DEV_TASK_RCV: 设备任务接收
- DEV_TASK_SCHED_EXEC: 设备任务调度执行
- DEV_TASK_SYNC_CORE_STOP: 设备任务同步核心停止

### 步骤 5：生成性能统计报告

#### 方法一：使用 PyPTO Toolkit（推荐）

1. 在泳道图界面中，点击工具栏的 "性能统计" 按钮
2. 查看性能指标阶梯图

**关键性能指标：**

| 指标 | 说明 |
|------|------|
| 计算时间 | 实际计算任务的执行时间 |
| 控制开销 | AICPU 控制流程总时间 |
| 线程利用率 | AIV/AIC 线程的工作时间占比 |
| 气泡率 | 线程等待调度时间占比 |
| 峰值内存使用 | OOO 内存使用峰值 |
| 内存效率 | 实际数据占用 / 峰值内存 |

#### 方法二：使用性能分析脚本

使用技能中的性能分析脚本自动生成报告：

```bash
python3 .opencode/skills/pypto-operator-perf-autotune/scripts/analyze_performance.py <output_dir>
```

示例：
```bash
python3 .opencode/skills/pypto-operator-perf-autotune/scripts/analyze_performance.py output/output_20260214_152549_401503_511667
```

脚本会自动分析所有性能数据文件并生成格式化的报告，包括：
- 线程性能统计
- 任务执行性能
- 控制开销分析
- 性能评级

### 步骤 6：性能优化建议

根据性能分析结果，提供优化建议：

#### 6.1 气泡优化

如果气泡率较高（>5%）：
- 检查任务调度策略
- 优化任务依赖关系
- 考虑任务并行化

#### 6.2 内存优化

如果内存利用率较低（<50%）：
- 调整 tile size 提高内存利用率
- 优化内存分配策略
- 考虑内存复用

#### 6.3 计算优化

如果计算时间较长：
- 优化算子实现逻辑
- 使用更高效的 API
- 考虑向量化优化

#### 6.4 控制开销优化

如果控制开销占比过高（>50%）：
- 增加数据规模降低控制开销占比
- 优化任务调度策略
- 减少不必要的同步操作

## 性能评级标准

| 评分 | 线程利用率 | 气泡率 | 内存效率 | 控制开销占比 |
|------|-----------|--------|---------|-------------|
| ⭐⭐⭐⭐⭐ | >98% | <2% | >70% | <30% |
| ⭐⭐⭐⭐ | >95% | <5% | >50% | <50% |
| ⭐⭐⭐ | >90% | <10% | >30% | <70% |
| ⭐⭐ | >80% | <20% | >20% | <80% |
| ⭐ | <80% | >20% | <20% | >80% |

## 常见问题

### Q1: 泳道图文件在哪里？

A: 泳道图文件在 `output/output_*/` 目录下，其中 `*` 是时间戳。

### Q2: 如何查看性能统计？

A: 使用 PyPTO Toolkit 打开 `merged_swimlane.json` 文件，然后点击 "查看性能报告" 按钮。

### Q3: 气泡是什么？

A: 气泡是指线程等待调度的时间，表示线程空闲的时间段。气泡率越低，说明调度效率越高。

### Q4: 控制开销占比过高怎么办？

A: 对于小数据量，控制开销占比高是正常现象。可以通过增加数据规模来降低控制开销占比。

## 参考资料

- [泳道图查看文档](../../docs/tools/swimlane_graph/查看泳道图.md)
- [配置泳道图系统参数](../../docs/tools/swimlane_graph/配置泳道图系统参数.md)
- [PyPTO Toolkit 使用指南](../../docs/tools/pypto_toolkit/)

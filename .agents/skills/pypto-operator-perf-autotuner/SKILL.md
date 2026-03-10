---
name: pypto-operator-perf-autotuner
description: PyPTO算子性能分析和自动调优技能。用于生成泳道图、分析性能数据、查看性能统计和提供优化建议。当用户需要分析 PyPTO 算子的性能、生成性能报告或进行性能调优时使用此技能。
---

# PyPTO 算子性能分析和自动调优

## 概述

此技能提供 PyPTO 算子性能调优的完整工作流程，包括性能数据采集、性能分析、优化策略和迭代调优。

## 工作流程

### 步骤 1：性能数据采集

#### 1.1 启用性能数据采集

在算子实现文件中，修改 `@pypto.frontend.jit` 装饰器，添加 `debug_options` 参数：

```python
@pypto.frontend.jit(
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

#### 1.2 重新编译并运行
如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/

# 编译 whl 包
python3 build_ci.py -f python3 --disable_auto_execute

# 运行算子（生成泳道图数据）
python3 custom/operator_name/operator.py --run-mode npu
```

执行后会在 `output/output_*/` 目录下生成泳道图数据文件：
- `merged_swimlane.json` - 泳道图数据文件
- `machine_runtime_operator_trace.json` - 性能追踪文件
- `bubble_analysis.log` - 气泡分析报告

### 步骤 2：检查精度
进行精度检查，根据检查结果选择执行下面步骤：
1. 如果检查通过，继续执行步骤3
2. 如果检查失败，是首轮失败，提示用户精度失败，是否选择修复或者继续进行优化
3. 如果检查失败，失败是由于上一轮修改导致，回退修改，尝试其它优化方案

### 步骤 3：分析性能数据
使用pypto-operator-perf-analyzer分析性能，生成性能报告和性能优化建议

### 步骤 4：执行性能优化
根据上一步的性能优化建议, 执行性能优化

### 步骤 5：迭代调优
1. 采集性能数据
2. 分析性能瓶颈
3. 应用优化策略
4. 重新编译运行
5. 检查精度
6. 对比性能提升，如果性能出现回退则回退修改
7. 重复步骤 1-6 直到达到目标性能

## 常见问题

### Q1: 泳道图文件在哪里？

A: 泳道图文件在 `output/output_*/` 目录下，其中 `*` 是时间戳。

### Q2: 如何查看性能统计？

A: 使用 PyPTO Toolkit 打开 `merged_swimlane.json` 文件，然后点击 "查看性能报告" 按钮。

### Q3: 气泡是什么？

A: 气泡是指线程等待调度的时间，表示线程空闲的时间段。气泡率越低，说明调度效率越高。

### Q4: 控制开销占比过高怎么办？

A: 对于小数据量，控制开销占比高是正常现象。可以通过增加数据规模来降低控制开销占比。

### Q5: 如何选择合适的 Tilesize？

A: 
- 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
- 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
- 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑

## 参考资料

- [性能调优文档](../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
- [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
- [功能调试](../../docs/tutorials/debug/debug.md)
- [精度调试](../../docs/tutorials/debug/precision.md)

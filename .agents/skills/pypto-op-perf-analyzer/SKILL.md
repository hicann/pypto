---
name: pypto-op-perf-analyzer
description: 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。
---

# PyPTO 性能指标分析技能

## 概述

此技能专注于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。

## 使用场景

当用户需要：
- 分析已生成的性能数据文件
- 评估算子性能表现
- 识别性能瓶颈
- 获取性能优化建议

## 工作流程

### 步骤 1：定位性能数据文件

性能数据文件位于 `output/output_*/` 目录下：

- `bubble_analysis.log` - 气泡分析报告
- `merged_swimlane.json` - 泳道图数据文件
- `machine_runtime_operator_trace.json` - 性能追踪文件

**查找最新输出目录：**
```bash
ls -lt output/ | head -n 2
```

### 步骤 2：提取核心性能指标

从 `bubble_analysis.log` 中提取以下指标：

**AIC 核心指标（AI Cube核心）：**
- Core Total Work Time: 核心总工作时间
- Total Wait Time: 总等待时间
- Wait Schedule Time: 等待调度时间（气泡）
- Wait Predecessor Time: 等待前驱时间

**AIV 核心指标（AI Vector核心）：**
- 同上指标

**算子实际执行时间：**
所有核心中 Core Total Work Time 的最大值

**AicoreTime(核心实际工作时间)：**
AicoreTime = 核心总工作时间 - 总等待时间


### 步骤 3：计算性能指标

#### 3.1 核心利用率

```
核心利用率 = AicoreTime / (AicoreTime + 等待总时间) × 100%
```

#### 3.2 气泡等待率

```
气泡率 = 等待调度时间 / (AicoreTime + 等待调度时间) × 100%
```

#### 3.3 平均核心利用率

```
平均核心利用率 = 所有核心核心利用率之和 / 核心数量
```

#### 3.4 平均气泡率

```
平均气泡率 = 所有核心气泡率之和 / 核心数量
```

### 步骤 4：性能评级

根据以下标准进行性能评级：

| 评分 | 核心利用率 | 气泡率 | 内存效率 | 控制开销占比 |
|------|-----------|--------|---------|-------------|
| ⭐⭐⭐⭐⭐ | >90% | <2% | >70% | <30% |
| ⭐⭐⭐⭐ | >80% | <5% | >50% | <50% |
| ⭐⭐⭐ | >60% | <10% | >30% | <70% |
| ⭐⭐ | >50% | <20% | >20% | <80% |
| ⭐ | <50% | >20% | <20% | >80% |

### 步骤 5：性能瓶颈分析

#### 5.1 气泡率分析

**高气泡率（>10%）可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**分析要点：**
- 识别气泡率最高的核心
- 检查 Top 3 tasks in waiting schedule time
- 分析任务分布情况

#### 5.2 核心利用率分析

**低核心利用率（<50%）可能原因：**
- 等待时间过长
- 任务调度不均衡
- 任务粒度过小
- 任务依赖过多

**分析要点：**
- 检查 Total Wait Time 是否过高
- 检查 Wait Predecessor Time 是否过长
- 检查核心间工作时间差异

#### 5.3 核心负载均衡分析

**分析方法：**
- 计算所有AicoreTime的标准差
- 识别工作时间最大和最小的核心
- 分析差异原因

#### 5.4 任务依赖分析

**分析方法：**
- 检查 Wait Predecessor Time
- 识别依赖关系复杂的任务
- 分析任务执行顺序

### 步骤 6：生成性能分析报告

性能分析报告应包含以下内容：

#### 6.1 核心性能指标

- 算子实际执行时间（所有核心最大工作时间）
- 各核心的AicoreTime
- 各核心的气泡时间
- 各核心的等待时间

#### 6.2 性能指标统计

- 平均核心利用率
- 平均气泡率
- 最大/最小核心利用率
- 最大/最小气泡率
- 核心负载均衡度

#### 6.3 性能评级

- 核心利用率评级
- 气泡率评级
- 综合评级

#### 6.4 性能瓶颈分析

- 主要瓶颈识别
- 瓶颈原因分析
- 影响程度评估

#### 6.5 性能优化建议

按优先级分类的优化建议：
- 高优先级优化
- 中优先级优化
- 低优先级优化

## 数据提取方法

使用技能中的性能分析脚本自动生成报告：

```bash
python3 scripts/analyze_perf.py <output_dir>
```

示例：
```bash
python3 scripts/analyze_perf.py output/output_20260214_152549_401503_511667
```

## 性能优化建议库

**⚠️ 重要提示**：优先采用高优先级的优化建议。

### 优化建议：气泡率高

**症状：** 气泡率 > 20%

**可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**优化建议：**
1. **Stitch 调优（优先级高）**
   ```python
   @pypto.frontend.jit(
      runtime_options={"stitch_function_max_num": 128}
   )
   ```

2. **对于循环类任务动态轴范围较广时开启loop_unroll（优先级高）**
   ```python
   for idx in pypto.loop(A.shape[0] // 64, unroll_list=[8, 4, 2, 1], name="A", idx_name='b'):
       offset = idx * s2_tile
   ```

   **参数说明：**
   - `loop_count`: 循环迭代次数
   - `unroll_list`: 展开因子列表，按优先级从高到低排列
   - `[8, 4, 2, 1]`: 常用配置，适应性强
   - `[16, 8, 4, 2, 1]`: 适用于更大循环
   - `[4, 2, 1]`: 适用于较小循环
   - `name`: 循环名称（用于调试）
   - `idx_name`: 循环索引变量名

   **⚠️ 重要原则：**
   - **loop_unroll 必须放在最内层循环！**
   - **不要在外层循环使用 unroll_list**
   - **循环迭代次数应足够大（建议 > 8）**

   **优化案例对比：**

   **原始代码（无优化）：**
   ```python
   for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
      for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
         for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
               for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                  for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):
                     # 计算逻辑
   ```

   **优化后代码（添加 loop_unroll）：**
   ```python
   for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx="b_idx"):
      for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
         for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
               for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                  # 最内层循环添加 unroll_list
                  for s2_idx in pypto.loop(s2_loop, unroll_list=[8, 4, 2, 1], name="LOOP_s2", idx_name="s2_idx"):
                     # 计算逻辑
   ```

3. **调整任务粒度**
   - 增大 loop 的 tile size
   - 减少 loop 层级

4. **优化调度策略**
    ```python
    @pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
    ```

5. **使用 L1Reuse 优化**
   ```python
   pypto.set_pass_options(cube_l1_reuse_setting={0: 8})
   ```

### 优化建议 2：核心利用率低

**症状：** 核心利用率 < 30%

**可能原因：**
- 等待时间过长
- 任务调度不均衡
- 内存访问冲突

**优化建议：**

1. **使用 L2 亲和调度**
    ```python
    @pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
    ```

2. **调整 Tilesize 增大算术强度**
   ```python
   # Cube Tilesize
   pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
   ```

3. **启用 CubeNBuffer 合并同构子图**
   ```python
   pypto.set_pass_options(cube_nbuffer_setting={0: 8})
   ```

### 优化建议 3：核心负载不均衡

**症状：** AicoreTime差异 > 20%

**可能原因：**
- 任务分配不均
- 任务执行时间差异大

**优化建议：**

1. **调整任务分配策略**
   - 使用更均匀的任务切分
   - 避免某些核心任务过多

2. **优化任务粒度**
   - 调整 tile size 使任务更均匀

3. **调整任务执行顺序**
   - 使用 sg_set_scope 合并子图
   ```python
   pypto.set_pass_options(sg_set_scope=1)
   # ... 操作 ...
   pypto.set_pass_options(sg_set_scope=-1)
   ```

## 性能分析报告模板

```markdown
## {算子名称} 性能分析报告

### 1. 核心性能指标

**算子实际执行时间:** {最大工作时间} us

**AIC AicoreTime:**
- AIC_1: {时间} us
- AIC_2: {时间} us
- ...

**AIV AicoreTime:**
- AIV_1: {时间} us
- AIV_2: {时间} us
- ...

### 2. 核心利用率分析

| 核心 | AIcoreTime | 等待时间 | 核心利用率 |
|------|---------|---------|-----------|
| AIC_1 | {时间} | {时间} | {利用率}% |
| ...

**平均核心利用率: {平均利用率}%**

### 3. 气泡分析

| 核心 | 气泡率 | 等待调度时间 |
|------|--------|------------|
| AIC_1 | {气泡率}% | {时间} us |
| ...

**平均气泡率: {平均气泡率}%**
**最大气泡率: {最大气泡率}% ({核心名称})**

### 4. 性能评级

| 指标 | 当前值 | 目标值(⭐⭐⭐⭐⭐) | 评级 |
|------|--------|----------------|------|
| 核心利用率 | {当前值}% | >98% | {星级} |
| 气泡率 | {当前值}% | <2% | {星级} |

**综合评级: {星级} ({评级描述})**

### 5. 性能瓶颈分析

1. **{瓶颈1名称}**
   - 描述: {描述}
   - 影响: {影响程度}

2. **{瓶颈2名称}**
   - 描述: {描述}
   - 影响: {影响程度}

### 6. 性能优化建议

**高优先级优化:**

1. **{优化1}**
   - 描述: {描述}
   - 代码: {代码示例}

**中优先级优化:**

2. **{优化2}**
   - 描述: {描述}
   - 代码: {代码示例}

### 7. 性能数据文件位置

- 泳道图: {泳道图路径}
- 气泡分析: {气泡分析路径}
- 性能追踪: {性能追踪路径}

可在 https://ui.perfetto.dev/ 上传泳道图文件进行可视化分析。
```

## 参考资料

- [性能调优文档](../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
- [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)

---
name: tune-swimlane
description: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch 优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。
---

# PyPTO 算子深度性能调优

## 概述

深度性能调优通过泳道图分析及调优性能，采用 man-in-loop 的方式，通过获取并分析当前算子性能数据，针对性调整各性能配置参数，经过迭代调优逐步逼近最佳性能。

## 前置条件

1. **完成开箱性能调优**：先进行代码级优化
2. **精度校验通过**：确保算子计算正确
3. **已采集性能数据**：生成泳道图和气泡分析报告

## 泳道图分析

### 泳道图文件位置

泳道图数据文件位于 `output/output_*/` 目录：
- `merged_swimlane.json` - 泳道图数据文件
- `bubble_analysis.log` - 气泡分析报告

### 查看泳道图

1. 通过 PyPTO Toolkit 插件查看
2. 或在 https://ui.perfetto.dev/ 上传泳道图文件
3. 查看泳道图文件及日志信息

### 泳道图关键信息

- 任务的执行顺序和耗时信息
- 各核心的工作时间和等待时间
- 气泡（线程等待调度的时间）
- 任务依赖关系

## 调优方向

### 1. Stitch 调优

Stitch 配置决定了多少个 root function 被同时下发调度。

#### 1.1 配置方法

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
```
**参考资料**
- [stitch_function_max_num 参数设置说明](../../../../docs/api/config/pypto-frontend-jit.md)

#### 1.2 参数影响

| 参数值 | 优点 | 缺点 |
|--------|------|------|
| 过小（如 1） | - | 每个任务需同步，调度开销大 |
| 适中（如 128） | 泳道图紧凑，调度开销低 | - |
| 过大（如 512） | 泳道图更紧凑 | 调度耗时增加，workspace 增加 |

#### 1.3 调优建议

在内存资源允许的前提下，逐步增大 Stitch 配置，结合泳道图和端到端总耗时数据调整参数。

### 2. TileShape 深度调优

#### 2.1 Matmul TileShape 深度调优

主要关注**减少重复载入**和**K轴分核**两个调优手段。
**减少重复载入**
```python
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 256])
```
每一个list中，前面的代表L0的切块大小，后面的代表L1的切块大小。L1的设置的大一些，可以减少重复载入。

**K轴分核**
```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256],
    enable_split_k=True)          # K轴分核
```

**参数说明**：
- `enable_split_k`：K轴分核

#### 2.2 Vector TileShape 深度调优

**原则 1**：下游 Vector Operation 的 TileShape 尽可能使用上游 Operation 的输出 TileShape

```python
# 上下游 TileShape 对齐时，可以合并在一个子图
# Transpose TileShape: (64, 128)
# Add TileShape 应优先选择: (128, 64)
```

**原则 2**：根据泳道图上的子图大小和并行度调整

- 并行核数较少（<一半 Vector 核）：减小 TileShape
- 子图耗时短、调度开销占比较高：增大 TileShape

**原则 3**：调整相邻 Cube 和 Vector Operation 的 TileShape，使依赖更简单


### 3. 合图调优

合图是指将计算图中多个逻辑上独立的 Task 合并为一个逻辑子图。

**⚠️⚠️⚠️ 关键原则：**
1. Key (hashorder) 等同于泳道图分析中得到的 **psgId**（`dyn_topo.txt` 中的 `psg_id_within_root`）。值 -1 是默认通配，非负整数匹配特定同构子图组。
2. Value (N) 是合并粒度，每 N 个同构子图合并为一个。设为 1 表示不合并。
3. 合并粒度应由 **t/iter**（单层循环相同 leafHash 的 task 数量）和核心数决定，常用值为 1/2/4/8/16。
4. **⚠️ 必须先用 analyze_swimlane.py 分析泳道图**：获取 psgId（hashorder）、core 类型（AIC/AIV）、t/iter（合图粒度参考），再据此配置。禁止盲猜配置。

**合图调优标准流程**：

```bash
# Step 1: 用 analyze_swimlane.py 分析泳道图数据
python3 .agents/skills/pypto-operator-auto-tuner/scripts/analyze_swimlane.py \
    output/output_<最新目录>

# Step 2: 从输出确定：
#   - psgId 列 → 即为 hashorder（set_pass_options 的 key）
#   - core 列 → AIC 对应 cube 配置，AIV 对应 vec 配置
#   - t/iter 列 → 合图粒度的参考值

# Step 3: 根据分析结果设置配置
```

#### 3.0 确定外层循环次数（outer_loops）

`t/iter = cnt / outer_loops`，其中 `outer_loops` 是外层循环的总迭代次数。

**自动检测**：脚本默认取所有 leafHash 的 cnt 的 **GCD** 作为 `outer_loops`。

**手动确认**：分析实现代码的 loop 嵌套和 tile 切块：

```python
# 示例：flash_attention_score_grad
# B=2, N=8, S=256, S_TILE=128, s_loop=S//S_TILE=2
# loop 嵌套: b(2) × n(8) × s1(s_loop=2) × s2(s_loop=2)
# 外层循环（s2 之外）: b × n × s1 = 2 × 8 × 2 = 32
for b_idx in pypto.loop(b, ...):
    for n_idx in pypto.loop(N, ...):
        for s1_idx in pypto.loop(s_loop, ...):
            for s2_idx in pypto.loop(s_loop, ...):   # 最内层
                ...
```

分析方法：
1. 找到 kernel 函数中所有 `pypto.loop()` 调用，确定嵌套层级
2. 最内层循环（通常是带 `unroll_list` 的那个）不参与 outer_loops 计算
3. `outer_loops = 各外层循环次数的乘积`

可用 `--outer-loops` 参数手动指定精确值：
```bash
python3 analyze_swimlane.py output/output_xxx --outer-loops 32
```

#### t/iter 对合图粒度的指导

| t/iter | 含义 | 合图粒度建议 |
|--------|------|-------------|
| 1 | 每外层迭代只执行 1 次 | 粒度 1（不合并），或跨实例尝试 2/4 |
| 2 | 每外层迭代执行 2 次 | 优先试 2，再试 4 |
| 4+ | 每外层迭代执行多次 | 可试 2/4/8，逐步增大 |

**注意**：合图粒度可以大于 t/iter（跨外层迭代合并），但不宜过大以免 L1/UB 内存溢出。

#### 输出解读示例

```
#   leafHash              cnt t/iter  ...  core psgId  ...  compute_ops
1   11267...              32   1.0    ...  AIV    1    ...  MULS+EXPAND+SUB+...
2   73361...              32   1.0    ...  AIC    0    ...  L1_TO_L0A+A_MUL_B+...
3   10262...              64   2.0    ...  AIC    2    ...  L1_TO_L0At+A_MULACC_B+...

outer_loops=32 (auto, GCD of counts)

[AIC] cube_l1_reuse_setting / cube_nbuffer_setting:
  psgId=0: cnt=32, t/iter=1, avg=8.77us
  psgId=2: cnt=64, t/iter=2, avg=2.51us  → try {2: 2/4/8}

[AIV] vec_nbuffer_setting:
  psgId=1: cnt=32, t/iter=1, avg=30.89us
```

**解读**：
- psgId=2 的 AIC 子图 t/iter=2（最内层循环执行 2 次），可设置 `cube_nbuffer_setting: {2: 2}` 将这两次合并
- psgId=0 的 AIC 子图 t/iter=1，但 total 耗时高，可设置 `cube_l1_reuse_setting: {0: 4}` 消除重复搬运
- AIV 子图 t/iter 均为 1，如需合图可尝试跨实例合并

#### 3.1 Vector 合图

##### 3.1.1 自动合图方案
**⚠️ 重要原则：**
vector合图往往需要 pg_upper_bound 和 vec_nbuffer_setting 配合使用，进行深度和广度的合图优化。

```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 500000,
        "vec_nbuffer_setting": {-2: 1, -1: 2}
        }
)
```
**适用场景**：自动切图的vector task之间有直接依赖关系，且每一个task耗时很短（<10us）

**参数说明**
- pg_upper_bound：代表广度方向合图的cycle数，这个参数在pass中会模拟进行合图。
- vec_nbuffer_setting：表示同构的并行子图合图的任务数量，-2:1不需要改变，-1:2代表，所有的vector均按照2的粒度进行合图

**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIV]` 部分的输出
2. 根据 `psgId` 确定 hashorder，根据 `t/iter` 确定粒度参考值
3. t/iter=1 的组先设为 1，t/iter≥2 的组设为对应值或更小
4. 可先用 `{-1: N}` 全局配置，再按 psgId 精细调优

**参考资料**
- [vec_nbuffer_setting 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)

##### 3.1.2 手动合图方案
```python
# 开始合图
pypto.set_pass_options(sg_set_scope=1)
# ... 操作 ...
# 结束合图
pypto.set_pass_options(sg_set_scope=-1)
```

**融合目标**：
- 上下游 Operation 间传输数据量较大
- 多个 Operation 切分后变成并行的连通分支

**注意**
- 当前主要考虑在连续的 Vector 计算过程中使用，暂不支持将 Matmul Operation 与 Vector Operation 进行合图。
- `sg_set_scope` 不适合包含 cube 操作的场景。

**参考资料**
- [sg_set_scope 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)


#### 3.2 Cube 合图

##### 3.2.1 L1Reuse 策略（默认开启，用于合并具有 L1 重复搬运的子图）

**适用场景**：matmul 的 M 或 N 轴进行了切分，存在重复搬运

```python
# 全局统一配置为 2
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)

# 全局配置为 2 的基础上，psgId=0 的子图配置为 8
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}
)
```
**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIC]` 部分的输出
2. 根据 `psgId` 确定 hashorder，优先对 total 耗时大且有重复搬运的子图调优
3. `t/iter` 越大（内层循环次数越多），L1 复用收益越高，可设更大粒度
4. 可先用 `{-1: N}` 全局配置，再按 psgId 精细调优

**参考资料**
- [cube_l1_reuse_setting 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)

##### 3.2.2 CubeNBuffer 策略（用于合并同构的子图）

**适用场景**：
- 同构子图数量很多，且每一个task的执行耗时很短（10us以下）
- K 轴很长且没有切 K

```python
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {-1: 2}}
)
```
**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIC]` 部分的输出
2. 根据 `psgId` 确定 hashorder，根据 `t/iter` 和 avg 耗时确定粒度
3. avg<10us 且 t/iter≥2 的组优先设置 `cube_nbuffer_setting: {psgId: t/iter}`
4. 可先用 `{-1: N}` 全局配置，再按 psgId 精细调优

**参考资料**
- [cube_nbuffer_setting 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)

##### 3.2.3 L1Reuse 与 CubeNBuffer 的协同使用

**⚠️ 重要：两者作用维度不同，需协同配置，不宜同时过大。**

| 参数 | 合并维度 | 核心目的 |
|------|---------|---------|
| cube_l1_reuse_setting | 消除 GM 数据重复搬运 | 多个子图复用同一份 L1 数据，减少搬运开销 |
| cube_nbuffer_setting | 合并结构相同的 AIC 子图 | 减少调度开销，提升核心利用率 |

**协同原则**：
1. **两者不宜同时设置过大**：cube_l1_reuse 通过消除重复搬运带来收益，合并力度越大 L1 复用越好；cube_nbuffer 通过合并同构子图减少调度开销。但两者同时过大会导致单个子图过大，占用过多 L1/UB 内存，反而引发性能退化。
2. **优先调 cube_l1_reuse_setting**：先确定 L1 数据复用的合并力度（消除重复搬运是更直接的收益），再调整 cube_nbuffer_setting。
3. **观察 Task Count 变化**：合图后 Total Task Count 应适度下降。如果 Task Count 不降反升（例如 1664→6400），说明合图配置过度，应回退。
4. **用 analyze_swimlane.py 确定参数**：psgId 即 hashorder，t/iter 指导粒度。

**反面案例**（flash_attention_score_grad 实测）：
```python
# baseline: 728us, Task=1664
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 8}
# → 711us ✅ (Task=1664, 利用率 58.8%→61.4%)

# 过度合图: 734us ❌ 性能退化
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 16}
```

**参考资料**
- [cube_l1_reuse_setting 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)
- [cube_nbuffer_setting 参数设置说明](../../../../docs/api/config/pypto-set_pass_options.md)

##### 3.2.4 自动合图模式（空字典 `{}`）的风险

**⚠️ 风险提示：自动模式可能过度合图导致性能严重退化，不建议直接使用。**

当设置为空字典 `{}` 时，Pass 会根据硬件核心数自动计算合并粒度。但自动模式不了解算子的实际数据流特征，可能将不应合并的子图强行合并，导致：
- Task 数暴增（如 1664→6400）
- 子图过大，L1/UB 内存争用
- 核心利用率大幅下降（如 58%→48%）

**反面案例**（flash_attention_score_grad 实测）：
```python
# 自动模式: 1038us ❌ 性能退化 42%
"cube_l1_reuse_setting": {},
"cube_nbuffer_setting": {}
# Task Count: 1664 → 6400, 利用率: 58.8% → 48.7%
```

**建议**：始终使用 [analyze_swimlane.py](../scripts/analyze_swimlane.py) 分析泳道图获取 psgId 和 t/iter 后手动精确配置，避免使用空字典 `{}` 自动模式。


### 4. 调度策略调优

当上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要时，推荐使用 L2 亲和调度。

```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**调优建议**：
- 尝试不同的调度策略，值域范围是[0, 3]

**注意事项**：综合考虑 L2 复用与负载均衡的影响，不同场景的最佳配置策略不同。

**参考资料**
- [device_sched_mode 参数设置说明](../../../../docs/api/config/pypto-frontend-jit.md)


## 性能优化建议库

### 建议 1：气泡率过高

**症状**：气泡率 > 10%

**可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**优化建议**：
1. **Stitch 调优（优先级高）**
   ```python
   @pypto.frontend.jit(
       runtime_options={"stitch_function_max_num": 128}
   )
   ```

2. **Loop Unroll（优先级高）**
   ```python
   for s2_idx in pypto.loop(s2_loop, unroll_list=[8, 4, 2, 1], name="LOOP_s2", idx_name="s2_idx"):
       # 计算逻辑
   ```

 3. **L1Reuse 优化**（需先分析泳道图）
    ```python
    # Step 1: 运行 analyze_swimlane.py 获取 psgId(=hashorder) 和 t/iter
    # Step 2: 根据输出设置
    pypto.set_pass_options(cube_l1_reuse_setting={0: 8})
    ```
    ⚠️ 使用 [analyze_swimlane.py](../scripts/analyze_swimlane.py) 获取 psgId 和 t/iter 后再设置，禁止盲猜。

4. **调整任务粒度**
- 增大 loop 的 tile size
- 减少 loop 层级

5. **vector 合图**
    ```python
    @pypto.frontend.jit(
        pass_options={
            "pg_upper_bound": 50000,
            "vec_nbuffer_setting": {-2: 1, -1: 2}
            }
    )
    ```

### 建议 2：核心利用率低

**症状**：核心利用率 < 50%

**可能原因：**
- 等待时间过长
- 任务调度不均衡
- 内存访问冲突

**优化建议**：

1. **L2 亲和调度**
   ```python
   @pypto.jit(runtime_options={"device_sched_mode": 1})
   ```

2. **调整 TileSize**
   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
   ```

3. **启用 CubeNBuffer 合并同构子图**
   ```python
   pypto.set_pass_options(cube_nbuffer_setting={-1: 4})
   ```

### 建议 3：核心负载不均衡

**症状**：AicoreTime 差异 > 20%

**可能原因：**
- 任务分配不均
- 任务执行时间差异大

**优化建议**：
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


## 调优流程

```
┌────────────────────────────────────────────────┐
│                深度性能调优流程                │
├────────────────────────────────────────────────┤
│                                                │
│  1. 采集泳道图数据                             │
│     └─ debug_options={"runtime_debug_mode": 1} │
│                                                │
│  2. 分析泳道图                                 │
│     ├─ 查看任务执行顺序                        │
│     ├─ 识别气泡（等待调度时间）                │
│     └─ 分析核心利用率                          │
│                                                │
│  3. 选择调优方向                               │
│     ├─ 气泡率高 → Stitch/Loop Unroll           │
│     ├─ 利用率低 → 调度策略/TileShape           │
│     └─ 负载不均 → 合图优化                     │
│                                                │
│  4. 应用优化                                   │
│     └─ 每次只修改一个参数                      │
│                                                │
│  5. 验证                                       │
│     ├─ 重新编译运行                            │
│     ├─ 检查精度                                │
│     └─ 对比性能数据                            │
│                                                │
│  6. 迭代直到达到目标性能                       │
│                                                │
└────────────────────────────────────────────────┘
```


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

* 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
* 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
* 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑


## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
- [GLM Attention 案例](../../../../models/glm_v4_5/glm_attention.py)
- [性能优化案例](../../../../docs/tutorials/debug/performance_case_quantindexerprolog.md)

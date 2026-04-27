# PyPTO 算子性能优化点索引

> 本文件是性能调优的**单一信息源**。编排器 ITER_START 阶段从这里选题，子技能 SKILL.md 只保留操作指南（怎么改），不重复优化点枚举。

## 使用方法

1. **编排器进入 PHASE 时**，按「一、按阶段分组」章节生成调优点清单
2. **ITER_START 选优化点时**，如已有性能数据，可先查「二、按症状索引」定位方向
3. **确定优化点编号后**，回到对应子技能 SKILL.md 的「调优方向」章节查看详细操作指南

---

## 一、按阶段分组

### 开箱调优（tune-frontend）

| 编号 | 优化方向 | 优先级 | 适用条件 | 详细指南 |
|------|---------|--------|---------|---------|
| F-1 | 任务粒度检查 | ⭐⭐⭐ | 含 Matmul 算子 | tune-frontend §1.2 |
| F-2 | 循环体计算量 | ⭐⭐⭐ | 含内层 loop | tune-frontend §1.2 |
| F-3 | 循环次数优化 | ⭐⭐⭐ | loop 次数 > 100 | tune-frontend §1.2 |
| F-4 | Reshape 全局优化 | ⭐⭐⭐ | 有 reshape 或 shape 维度大于 2D | tune-frontend §2 |
| F-5 | 静态轴改 Python for | ⭐⭐ | 有静态轴用 pypto.loop | tune-frontend §1.1 |
| F-6 | 合并独立 loop | ⭐⭐ | 有多个独立 loop | tune-frontend §1.3 |
| F-7 | 外层动态轴切块 | ⭐⭐ | 外层动态轴范围大 | tune-frontend §1.2.1 |
| F-8 | 内层 unroll | ⭐⭐ | 内层动态轴范围大 | tune-frontend §1.2.2 |
| F-9 | Cube TileShape 设置 | ⭐⭐ | 含 Matmul | tune-frontend §3.2 |
| F-10 | Vector TileShape 设置 | ⭐⭐ | 含 Vector 计算 | tune-frontend §3.3 |
| F-11 | 常量配置调整 | ⭐ | 算子有 BLOCK_SIZE 等常量 | tune-frontend 建议 3 |
| F-12 | 输入矩阵 NZ 格式 | ⭐ | 权重矩阵较大 | tune-frontend 局部§1 |
| F-13 | Transpose 优化 | ⭐ | 含 transpose+matmul | tune-frontend 局部§2 |
| F-14 | 冗余搬运消除 | ⭐ | 有 concat/assemble 搬运 | tune-frontend 局部§3 |

### 深度调优（tune-swimlane）

| 编号 | 优化方向 | 优先级 | 适用条件 | 详细指南 | 前置条件 |
|------|---------|--------|---------|---------|---------|
| S-1 | 核使用率分析 | ⭐⭐⭐ | 所有算子 | tune-swimlane §3.1 | 无（强制首选） |
| S-2 | 核填充（TileShape 调整增任务数） | ⭐⭐⭐ | 核未满 | tune-swimlane §3.2 | S-1 完成 |
| S-3 | 负载均衡分析 | ⭐⭐⭐ | 多子图算子 | tune-swimlane §3.3 | S-2 完成 |
| S-4 | Vector 手动合图（sg_set_scope） | ⭐⭐⭐ | 有连续 AIV 操作 | tune-swimlane §4.1.2 | S-3 完成 |
| S-5 | Vector 自动合图（nbuffer） | ⭐⭐ | 短耗时 AIV 任务 | tune-swimlane §4.1.1 | S-3 完成 |
| S-6 | Cube L1Reuse（消除重复搬运） | ⭐⭐ | AIC 核满、有重复搬运 | tune-swimlane §4.2.1 | S-3 完成 |
| S-7 | Cube CubeNBuffer（合并同构子图） | ⭐⭐ | AIC 核满、短耗时 | tune-swimlane §4.2.2 | S-3 完成 |
| S-8 | L1Reuse + CubeNBuffer 协同 | ⭐ | 已有 S-6/S-7 基础 | tune-swimlane §4.2.3 | S-6 或 S-7 |
| S-9 | Stitch 调优 | ⭐ | 所有算子 | tune-swimlane §1 | 无 |
| S-10 | 调度策略 | ⭐ | 依赖简单的上下游 | tune-swimlane §5 | 无 |
| S-11 | TileShape 深度调优 | ⭐⭐ | 需减少重复载入/K 轴分核 | tune-swimlane §2 | S-2 完成 |

### 核内调优（tune-incore）

| 编号 | 优化方向 | 优先级 | 适用条件 | 详细指南 |
|------|---------|--------|---------|---------|
| I-1 | 小 Shape 矩阵乘 | ⭐⭐⭐ | Matmul Shape 特殊 | tune-incore §1 |
| I-2 | L2 Cache 策略（权重矩阵 NONE_CACHEABLE） | ⭐⭐ | 含大型权重矩阵 / 融合算子 | tune-incore §2 |
| I-3 | 冗余计算消依赖 | ⭐⭐ | 一对多子图依赖 | tune-incore §3 |
| I-4 | 尾轴长度优化 | ⭐⭐ | 尾轴 < 32B 对齐 | tune-incore §4 |
| I-5 | TileOperation 实现检查 | ⭐ | 上述优化无效时 | tune-incore §5 |

---

## 二、按症状索引

> 供 ITER_START 阶段根据性能指标快速定位优化方向。编号指向「一、按阶段分组」中的条目。

### 症状 A：气泡率 > 10%

| 优先级 | 优化点 | 所在阶段 | 操作速览 |
|--------|--------|---------|---------|
| ⭐⭐⭐ | F-3 循环次数优化 | 开箱 | 增大 tile size 或切块 |
| ⭐⭐⭐ | F-8 内层 unroll | 开箱 | `unroll_list=[64,16,4]` |
| ⭐⭐ | S-9 Stitch 调优 | 深度 | `stitch_function_max_num: 128` |
| ⭐⭐ | S-4 / S-5 合图 | 深度 | sg_set_scope 或 nbuffer |
| ⭐ | S-10 调度策略 | 深度 | `device_sched_mode` 调整 |

### 症状 B：核心利用率 < 50%

| 优先级 | 优化点 | 所在阶段 | 操作速览 |
|--------|--------|---------|---------|
| ⭐⭐⭐ | F-1 任务粒度 | 开箱 | 增大 Matmul M/N 轴 |
| ⭐⭐⭐ | F-9 Cube TileShape | 开箱 | 推荐配置 |
| ⭐⭐⭐ | S-1 核使用率分析 | 深度 | `analyze_core_usage.py` |
| ⭐⭐⭐ | S-2 核填充 | 深度 | 减小 L0/L1 增加任务数 |
| ⭐⭐ | F-4 Reshape 全局优化 | 开箱 | `reshape(inplace=True)` 外提 + 合轴 |
| ⭐ | S-10 调度策略 | 深度 | `device_sched_mode` |
| ⭐ | S-6 / S-7 Cube 合图 | 深度 | 核满后再启用 |

### 症状 C：负载不均衡（AicoreTime 差异 > 20%）

| 优先级 | 优化点 | 所在阶段 | 操作速览 |
|--------|--------|---------|---------|
| ⭐⭐⭐ | S-3 负载均衡分析 | 深度 | 按 total(us) 排序 → 调整瓶颈 |
| ⭐⭐ | S-11 TileShape 深度调优 | 深度 | 减小瓶颈子图 L0/L1 |
| ⭐ | S-4 手动合图 | 深度 | sg_set_scope 合并子图 |

### 症状 D：单 task 耗时过长

| 优先级 | 优化点 | 所在阶段 | 操作速览 |
|--------|--------|---------|---------|
| ⭐⭐⭐ | I-1 小 Shape 矩阵乘 | 核内 | Vector 预处理 reshape |
| ⭐⭐ | I-2 L2 Cache（融合算子批量设置权重） | 核内 | 所有权重同时 `NONE_CACHEABLE` |
| ⭐⭐ | I-3 冗余计算消依赖 | 核内 | 复制数据使分支独立 |
| ⭐⭐ | I-4 尾轴长度优化 | 核内 | concat/transpose 增大尾轴 |
| ⭐ | I-5 Operation 检查 | 核内 | 与 Ascend C 对比 |

---

## 三、优化点全表（供调优点清单生成）

> 每个优化点的完整信息，编排器据此生成调优点清单。

### [F-1] 任务粒度检查

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 算子包含 Matmul
- **检查方法**: 检查 Matmul 的 M/N/K 轴是否充分利用硬件，M 轴 < 8 是常见问题
- **操作指南**: tune-frontend SKILL.md §1.2（切块 + unroll）
- **典型收益**: 5-50%（取决于粒度差异）
- **约束**: 切块大小不应超过 shape 中该维度的大小
- **关联优化**: F-7（外层切块）、F-8（内层 unroll）

### [F-2] 循环体计算量

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 含内层 loop
- **检查方法**: 检查循环体内部的计算量是否太小，用不满算力
- **操作指南**: tune-frontend SKILL.md §1.2
- **典型收益**: 10-30%
- **解决方案**: 开启 loop_unroll 或增加切分块大小
- **关联优化**: F-1（任务粒度）、F-8（内层 unroll）

### [F-3] 循环次数优化

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: loop 次数 > 100
- **检查方法**: 检查循环总次数，过多会导致调度开销大
- **操作指南**: tune-frontend SKILL.md §1.2
- **典型收益**: 5-20%
- **解决方案**: 切块减少循环次数
- **关联优化**: F-7（外层切块）、F-1（任务粒度）

### [F-4] Reshape 全局优化

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 算子中存在 `pypto.reshape` 调用，或循环体内参与计算的 tensor shape 维度超过 2D
- **检查方法**: 逐行扫描算子代码中所有 `pypto.reshape` 调用，分析每个 reshape 是否必须出现在最内层循环体中（参考 tune-frontend §2.1 分析表格）
- **操作指南**: tune-frontend SKILL.md §2
- **典型收益**: 5-30%
- **优化方式**:
  - **方式 1（原始输入 reshape 外提）**：对原始输入（函数参数）的 reshape，挪到算子入口（所有 loop 之前），使用 `inplace=True`，避免循环内重复数据拷贝
  - **方式 2（高维计算提前合轴）**：循环体内计算超过 2D 时，进入循环前对原始输入 `reshape inplace` 合轴为 2D，避免循环体内出现 reshape
- **约束**: 只有原始输入（函数参数）可用 `reshape(inplace=True)`，中间结果和输出 tensor 不能 inplace reshape；输出 tensor inplace reshape 会导致切片写入索引断裂（输出全零）
- **关联优化**: F-9（Cube TileShape）、F-10（Vector TileShape）

### [F-5] 静态轴改 Python for

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 静态轴使用了 `pypto.loop`
- **检查方法**: 搜索代码中 `pypto.loop` 调用，判断循环轴是否为静态（编译期可知的常量）
- **操作指南**: tune-frontend SKILL.md §1.1
- **典型收益**: 3-10%
- **代码示例**: `for i in pypto.loop(n, ...)` → `for i in range(n):`

### [F-6] 合并独立 loop

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 有多个独立的 pypto.loop
- **检查方法**: 检查是否有循环体无数据依赖的独立 loop
- **操作指南**: tune-frontend SKILL.md §1.3
- **典型收益**: 3-15%
- **约束**: 合并的操作之间无数据依赖冲突

### [F-7] 外层动态轴切块

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 外层动态轴范围较大
- **检查方法**: 检查外层 loop 的动态轴数值范围
- **操作指南**: tune-frontend SKILL.md §1.2.1
- **典型收益**: 10-40%
- **约束**: 切块大小从较大值开始尝试（如 64, 32, 16），不应超过 shape 中该维度的大小
- **代码示例**: `pypto.loop(b // b_block_size, ...)`

### [F-8] 内层 unroll

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 内层动态轴范围较大
- **检查方法**: 检查最内层 loop 的动态轴数值范围
- **操作指南**: tune-frontend SKILL.md §1.2.2
- **典型收益**: 5-30%
- **约束**: loop_unroll 必须放在最内层循环；unroll_list 最大值不要超过循环次数
- **代码示例**: `pypto.loop(n, unroll_list=[8, 4, 2, 1], ...)`

### [F-9] Cube TileShape 设置

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 含 Matmul（Cube 计算）
- **检查方法**: 检查是否设置了 `set_cube_tile_shapes`，配置是否为推荐值
- **操作指南**: tune-frontend SKILL.md §3.2
- **典型收益**: 5-20%
- **推荐配置**: `[128, 128], [64, 256], [256, 256]` 或 `[256, 256], [64, 256], [128, 128]`
- **约束**: L1 不超过实际轴长；`L0 <= L1` 且 `L1 % L0 == 0`；BF16 下 L0/L1 需 16 元素对齐；多 Matmul 时每个独立设置

### [F-10] Vector TileShape 设置

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 含 Vector 计算
- **检查方法**: 检查是否设置了 `set_vec_tile_shapes`
- **操作指南**: tune-frontend SKILL.md §3.3
- **典型收益**: 3-10%
- **推荐配置**: `pypto.set_vec_tile_shapes(64, 512)`
- **约束**: 优先用满尾轴；尾轴过大必须切分时按 512B 对齐切分；尾轴 32B 对齐（最低要求）；归约类计算不在归约轴上切分

### [F-11] 常量配置调整

- **阶段**: 开箱调优
- **优先级**: ⭐ P3
- **适用条件**: 算子中有 BLOCK_SIZE 等硬编码常量
- **检查方法**: 搜索算子代码中的常量定义（如 BLOCK_SIZE_KV、TILE_SIZE 等）
- **操作指南**: 在算子中写死的部分常量配置参数，可以尝试调整优化
- **典型收益**: 3-15%
- **常见调整值**: BLOCK_SIZE 可尝试 16/32/64/128

### [F-12] 输入矩阵 NZ 格式

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 权重矩阵 Shape 较大
- **检查方法**: 检查输入矩阵是否可以提前以 NZ 格式存储
- **操作指南**: tune-frontend SKILL.md 局部§1
- **典型收益**: 5-15%
- **原理**: NZ 格式的数据搬运到 L1 的带宽更高

### [F-13] Transpose 优化

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 含 transpose + matmul 结构
- **检查方法**: 搜索 transpose 操作后紧跟 matmul 的模式
- **操作指南**: tune-frontend SKILL.md 局部§2
- **典型收益**: 3-10%
- **解决方案**: 通过 matmul 的 `a_trans` / `b_trans` 参数融合 transpose，当 M 轴较大 N 轴较小时更换左右矩阵并使用转置配置

### [F-14] 冗余搬运消除

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 有 concat / assemble 等数据搬运操作
- **检查方法**: 检查是否有不合理数据操作导致的冗余搬运
- **操作指南**: tune-frontend SKILL.md 局部§3
- **典型收益**: 3-10%
- **解决方案**: 更换 concat 为 assemble

---

### [S-1] 核使用率分析

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P0（强制首选）
- **适用条件**: 所有算子
- **检查方法**: 运行 `analyze_core_usage.py` 统计每个 leafHash 占用的 core 数量
- **操作指南**: tune-swimlane SKILL.md §3.1
- **输出**: 每个 leafHash 的核使用率（used/total），判定 FULL / NOT FULL
- **后续路径**: 核未满 → S-2；核已满 → S-3
- **关联优化**: S-2（核填充）、S-11（TileShape 深度调优）

### [S-2] 核填充（TileShape 调整增任务数）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: S-1 分析后有 NOT FULL 子图
- **前置条件**: S-1 完成
- **检查方法**: 对 NOT FULL 子图，运行 `leafhash_to_code.py` 定位代码，减小 nL0/nL1 增加任务数
- **操作指南**: tune-swimlane SKILL.md §3.2
- **约束**: `nL0 <= nL1 && nL1 % nL0 == 0`；逐步减小（如 256→128→64），每步实测
- **⛔ 禁止**: 以结构限制为由跳过核填充，必须尝试完所有轴的 TileShape 调整

### [S-3] 负载均衡分析

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P1（核填充后强制执行）
- **适用条件**: 多子图算子
- **前置条件**: S-2 完成
- **检查方法**: 按 total(us) 降序排列所有子图，识别瓶颈子图，量化差距（>20% 必须优化）
- **操作指南**: tune-swimlane SKILL.md §3.3
- **达标条件**: 瓶颈子图 total 与次大 total 差距 < 20%，或连续 3 轮调整无法改善
- **约束**: 每次只调整一个子图的 TileShape

### [S-4] Vector 手动合图（sg_set_scope）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P2
- **适用条件**: 有连续 AIV 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_aiv_dep_chains.py` 分析 AIV 依赖链，用 `leafhash_to_code.py` 映射到代码行
- **操作指南**: tune-swimlane SKILL.md §4.1.2
- **约束**: 仅对有直接上下游数据依赖的 Vector 操作生效；不包裹 Cube 操作；不跨 loop 边界
- **⛔ 最易跳过的优化项**: 如果跳过此项，必须说明具体原因

### [S-5] Vector 自动合图（nbuffer）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: 短耗时（<10us）AIV 任务
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py` 查看 `[AIV]` 部分，确认短耗时任务
- **操作指南**: tune-swimlane SKILL.md §4.1.1
- **配置示例**: `pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}}`
- **调优方法**: 可先用 `{-1: N}` 全局配置，再按 psgId 精细调优

### [S-6] Cube L1Reuse（消除重复搬运）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: AIC 核满、matmul 的 M 或 N 轴进行了切分（存在重复搬运）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py` 查看 `[AIC]` 部分，优先对 total 耗时大且有重复搬运的子图调优
- **操作指南**: tune-swimlane SKILL.md §4.2.1
- **配置示例**: `pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}`
- **调优方法**: t/iter 越大，L1 复用收益越高，可设更大粒度

### [S-7] Cube CubeNBuffer（合并同构子图）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: AIC 核满、同构子图数量多且每个 task 执行耗时很短（<10us）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py`，avg<10us 且 t/iter≥2 的组优先设置
- **操作指南**: tune-swimlane SKILL.md §4.2.2
- **配置示例**: `pass_options={"cube_nbuffer_setting": {-1: 2}}`
- **⛔ 风险**: 不要使用空字典 `{}` 自动模式，可能过度合图导致性能严重退化

### [S-8] L1Reuse + CubeNBuffer 协同

- **阶段**: 深度调优
- **优先级**: ⭐ P3
- **适用条件**: 已有 S-6 或 S-7 的配置基础
- **前置条件**: S-6 或 S-7 完成
- **操作指南**: tune-swimlane SKILL.md §4.2.3
- **约束**: 两者不宜同时设置过大，会导致单个子图过大、L1/UB 内存争用；优先调 cube_l1_reuse_setting，再调整 cube_nbuffer_setting

### [S-9] Stitch 调优

- **阶段**: 深度调优
- **优先级**: ⭐ P4
- **适用条件**: 所有算子
- **检查方法**: 查看当前 `stitch_function_max_num` 配置
- **操作指南**: tune-swimlane SKILL.md §1
- **配置示例**: `runtime_options={"stitch_function_max_num": 128}`
- **调优方法**: 在内存资源允许的前提下逐步增大，结合泳道图和端到端耗时调整

### [S-10] 调度策略

- **阶段**: 深度调优
- **优先级**: ⭐ P4
- **适用条件**: 上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要
- **操作指南**: tune-swimlane SKILL.md §5
- **配置示例**: `runtime_options={"device_sched_mode": 1}`
- **调优方法**: 尝试不同调度策略，值域范围 [0, 3]

### [S-11] TileShape 深度调优

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 需减少重复载入或 K 轴分核
- **前置条件**: S-2 完成
- **操作指南**: tune-swimlane SKILL.md §2
- **优化手段**: 减少重复载入（增大 L1）和 K 轴分核（`enable_split_k=True`）
- **Vector 原则**: 下游 TileShape 尽可能使用上游输出 TileShape；归约轴不切分

---

### [I-1] 小 Shape 矩阵乘

- **阶段**: 核内调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: Matmul 的 Shape 特殊（如 M 很大 N 很小）
- **检查方法**: 通过泳道图定位到耗时较长的 task，检查其 Matmul 的 M/N/K 是否特殊
- **操作指南**: tune-incore SKILL.md §1
- **解决方案**: 使用 Vector 操作提前处理输入矩阵，通过 concat/reshape 构造标准 Shape
- **典型案例**: 左右矩阵 (884736, 16) × (16, 16) → 从 500us 优化到 40us

### [I-2] L2 Cache 策略

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1（融合算子中效果显著）
- **适用条件**: 算子包含大型权重矩阵（matmul 权重），或有过大的输出 Tensor
- **API**: `tensor.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)`
- **API 文档**: `docs/api/tensor/pypto-Tensor-set_cache_policy.md`
- **检查方法**: 分析算子中所有权重矩阵的访问模式，识别只读一次且不复用的大 Tensor
- **操作指南**: tune-incore SKILL.md §2
- **调优策略**:
  - 简单算子：逐个对候选 Tensor 设置 NONE_CACHEABLE，每次实测对比
  - 融合算子（含多个大权重矩阵）：**同时对所有权重设置 NONE_CACHEABLE**，避免 L2 争用失衡
- **⛔ 不适用场景**:
  - 输入 Tensor（数据量小，硬件预取已足够，绕过反增延迟）
  - 输出 Tensor（增加写回延迟）
  - 融合算子中单独对某个权重设置（打破 L2 平衡，可能恶化）
- **典型收益**: 10-20%（融合算子中批量设置所有权重）
- **典型案例**: Pangu 7B Fused Layer，5 个权重矩阵同时设置 NONE_CACHEABLE → 437.28us→354us（-19.1%）
- **代码示例**:
  ```python
  # 融合算子中：对所有权重矩阵同时设置
  qkv_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  o_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  gate_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  up_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  down_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  ```

### [I-3] 冗余计算消依赖

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 一对多的子图依赖（一个 tensor 被多个下游消费）
- **检查方法**: 分析 `dyn_topo.txt` 中是否存在一个节点有多个不同 psgId 的后继
- **操作指南**: tune-incore SKILL.md §3
- **解决方案**: 增加冗余计算（复制数据），使每个分支独立，避免一对多的子图依赖
- **典型案例**: GLM MoE Fusion 中将 e_score_bias_2d 复制 tile_batch 份

### [I-4] 尾轴长度优化

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1
- **适用条件**: Operation 输入 Tensor 尾轴 < 32B 对齐
- **检查方法**: 检查参与计算的关键 tensor 的最后一个维度大小
- **操作指南**: tune-incore SKILL.md §4
- **解决方案**: 使用 concat 增大尾轴、transpose 调整轴顺序、reshape 调整 Shape

### [I-5] TileOperation 实现检查

- **阶段**: 核内调优
- **优先级**: ⭐ P2
- **适用条件**: 上述所有优化手段都已尝试，单 task 耗时仍然过长
- **检查方法**: 构造单独 Operation 的测试用例，与 Ascend C 小算子性能对比
- **操作指南**: tune-incore SKILL.md §5
- **解决方案**: 确认性能差距后检查是否使用了更优指令，或考虑使用其他 Operation 组合替代

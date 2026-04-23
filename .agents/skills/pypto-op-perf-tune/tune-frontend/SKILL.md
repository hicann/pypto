---
name: tune-frontend
description: PyPTO 算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括 loop 写法优化、TileShape 设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开箱性能调优时使用此技能。触发词：开箱性能调优、代码级优化、loop 优化、TileShape 设置、前端优化。
---

# PyPTO 算子开箱性能调优

## 概述

开箱性能调优主要关注代码级的调优、前端写法不同导致的性能差异。在算子初始编写过程中直接得到较好的开箱性能。


## ⚡ 代码分析（最重要！）

**必须按顺序检查以下问题**：

### 🔥 P0 - 最常见问题（80%的性能问题都在这里）

1. **任务粒度是否足够大？**
   - 检查最内层循环的任务粒度（如 Matmul 的 M/N/K 轴）
   - ❌ 常见错误：Matmul 的 M 轴只有 1，无法利用 Cube 计算能力
   - ✅ 解决方案：对外层轴切块，增大任务粒度

2. **循环体一次的计算量是否太小？**
   - 检查循环体内部的计算量，如果太小，用不满算力。
   - ✅ 解决方案：
   - 考虑开启 loop_unroll
   - 分析循环轴的切块是否合理。太小的话，需要增加切分块大小

3. **循环次数是否过多？**
   - 循环次数过多会导致调度开销大
   - ✅ 解决方案：切块，减少循环次数

4. **shape 是否可以提前合轴？**
   - 如果 shape 是 2 维以上，性能会比较差。因为 npu 指令支持的维度是两维的。考虑在进入循环前，先进行合轴处理
   - ✅ 解决方案：进入循环前，对原始输入使用 `reshape inplace` 进行合轴

5. **原始输入的 reshape 是否在 loop 外层执行？**
   - 对原始输入（函数参数）的 reshape 必须放在所有 loop 之前，并使用 `inplace=True`，避免冗余数据拷贝和循环内重复执行
   - ✅ 解决方案：将原始输入的 reshape 提取到 loop 最外层之前，统一用 `pypto.reshape(tensor, [...], inplace=True)` 处理

### P1 - 其他常见问题

1. loop 层级是否太深，考虑合并 loop
2. 计算 op 是否冗余，考虑使用更高效的 operation

## 💡 关键启发

### 1. loop_unroll 的本质
- **目的**：增加并行度，让多个循环任务并行执行
- **unroll_list 数值含义**：代表并行块的大小（如[8,4,2,1]表示优先并行 8 个块）
- **关键**：数值代表并行块的大小，不是简单的循环展开次数

### 2. 优化循环结构的思路
- **合并循环** - 减少嵌套层级，增加单层迭代次数
- **外层动态轴** → 使用切块（静态切分）
- **内层动态轴** → 使用loop_unroll（动态展开）
- **黄金组合** → 外层切块 + 内层 unroll = 最优并行度

### 3. 性能优化三要素
1. **任务粒度** - 每个任务的计算量（越大越好）
2. **并行度** - 可以并行执行的任务数（越多越好）
3. **调度开销** - 任务切换的时间成本（越小越好）

**优化目标**：在保证任务粒度的前提下，最大化并行度，最小化调度开销。

## 调优方向

调优方向分为**全局性能优化**和**局部性能优化**两个层次：

- **全局性能优化**：从算子整体结构出发，关注循环组织方式和每个 operation 的基本块设置是否合理
- **局部性能优化**：针对具体 operation 的写法优化，包括数据操作、合轴等细节

**执行顺序：先完成全局优化，再进行局部优化。**

### 全局性能优化

#### 1. Loop 写法优化

**增加 root function 的大小，减少它们的个数**
由于不同 root function 之间的子图不能合并，而子图合并是 PyPTO 优化性能的关键手段。

##### 1.1 静态轴使用 Python for 循环

`pypto.loop` 方法会按当前轴循环展开成不同的 root function。因此静态轴上的循环应使用 Python 的 for 循环。

```python
# ✅ 推荐：静态轴使用 Python for
for i in range(batch_size):
    result[i] = process(data[i])

# ❌ 避免：静态轴使用 PyPTO loop
for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
    result[i] = process(data[i])
```

##### 1.2 减少循环次数，增加并行度
###### 1.2.1 如果外层的动态轴范围很大，使用切块处理（高优先级）

算子的循环轴的 dim 数值范围往往较广，往往需要对其进行静态切分，否则循环次数太大。

示例：
```python
# 推荐：动态轴使用 loop 切块，对b_loop轴按128进行切分，提高并行度
bsz, h = x.shape
b = 128
b_loop = (bsz + b - 1) // b
for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
    b_valid = (bsz - b_idx * b).min(b)
    x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
    # Matmul
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    y = pypto.matmul(x_view, W)
```

###### 1.2.2 如果内层的动态轴范围很大，调整切块大小或使用loop_unroll展开，增加并行度
   ```python
   for idx in pypto.loop(A.shape[0] // 64, unroll_list=[128, 64, 8, 1], name="A", idx_name='b'):
       offset = idx * s2_tile
   ```

   **参数说明：**
   - `unroll_list=[8, 4, 2, 1]`: 展开因子列表，按优先级从高到低排列，数值代表并行块大小，优先尝试并行 8 个块
   - `[8, 4, 2, 1]`: 常用配置，适应性强
   - `[128, 64, 16, 8, 1]`: 适用于更大循环
   - `name`: 循环名称（用于调试）
   - `idx_name`: 循环索引变量名

    **⚠️ 重要原则：**
    - **loop_unroll 必须放在最内层循环！不要在外层循环使用 unroll_list**
    - 数值代表**并行块大小**，不是简单的展开次数
    - 目的是**增加并行度**，让多个任务可以并行执行
    - **unroll_list 的最大值，不要超过循环次数**

###### 1.2.3 切块优化策略：外层切块 + 内层unroll

**核心思路**：对循环轴切块，减少循环次数，增大任务粒度，然后在最内层使用unroll增加并行度。

**适用场景**：当最内层循环次数过少（<8）无法有效unroll时，应考虑对外层轴切块。

**🔥 典型案例：Flash Attention（最常见错误）**

**❌ 错误实现（任务粒度过小）**：
```python
for q_idx in pypto.loop(SEQ_LEN_Q, name="LOOP_Q"):  # 64 次
    q_vec = pypto.view(query, [1, HEAD_DIM], [q_idx, 0])  # ❌ M 轴 = 1
    for kv_idx in pypto.loop(num_kv_blocks, name="LOOP_KV"):  # 1 次
        scores = pypto.matmul(q_vec, k_block, ...)  # ❌ [1, 128]
```

**问题**：
- Matmul M 轴只有 1，浪费 Cube 计算能力
- 循环 64 次，调度开销极大
- 每个任务计算量太小，任务粒度过细

**✅ 正确实现（Query 切块）**：
```python
Q_BLOCK_SIZE = 16
num_q_blocks = SEQ_LEN_Q // Q_BLOCK_SIZE  # 4 次

for q_block_idx in pypto.loop(num_q_blocks, name="LOOP_Q"):  # 4 次
    q_start = q_block_idx * Q_BLOCK_SIZE
    cur_q_size = pypto.min(Q_BLOCK_SIZE, SEQ_LEN_Q - q_start)

    # ✅ 批量获取 16 个 query
    q_block = pypto.view(query, [Q_BLOCK_SIZE, HEAD_DIM], [q_start, 0],
                        valid_shape=[cur_q_size, HEAD_DIM])

    for kv_block_idx in pypto.loop(num_kv_blocks, unroll_list=[4,2,1], name="LOOP_KV"):  # ✅ 可 unroll
        # ✅ Matmul M 轴 = 16
        scores = pypto.matmul(q_block, k_block, ...)  # ✅ [16, 128]
```

**收益来源**：
- ✅ 任务数: 64 → 4（减少 16 倍）
- ✅ Matmul M 轴: 1 → 16（计算量增大 16 倍）

**⚠️ 切块大小建议**:
- 从较大值开始尝试（如 64, 32, 16），但切开的值不应该超过shape中该维度的大小
- 平衡任务粒度和内存占用
- 调整中间 tensor 的 shape

##### 1.3 尽可能合并 loop

检查算子代码是否有可以合并的 loop 块：

```python
# ❌ 不推荐：两个独立的 loop
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
    out_2 = Operation2(x2[b_idx, :], y)

# ✅ 推荐：合并 loop
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
    out_2 = Operation2(x2[b_idx, :], y)
```

#### 2. 基本块优化

对算子中所有 operation 逐个审查 shape 与基本块（TileShape）设置，识别维度不匹配、设置不合理等问题，产出审查表格后再进行优化。

##### 2.1 逐 Operation Shape 与基本块审查

**分析方法**：
1. 逐行阅读算子 kernel 代码，记录每个 operation（matmul、cast、reshape、view、concat、add、mul、sum 等）
2. 统计每个 operation 的：输入 shape、输出 shape、TileShape 设置（cube_tile / vec_tile）
3. 检查每个 TileShape 是否与实际 shape 匹配、是否合理
4. 汇总问题清单，按优先级排序

**审查表格模板**：

| # | Operation | 输入 Shape | 输出 Shape | TileShape 设置 | 合理? | 问题与建议 |
|---|-----------|-----------|-----------|---------------|------|-----------|
| 1 | `matmul(A, B, b_trans=True)` | A:`[1,4096]` B:`[6144,4096]` → `[1,6144]` FP32 | `cube([16,16],[64,256],[256,256])` | ⚠️ | M=1, K 轴未驻留，建议 `[16,16],[128,4096,256],[256,256]` |

**关键检查项**（对每个 operation 逐一检查）：
1. **TileShape 维度与 Shape 维度是否匹配**：vec_tile_shapes 的每个维度不应超过对应 tensor 维度
2. **Cube TileShape 对齐约束**：L0 各维度必须 16 元素对齐（BF16/FP16 场景）
3. **L1 是否超过实际轴长**：mL1/kL1/nL1 超过对应维度实际大小是无意义的
4. **大轴的 L1 是否过小**：导致过多的重复载入次数
5. **Reduce 轴是否被不必要地切分**：归约计算尽量不对归约轴切分
6. **多个 matmul 是否共用同一 TileShape**：不同 shape 的 matmul 必须独立配置
7. **reshape/view 前后 vec_tile 是否匹配对应阶段的 tensor shape**：reshape 前按源 shape 设，reshape 后按目标 shape 重设；特别关注 reshape 后紧跟 assemble 的场景

##### 2.2 Cube TileShape 设置规范

**函数原型**：
```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
# 高级用法：A/B 矩阵独立设置 K 轴切分
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1])
```

**参数说明**：
- `mL0/mL1`：M 维度在 L0/L1 上的切分大小
- `kL0/kL1`：K 维度在 L0/L1 上的切分大小；三维 `[kL0, kAL1, kBL1]` 可分别设置 A/B 矩阵 K 轴切分
- `nL0/nL1`：N 维度在 L0/L1 上的切分大小

**对齐约束**（BF16/FP16 场景）：
- L0 各维度必须 16 元素对齐（即 L0_M, L0_K, L0_N 均为 16 的倍数）
- kL0, kL1, nL0, nL1 需满足 32 字节对齐
- `L0 <= L1` 且 `L1 % L0 == 0`

**训练场景（M/N 较大）推荐初始配置**（A2/A3 平台）：
```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

优点：在满足 L0 Buffer 约束下达到较大算数强度，可开启 Double Buffer。

**推理/Decode 场景（M=1 或 M 较小）**：

M 轴较小时无法通过 M/N 切分提高算数强度，优化思路是**减少重复载入**：
- 使用 K 轴三维配置 `[kL0, kAL1, kBL1]`，让 A 矩阵一次搬入 L1 驻留
- 尽量增大 nL1，减少 N 轴切分次数

```python
# Decode 场景 matmul 示例：A=[1,K] × B^T=[N,K]
pypto.set_cube_tile_shapes([16, 16], [16, K, 256], [256, 256])
# A 矩阵 K 轴整体驻留 L1(kAL1=K)，B 矩阵 K 轴按 256 分批载入(kBL1=256)
```

**配置要点**：
1. **L1 不超过实际轴长**：mL1/kL1/nL1 不应超过对应维度实际大小，超过无意义且浪费 L1 空间
2. **小轴不切**：某轴本身较小（如 K=128），L0=L1=min(实际值, 最大对齐值)，不切分
3. **大轴大 tile**：某轴较大（如 N=2048 或 K=4096），用较大 L1 减少切分次数

**⚠️ 独立设置原则**：同一算子中多个不同 shape 的 matmul，**必须在每个 matmul 前分别调用 `set_cube_tile_shapes`**，不要用统一值。

**🔥 案例**：[多 Matmul 独立 TileShape 优化](cases/per-matmul-tile-shapes.md)（3 个 matmul 独立设 tile，-46.1%）

##### 2.3 Vector TileShape 设置规范

**配置原则**：
1. 满足特定 Operation 对 TileShape 的规格约束
2. 保证 Operation 的输入与输出 Tensor 可以在 UB 中分配内存
3. TileShape 不能过大也不能过小（数据块大小在 16 到 64KB 之间）
4. 尾轴 32B 对齐
5. 归约类计算尽可能不要在归约轴上进行切分

```python
pypto.set_vec_tile_shapes(64, 512)
```

**⚠️ 维度匹配检查**：vec_tile_shapes 每个维度的值不应超过对应 tensor 的实际维度大小。例如 tensor shape 为 `[1, 1024]`，不应设置 `vec(8, 128)`（第 1 维 8 > 实际 1）。

**归约轴切分问题**：
- ❌ 对 reduce 轴切分：多个子图的输出需要在同一个子图进行 reduce 操作，产生 GM 搬运和调度开销
- ✅ 不对 reduce 轴切分：上下游子图合并，没有额外开销

**⚠️ reshape 前后的 vec_tile_shapes 设置规则**：

`reshape` / `view` 操作会改变 tensor 的 shape，但 vec_tile_shapes 需要与当前操作的实际 tensor shape 匹配。因此必须遵循：

1. **reshape 前**：vec_tile_shapes 按源 tensor（reshape 前）的 shape 设置
2. **reshape 后**：在操作 reshape 后的 tensor 之前，重新设置 vec_tile_shapes 按目标 tensor（reshape 后）的 shape

```python
# ✅ 正确：reshape 前按源 shape 设，reshape 后按目标 shape 重新设
pypto.set_vec_tile_shapes(8, 128)          # 源 tensor [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.set_vec_tile_shapes(8, 1, 128)      # 目标 tensor [8, 1, 128]
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 前就按目标 shape 设了 vec_tile
pypto.set_vec_tile_shapes(8, 1, 128)      # ❌ 此时 tensor 还是 [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 后未重新设 vec_tile，沿用旧的设置
v_cur = pypto.reshape(v_bf16, [8, 1, 128])  # 源 [1, 1024] → 目标 [8, 1, 128]
# ❌ 缺少 set_vec_tile_shapes(8, 1, 128)
pypto.assemble(v_cur, [0, pos, 0], cache)
```

**常见遗漏场景**：
- reshape 后紧跟 `assemble`：assemble 消费的是目标 shape，必须在 reshape 后、assemble 前设置匹配目标 shape 的 vec_tile
- reshape 后紧跟 `matmul`：matmul 由 cube_tile 控制，vec_tile 影响较小，但仍建议按目标 shape 设置
- 多个连续 reshape：每次 reshape 后都需确认 vec_tile 是否匹配

##### 2.4 常见基本块问题检查清单

| # | 检查项 | 症状 | 修复方法 |
|---|--------|------|---------|
| 1 | vec_tile 维度超过 tensor 实际维度 | 运行时异常或性能异常 | vec_tile 每维 ≤ 对应 tensor 维度 |
| 2 | cube L1 超过实际轴长 | L1 空间浪费，其他轴可用空间减少 | L1 设为 min(推荐值, 实际轴长)，并满足对齐 |
| 3 | K 轴 L1 过小导致重复载入 | GM 搬运开销大（尤其 K≥4096 时） | 增大 kL1，或用 `[kL0, kAL1, kBL1]` 独立配置 |
| 4 | N 轴 L1 过小导致切分次数多 | 任务数过多，调度开销大 | 增大 nL1 减少切分 |
| 5 | 多个 matmul 共用同一 TileShape | 不同 shape 的 matmul 用统一 tile 导致次优 | 每个 matmul 前独立设置 |
| 6 | reduce 轴被切分 | 额外 GM 搬运和调度开销 | 归约轴 tile 设为全长 |
| 7 | Decode M=1 未利用 K 轴驻留 | A 矩阵重复载入 | 使用 `[kL0, kAL1, kBL1]` 让 A 矩阵驻留 L1 |
| 8 | reshape 前按目标 shape 设 vec_tile | reshape 操作使用了不匹配的 tile 配置 | reshape 前按源 tensor shape 设，reshape 后按目标 shape 重设 |
| 9 | reshape 后 assemble 前缺少 vec_tile | assemble 使用了不匹配的 tile 配置 | reshape 后、assemble 前显式设置匹配目标 shape 的 vec_tile |

### 局部性能优化

#### 1. 输入矩阵格式优化

检查输入矩阵、尤其是 Shape 较大的权重矩阵是否可以提前以 NZ 格式存储。

**NZ 格式的数据搬运到 L1 的带宽更高。**

#### 2. Transpose 优化

矩阵乘前后有 transpose 时，可以尝试更换左右矩阵并使用左右矩阵转置的配置。

当 M 轴较大、N 轴较小时，使得左右矩阵有更大的尾轴，提升搬运带宽。

**⚠️ 重要原则**
- `transpose + matmul` 的结构，可以通过 matmul 的 `a_trans` 及 `b_trans` 参数进行配置，完成 op 融合。好处是，matmul 运算时，可以随路 transpose

#### 3. 原始输入 reshape 优化

**原则**：对原始输入（函数参数）的 reshape，必须挪到 loop 最外层之前执行，并使用 `inplace=True`。

```python
# ✅ 正确：reshape 挪到 loop 之前，inplace=True
q_grouped = pypto.reshape(query, [num_kv_heads, num_heads_per_group, head_dim], inplace=True)
k_cache = pypto.reshape(key_cache, [kv_len, num_kv_heads, head_dim], inplace=True)

for i in pypto.loop(num_blocks, ...):
    # loop 内直接使用已 reshape 的 tensor
    scores = pypto.matmul(q_grouped, k_cache_block, ...)

# ❌ 错误：reshape 放在 loop 内部，每次循环重复执行
for i in pypto.loop(num_blocks, ...):
    q_grouped = pypto.reshape(query, [num_kv_heads, num_heads_per_group, head_dim])  # 冗余
```

**注意**：只有原始输入可用 `inplace=True`，中间结果和输出 tensor 不能 inplace reshape。

#### 4. 冗余搬运优化

检查是否有不合理数据操作导致的冗余搬运：

- 更换 concat 为 assemble
- 尝试对 reshape 配置 `inplace = True` 参数

#### 5. 合轴优化

##### 5.1 尽可能减少循环体中 shape 的维度
**症状**
循环体内参与计算的 tensor 的 shape 的维度超过两维
**原因**
shape 维度太多，会导致处理复杂，此外，pto 指令对多维的 tensor 处理不友好，性能较差
**解决**
在循环体外部对输入先进行 `reshape`，并配置`inplace = True` 参数，对多维的 tensor 进行合轴处理。输出保持原有 shape 维度不变。

##### 5.2 合轴的输入输出分离原则

**只有原始输入（函数参数）可使用 `reshape(inplace=True)`，中间结果和输出 tensor 不能 inplace reshape。对原始输入的 reshape 必须使用 `inplace=True`，避免冗余数据拷贝。**

```python
# ✅ 正确：原始输入合轴时 inplace=True
query_2d = pypto.reshape(query, [batch * heads * seq_q, dim], inplace=True)
key_2d = pypto.reshape(key, [batch * heads * seq_kv, dim], inplace=True)
value_2d = pypto.reshape(value, [batch * heads * seq_kv, dim], inplace=True)

for b_idx in pypto.loop(batch, ...):
    for n_idx in range(heads):
        q_offset = b_idx * heads * seq_q + n_idx * seq_q + q_start
        q_block = pypto.view(query_2d, [BLOCK, dim], [q_offset, 0], ...)
        # ...
        # output 保持 4D 切片写入
        output[b_idx:b_idx+1, n_idx:n_idx+1, ...] = result_4d
```

```python
# ❌ 错误：output 也合轴为 2D，切片写入会得到全零结果
output_2d = pypto.reshape(output, [batch * heads * seq, dim], inplace=True)
output_2d[offset:offset+block, :] = result_2d  # 写入无效，输出全零
```

**原因**：inplace reshape 改变了 tensor 的内存视图，output 的切片写入依赖原始 shape 索引，reshape 后索引关系断裂导致写入失败。

##### 5.3 Vector 合轴 + 合图

**目标场景**：连续 vector 操作（含 reduce）的 tensor shape 超过 2D

**优化措施**：reshape 合轴为 2D → 设置适配 2D 的 vec_tile_shapes → `sg_set_scope` 强制合图 → reshape 回原维度

**⚠️ 关键约束**：
- Reduce op 尾轴必须 32B 对齐（FP32 下为 8 的倍数）
- vec_tile_shapes 归约轴不切分（第二维 = 归约轴全长）
- tile 数据量不超 UB：`第一维 × 第二维 × dtype字节数 × 3 ≤ 128KB`
- 合轴后必须显式设置 vec_tile_shapes

**🔥 案例**：[Decode Attention Vector 合轴优化](cases/vector-axis-merge-softmax.md)（-6.0%，任务数 -18.5%，含 4 轮迭代失败分析）


## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [GDR 算子案例](../../../../docs/tutorials/debug/performance_case_GDR.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
- [典型案例库](cases/README.md)

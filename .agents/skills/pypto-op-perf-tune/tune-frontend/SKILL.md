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

### 1. Loop 写法优化

**增加 root function 的大小，减少它们的个数**
由于不同 root function 之间的子图不能合并，而子图合并是 PyPTO 优化性能的关键手段。

#### 1.1 静态轴使用 Python for 循环

`pypto.loop` 方法会按当前轴循环展开成不同的 root function。因此静态轴上的循环应使用 Python 的 for 循环。

```python
# ✅ 推荐：静态轴使用 Python for
for i in range(batch_size):
    result[i] = process(data[i])

# ❌ 避免：静态轴使用 PyPTO loop
for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
    result[i] = process(data[i])
```

#### 1.2 减少循环次数，增加并行度
##### 1.2.1 如果外层的动态轴范围很大，使用切块处理（高优先级）

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

##### 1.2.2 如果内层的动态轴范围很大，调整切块大小或使用loop_unroll展开，增加并行度
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

##### 1.2.3 切块优化策略：外层切块 + 内层unroll

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

#### 1.3 尽可能合并 loop

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

### 2. TileShape 设置优化

TileShape 切分大小直接决定：
- 算子切分后的任务数量
- 实际执行时的分核数、计算轮次
- 算子的算数强度

**优化关键**：优化 Tiling 配置

#### 2.1 Matmul 初始 Tiling 配置

针对矩阵运算场景（A、B 矩阵均为 DT_BF16 或 DT_FP16 类型）：

```python
# Cube 的相关计算建议采用如下的 TileShape
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

**优点**：
- 在满足 L0 Buffer 约束的条件下达到较大的算数强度
- 后续进一步使用合图相关接口进行深度调优时，有机会开启 Double Buffer

**⚠️ 独立设置原则**：

同一算子中存在多个不同 shape 特征的 matmul 时，**必须在每个 matmul 前分别调用 `set_cube_tile_shapes`**，根据各 matmul 的实际 M/K/N 独立配置。不要用统一值，不同 shape 的最优 tile 不同。

**配置要点**：
1. **L1 不超过实际轴长**：`mL1/kL1/nL1` 不应超过对应维度实际大小，超过无意义（实际数据不够一个 tile，反而浪费 L1 空间导致其他轴可用空间减少）
2. **约束**：`L0 <= L1` 且 `L1 % L0 == 0`；BF16 下 L0/L1 需 16 元素对齐（32B）
3. **小轴不切**：如果某个轴本身很小（如 K=128），L0=L1=实际值，不切分
4. **大轴大 tile**：如果某个轴很大（如 N=2048 或 K=2048），用较大的 L1 减少任务数

**🔥 案例**：[多 Matmul 独立 TileShape 优化](cases/per-matmul-tile-shapes.md)（3 个 matmul 独立设 tile，-46.1%）

#### 2.2 Vector 初始 Tiling 配置

针对向量运算场景：

**配置原则**：
1. 满足特定 Operation 对 TileShape 的规格约束
2. 保证 Operation 的输入与输出 Tensor 可以在 UB 中分配内存
3. TileShape 不能过大也不能过小（数据块大小在 16 到 64KB 之间）
4. 尾轴 32B 对齐
5. 归约类计算尽可能不要在归约轴上进行切分

```python
# Vector 的相关计算建议采用如下的 TileShape
pypto.set_vec_tile_shapes(64, 512)
```

**归约轴切分问题示例**：

对于输入 Shape 为 (56, 1024) 的 RMSNorm：
- ❌ 对 reduce 轴切分：多个子图的输出需要在同一个子图进行 reduce 操作，产生 GM 搬运和调度开销
- ✅ 不对 reduce 轴切分：上下游子图合并，没有 GM 搬运和调度开销

### 3. 数据操作优化

#### 3.1 输入矩阵格式优化

检查输入矩阵、尤其是 Shape 较大的权重矩阵是否可以提前以 NZ 格式存储。

**NZ 格式的数据搬运到 L1 的带宽更高。**

#### 3.2 Transpose 优化

矩阵乘前后有 transpose 时，可以尝试更换左右矩阵并使用左右矩阵转置的配置。

当 M 轴较大、N 轴较小时，使得左右矩阵有更大的尾轴，提升搬运带宽。

**⚠️ 重要原则**
- `transpose + matmul` 的结构，可以通过 matmul 的 `a_trans` 及 `b_trans` 参数进行配置，完成 op 融合。好处是，matmul 运算时，可以随路 transpose

#### 3.3 原始输入 reshape 优化

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

#### 3.4 冗余搬运优化

检查是否有不合理数据操作导致的冗余搬运：

- 更换 concat 为 assemble
- 尝试对 reshape 配置 `inplace = True` 参数

### 4. ⚠️ 合轴优化

#### 4.1 尽可能减少循环体中 shape 的维度
**症状**
循环体内参与计算的 tensor 的 shape 的维度超过两维
**原因**
shape 维度太多，会导致处理复杂，此外，pto 指令对多维的 tensor 处理不友好，性能较差
**解决**
在循环体外部对输入先进行 `reshape`，并配置`inplace = True` 参数，对多维的 tensor 进行合轴处理。输出保持原有 shape 维度不变。

#### 4.2 合轴的输入输出分离原则

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

#### 4.3 Vector 合轴 + 合图

**目标场景**：连续 vector 操作（含 reduce）的 tensor shape 超过 2D

**优化措施**：reshape 合轴为 2D → 设置适配 2D 的 vec_tile_shapes → `sg_set_scope` 强制合图 → reshape 回原维度

**⚠️ 关键约束**：
- Reduce op 尾轴必须 32B 对齐（FP32 下为 8 的倍数）
- vec_tile_shapes 归约轴不切分（第二维 = 归约轴全长）
- tile 数据量不超 UB：`第一维 × 第二维 × dtype字节数 × 3 ≤ 128KB`
- 合轴后必须显式设置 vec_tile_shapes

**🔥 案例**：[Decode Attention Vector 合轴优化](cases/vector-axis-merge-softmax.md)（-6.0%，任务数 -18.5%，含 4 轮迭代失败分析）


## 性能优化建议库

### 建议 1：Loop 优化

| 问题 | 解决方案 | 代码示例 |
|------|---------|---------|
| 静态轴使用 pypto.loop | 改用 Python for | `for i in range(n):` |
| 多个独立 loop | 合并 loop | 合并到同一个 loop 内 |
| 内层动态轴切分 | 使用 loop_unroll | `pypto.loop_unroll(..., unroll_list=[64, 16, 4])` |
| 外层动态轴切分 | 使用静态切块 | `pypto.loop(b // b_block_size, ...)` |

### 建议 2：TileShape 优化

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| Cube 计算 | `[128, 128], [64, 256], [256, 256]` | 高算数强度 |
| Cube 计算（多 matmul） | 每个 matmul 前独立设置 | 按实际 M/K/N 分别配置，L1 不超实际轴长 |
| Vector 计算 | `64, 512` | UB 利用率高 |
| Reduce 操作 | 不切归约轴 | 避免额外 GM 搬运 |

### 建议 3：常量配置优化
在算子中写死的部分常量配置参数，可以尝试调整优化

### 建议 4：数据操作优化

| 问题 | 解决方案 |
|------|---------|
| 大矩阵搬运慢 | 使用 NZ 格式存储 |
| transpose 性能差 | 调整左右矩阵顺序 |
| concat 冗余搬运 | 使用 assemble |
| reshape 冗余搬运 | 配置 `inplace=True` |

### 建议 5：合轴优化

| 问题 | 解决方案 | 关键约束 |
|------|---------|---------|
| 计算 shape 超过 2D | 算子入口对输入 reshape 合轴 | 只读输入可合轴，输出不能 inplace reshape |
| 连续 vector 操作在 3D 下性能差 | 合轴为 2D + sg_set_scope 合图 | vec_tile_shapes 归约轴不切分，tile 数据量不超 UB |
| vector 操作产生过多子图 | sg_set_scope 强制合图 | 相邻且有上下游依赖的 op 才能合图 |


**优化优先级**：
1. ⭐⭐⭐ **任务粒度优化**（切块、合并loop、合轴） - **最重要**
2. ⭐⭐ **TileShape 优化**（Cube/Vector 推荐配置） - **很重要**
3. ⭐⭐ **loop_unroll 配置**（最内层）
4. ⭐ **常量配置调整**（BLOCK_SIZE 等）

## 调优流程

**⚠️ 重要：开箱性能调优不需要查看性能报告的详细分析，但需要对比基准执行时间！**

### 1. 建立性能基准

**首次运行算子用例**，记录基准性能：
```bash
python3 custom/operator_name/operator.py --run-mode npu
```

**记录基准执行时间**：
```
基准执行时间: XXX us
```

### 2. 迭代优化循环

```
┌───────────────────────────────────┐
│     开箱调优迭代流程              │
├───────────────────────────────────┤
│                                   │
│  1. 选择一个优化点                │
│     ├─ Loop 写法优化              │
│     ├─ TileShape 设置优化         │
│     └─ 数据操作优化               │
│                                   │
│  2. 修改代码                      │
│     └─ 每次只修改一个参数         │
│                                   │
│  3. 验证精度 ⭐                   │
│     ├─ 运行测试用例               │
│     └─ 失败，尝试解决，不行则回退 │
│                                   │
│  4. 对比性能 ⭐                   │
│     ├─ 记录新执行时间             │
│     ├─ 对比基准执行时间           │
│     └─ 计算提升百分比             │
│                                   │
│  5. 判断是否保留                  │
│     ├─ 性能提升：保留修改         │
│     └─ 性能下降：回退修改         │
│                                   │
│  6. 检查终止条件                  │
│     ├─ 达到性能目标               │
│     └─ 连续5次优化无提升          │
│                                   │
└───────────────────────────────────┘
```

### 3. 优化检查清单

**🔥 P0 - 任务粒度（最重要）**：
- [ ] **Matmul 的 M/N/K 轴是否充分利用硬件？**（M 轴 < 8 是常见问题）
- [ ] **任务总数是否过多？**（> 1000 可能调度开销大）
- [ ] **合轴优化：shape 的维度是否超过 2 维？**（超过两维，搬运及计算的开销较大）

**P1 - Loop 写法**：
- [ ] 静态轴是否使用 Python for
- [ ] 是否可以合并独立 loop

**P2 - TileShape 设置**：
- [ ] Cube 计算：是否使用推荐配置
- [ ] Vector 计算: 是否使用推荐配置
- [ ] 归约轴是否避免切分

**P3 - 常量配置**：
- [ ] BLOCK_SIZE 是否合理（16/32/64）

**P4 - 数据操作**：
- [ ] 输入矩阵格式是否优化（NZ 格式）
- [ ] transpose 配置是否合理
- [ ] reshape 操作是否可以消除
- [ ] 是否存在冗余搬运

### 4. 性能对比示例

```markdown
## 优化记录

| 轮次 | 优化内容 | 执行时间(us) | 提升比例 | 精度结果 |
|------|---------|-------------|---------|---------|
| 基准 | 无优化 | 27469.66 | - | 通过 |
| 1 | 静态轴改用Python for | 25123.45 | 8.5% | 通过 |
| 2 | TileShape优化 | 22456.78 | 10.6% | 通过 |
| 3 | 合并loop | 21345.12 | 4.9% | 通过 |
```

## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [GDR 算子案例](../../../../docs/tutorials/debug/performance_case_GDR.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
- [典型案例库](cases/README.md)

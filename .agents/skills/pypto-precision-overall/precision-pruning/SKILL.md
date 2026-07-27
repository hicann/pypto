---
name: precision-pruning
description: PyPTO 用例剪枝技能。在能复现精度问题的前提下，通过缩减 tile 循环次数简化测试用例。触发词：用例剪枝、剪枝、化简用例、缩减循环、pruning。
---

# 用例剪枝

缩减 tile 循环次数以加速精度调试，同时**保持原用例尾块行为不变**。

## 核心公式

对每个 tile 维度：`tail = dim % tile`

| tail | 行为 | new dim | 迭代数 |
|------|------|---------|--------|
| `= 0` | 无尾块 | `tile` | 1 轮 |
| `> 0` | 有尾块，保留 | `tile + tail` | 2 轮（1 满 + 1 尾） |

loop_unroll：`effective_tile = tile × max_unroll`，套用同上公式，tail 是元素级余数。
硬编码 `pypto.loop(N, ...)`：shape 同上，loop 字面量同步改为 1（无尾）或 2（有尾）。

## 工作流程

### 1. 分析

读代码，提取每个 tile 维度的 `dim`、`tile`、`max_unroll`：

| 循环写法 | 如何提取 |
|----------|----------|
| `m_loop = (m + tile_m - 1) // tile_m` | dim=`m`, tile=`tile_m` |
| `m_loop = (m + tile_m*4 - 1) // (tile_m*4)` + `loop_unroll` | dim=`m`, tile=`tile_m`, unroll=4 |
| `pypto.loop(2, name="B", ...)` | 硬编码，需要从测试参数中找到对应 dim/tile |

### 2. 计算

```
effective_tile = tile × max_unroll          (普通 loop: max_unroll=1)
unroll_iters   = ⌈dim / effective_tile⌉

if unroll_iters <= 1 → dim 已是最小，跳过
tail = dim % effective_tile
new_dim = effective_tile + tail            (tail=0 时即 effective_tile)
if new_dim == dim → 已是最小，跳过
```

对于 `loop_unroll(m_loop, unroll_list=[4,2,1])`：提取最大的 unroll 值作为 `max_unroll`，套同上公式。`effective_tile = tile × max_unroll`，tail 是元素级余数，保证不会膨胀。

对于硬编码 `pypto.loop(N, ...)`：shape 同上公式，loop 字面量 `N` 改为：
```
dim <= tile           → 1
dim > tile 且无尾块    → 1
dim > tile 且有尾块    → 2
```

### 3. 修改

- **shape 驱动的循环**：只改测试入口的 shape 字面量，kernel 不动
- **硬编码循环**：改 shape + 改 `pypto.loop(N, ...)` 中的 `N`
- golden 和 tile_fwk_config 中依赖 shape 的参数同步修改
- tile size 保持不动

### 4. 验证

**必须上板跑测试**，对比剪枝前后的精度行为：

| 原用例 | 剪枝成功 ✓ | 剪枝失败 ✗ |
|--------|-----------|-----------|
| 原本正确 | 仍正确 | 出现差异 |
| 原本错误 | 仍错误 | 差异消失 → 问题在循环迭代过程中 |

失败时逐步增加迭代：`dim = tile * N + tail`，N 从 2 递增直到差异复现。

### 5. 提交

另存为 `*_pruned.py`，头部标注剪枝信息和验证状态。

## 循环模式速查

| 模式 | kernel 写法 | 剪枝要点 |
|------|------------|----------|
| tile 循环 | `m_loop = (m + tile - 1) // tile` | 改 shape，kernel 不动 |
| loop_unroll | `pypto.loop_unroll(N, unroll_list=[...])` | `effective_tile = tile × max_unroll` |
| 硬编码 | `pypto.loop(2, name="B", ...)` | 改 shape + 改 loop 字面量 |
| 非 tile | `for i in range(32)` | 跳过，无法剪枝 |

## 常见问题

**剪枝后精度差异消失**：问题在循环迭代过程中（累积误差、状态传递），不是尾块问题。按 `dim = tile * N + tail` 逐步增加迭代直到差异复现。

**buffer 不够**：检查 tile_fwk_config，按比例缩小依赖 shape 的 buffer 参数。

**多个 kernel**：分别分析，确保 shape 一致。

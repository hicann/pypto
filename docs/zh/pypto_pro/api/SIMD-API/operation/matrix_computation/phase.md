# phase 使用约束（AccPhase / STPhase）

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

`matmul` / `matmul_acc` 的 `phase` 参数（`pl.AccPhase`）与 `store` / `store_tile` 的 `phase` 参数（`pl.STPhase`）共同控制 Cube（矩阵乘）与 FixPipe（L0C→GM 搬运）之间的 **unit_flag 硬件握手**。正确使用 phase 可以省去软件同步、提升流水并行度；使用不当则会导致精度问题或设备卡死。

## 硬件 unit_flag 机制

### matmul / matmul_acc（AccPhase）

`phase=pl.AccPhase.Partial` 或 `phase=pl.AccPhase.Final` 均会使能硬件的 unitFlag 功能：

- **unit_flag = 0**：硬件直接写入 Acc（L0C）。
- **unit_flag = 1**：硬件写入 Acc 的操作会被暂停，直到 unit_flag 被设置回 0。

两者的区别：

| 模式 | 检查 unit_flag | 设置 unit_flag |
|---|---|---|
| `Partial` | 是（等待 unit_flag=0 才写入） | 否（不改变 unit_flag） |
| `Final` | 是（等待 unit_flag=0 才写入） | 是（写入后将 unit_flag 置为 1） |

### store / store_tile（STPhase）

`phase=pl.STPhase.Partial` 或 `phase=pl.STPhase.Final` 均会使能硬件的 unitFlag 功能：

- **unit_flag = 1**：硬件直接读 Acc（L0C）。
- **unit_flag = 0**：硬件读 Acc 的操作会被暂停，直到 unit_flag 被设置为 1。

两者的区别：

| 模式 | 检查 unit_flag | 设置 unit_flag |
|---|---|---|
| `Partial` | 是（等待 unit_flag=1 才读取） | 否（不改变 unit_flag） |
| `Final` | 是（等待 unit_flag=1 才读取） | 是（读取后将 unit_flag 置为 0） |

## phase 与自动同步的关系

| 配置 | 自动同步 | 同步机制 |
|---|---|---|
| 配置了 phase | **不自动插入同步** | 靠硬件 unit_flag 实现 Matmul（M 流水）与 FixPipe 之间的同步 |
| 未配置 phase | **自动插入同步** | 框架自动插入 M 流水与 FixPipe 流水的软件同步 |

## 使用约束

如果 phase 使用不当，可能会导致精度问题或者卡死现象。使用时必须保证：

1. **配对使用**：如果 `matmul` 或 `matmul_acc` 使用了 phase，对应的 `store` 或 `store_tile` 也需要使用 phase。
2. **Final 收尾**：对于同一块 L0C，`matmul` 或 `matmul_acc` 的最后一轮写操作，以及 `store` 或 `store_tile` 的最后一轮读操作，必须使用 `Final` 模式。

## 错误案例

### 错误案例一：matmul 无 Final 导致卡死

```python
pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

**现象**：卡死。

**原因**：`matmul` 使用 `Partial` 只检查 unit_flag 不会设置 unit_flag，unit_flag 始终为 0。`store` 使用 `Final` 等待 unit_flag 被设置成 1 才能读取，但 unit_flag 永远不会被置 1，FixPipe 一直等待 → 卡死。

### 错误案例二：store 未配置 phase 导致精度问题

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    if ki == 0:
        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
    else:
        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
pl.store(out, ac, [0, 0])
```

**现象**：精度问题。

**原因**：

- **软件同步角度**：`store` 未配置 phase，框架会自动插入 FixPipe 流水同步；但 `matmul` 配置了 phase，不会自动插入 M 流水同步。两种同步机制不匹配。
- **硬件 unit_flag 角度**：`store` 未配置 phase，不受硬件 unit_flag 值影响，FixPipe 不会等待 unit_flag。

上述两种情况，FixPipe 搬运 L0C 数据都不会严格等待 Matmul 计算完成，导致读到未完成的数据。

### 错误案例三：循环内 store(Final) 后 matmul 卡死

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
    pl.store(out, ac, [0, 0], phase=pl.AccPhase.Partial)
```

**现象**：卡死。

**原因**：

- 第一轮循环：`matmul(Final)` 将 unit_flag 设置成 1，`store(Partial)` 能将 L0C 数据搬运出去，但未改变 unit_flag 的值（仍为 1）。
- 第二轮循环：由于共用同一块 L0C 内存，`matmul` 等待 unit_flag 变更为 0，但 unit_flag 始终为 1 → 卡死。

## 正确用法示例

### 单次 matmul（无 K 维累加）

不传 phase，框架自动插入同步：

```python
pl.matmul(ac, al, br)
pl.store(out, ac, [0, 0])
```

### K 维分块累加（多块）

首块 `Partial`，中间块 `Partial`，末块 `Final`，store 用 `STPhase.Final`：

```python
with pl.section_cube():
    ac = acc.current()
    for k in pl.range(0, K_TOTAL, TILE_K):
        ...
        if k == 0:
            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)        # 首块
        elif k < K_TOTAL - TILE_K:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial) # 中间块
        else:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)   # 末块
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)                # Final 收尾
```

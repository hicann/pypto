# pypto_pro.language.matmul_acc

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

在已有累加器值的基础上继续累加一次矩阵乘积：`dst_tile = acc_tile + lhs_tile × rhs_tile`，数据通路为 L0C(Acc) + L0A(Left) × L0B(Right) → L0C(Acc)。主要用于 K 维分块累加——把大 K 切成多块，首块用 [`pypto_pro.language.matmul`](matmul.md) 写入，其余块用 `matmul_acc` 累加到同一个 L0C 累加器。

## 函数原型

```python
pypto_pro.language.matmul_acc(dst_tile, acc_tile, lhs_tile, rhs_tile, *, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是Acc/L0C Tile，存放累加结果 |
| `acc_tile` | 输入 | 只能是Acc/L0C Tile，已有的累加器值（通常与dst_tile为同一tile） |
| `lhs_tile` | 输入 | 只能是L0A/Left Tile，左矩阵 |
| `rhs_tile` | 输入 | 只能是L0B/Right Tile，右矩阵 |
| `phase` | 输入 | 可选，用于K维分块累加场景 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：FP16、BF16、FP32、INT32<br>shape：`[M, N]`，需与 `acc_tile` 一致<br>地址配置：<br>• 只能是 Acc/L0C 内存空间，其他空间报错<br>• `layout=pl.NZ`；FP32/INT32 需设 `fractal`（FP32 默认 1024）<br>• 通常与 `acc_tile` 为同一个 tile，实现 in-place 累加 |
| `acc_tile` | 输入 | 数据类型：FP16、BF16、FP32、INT32<br>shape：`[M, N]`，需与 `dst_tile` 一致<br>地址配置：<br>• 只能是 Acc/L0C 内存空间，其他空间报错<br>• 通常与 `dst_tile` 为同一个 tile（K 维循环里持续累加）<br>• 其值应为前一步累加的结果；首块请改用 `matmul` 而非 `matmul_acc` |
| `lhs_tile` | 输入 | 数据类型：FP16、BF16、FP32、INT8<br>shape：`[M, K]`<br>地址配置：<br>• 只能是 L0A/Left 内存空间，其他空间报错<br>• A3 默认 `layout=pl.ZZ`；A5 默认 `layout=pl.NZ` |
| `rhs_tile` | 输入 | 数据类型：与 `lhs_tile` 一致<br>shape：`[K, N]`，K 维需与 `lhs_tile` 的 K 一致<br>地址配置：<br>• 只能是 L0B/Right 内存空间，其他空间报错<br>• `layout=pl.ZN` |
| `phase` | 输入 | 可选，K 维分块累加时控制 fixpipe 写回 GM 的 unit_flag：<br>• 不传（默认）：单次累加，无分块控制<br>• `pl.AccPhase.Partial`：中间累加步，表示后续还有 K 块<br>• `pl.AccPhase.Final`：最终步，表示 K 累加结束、可写回 GM |

## 调用示例

完整 kernel：计算 `C[128,128] = A[128,K] @ B[K,N]`，K 维按 128 分块累加。cube kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。首块用 `matmul` 写入累加器，其余块用 `matmul_acc` 累加到同一个 `ac`，K 循环结束后写回 GM。

K 维分块累加链对正确性有三个硬性要求（缺一会精度错，已上板验证）：

1. **每步 matmul / matmul_acc 都要传 `phase`**：首/中间块 `phase=pl.AccPhase.Partial`，最后一块 `phase=pl.AccPhase.Final`；写回 GM 的 `store` 也传 `phase=pl.STPhase.Final`。
2. **L0C 累加器设 `fractal=1024`**（FP32）。
3. **cube 段用 `pypto_pro.language.system.set_mm_layout_transform(enabled=True)` 开启**，段末 `enabled=False` 关闭。

此外 L1 与 L0A/L0B 都用 `next()` 轮转（双 mutex_id），让相邻 K 块的搬运与计算在不同 buffer 上并行。

```python
import pypto_pro.language as pl

TILE = 128
K_SIZE_ACC = 256     # 分 2 个 TILE 块累加


@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    # L1 / L0 双缓冲（next() 轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    # Acc：K 累加要求 fractal=1024
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE):     # K 维分块（累加）
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)          # 首块写入累加器
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)    # 末块累加（K=2 块）
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```

> 上例 K 恰为 2 块，故 `k == 0` 首块传 `pl.AccPhase.Partial`、否则末块传 `pl.AccPhase.Final`。若 K 超过 2 块，应让所有中间块传 `phase=pl.AccPhase.Partial`，仅最后一块传 `phase=pl.AccPhase.Final`。

# pypto_pro.language.matmul

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

完成一次矩阵乘法 `dst_tile = lhs_tile × rhs_tile`，数据通路为 L0A(Left) × L0B(Right) → L0C(Acc)。输入矩阵需先搬运到 L0A/L0B 内存空间（一般经 GM → L1 → L0A/L0B 两跳），结果写入 L0C 累加器。

如果要在已有累加结果上继续累加（K 维分块的非首块），使用 [`pypto_pro.language.matmul_acc`](matmul_acc.md)。

## 函数原型

```python
pypto_pro.language.matmul(dst_tile, lhs_tile, rhs_tile, *, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是Acc/L0C Tile，存放矩阵乘法结果 |
| `lhs_tile` | 输入 | 只能是L0A/Left Tile，左矩阵 |
| `rhs_tile` | 输入 | 只能是L0B/Right Tile，右矩阵 |
| `phase` | 输入 | 可选，用于K维分块累加场景 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：FP16、BF16、FP32、INT32（累加器精度通常高于输入，如 FP16 输入对应 FP32 累加）<br>shape：`[M, N]`<br>地址配置：<br>• 只能是 Acc/L0C 内存空间，其他空间报错<br>• `layout=pl.NZ`；FP32/INT32 累加器需设 `fractal`（FP32 默认 1024）<br>• 支持通过 `valid_shape=[-1, -1]` + `set_validshape` 设置尾块有效大小 |
| `lhs_tile` | 输入 | 数据类型：FP16、BF16、FP32、INT8<br>shape：`[M, K]`<br>地址配置：<br>• 只能是 L0A/Left 内存空间，其他空间报错<br>• A3 默认 `layout=pl.ZZ`；A5 默认 `layout=pl.NZ` |
| `rhs_tile` | 输入 | 数据类型：与 `lhs_tile` 一致<br>shape：`[K, N]`，K 维需与 `lhs_tile` 的 K 一致<br>地址配置：<br>• 只能是 L0B/Right 内存空间，其他空间报错<br>• `layout=pl.ZN` |
| `phase` | 输入 | 可选，K 维分块累加时控制 fixpipe 写回 GM 的 unit_flag：<br>• 不传（默认）：单次乘法，无分块累加<br>• `pl.AccPhase.Partial`：中间累加步，表示后续还有 K 块<br>• `pl.AccPhase.Final`：最终步，表示 K 累加结束、可写回 GM<br>详见 [`matmul_acc`](matmul_acc.md) 的分块累加用法 |

## 调用示例

完整 kernel 计算 `C[M,N] = A[M,K] @ B[K,N]`，用 `make_tile_group` + `auto_mutex` 管理 L1/L0A/L0B/L0C 缓冲。L1 暂存用 `next()` 轮转开 ping-pong 双缓冲，L0A/L0B/L0C 用单 mutex_id 的 group 配 `current()`。开启 `auto_mutex=True` 后，相邻搬运与计算间的同步由框架按 tile 的 mutex 自动插入，无需手写 `sync_src`/`sync_dst`。

下面是 K 恰好为一个 tile（128）的单次 matmul：每个 `[i, j]` 块一次 `matmul` 直接写回 GM，不涉及 K 维累加，因此不需要 `phase` / `fractal`。

> K 维分块累加（`phase` + `fractal` + `set_mm_layout_transform`）见 [`pypto_pro.language.matmul_acc`](matmul_acc.md) 的调用示例。

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128      # K 恰好一个 tile，无需分块累加
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A / L0B / Acc 单 tile group（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):          # M 维分块
            for j in pl.range(0, N_SIZE, TILE):      # N 维分块
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

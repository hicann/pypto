# pypto_pro.language.move

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

在 L1、L0A/L0B、L0C、UB 等各级内存之间搬运 tile（如 L1→L0A/L0B、L0C(Acc)→UB(Vec)、UB→L1 等）。具体走哪条硬件搬运通路、用哪条流水，由**源与目的 tile 的内存空间**决定。

可在搬运的同时融合 ReLU、预量化，或经 fixpipe 做量化（`fp_tile`）；也可通过 `offset` 从一块较宽的源 tile 中提取子块（对应后端 `pto::TEXTRACT`）。

与 [`pypto_pro.language.load`](load.md)/[`pypto_pro.language.store`](store.md)（tensor↔tile，跨 GM）不同，`move` 是 tile↔tile，不涉及 GM。

## 函数原型

```python
pypto_pro.language.move(dst_tile, src_tile, offset=None, *, acc_to_vec_mode=None,
        relu_pre_mode=None, pre_quant_scalar=None, fp_tile=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 目标 tile，搬入目的地；其内存空间决定 TMOV 变体 |
| `src_tile` | 输入 | 源 tile |
| `offset` | 输入 | 可选，`[offset_m, offset_k]`，从较宽的源 tile 中提取子块 |
| `acc_to_vec_mode` | 输入 | 可选，Acc→Vec 搬运模式 |
| `relu_pre_mode` | 输入 | 可选，搬运时融合 ReLU |
| `pre_quant_scalar` | 输入 | 可选，预量化标量 |
| `fp_tile` | 输入 | 可选，fixpipe 量化参数 tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>内存空间与源共同决定流水（见“流水类型”）；首地址必须 32 字节对齐 |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>使用 `offset` 提取子块时，源 tile 须不小于目的 tile |
| `offset` | 输入 | 二维 `[offset_m, offset_k]`，单位为元素个数<br>以源 tile 声明的物理 `shape` 为坐标系；`valid_shape` 不改变 offset 的坐标原点和计量单位，每一维须满足 `0 <= offset < src_tile.shape`<br>实际提取范围为 `[offset, offset + dst_tile.valid_shape)`。使用尾块时，完整的 `dst_tile.shape` 可以超出 offset 后的剩余范围，但实际提取范围不得超出 `src_tile.shape`；若源 tile 设置了更小的 `valid_shape`，实际提取范围还须位于该有效区域内 |
| `acc_to_vec_mode` | 输入 | 取 `pl.AccToVecMode.SingleModeVec0`/`pl.AccToVecMode.DualModeSplitM`/`pl.AccToVecMode.DualModeSplitN`；仅在源为 `Acc`、目的为 `Vec` 时有意义。`fp_tile` 存在时只支持单 vec 模式（`DualModeSplitM`/`DualModeSplitN` 报错） |
| `relu_pre_mode` | 输入 | 默认 `None`（不融合 ReLU）；可取 `pl.ReluPreMode.NormalRelu` |
| `pre_quant_scalar` | 输入 | 整数预量化标量；与 `fp_tile` 互斥 |
| `fp_tile` | 输入 | 提供时改走 `move_fp` 路径（fixpipe 量化）；与 `pre_quant_scalar` 互斥，且只支持单 vec 模式 |

## 流水类型

由源/目的内存空间决定：

| 源 → 目的 | 流水 |
|---|---|
| Acc(L0C) → Vec(UB) | FIX（fixpipe） |
| Mat(L1) → Left/Right(L0A/L0B) | MTE1 |
| Mat(L1) → Vec(UB) | V |
| Mat(L1) → 其他 | FIX |
| Vec(UB) → Mat(L1) | MTE3 |
| 其余 | V |

## 调用示例

下面是一个完整 matmul kernel：`pypto_pro.language.load` 把左右矩阵从 GM 载入 L1，`pypto_pro.language.move` 再把 L1 数据搬到 L0A/L0B 供 cube 计算——`move` 在此承担 L1→L0A/L0B 这一步。cube kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def matmul_move_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    tt_right = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    tt_acc = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)

    a_l1 = pl.make_tile_group(type=tt_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_mat, addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_c_l0c = c_l0c.current()
        pl.load(cur_a_l1, a, [0, 0])     # GM -> L1
        pl.load(cur_b_l1, b, [0, 0])
        pl.move(cur_a_l0a, cur_a_l1)            # L1 -> L0A
        pl.move(cur_b_l0b, cur_b_l1)            # L1 -> L0B
        pl.matmul(cur_c_l0c, cur_a_l0a, cur_b_l0b)
        pl.store(out, cur_c_l0c, [0, 0])     # L0C -> GM（源在 Acc，走 FIX 流水）
```

其他典型用法（节选）：

```python
# L0C(Acc) → UB(Vec)，搬运时预量化
pl.move(vec_tile, acc, pre_quant_scalar=DEQ_SCALAR_BITS, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)

# 从宽源 tile 提取子块（TEXTRACT）
pl.move(cur_a_left, a_wide_slot, offset=[0, KL0])
```

## pl.AccToVecMode.DualModeSplitM / pl.AccToVecMode.DualModeSplitN 尾块处理

### 硬件约束

- **pl.AccToVecMode.DualModeSplitM**：对 M 轴进行切分，硬件会往每一块 Vec 写 `M/2 * N` 数据，其中 M 必须是 2 的倍数。
- **pl.AccToVecMode.DualModeSplitN**：对 N 轴进行切分，硬件会往每一块 Vec 写 `M * N/2` 数据，其中 N 必须是 32 的倍数。

这里的M和N指的是元素个数，与数据类型无关。

从 GM→L1→L0A/L0B→L0C 的过程中，所有 tile 的 `valid_shape` 始终使用**实际尾块大小**。在 L0C→Vec 的 `move` 搬运阶段，如果使用 `pl.AccToVecMode.DualModeSplitM` 或 `pl.AccToVecMode.DualModeSplitN`，框架会**自动**在 move 之前对 L0C 的 `valid_shape` 做向上对齐，用户无需手动设置对齐后的值，但需要了解在Vec侧的切分策略：

- **M轴 切分策略**：M 轴向上对齐到2的倍数，Vec0（sub_id=0）得到前 `aligned_M / 2` 行，Vec1（sub_id=1）得到剩余 `原始M - aligned_M / 2` 行。例如 M=33，框架对齐到 34，Vec0 得到 17 行，Vec1 得到 33-17=16 行。
    - 当尾块 M = 1 时，Vec0 直接取实际尾块大小，Vec1 为 0。
- **N轴 切分策略**：N 轴向上对齐到32的倍数，Vec0（sub_id=0）得到前 `aligned_N / 2` 列，Vec1（sub_id=1）得到剩余 `原始N - aligned_N / 2` 列。例如 N=33，框架对齐到 64，Vec0 得到 32 列，Vec1 得到 33-32=1 列。再例如 N=65 时，框架对齐到 96，Vec0 得到 48 列，Vec1 得到 65-48=17 列。
    - 当尾块 N ≤ 16 时，Vec0 直接取实际尾块大小，Vec1 为 0。

### 总体流程

```text
GM → L1 (load)     : valid_shape = 实际尾块大小 (如 33)
L1 → L0A/L0B (move): valid_shape = 实际尾块大小 (如 33)
L0A/L0B → L0C (matmul): Acc valid_shape = 实际尾块大小 (如 33)
L0C → Vec (move)   : 框架自动对齐 Acc valid_shape（用户无需感知）

Vec侧              : Vec0/Vec1 valid_shape = 切分后的实际份额（用户需自行计算）
```

### pl.AccToVecMode.DualModeSplitM 尾块（M 轴按2对齐对半切分）

`pl.AccToVecMode.DualModeSplitM` 要求 M 为偶数。尾块 M 为奇数时，框架在 `move` 之前自动将 Acc 的 `validRow` 向上对齐到 2 的倍数。**用户无需手动对齐。**

- **框架自动对齐**：`validRow` 从 `valid_m` 变为 `(valid_m + 1) / 2 * 2`（如 33 → 34）
- **V 侧切分计算**：`v0 = (valid_m + 1) // 2 * 2 // 2`（如 34 → v0=17），`v1 = valid_m - v0`（如 33-17=16）

```python
# cube section: 用户只需设置实际 valid_m，无需手动对齐
pl.set_validshape(ac, [valid_m, N])       # valid_m=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # 框架自动对齐

# vector section: V 侧用户自行计算 Vec0/Vec1 中 Vec 实际大小
v0 = (valid_m + 1) // 2 * 2 // 2          # v0 = 17
v1 = valid_m - v0                         # v1 = 16
if sub_id == 0:
    pl.set_validshape(vec, [v0, N])
else:
    pl.set_validshape(vec, [v1, N])
```

### pl.AccToVecMode.DualModeSplitN 尾块（N 轴按 32 对齐切分）

`pl.AccToVecMode.DualModeSplitN` 要求 N 为 32 的倍数。尾块 N 不满足时，框架在 `move` 之前自动将 Acc 的 `validCol` 向上对齐到 32 的倍数。**用户只需在 matmul 前设置实际 valid_n，无需手动对齐。**

- **框架自动对齐**：`validCol` 从 `valid_n` 变为 `(valid_n + 31) / 32 * 32`（如 33 → 64）
- **V 侧切分计算**：`v0 = (valid_n + 31) // 32 * 32 // 2`（如 64 → v0=32），`v1 = valid_n - v0`（如 33-32=1）
- **尾块 ≤ 16 的特殊情况**：当尾块 N ≤ 16 时，`v0` 直接取实际尾块大小（`v0 = valid_n`），`v1 = 0`。

```python
# cube section: 用户只需设置实际 valid_n，无需手动对齐
pl.set_validshape(ac, [TILE, valid_n])     # valid_n=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)  # 框架自动对齐

# vector section: V 侧用户自行计算 v0/v1
v0 = (valid_n + 31) // 32 * 32 // 2        # v0 = 32
v1 = valid_n - v0                          # v1 = 1
if sub_id == 0:
    pl.set_validshape(vec, [TILE, v0])
else:
    pl.set_validshape(vec, [TILE, v1])
```

### 尾块切分规则总结

| 模式 | 框架自动对齐 | Vec0 份额 | Vec1 份额 | 非对齐尾块示例（M/N=33） | 对应用例 |
|------|------------|----------|----------|---------------------|---------|
| `pl.AccToVecMode.DualModeSplitM` | M 向上对齐到 2 的倍数 | `aligned_M // 2` | `原始M - v0` | M=33→对齐34→v0=17, v1=16 | `test_split_m_odd_tail` |
| `pl.AccToVecMode.DualModeSplitN` | N 向上对齐到 32 的倍数 | `aligned_N // 2` | `原始N - v0` | N=33→对齐64→v0=32, v1=1 | `test_split_n_odd_tail` |

> **注意**：
>
> - GM→L1→L0A/L0B→L0C 全程使用**实际尾块大小**的 `valid_shape`，确保 matmul 只计算有效数据。
> - L0C→Vec 的 `move` 时，框架自动对 Acc 的 `valid_shape` 做对齐，用户无需手动设置对齐值。
> - V 侧（vector section）用户需自行计算 Vec0/Vec1 并设置 Vec 的 `valid_shape`，框架不自动处理。
> - 切 M 轴时，若 M=1，则 Vec0 = 1，Vec1 = 0。
> - 切 N 轴时，若 N<=16，则 Vec0 = N，Vec1 = 0。

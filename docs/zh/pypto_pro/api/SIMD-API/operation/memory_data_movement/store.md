# pypto_pro.language.store

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

把 L1/UB Tile 的结果按**绝对元素坐标**写回 GM，是 [`pypto_pro.language.load`](load.md) 的反向操作。写出过程中可顺带融合 ReLU、预量化，或对目的地址做原子累加。

源 tile 可以来自不同内存空间（如 `Vec`(UB)、`Acc`(L0C)）：源在 `Acc` 时由 FIX（fixpipe）流水写回 GM，其余情况走 MTE3 流水。

如果希望按"第几块 tile"定位写出位置，需要使用 [`pypto_pro.language.store_tile`](store_tile.md)。

## 函数原型

```python
pypto_pro.language.store(dst_tensor, src_tile, offsets, *, relu_pre_mode=None, pre_quant_scalar=None,
         fp_tile=None, tile_dims=None, atomic=pl.AtomicType.AtomicNone, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 目标 GM tensor，写出目的地 |
| `src_tile` | 输入 | 源 tile，待写回 GM 的数据 |
| `offsets` | 输入 | GM 写入的各维**绝对元素偏移**，长度等于 `dst_tensor` 维数 |
| `relu_pre_mode` | 输入 | 可选，写出前融合 ReLU |
| `pre_quant_scalar` | 输入 | 可选，写出前预量化标量 |
| `fp_tile` | 输入 | 可选，fixpipe 量化参数 tile |
| `tile_dims` | 输入 | 可选，Tile 维度在目标 tensor 维度中对应哪几根轴 |
| `atomic` | 输入 | 原子写模式 |
| `phase` | 输入 | 可选，matmul 累加链路的 fixpipe 写回 GM 阶段 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 数据类型：b8、b16、b32、b64<br>layout：支持 `ND`、`DN`、`NZ`<br>offsets 换算后的写入范围不得越过对应维度 shape<br>原子累加时须预先初始化为零（或已知初值），累加在其上叠加 |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>内存空间：源在 `Acc`(L0C) 时由 FIX 流水写回 GM，其余（如 `Vec`/UB）走 MTE3 流水<br>首地址必须 32 字节对齐 |
| `offsets` | 输入 | 单位元素个数，大小不超过对应维度的 shape，不支持负数索引 |
| `relu_pre_mode` | 输入 | 默认 `None`（不融合 ReLU）；可取 `pl.ReluPreMode.NormalRelu`；与 `fp_tile` 互斥 |
| `pre_quant_scalar` | 输入 | 整数，以 i64 位模式编码；与 `fp_tile` 互斥 |
| `fp_tile` | 输入 | 提供时改走 `store_fp` 路径，与 `relu_pre_mode`、`pre_quant_scalar`、`phase` 均互斥 |
| `tile_dims` | 输入 | 只支持配置 tensor 维度范围内的 dim，只支持二维数组配置，其余配置报错 |
| `atomic` | 输入 | `pl.AtomicType.AtomicNone`（默认，覆盖写）或 `pl.AtomicType.AtomicAdd`（原子累加，硬件对每个目的地址做元素级加法） |
| `phase` | 输入 | `pl.STPhase.Unspecified`/`pl.STPhase.Partial`（中间累加步）/`pl.STPhase.Final`（最终步）；用硬件 unit_flag 接管 cube/fixp 间的握手，仅在 matmul 多步累加写回 GM 时使用；与 `fp_tile` 互斥 |

## 流水类型

MTE3（L1/UB → GM 的搬出流水）；当 `src_tile` 位于 `Acc`(L0C) 时为 FIX（fixpipe）。

## 调用示例

下面是一个完整 kernel：从 GM 载入两个输入，相加后用 `pypto_pro.language.store` 把 UB 上的结果写回 GM。vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

其他典型用法（节选）：

```python
# 写出前融合 ReLU
pl.store(relu_out, acc, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)

# 原子累加（reduce 场景，多核/多步累加到同一 GM 位置）
pl.store(dk_out, fp32_row_tile, [b_id, g_id, single_indice, 0], atomic=pl.AtomicType.AtomicAdd)

# matmul 累加链路最终步写回 GM
pl.store(out_tensor, acc.current(), [i, j], phase=pl.STPhase.Final)
```

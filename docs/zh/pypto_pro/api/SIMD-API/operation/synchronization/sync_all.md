# pypto_pro.language.system.sync_all

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

全核同步：hard 模式用 FFTS 硬件信号无需 workspace；soft 模式用 GM 轮询需要 workspace。

## 函数原型

```python
pypto_pro.language.system.sync_all(workspaces=None, *, core_type=pl.SyncCoreType.MIX, mode=pl.SyncAllMode.HARD)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `workspaces` | 输入 | 仅 soft 模式需要的 workspace 列表 |
| `core_type` | 输入 | 同步涉及的核类型 |
| `mode` | 输入 | 同步模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `workspaces` | 输入 | hard 模式时传 `None`（默认），无需 workspace<br>soft 模式时传 workspace 列表，可包含 GM tensor、UB tile、L1 tile 等<br>列表长度不限，每个元素须为合法的 tensor 或 tile |
| `core_type` | 输入 | `pl.SyncCoreType.MIX`（默认，AIV + AIC 全部核）/ `pl.SyncCoreType.AIV_ONLY`（仅 AIV 核）/ `pl.SyncCoreType.AIC_ONLY`（仅 AIC 核） |
| `mode` | 输入 | `pl.SyncAllMode.HARD`（默认，用 FFTS 硬件信号，无需 workspace）/ `pl.SyncAllMode.SOFT`（用 GM 轮询，需要 workspace）<br>hard 模式性能更优，推荐优先使用 |

## 调用示例

下面是一个完整 kernel：用 `pypto_pro.language.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)` 替代 `sync_src`/`sync_dst` 做全流水同步。`sync_all()` 会等待所有流水线（MTE1/MTE2/MTE3/V/M/FIX）前序操作完成，写法更简洁但同步粒度更粗。

> **注意**：纯 vector kernel 须指定 `core_type=pl.SyncCoreType.AIV_ONLY`，避免同步不存在的 cube 核导致设备错误。默认 `core_type=pl.SyncCoreType.MIX` 适用于同时包含 vector 和 cube 的 kernel。

```python
import pypto_pro.language as pl


@pl.jit()
def sync_all_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000, size=16384)
    tile_b = pl.make_tile(tt, addr=0x4000, size=16384)
    tile_out = pl.make_tile(tt, addr=0x8000, size=16384)
    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.store(out, tile_out, [0, 0])
```

soft 模式需要传入 workspace 列表：

```python
pl.system.sync_all([sync_gm, sync_ub], mode=pl.SyncAllMode.SOFT, core_type=pl.SyncCoreType.AIV_ONLY)
```

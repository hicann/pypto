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

全核同步：hard模式用FFTS硬件信号无需workspace；soft模式用GM轮询需要workspace。

## 函数原型

```python
pypto_pro.language.system.sync_all(workspaces=None, *, core_type=pl.SyncCoreType.MIX, mode=pl.SyncAllMode.HARD)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `workspaces` | 输入 | soft模式使用的workspace列表；hard模式不接受非空列表 |
| `core_type` | 输入 | 同步涉及的核类型 |
| `mode` | 输入 | 同步模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `workspaces` | 输入 | hard模式时传`None`（默认）或空列表。soft模式时必须传非空列表，列表中的元素按类型解释：GM `Tensor`为`gm_workspace`，Vec空间的`Tile`为`ub_workspace`，Mat空间的`Tile`为`l1_workspace`，整数常量或整数标量表达式为可选的`used_cores`（省略或取0表示使用全部核）。每种元素最多出现一次，其他类型或其他本地存储空间不支持。不同`core_type`要求的组合见下表。 |
| `core_type` | 输入 | `pl.SyncCoreType.MIX`（默认，AIV + AIC全部核）/ `pl.SyncCoreType.AIV_ONLY`（仅AIV核）/ `pl.SyncCoreType.AIC_ONLY`（仅AIC核） |
| `mode` | 输入 | `pl.SyncAllMode.HARD`（默认，用FFTS硬件信号，无需workspace）/ `pl.SyncAllMode.SOFT`（用GM轮询，需要workspace）<br>hard模式性能更优，推荐优先使用 |

soft模式的workspace组合如下：

| `core_type` | 必需workspace | 禁止的workspace |
|---|---|---|
| `pl.SyncCoreType.AIV_ONLY` | 1个GM `Tensor` + 1个Vec `Tile` | Mat `Tile` |
| `pl.SyncCoreType.AIC_ONLY` | 1个GM `Tensor` + 1个Mat `Tile` | Vec `Tile` |
| `pl.SyncCoreType.MIX` | 1个GM `Tensor` + 1个Vec `Tile` + 1个Mat `Tile` | 无 |

以上各组合均可再加入至多1个整数`used_cores`。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)`替代`sync_src`/`sync_dst`做全流水同步。`sync_all()`会等待所有流水线（MTE1/MTE2/MTE3/V/M/FIX）前序操作完成，写法更简洁但同步粒度更粗。

> [!CAUTION]注意
> 纯vector kernel须指定`core_type=pl.SyncCoreType.AIV_ONLY`，避免同步不存在的cube核导致设备错误。默认`core_type=pl.SyncCoreType.MIX`适用于同时包含vector和cube的kernel。

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

soft模式需要传入workspace列表：

```python
pl.system.sync_all([sync_gm, sync_ub], mode=pl.SyncAllMode.SOFT, core_type=pl.SyncCoreType.AIV_ONLY)
```

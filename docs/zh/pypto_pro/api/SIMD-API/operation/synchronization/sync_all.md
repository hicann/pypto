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

在多个AIV核、多个AIC核，或AIV与AIC核之间建立核间屏障。参与同步的核到达sync_all后等待，直到本轮所有参与核均已到达，再继续执行。

HARD模式使用FFTS硬件同步，不需要workspace；SOFT模式使用GM共享状态实现同步，需要根据参与同步的核类型提供GM和本地workspace。两种模式的屏障语义相同。

## 函数原型

```python
pypto_pro.language.system.sync_all(
    workspaces: Optional[List] = None,
    *,
    core_type: SyncCoreType = pypto_pro.language.SyncCoreType.MIX,
    mode: SyncAllMode = pypto_pro.language.SyncAllMode.HARD,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| workspaces | 输入 | 可选，SOFT模式使用的workspace列表，根据core_type传入GM Tensor、UB Tile、L1 Tile，以及可选的used_cores。HARD模式不使用workspace，保持默认值None或传空列表。 |
| core_type | 输入 | 可选，[pypto_pro.language.SyncCoreType](../../basic_data_structures/SyncCoreType.md)枚举值，指定参与屏障的核类型，默认为pl.SyncCoreType.MIX。该参数不指定参与核数量。 |
| mode | 输入 | 可选，[pypto_pro.language.SyncAllMode](../../basic_data_structures/SyncAllMode.md)枚举值，指定同步实现模式，默认为pl.SyncAllMode.HARD。 |

### workspaces参数组合

SOFT模式下，workspaces按下表组合。列表元素按类型和Tile所在内存空间识别，因此顺序不影响识别；GM、UB、L1和used_cores每类最多传入1个。

| core_type | workspaces |
|---|---|
| AIV_ONLY | [gm_workspace, ub_workspace]或[gm_workspace, ub_workspace, used_cores] |
| AIC_ONLY | [gm_workspace, l1_workspace]或[gm_workspace, l1_workspace, used_cores] |
| MIX | [gm_workspace, ub_workspace, l1_workspace]或[gm_workspace, ub_workspace, l1_workspace, used_cores] |

- gm_workspace：Tensor类型，存储空间为GM，数据类型为DT_INT32。作为所有参与核共享的同步空间，仅供本组SOFT屏障使用；首次执行Kernel前将其初始化为0。
- ub_workspace：Tile类型，存储空间为UB，数据类型为DT_INT32。作为AIV核本地使用的同步空间，用于AIV_ONLY和MIX。
- l1_workspace：Tile类型，存储空间为L1 Buffer，数据类型为DT_INT32。作为AIC核本地使用的同步空间，用于AIC_ONLY和MIX。
- used_cores：可选的Python int或整数Scalar表达式。AIV_ONLY时表示参与的AIV数量，AIC_ONLY时表示参与的AIC数量，MIX时表示参与的AIC和AIV总数。省略或传0时，根据Kernel的编译和启动配置确定参与核数。

GM workspace按参与核数每核至少预留32字节；UB workspace至少与GM workspace等大；L1 workspace至少预留32字节。例如8个核参与同步，GM和UB workspace分别至少需要8 * 32 = 256字节。GM workspace不得与业务数据或其他同时执行的屏障复用，Kernel执行期间不得由其他逻辑修改。

## 返回值说明

无。

## 约束说明

- 所有参与同步的核必须以相同顺序执行相同次数的sync_all。若循环次数或分支条件不一致，导致部分核少执行或多执行sync_all，可能发生死锁。MIX模式下，AIC侧与AIV侧的调用必须一一对应。
- 纯Vector Kernel使用AIV_ONLY，纯Cube Kernel使用AIC_ONLY。MIX模式要求Cube侧AIC和Vector侧AIV都执行对应的sync_all；只在一侧调用会使另一侧无法到达屏障，导致Kernel超时。
- used_cores小于已启动核数时，只能由选定的used_cores个核调用该SOFT屏障，未参与的核不得进入同一屏障。
- sync_all建立参与核之间的屏障。屏障前后需要跨核读写GM数据时，还需满足相应的数据可见性要求。
- 与set_cross_core/wait_cross_core并用时，两侧的MIX屏障必须位于该SET/WAIT对的同一侧；禁止Cube侧先执行sync_all再SET、Vector侧先WAIT再执行sync_all，否则会形成环形等待。

## 调用示例

下面是纯Vector Kernel展示HARD AIV_ONLY屏障的放置方式。各AIV按核号处理互不重叠的行，TileGroup和auto_mutex负责核内pipe依赖；sync_all位于循环外，使所有参与AIV在本阶段结束后再越过屏障。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP32],
    out: pl.Tensor[[2048, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    input_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    output_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile_x = input_tiles.next()
            tile_out = output_tiles.next()
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        pl.system.sync_all(
            mode=pl.SyncAllMode.HARD,
            core_type=pl.SyncCoreType.AIV_ONLY,
        )
```

该示例使用HARD AIV_ONLY屏障，不需要传入workspaces。

### HARD模式

HARD模式的三种core_type调用方式如下。以下片段分别用于对应类型的Kernel。

```python
# 纯Vector Kernel：所有参与AIV都执行
with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.AIV_ONLY,
    )

# 纯Cube Kernel：所有参与AIC都执行
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.AIC_ONLY,
    )

# Cube、Vector共存的Kernel：AIC和AIV必须到达同一个MIX屏障
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.MIX,
    )

with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.MIX,
    )
```

### SOFT模式

下面示例展示纯Vector Kernel使用SOFT AIV_ONLY屏障的完整构造。sync_gm由调用方分配，并在首次启动Kernel前初始化为0；sync_ub是每个AIV核使用的本地workspace。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def soft_sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP16],
    out: pl.Tensor[[2048, 64], pl.DT_FP16],
    sync_gm: pl.Tensor[[384], pl.DT_INT32],
):
    data_type = pl.TileType(
        shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    data_tiles = pl.make_tile_group(
        type=data_type, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])

    sync_ub_type = pl.TileType(
        shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    sync_ub = pl.make_tile(sync_ub_type, addr=0x3000, size=256)

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile = data_tiles.next()
            pl.load(tile, x, [row, 0])
            pl.store(out, tile, [row, 0])

        pl.system.sync_all(
            [sync_gm, sync_ub],
            mode=pl.SyncAllMode.SOFT,
            core_type=pl.SyncCoreType.AIV_ONLY,
        )
```

SOFT模式的其他core_type按下列方式构造workspace并调用。sync_gm是Kernel的INT32 Tensor入参，并已在首次执行前初始化为0；下面的sync_ub和sync_l1在Kernel内构造。实际使用时，只需要构造当前core_type要求的本地workspace。

```python
# 本地workspace示例：均预留256字节
sync_ub_type = pl.TileType(
    shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
sync_ub = pl.make_tile(sync_ub_type, addr=0x3000, size=256)

sync_l1_type = pl.TileType(
    shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Mat)
sync_l1 = pl.make_tile(sync_l1_type, addr=0x4000, size=256)

# AIV_ONLY：GM + UB，全部AIV参与
pl.system.sync_all(
    [sync_gm, sync_ub],
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.SyncCoreType.AIV_ONLY,
)

# AIV_ONLY：仅前used_cores个AIV进入屏障
if pl.get_block_idx() < used_cores:
    pl.system.sync_all(
        [sync_gm, sync_ub, used_cores],
        mode=pl.SyncAllMode.SOFT,
        core_type=pl.SyncCoreType.AIV_ONLY,
    )

# AIC_ONLY：GM + L1
pl.system.sync_all(
    [sync_gm, sync_l1],
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.SyncCoreType.AIC_ONLY,
)

# MIX：GM + UB + L1，Cube侧和Vector侧使用同一组workspace
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        [sync_gm, sync_ub, sync_l1],
        mode=pl.SyncAllMode.SOFT,
        core_type=pl.SyncCoreType.MIX,
    )

with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        [sync_gm, sync_ub, sync_l1],
        mode=pl.SyncAllMode.SOFT,
        core_type=pl.SyncCoreType.MIX,
    )
```

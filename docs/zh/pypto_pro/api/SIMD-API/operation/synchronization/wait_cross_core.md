# pypto_pro.language.system.wait_cross_core

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

核间同步控制接口，与[pypto_pro.language.system.set_cross_core](set_cross_core.md)配合使用。接口等待并消费指定事件ID对应的同步信号，具体同步机制和使用方法参见pypto_pro.language.system.set_cross_core。

## 函数原型

```python
pypto_pro.language.system.wait_cross_core(
    *,
    pipe: PipeType,
    event_id: Union[int, Scalar],
    sync_mode: CrossCoreSyncMode = pypto_pro.language.CrossCoreSyncMode.INTRA_BLOCK,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| pipe | 输入 | [pypto_pro.language.PipeType](../../basic_data_structures/PipeType.md)枚举值，表示等待期间被阻塞的硬件流水。接口只阻塞该流水中尚未下发的后续指令，已经下发的指令仍可继续执行。等待完成后，该流水才能继续执行后续指令。sync_mode为INTER_BLOCK、INTER_SUBBLOCK或INTRA_BLOCK时，可取M、V、MTE1、MTE2、MTE3、FIX，不支持S和ALL；sync_mode为UNICAST_BLOCK时还可取S，但仍不支持ALL。该值可以与配对的pypto_pro.language.system.set_cross_core的pipe不同。 |
| event_id | 输入 | 核间同步事件ID。支持Python整型常量或运行时整数Scalar表达式。Python整型常量当前只能取0～15。动态表达式须由调用方保证运行时取值合法：INTER_BLOCK、INTER_SUBBLOCK、INTRA_BLOCK取0～15；UNICAST_BLOCK在AIV侧取0～15，在AIC侧取0～31。UNICAST_BLOCK中AIC侧0～15对应AIV0，16～31对应AIV1；AIV侧始终取0～15。除该映射外，配对的SET和WAIT使用相同事件号。[pypto_pro.language.system.sync_all](sync_all.md)内部使用事件ID 11～14，与本接口同时使用时不得将这些ID用于尚未完成的手工核间同步，并应避免与自动流水编排分配的事件ID冲突。 |
| sync_mode | 输入 | 核间同步模式，用于指定参与同步的核以及SET/WAIT信号的配对方式。须与配对的pypto_pro.language.system.set_cross_core使用相同模式，取值参见[pypto_pro.language.CrossCoreSyncMode](CrossCoreSyncMode.md)。 |

## 约束说明

- 必须存在与当前调用匹配的pypto_pro.language.system.set_cross_core，并保证所有参与同步的核均能到达同步点，否则可能发生死锁。多流或多算子并发时，须保证同步所需的核能够同时执行。

## 返回值说明

无返回值。接口等待并消费一个配对信号。

## 调用示例

### INTER_BLOCK

```python
with pl.section_vector():
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=0,
        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
    )
    # 所有AIV均到达同步点后执行的操作。
```

### INTER_SUBBLOCK

```python
with pl.section_vector():
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V,
        event_id=1,
        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
    )
    # 同一AI Core内的AIV0和AIV1均到达后继续。
```

### INTRA_BLOCK

```python
with pl.section_cube():
    # 等待AIV0和AIV1的信号。
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE1,
        event_id=2,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
    )
```

### UNICAST_BLOCK

```python
with pl.section_cube():
    # 仅等待AIV0的信号。
    pl.system.wait_cross_core(
        pipe=pl.PipeType.S,
        event_id=15,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
    )
```

### 完整Kernel示例

```python
import pypto_pro.language as pl


@pl.jit()
def cross_core_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ),
        addr=0x10000, size=16384)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x0000, size=8192)
        tile_y = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x2000, size=8192)
        tile_sum = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x4000, size=8192)
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                        layout=pl.NZ),
            addr=0x6000, size=8448)

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                        layout=pl.NZ),
            addr=0x0000, size=16384)
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                        layout=pl.NZ),
            addr=0x0000, size=16384)
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                        layout=pl.ZN),
            addr=0x0000, size=16384)
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                        layout=pl.NZ, fractal=1024),
            addr=0x0000, size=16384)

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.move(v1_left, v1_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])
```

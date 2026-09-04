# pypto_pro.language.system.sync_src / pypto_pro.language.system.sync_dst

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

AI Core内部的flag式流水同步。sync_src在set_pipe上置位flag，sync_dst使wait_pipe等待同一flag，用于约束两条具体pipe之间的执行顺序。两个接口必须成对使用。

## 函数原型

```python
pypto_pro.language.system.sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: Union[int, Scalar],
) -> None

pypto_pro.language.system.sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| set_pipe | 输入 | pypto_pro.language.PipeType枚举值，置位flag的pipe。可取MTE2（GM→UB/L1搬入）/ V（向量计算）/ MTE3（UB→GM搬出）/ S（标量流水）/ MTE1（L1→L0搬运）/ M（矩阵计算）/ FIX（fixpipe）。必须是一条具体pipe，不允许PipeType.ALL。 |
| wait_pipe | 输入 | pypto_pro.language.PipeType枚举值，等待flag的pipe。取值同set_pipe；必须与set_pipe不同，否则前端报错。 |
| event_id | 输入 | 编译时可确定的整数值，或整数类型的运行时标量表达式。静态ID可以是Python int或结果为整数的常量表达式，不接受bool，取值范围为[0, 7]；动态ID必须是整数类型的Scalar表达式，运行时取值需在[0, 7]范围内。 |

## 返回值说明

无。

## 约束说明

- sync_src与sync_dst必须成对使用，且set_pipe、wait_pipe和event_id必须完全一致，必须先置位、后等待。
- set_pipe和wait_pipe必须组成当前执行侧支持的核内同步路径。
- 同一event ID只能在前一次sync_dst完成等待后复用。过早复用、漏写任一侧或两侧所在控制流路径不一致，都可能造成数据竞争或死锁。
- 与auto_mutex=True并用时，显式flag同步仍会保留；应确保它与自动mutex分别负责明确的依赖，不要为同一依赖重复同步。

### 典型同步模式

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load后V才能计算 | MTE2 | V | 确保GM→UB搬运完成 |
| 计算后MTE3才能store | V | MTE3 | 确保向量计算完成 |
| store后MTE2才能load | MTE3 | MTE2 | 确保UB→GM搬出完成再搬入新数据 |

## 调用示例

下面是一个完整Kernel：从GM载入两个FP32输入，用sync_src/sync_dst约束MTE2（load）→ V（计算）→ MTE3（store）的执行顺序。该示例为纯Vector Kernel，使用sync_src/sync_dst手动同步。

```python
import pypto_pro.language as pl


@pl.jit()
def sync_src_dst_kernel(
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
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```

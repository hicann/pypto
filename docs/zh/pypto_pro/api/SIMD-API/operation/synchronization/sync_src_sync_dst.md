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

flag 式流水同步：置位/等待 flag，必须成对使用，约束 pipe 之间的执行顺序。

## 函数原型

```python
pypto_pro.language.system.sync_src(*, set_pipe, wait_pipe, event_id)
pypto_pro.language.system.sync_dst(*, set_pipe, wait_pipe, event_id)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `set_pipe` | 输入 | 置位 flag 的 pipe |
| `wait_pipe` | 输入 | 等待 flag 的 pipe |
| `event_id` | 输入 | 事件 id |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `set_pipe` | 输入 | `pypto_pro.language.PipeType.MTE2`（GM→UB 搬入）/ `pypto_pro.language.PipeType.V`（向量计算）/ `pypto_pro.language.PipeType.MTE3`（UB→GM 搬出）/ `pypto_pro.language.PipeType.S`（标量流水）/ `pypto_pro.language.PipeType.MTE1`（L1→L0 搬运）/ `pypto_pro.language.PipeType.M`（矩阵计算）/ `pypto_pro.language.PipeType.FIX`（fixpipe） |
| `wait_pipe` | 输入 | 取值同 `set_pipe`<br>须与 `set_pipe` 不同，否则无意义 |
| `event_id` | 输入 | 整型常量（静态）或运行时 Expr（动态）<br>同一对 pipe 之间不同 event_id 互不干扰，可用于区分多步同步 |

## 典型同步模式

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load 后 V 才能计算 | MTE2 | V | 确保 GM→UB 搬运完成 |
| 计算后 MTE3 才能 store | V | MTE3 | 确保向量计算完成 |
| store 后 MTE2 才能 load | MTE3 | MTE2 | 确保 UB→GM 搬出完成再搬入新数据 |

## 调用示例

下面是一个完整 kernel：从 GM 载入两个 FP32 输入，用 `sync_src`/`sync_dst` 约束 MTE2（load）→ V（计算）→ MTE3（store）的执行顺序。纯 vector kernel，同步用 `sync_src`/`sync_dst` 手写。

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

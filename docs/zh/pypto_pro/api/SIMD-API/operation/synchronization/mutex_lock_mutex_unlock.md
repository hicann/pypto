# pypto_pro.language.system.mutex_lock / pypto_pro.language.system.mutex_unlock

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

基于 buffer-id 的互斥加锁 / 解锁，用于 A5 架构上多 pipe 共享缓冲区的安全访问。

## 函数原型

```python
pypto_pro.language.system.mutex_lock(*, pipe, mutex_id, mode=0, max_mutex_id=2, mutex_ids=None)
pypto_pro.language.system.mutex_unlock(*, pipe, mutex_id, mode=0, max_mutex_id=2, mutex_ids=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pipe` | 输入 | 加 / 解锁所在的 pipe |
| `mutex_id` | 输入 | MutexID |
| `mode` | 输入 | 模式属性 |
| `max_mutex_id` | 输入 | 动态 id 时展开上界 |
| `mutex_ids` | 输入 | 动态 id 时 if-chain 的比较目标列表 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pipe` | 输入 | `pypto_pro.language.PipeType.MTE1` / `pypto_pro.language.PipeType.MTE2` / `pypto_pro.language.PipeType.MTE3` 等<br>加锁与解锁须在同一 pipe |
| `mutex_id` | 输入 | 整型常量，取值 0~31<br>同一 pipe 内不同 mutex_id 互不干扰 |
| `mode` | 输入 | 默认 0<br>高级用法，一般场景无需修改 |
| `max_mutex_id` | 输入 | 默认 2<br>仅动态 mutex_id 时生效，静态 id 时忽略 |
| `mutex_ids` | 输入 | 整数列表<br>仅动态 mutex_id 时生效，静态 id 时忽略 |

## 使用说明

手动 `mutex_lock`/`mutex_unlock` 用于 `auto_mutex=False` 场景下手动管理缓冲区互斥。推荐使用 `auto_mutex=True`（配合 `make_tile_group`），由框架自动插入锁，无需手动调用。

## 调用示例

下面是一个完整 kernel：用手动 `mutex_lock`/`mutex_unlock` 保护 load 和 store 的缓冲区访问，替代 `auto_mutex`。mutex 设计用于 `make_tile_group` 缓冲区，每个 buffer 用独立的 mutex_id 加锁。纯 vector kernel，同步用 `sync_src`/`sync_dst` 手写。

```python
import pypto_pro.language as pl


@pl.jit()
def mutex_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()

        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(cur_a, a, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=1)
        pl.load(cur_b, b, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=1)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(cur_out, cur_a, cur_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=2)
        pl.store(out, cur_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=2)
```

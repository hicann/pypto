# pypto_pro.language.ssbuf_store

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

把一个具名 struct 的内容，按字节地址偏移写入共享标量缓冲区（SSBUF），用于在不同 pipe / 不同核之间传递少量元数据（如批次号、块号、地址偏移等）。

SSBUF 是核内的标量便签缓冲区，按字节寻址，不是 tile 内存、不经硬件数据搬运通路，而是由标量（S）流水逐字拷贝。常用于跨核通信：一个核写入，另一些核读取。需配合 [`set_cross_core`/`wait_cross_core`](../synchronization/set_cross_core_wait_cross_core.md) 做同步。

## 函数原型

```python
pypto_pro.language.ssbuf_store(struct_var, offset)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `struct_var` | 输入 | 要写入的具名 struct（由 `pypto_pro.language.struct` 创建） |
| `offset` | 输入 | SSBUF 字节地址偏移 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `struct_var` | 输入 | 须为布局确定的 POD 具名 struct；按 `sizeof(struct)` 逐 u32 字拷贝 |
| `offset` | 输入 | 单位为字节（int 或 Expr），须落在本核 SSBUF 分区内，且不与其他核的并发写入区间重叠 |

## 流水类型

S（标量/系统流水）。

## 调用示例

下面是一个完整 kernel：vector 侧把元数据写入 SSBUF 并发跨核事件，cube 侧等待后读取。`pypto_pro.language.ssbuf_store` 负责写入。

```python
import pypto_pro.language as pl


@pl.jit()
def ssbuf_copy_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    message = pl.struct("Message", batch=0, block=0, offset=0)

    with pl.section_vector():
        message.batch = 8
        message.block = 1
        message.offset = 32768
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf message: batch=%d, block=%d, offset=%d",
                  message.batch, message.block, message.offset)
```

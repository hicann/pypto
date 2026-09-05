# pypto_pro.language.ssbuf_load

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

按字节地址偏移从SSBuffer读取数据并填入具名struct，用于在不同硬件流水或不同计算核之间接收少量元数据，例如批次号、块号、地址偏移等。

## 函数原型

```python
pypto_pro.language.ssbuf_load(
    struct_var: Struct,
    offset: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| struct_var | 输出 | 接收读取结果的参数，Struct类型，须由[pypto_pro.language.struct](../utilities/struct.md)创建。sizeof(struct_var)必须是4的倍数。接口每次读取4字节，共读取sizeof(struct_var) / 4次，不读取末尾不足4字节的数据。发送端和接收端必须使用字段顺序、字段类型、数组长度及C++对齐方式完全一致的struct定义。 |
| offset | 输入 | 起始字节地址偏移，int或Scalar类型，读取区间为[offset, offset + sizeof(struct_var))。取值必须非负并按4字节对齐。读取区间必须位于目标平台的有效SSBuffer地址范围内；PyPTO不对offset执行越界检查，开发者需要根据目标平台和Kernel通信方案规划地址。 |

## 约束说明

- pypto_pro.language.ssbuf_load本身不等待发送端完成写入。跨硬件流水或跨计算核通信时，接收端必须先调用与发送端匹配的[pypto_pro.language.system.wait_cross_core](../synchronization/wait_cross_core.md)，再调用pypto_pro.language.ssbuf_load读取数据。
- 读取期间不得有其他执行单元改写同一区间，否则读取结果无法保证一致。

## 返回值说明

无。

## 调用示例

### 跨核传递元数据

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
            pl.system.set_cross_core(
                pipe=pl.PipeType.S,
                event_id=15,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
            )

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf message: batch=%d, block=%d, offset=%d",
                  message.batch, message.block, message.offset)
```

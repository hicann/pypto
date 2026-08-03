# vf.reduce_max

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

向量寄存器最大值归约：将源寄存器中的所有有效元素（mask选中的元素）求最大值，结果写入目标寄存器的第一个元素`dst[0]`，第一个最大值所在索引写入`dst[1]`，其余元素置零（`ZEROING`模式）或保留原值（`MERGING`模式）。对应硬件`vcmax`指令（AscendC ReduceMax）。

当`datablock=True`时，启用datablock粒度归约（对应`vcgmax`指令，AscendC ReduceMaxDatablock），每个datablock独立归约：b32类型每16个元素为一个datablock，b16类型每32个元素为一个datablock，各datablock分别求最大值并将结果写入各自datablock的第一个元素。

必须在`@pl.vector_function`函数内使用。

## 函数原型

```python
dst = vf.reduce_max(src, preg, *, datablock=False, merge_mode=pl.MergeMode.ZEROING)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器，归约结果写入第一个元素`dst[0]`，索引写入`dst[1]` |
| `src` | 输入 | 源向量寄存器 |
| `preg` | 输入 | 掩码寄存器 |
| `datablock` | 输入 | 可选，``True``时按datablock粒度归约（对应`vcgmax`指令），默认``False`` |
| `merge_mode` | 输入 | 可选，合并模式：``pl.MergeMode.ZEROING``（默认）或``pl.MergeMode.MERGING`` |

## 数据类型

**普通归约（`datablock=False`，对应`vcmax`）**：

| src | dst |
|---|---|
| FP16 | FP16 |
| FP32 | FP32 |
| INT16 | INT16 |
| UINT16 | UINT16 |
| INT32 | INT32 |
| UINT32 | UINT32 |
| INT8 | INT8 |
| UINT8 | UINT8 |

**Datablock归约（`datablock=True`，对应`vcgmax`）**：

| src | dst |
|---|---|
| FP16 | FP16 |
| FP32 | FP32 |
| INT32 | INT32 |
| UINT32 | UINT32 |

## 返回值说明

返回目标向量寄存器（`RegTensor`类型）。最大值存储在`dst[0]`，第一个最大值元素的索引存储在`dst[1]`，其余元素根据`merge_mode`置零或保留原值。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。
- 源操作数与目标操作数的数据类型需要保持一致。
- 当所有元素均不参与计算时（mask为空），将该数据类型的最小值写入`dst[0]`。
- 当存在多个最大值时，将第一个最大值的索引保存在`dst[1]`中。
- `max(-0, +0) = +0`。
- 如果输入数据存在nan，将该数据类型的nan写入`dst[0]`，并将第一个nan的索引保存在`dst[1]`中。
- datablock归约模式（`datablock=True`）仅支持b16/b32宽度的数据类型（FP16/FP32/INT32/UINT32），不支持b8类型。

## 关键特性

**索引值需要强制类型转换**

dstReg的最大值索引按照dstReg的数据类型存储，比如dstReg为half类型时，索引按照half类型存储，因此读取索引需要使用reinterpret_cast方法转换到整数类型。若数据类型是half，需要使用reinterpret_cast<uint16_t*>；若数据类型是float，需要使用reinterpret_cast<uint32_t*>。

归约求最大值的计算过程及索引保存方式如下图所示：

**图1**ReduceMax归约索引示意图

![reg_reduce_index示意图](../../../../figures/reg_reduce_index.jpg)

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src0 = vf.load_align(src_tile, 0)
    max0 = vf.reduce_max(src0, preg_all)
    vf.store_align(dst_tile, max0, preg_all)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    t_out = pl.make_tile(tf, addr=256, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[0, 0], torch.max(a), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

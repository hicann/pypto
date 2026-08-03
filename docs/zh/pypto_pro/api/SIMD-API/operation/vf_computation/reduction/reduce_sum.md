# vf.reduce_sum

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

向量寄存器求和归约：将源寄存器中的所有有效元素（mask选中的元素）求和，结果写入目标寄存器的第一个元素（`dst[0]`），其余元素置零（`ZEROING`模式）或保留原值（`MERGING`模式）。对应硬件`vcadd`指令（AscendC ReduceSum）。

当`datablock=True`时，启用datablock粒度归约（对应`vcgadd`指令，AscendC ReduceSumDatablock），每个datablock独立归约：b32类型每16个元素为一个datablock，b16类型每32个元素为一个datablock，各datablock分别求和并将结果写入各自datablock的第一个元素。

必须在`@pl.vector_function`函数内使用。

## 函数原型

```python
dst = vf.reduce_sum(src, preg, *, datablock=False, merge_mode=pl.MergeMode.ZEROING)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器，归约结果写入第一个元素 |
| `src` | 输入 | 源向量寄存器 |
| `preg` | 输入 | 掩码寄存器 |
| `datablock` | 输入 | 可选，``True``时按datablock粒度归约（对应`vcgadd`指令），默认``False`` |
| `merge_mode` | 输入 | 可选，合并模式：``pl.MergeMode.ZEROING``（默认）或``pl.MergeMode.MERGING`` |

## 数据类型

**普通归约（`datablock=False`，对应`vcadd`）**：

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

**Datablock归约（`datablock=True`，对应`vcgadd`）**：

| src | dst |
|---|---|
| FP16 | FP16 |
| FP32 | FP32 |
| INT32 | INT32 |
| UINT32 | UINT32 |

## 返回值说明

返回目标向量寄存器（`RegTensor`类型），归约结果存储在第一个元素`dst[0]`中，其余元素根据`merge_mode`置零或保留原值。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。
- 源操作数与目标操作数的数据类型需要保持一致。
- 当所有元素均不参与计算时（mask为空），将目的操作数数据类型的0写入dstReg。
- 指令内累加顺序采用二叉树累加方式，结果具有确定性。
- 当源操作数数据类型为FP16时，中间累加过程在FP32精度下进行，最终结果再舍入为FP16，因此精度高于先逐对`vf.add`再手动归约的写法。
- datablock归约模式（`datablock=True`）仅支持b16/b32宽度的数据类型（FP16/FP32/INT32/UINT32），不支持b8类型。

## 关键特性

**ReduceSum累加顺序**

以二叉树累加的方式计算源操作数srcReg内有效元素的数据总和。以half类型的数据求和为例，在srcReg内有128个数，通过二叉树的方式，两两相加，最终得到目的操作数为1个half类型的数据sum，计算过程如下图所示：

**图1**ReduceSum累加顺序

![ReduceSum累加顺序](../../../../figures/reduce_sum_accum_order.jpg)

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
    sum0 = vf.reduce_sum(src0, preg_all)
    vf.store_align(dst_tile, sum0, preg_all)


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
    torch.testing.assert_close(out[0, 0], a.sum(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

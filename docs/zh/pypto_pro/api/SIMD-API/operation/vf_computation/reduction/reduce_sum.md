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

向量寄存器求和归约：将源寄存器中的所有有效元素求和，结果写入目标寄存器的第一个元素。

必须在 `@pl.vector_function` 函数内使用。

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
| `datablock` | 输入 | 可选，``True`` 时按 datablock 粒度归约，默认 ``False`` |
| `merge_mode` | 输入 | 可选，合并模式：``pl.MergeMode.ZEROING``（默认）或 ``pl.MergeMode.MERGING`` |

## 数据类型

| src | dst |
|---|---|
| FP16 | FP16 |
| FP32 | FP32 |
| BF16 | BF16 |
| INT32 | INT32 |
| UINT32 | UINT32 |

## 返回值说明

返回目标向量寄存器（`RegTensor` 类型）。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。
- 源操作数与目标操作数的数据类型需要保持一致。
- 当所有元素均不参与计算时（mask 为空），将目的操作数数据类型的 0 写入 dstReg。
- 指令内累加顺序采用二叉树累加方式。

## 关键特性

**ReduceSum 累加顺序**

以二叉树累加的方式计算源操作数 srcReg 内有效元素的数据总和。以 half 类型的数据求和为例，在 srcReg 内有 128 个数，通过二叉树的方式，两两相加，最终得到目的操作数为 1 个 half 类型的数据 sum，计算过程如下图所示：

**图 1** ReduceSum 累加顺序

![ReduceSum累加顺序](../../../../figures/reduce_sum_accum_order.jpg)

## 调用示例

```python
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
    device = "npu:0"
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

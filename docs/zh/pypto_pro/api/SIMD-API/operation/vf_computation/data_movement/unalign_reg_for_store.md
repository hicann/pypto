# vf.unalign_reg_for_store

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

为非对齐存储分配 alignment tracker 寄存器。该寄存器贯穿后续的 `vf.store_unalign` / `vf.store_unalign_post` 调用链，用于追踪未对齐字节的累积状态。

## 函数原型

```python
align_reg = vf.unalign_reg_for_store()
```

## 参数说明

无参数。

## 数据类型

不涉及数据类型。

## 返回值说明

返回 UnalignReg 类型，供 `vf.store_unalign` 和 `vf.store_unalign_post` 使用。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 后续 `vf.store_unalign` 和 `vf.store_unalign_post` 必须使用带步长形式（4 参数 / 3 参数），无步长 legacy 形式在当前硬件上可能导致挂死。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 分配 alignment tracker 寄存器，贯穿后续非对齐存储
    align_reg = vf.unalign_reg_for_store()
    reg = vf.load_align(src_tile, 0)
    # 4 参数形式（带步长）：stride=64 存储 64 个 FP32 元素，post_update=True
    vf.store_unalign(dst_tile, reg, align_reg, 64, post_update=True)
    # 3 参数形式（带步长）：stride=0，post_update=True
    vf.store_unalign_post(dst_tile, align_reg, 0, post_update=True)


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
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

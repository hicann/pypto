# pl.get_spr

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

读取指定特殊寄存器的值，返回标量值。当前仅支持 AR 寄存器（`get_ar` 指令）。

AR 寄存器是一个特殊的地址寄存器，通常配合 `vf.squeeze` 使用——`vf.squeeze` 会将有效元素的总字节数存储到 AR 寄存器中，随后可通过 `pl.get_spr` 读取该值。

## 函数原型

```python
ar_value = pl.get_spr()
```

## 参数说明

无参数。当前仅支持读取 AR 寄存器。

## 数据类型

返回 int64_t 类型的标量值。

## 返回值说明

返回 `int64_t` 类型的标量值，为 AR 寄存器中的数值。

## 约束说明

- 当前仅支持读取 AR 寄存器，不支持其他特殊寄存器。
- AR 寄存器的值由 `vf.squeeze` 写入，需在调用 `pl.get_spr` 之前先执行 `vf.squeeze`。
- `get_ar()` 为 `__aicore__` 指令，不能在 `@pl.vector_function` 函数体内使用。应在 `@pl.jit` kernel 的非 VF 区域调用。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # Squeeze 会将有效元素字节数写入 AR 寄存器
    reg_sq = vf.squeeze(reg, preg)
    vf.store_align(dst_tile, reg_sq, preg)


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
    # 在 kernel 的非 VF 区域读取 AR 寄存器
    ar_value = pl.get_spr()


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

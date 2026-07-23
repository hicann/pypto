# vf.leaky_relu

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

该接口用于按元素执行Leaky ReLU(Leaky Rectified Linear Unit)操作：当源操作数中某元素大于等于0时，直接将该元素写入目的操作数；否则将该元素乘以标量值scalarValue后写入目的操作数。计算公式如下：

$$dstReg_i = \begin{cases} srcReg_i & srcReg_i \geq 0 \\ srcReg_i \times scalarValue & srcReg_i < 0 \end{cases}$$

## 函数原型

```python
dst = vf.leaky_relu(src, scalar, preg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器，向量寄存器 |
| `src` | 输入 | 源操作数，向量寄存器。源操作数的数据类型需要与目的操作数保持一致。 |
| `scalar` | 输入 | 标量源操作数，类型为标量，数据类型需与目的操作数保持一致。 |
| `preg` | 输入 | 掩码寄存器，类型为 `MaskReg` |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：FP16、FP32。

## 返回值说明

返回目标向量寄存器（`RegTensor` 类型）。

## 约束说明

无

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.leaky_relu(reg_a, 0.01, preg)
    vf.store_align(dst_tile, reg_out, preg)


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
    expected = torch.nn.functional.leaky_relu(a, 0.01)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

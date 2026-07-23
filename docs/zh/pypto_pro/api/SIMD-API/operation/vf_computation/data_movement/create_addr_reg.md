# vf.create_addr_reg

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

`vf.create_addr_reg` 用于创建地址偏移量寄存器（AddrReg），在多维循环中逐层累加地址偏移。AddrReg 可作为 `vf.load_align` 和 `vf.store_align` 的地址偏移参数，替代直接传入整数偏移量。

偏移量计算公式为 `offset = index0 * stride0 + index1 * stride1 + ...`，支持 1-4 层循环轴。在循环中，index 每次递增 1，AddrReg 的偏移量自动增加对应的 stride。

## 函数原型

```python
# 1 层循环: offset = index0 * stride0
a_reg = vf.create_addr_reg(index0, stride0, *, dtype=pl.DT_FP32)

# 2 层循环: offset = index0 * stride0 + index1 * stride1
a_reg = vf.create_addr_reg(index0, stride0, index1, stride1, *, dtype=pl.DT_FP32)

# 3 层循环: offset = index0 * stride0 + index1 * stride1 + index2 * stride2
a_reg = vf.create_addr_reg(index0, stride0, index1, stride1, index2, stride2, *, dtype=pl.DT_FP32)

# 4 层循环: offset = index0 * stride0 + index1 * stride1 + index2 * stride2 + index3 * stride3
a_reg = vf.create_addr_reg(index0, stride0, index1, stride1, index2, stride2, index3, stride3, *, dtype=pl.DT_FP32)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `a_reg` | 输出 | AddrReg 地址偏移量寄存器 |
| `index0` | 输入 | 最外层循环轴索引（循环变量） |
| `stride0` | 输入 | 最外层循环轴对应的地址偏移量，单位为元素个数 |
| `index1` | 输入 | 可选，第二层循环轴索引 |
| `stride1` | 输入 | 可选，第二层循环轴对应的地址偏移量 |
| `index2` | 输入 | 可选，第三层循环轴索引 |
| `stride2` | 输入 | 可选，第三层循环轴对应的地址偏移量 |
| `index3` | 输入 | 可选，第四层循环轴索引 |
| `stride3` | 输入 | 可选，第四层循环轴对应的地址偏移量 |
| `dtype` | 输入 | 可选，模板参数对应的数据类型（默认 `pl.DT_FP32`）。决定元素宽度：b8/b16/b32/b64 |

## 数据类型

| dtype | 元素宽度 |
|---|---|
| b8 | 1 字节 |
| b16 | 2 字节 |
| b32 | 4 字节 |
| b64 | 8 字节 |

## 返回值说明

返回 AddrReg 类型

## 约束说明

- AddrReg 数量上限为 8。
- 由于硬件循环（HardwareLoop）限制，AddrReg 最多支持 4 层循环轴。
- AddrReg 仅支持 `vf.load_align` 和 `vf.store_align` 搬运指令使用。
- 通过 AddrReg 设置地址偏移进行搬运时，需要满足对应搬运指令的地址对齐约束。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_repeat_size = 64
    repeat_times = 2
    for i in pl.range(0, repeat_times, 1):
        # offset = i * one_repeat_size
        a_reg = vf.create_addr_reg(i, one_repeat_size, dtype=pl.DT_FP32)
        reg = vf.load_align(src_tile, a_reg)
        vf.store_align(dst_tile, reg, preg, a_reg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=512)
    t_out = pl.make_tile(tf, addr=512, size=512)
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
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

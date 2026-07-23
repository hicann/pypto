# vf.shift_left

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

`shift_left`指令根据mask对源操作数src进行左移操作，将结果写入目的操作数dst。移位量`shift`既可以是**标量**（所有元素移动相同位数），也可以是**向量寄存器**（每个元素按对应lane的位数移动）。接口会根据`shift`参数的类型自动选择：

- `shift`为标量（整型字面量或标量变量）：所有元素统一左移。
- `shift`为向量寄存器：逐元素左移。

> 说明：原标量版接口`vf.shift_lefts`已合并进`vf.shift_left`，无需再区分接口名，直接由`shift`参数类型分派。

根据源操作数的数据类型，左移操作分为以下两种情况：

- **数据类型为无符号类型：执行逻辑左移。**

  逻辑左移会将二进制数整体向左移动指定的位数，最高位被丢弃，最低位用0填充。例如，二进制数1010101010101010（UINT16类型）逻辑左移1位后，结果为0101010101010100。
- **数据类型为有符号类型：执行算术左移。**

  算术左移会将二进制数整体向左移动指定的位数，次高位被丢弃，最低位用0填充。例如，二进制数1010101010101010（INT16类型）算术左移1位后，结果为1101010101010100；算术左移3位后，结果为1101010101010000。

$$
dst_i = src_i \ll shift_i
$$

## 函数原型

```python
# shift 为标量：所有元素统一左移
dst = vf.shift_left(src, shift_bits, preg)

# shift 为向量寄存器：逐元素左移
dst = vf.shift_left(src, shift_reg, preg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器 |
| `src` | 输入 | 源操作数 |
| `shift` | 输入 | 左移位数。标量（整型，所有元素统一移位）或向量寄存器（逐元素移位）。不支持设置为负数，负数行为未定义 |
| `preg` | 输入 | 掩码寄存器 |

## 数据类型

`shift`为向量寄存器时：

| dst | src | shift（寄存器） |
|---|---|---|
| INT8 | INT8 | INT8 |
| UINT8 | UINT8 | INT8 |
| INT16 | INT16 | INT16 |
| UINT16 | UINT16 | INT16 |
| INT32 | INT32 | INT32 |
| UINT32 | UINT32 | INT32 |
| INT64 | INT64 | INT64 |
| UINT64 | UINT64 | INT64 |

`shift`为标量时，src/dst 支持 INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64，移位量为标量整型。

## 返回值说明

返回目标向量寄存器（`RegTensor` 类型）。

## 约束说明

- 对于逻辑位移（无符号数据类型），如果位移量大于数据类型位宽，则输出为0。
- 对于算术位移（有符号数据类型），如果位移量大于数据类型位宽，则输出0。
- 移位量不支持设置为负数，负数行为未定义。

## 调用示例

标量移位（所有元素统一左移）：

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_scalar(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_out = vf.shift_left(reg_src, 4, preg)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    t_out = pl.make_tile(tf, addr=256, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_scalar(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

向量移位（逐元素左移）：

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_vector(src_tile, shift_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_shift = vf.load_align(shift_tile, 0)
    reg_out = vf.shift_left(reg_src, reg_shift, preg)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel_vector(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    shift: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf_u32 = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tf_i32 = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf_u32, addr=0, size=256)
    in_shift = pl.make_tile(tf_i32, addr=256, size=256)
    t_out = pl.make_tile(tf_u32, addr=512, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_shift, shift, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_vector(in_a, in_shift, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_2():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    shift = torch.full([1, 64], 4, device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_vector[None, core_nums](a, shift, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)


if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
